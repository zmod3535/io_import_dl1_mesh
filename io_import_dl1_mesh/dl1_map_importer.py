import os
import re
import struct
import bpy
import mathutils
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, IntProperty
from . import dl1_parser, dl1_mesh_builder


def _parse_sobj_file(filepath: str):
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 0x58 or data[:4] != b'SO18':
        raise RuntimeError("Неверный формат файла: ожидается заголовок SO18 (.sobj)")

    defs_offset, inst_offset, sec3_offset = struct.unpack_from('<3I', data, 0x04)
    inst_count, _, defs_count = struct.unpack_from('<3I', data, 0x1C)

    # 1. Читаем словарь префабов/мешей (с defs_offset по inst_offset)
    pos = defs_offset
    mesh_defs = list()
    for _ in range(defs_count):
        if pos + 2 > inst_offset:
            break
        m_len = struct.unpack_from('<H', data, pos)[0]
        pos += 2
        m_name = data[pos:pos + m_len].decode('latin1', errors='ignore').strip('\x00')
        pos += m_len

        s_len = struct.unpack_from('<H', data, pos)[0]
        pos += 2
        s_name = data[pos:pos + s_len].decode('latin1', errors='ignore').strip('\x00')
        pos += s_len

        # 28 байт служебных данных (флаги, хеш, 0xFFFFFFFF)
        pos += 28

        n_len = struct.unpack_from('<H', data, pos)[0]
        pos += 2
        n_name = data[pos:pos + n_len].decode('latin1', errors='ignore').strip('\x00')
        pos += n_len

        # 2 байта выравнивания/флагов в конце записи
        pos += 2

        mesh_defs.append({
            'mesh_name': os.path.basename(m_name.replace('\\', '/')),
            'skin_name': s_name,
            'node_name': n_name,
        })

    # 2. Читаем массив инстансов (каждый инстанс строго 48 байт)
    instances = list()
    stride = 48
    for i in range(inst_count):
        base = inst_offset + i * stride
        if base + stride > len(data):
            break

        px, py, pz = struct.unpack_from('<3f', data, base)
        ex, ey, ez = struct.unpack_from('<3f', data, base + 12)
        qx, qy, qz, qw = struct.unpack_from('<4h', data, base + 24)
        flags, mesh_idx = struct.unpack_from('<2H', data, base + 32)

        instances.append({
            'pos': (px, py, pz),
            'ext': (ex, ey, ez),
            'quat': (qx / 32767.0, qy / 32767.0, qz / 32767.0, qw / 32767.0),
            'flags': flags,
            'mesh_idx': mesh_idx,
        })

    return mesh_defs, instances


def _index_mesh_folder(mesh_dir: str) -> dict:
    """Индексирует папку с мешами: сопоставляет имя .msh (в нижнем регистре) с путём к файлу или _part00.bin"""
    index = dict()
    if not mesh_dir or not os.path.isdir(mesh_dir):
        return index

    for root, _, files in os.walk(mesh_dir):
        for fname in files:
            low = fname.lower()
            full_path = os.path.join(root, fname)

            # Вариант 1: цельный .msh
            if low.endswith('.msh') and '_part' not in low:
                index[low] = full_path
                continue

            # Вариант 2: дампы RP6_UnpackCLI вида "1524_int_garage_wall_a_part00.bin" или "int_garage_wall_a_part03.bin"
            m = re.match(r'^(?:\d+_)?(.+?)_part_?(\d+)\.(bin|msh)$', low)
            if m:
                clean_msh = m.group(1) + '.msh'
                part_num = int(m.group(2))
                if clean_msh not in index or part_num == 3:
                    index[clean_msh] = full_path

    return index


class IMPORT_OT_dl1_map(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.dl1_sobj_map"
    bl_label = "Import Dying Light 1 Map (.sobj)"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(default="*.sobj", options={'HIDDEN'})

    meshes_folder: StringProperty(
        name="Meshes Folder",
        description="Папка с распакованными мешами (.msh или _part00..04.bin)",
        subtype='DIR_PATH',
        default="",
    )
    convert_z_up: BoolProperty(
        name="Convert Y-up to Blender Z-up",
        description="Преобразовать координаты карты и мешей из Y-up в Z-up",
        default=True,
    )
    split_by_material: BoolProperty(
        name="Split by Material",
        description="Разделять каждый префаб на сабмеши по материалам",
        default=True,
    )
    place_placeholders: BoolProperty(
        name="Create Empties for Missing Meshes",
        description="Ставить Empty-точки на месте мешей, не найденных в папке",
        default=True,
    )
    max_instances: IntProperty(
        name="Max Instances (0 = All)",
        description="Ограничить количество импортируемых объектов для быстрой проверки",
        default=0,
        min=0,
    )

    def execute(self, context):
        if not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "Файл .sobj не найден")
            return {'CANCELLED'}

        try:
            mesh_defs, instances = _parse_sobj_file(self.filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка чтения .sobj: {e}")
            return {'CANCELLED'}

        # Если папка с мешами не задана вручную, пробуем искать рядом с картой или на уровень выше
        search_dir = self.meshes_folder
        if not search_dir or not os.path.isdir(search_dir):
            search_dir = os.path.dirname(self.filepath)

        mesh_index = _index_mesh_folder(search_dir)

        map_name = os.path.splitext(os.path.basename(self.filepath))[0]
        map_col = bpy.data.collections.new(f"Map_{map_name}")
        context.scene.collection.children.link(map_col)

        # Кэш загруженных мешей по индексу префаба (чтобы не грузить одну стену 500 раз)
        prefab_cache = dict()
        missing_defs = set()

        limit = len(instances) if self.max_instances <= 0 else min(len(instances), self.max_instances)
        placed_count = 0

        for idx in range(limit):
            inst = instances[idx]
            m_idx = inst['mesh_idx']
            if m_idx >= len(mesh_defs):
                continue

            m_def = mesh_defs[m_idx]
            m_key = m_def['mesh_name'].lower()

            if m_idx not in prefab_cache:
                src_path = mesh_index.get(m_key)
                if src_path and os.path.isfile(src_path):
                    try:
                        bundle = dl1_parser.load_dl1_bundle(src_path)
                        built_meshes, _ = dl1_mesh_builder.build_blender_meshes(
                            bundle,
                            convert_z_up=self.convert_z_up,
                            split_by_material=self.split_by_material,
                        )
                        prefab_cache[m_idx] = built_meshes
                    except Exception as e:
                        print(f"[DL1 Map] Ошибка загрузки меша '{m_def['mesh_name']}': {e}")
                        prefab_cache[m_idx] = None
                        missing_defs.add(m_def['mesh_name'])
                else:
                    prefab_cache[m_idx] = None
                    missing_defs.add(m_def['mesh_name'])

            cached_meshes = prefab_cache[m_idx]

            px, py, pz = inst['pos']
            qx, qy, qz, qw = inst['quat']

            if self.convert_z_up:
                loc = mathutils.Vector((px, pz, py))
                rot = mathutils.Quaternion((qw, -qx, -qz, -qy)).normalized()
            else:
                loc = mathutils.Vector((px, py, pz))
                rot = mathutils.Quaternion((qw, qx, qy, qz)).normalized()

            base_obj_name = f"{m_def['node_name']}_{idx:04d}"

            if cached_meshes:
                for sub_i, b_mesh in enumerate(cached_meshes):
                    obj_name = base_obj_name if len(cached_meshes) == 1 else f"{base_obj_name}_sub{sub_i}"
                    obj = bpy.data.objects.new(obj_name, b_mesh)
                    obj.location = loc
                    obj.rotation_mode = 'QUATERNION'
                    obj.rotation_quaternion = rot
                    map_col.objects.link(obj)
                placed_count += 1
            elif self.place_placeholders:
                empty = bpy.data.objects.new(f"MISSING_{base_obj_name}", None)
                empty.empty_display_type = 'PLAIN_AXES'
                empty.empty_display_size = 0.5
                empty.location = loc
                empty.rotation_mode = 'QUATERNION'
                empty.rotation_quaternion = rot
                map_col.objects.link(empty)
                placed_count += 1

        msg = (
            f"Карта '{map_name}': расставлено {placed_count} объектов | "
            f"Найдено префабов: {sum(1 for v in prefab_cache.values() if v)}/{len(prefab_cache)}"
        )
        if missing_defs:
            print(f"[DL1 Map] Не найдены меши ({len(missing_defs)}): {', '.join(sorted(missing_defs)[:20])}")

        self.report({'INFO'}, msg)
        return {'FINISHED'}
