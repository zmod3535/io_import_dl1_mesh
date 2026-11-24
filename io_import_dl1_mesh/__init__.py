bl_info = {
    "name": "Dying Light 1 Mesh & Map Importer (.bin / .msh / .sobj)",
    "author": "Reverse Engineering Assistant",
    "version": (1, 3, 0),
    "blender": (5, 1, 0),
    "location": "File > Import > Dying Light 1",
    "description": "Imports Dying Light 1 meshes (_part00..04, .msh) and static map layouts (.sobj)",
    "category": "Import-Export",
}

import os
import re
import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, CollectionProperty
from . import dl1_parser, dl1_mesh_builder, dl1_map_importer


def _get_bundle_key(filepath: str) -> str:
    folder = os.path.dirname(filepath)
    fname = os.path.basename(filepath)
    m = re.match(r'^(.+)_part_?(\d+)\.bin$', fname, flags=re.IGNORECASE)
    base = m.group(1) if m else os.path.splitext(fname)[0]
    return os.path.normcase(os.path.join(folder, base))


class IMPORT_OT_dl1_mesh(bpy.types.Operator, ImportHelper):
    bl_idname = "import_mesh.dl1_bin"
    bl_label = "Import Dying Light 1 Mesh(es)"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(default="*.bin;*.msh", options={'HIDDEN'})

    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
    )
    directory: StringProperty(subtype='DIR_PATH')

    convert_z_up: BoolProperty(
        name="Convert Y-up to Blender Z-up",
        description="Преобразовать координаты из Chrome Engine (Y-up) в Blender (Z-up)",
        default=True,
    )
    split_by_material: BoolProperty(
        name="Split by Material",
        description="Разделять каждую модель на отдельные объекты по материалам",
        default=True,
    )

    def execute(self, context):
        target_files = list()
        if self.files:
            for f in self.files:
                if f.name:
                    target_files.append(os.path.join(self.directory, f.name))
        if not target_files and self.filepath:
            target_files.append(self.filepath)

        if not target_files:
            self.report({'ERROR'}, "Файлы не выбраны")
            return {'CANCELLED'}

        unique_paths = list()
        seen_keys = set()
        for path in target_files:
            if not os.path.isfile(path):
                continue
            key = _get_bundle_key(path)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_paths.append(path)

        imported_models = 0
        failed_models = 0
        total_verts = 0
        total_polys = 0
        all_created_objs = list()

        for path in unique_paths:
            try:
                bundle = dl1_parser.load_dl1_bundle(path)
                meshes, layout = dl1_mesh_builder.build_blender_meshes(
                    bundle,
                    convert_z_up=self.convert_z_up,
                    split_by_material=self.split_by_material,
                )

                if self.split_by_material and len(meshes) > 1:
                    target_col = bpy.data.collections.new(bundle['name'])
                    context.scene.collection.children.link(target_col)
                else:
                    target_col = context.collection

                for mesh in meshes:
                    obj = bpy.data.objects.new(mesh.name, mesh)
                    target_col.objects.link(obj)
                    all_created_objs.append(obj)
                    total_verts += len(mesh.vertices)
                    total_polys += len(mesh.polygons)

                imported_models += 1

            except Exception as e:
                failed_models += 1
                print(f"[DL1 Importer] Пропущен файл '{os.path.basename(path)}': {e}")

        if imported_models == 0:
            self.report({'ERROR'}, "Не удалось импортировать ни одной модели (подробности в System Console)")
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')
        for obj in all_created_objs:
            obj.select_set(True)
        if all_created_objs:
            context.view_layer.objects.active = all_created_objs[0]

        msg = (
            f"Загружено моделей: {imported_models} | Объектов: {len(all_created_objs)} | "
            f"Вершин: {total_verts} | Полигонов: {total_polys}"
        )
        if failed_models > 0:
            msg += f" | Пропущено: {failed_models}"

        self.report({'INFO'}, msg)
        return {'FINISHED'}


def menu_func_import_mesh(self, context):
    self.layout.operator(IMPORT_OT_dl1_mesh.bl_idname, text="Dying Light 1 Mesh (.bin / .msh)")


def menu_func_import_map(self, context):
    self.layout.operator(dl1_map_importer.IMPORT_OT_dl1_map.bl_idname, text="Dying Light 1 Map (.sobj)")


def register():
    bpy.utils.register_class(IMPORT_OT_dl1_mesh)
    bpy.utils.register_class(dl1_map_importer.IMPORT_OT_dl1_map)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_mesh)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_map)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_map)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_mesh)
    bpy.utils.unregister_class(dl1_map_importer.IMPORT_OT_dl1_map)
    bpy.utils.unregister_class(IMPORT_OT_dl1_mesh)


if __name__ == "__main__":
    register()
