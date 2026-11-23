bl_info = {
    "name": "Dying Light 1 Mesh/Part Importer (.bin / .msh)",
    "author": "Reverse Engineering Assistant",
    "version": (1, 1, 0),
    "blender": (5, 1, 0),
    "location": "File > Import > Dying Light 1 (.bin, .msh)",
    "description": "Imports Dying Light 1 segmented meshes (_part00..04) and single .msh/.bin files",
    "category": "Import-Export",
}

import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty
from . import dl1_parser, dl1_mesh_builder


class IMPORT_OT_dl1_mesh(bpy.types.Operator, ImportHelper):
    bl_idname = "import_mesh.dl1_bin"
    bl_label = "Import Dying Light 1 Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(default="*.bin;*.msh", options={'HIDDEN'})
    convert_z_up: BoolProperty(
        name="Convert Y-up to Blender Z-up",
        description="Преобразовать координаты из Chrome Engine (Y-up) в Blender (Z-up)",
        default=True,
    )
    split_by_material: BoolProperty(
        name="Split by Material",
        description="Разделять модель на отдельные объекты по каждому материалу",
        default=True,
    )

    def execute(self, context):
        try:
            bundle = dl1_parser.load_dl1_bundle(self.filepath)
            meshes, layout = dl1_mesh_builder.build_blender_meshes(
                bundle,
                convert_z_up=self.convert_z_up,
                split_by_material=self.split_by_material,
            )

            # Если разделено на несколько частей, создаём для модели отдельную коллекцию
            if self.split_by_material and len(meshes) > 1:
                target_col = bpy.data.collections.new(bundle['name'])
                context.scene.collection.children.link(target_col)
            else:
                target_col = context.collection

            total_verts = 0
            total_polys = 0
            created_objs = []

            for mesh in meshes:
                obj = bpy.data.objects.new(mesh.name, mesh)
                target_col.objects.link(obj)
                created_objs.append(obj)
                total_verts += len(mesh.vertices)
                total_polys += len(mesh.polygons)

            for obj in created_objs:
                obj.select_set(True)
            if created_objs:
                context.view_layer.objects.active = created_objs[0]

            self.report(
                {'INFO'},
                f"Импортировано: {bundle['name']} | Частей: {len(meshes)} | "
                f"Вершин: {total_verts} | Полигонов: {total_polys} | Stride: {layout['stride']}"
            )
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Ошибка импорта DL1: {e}")
            return {'CANCELLED'}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_dl1_mesh.bl_idname, text="Dying Light 1 Mesh (.bin / .msh)")


def register():
    bpy.utils.register_class(IMPORT_OT_dl1_mesh)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(IMPORT_OT_dl1_mesh)


if __name__ == "__main__":
    register()
