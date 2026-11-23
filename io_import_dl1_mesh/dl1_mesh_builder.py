import os
import struct
import bpy
from . import dl1_parser


def read_vertices(vb: bytes, vtx_cnt: int, is_part: bool, convert_z_up: bool):
    layout = None
    if is_part:
        for pad in range(0, 129, 4):
            layout = dl1_parser.detect_from_offset(vb, pad, vtx_cnt, True)
            if layout:
                break
    else:
        dk1 = vb.find(b'dkcP')
        c_start = dk1 if dk1 != -1 else 0
        end_search = min(len(vb) - 32, c_start + 16384) if len(vb) > 32 else 0
        for off in range(c_start + 16, end_search, 4):
            layout = dl1_parser.detect_from_offset(vb, off, vtx_cnt, False)
            if layout:
                break

    if not layout:
        return None, None, None

    verts = list()
    uvs = list()
    fmt = '<3e' if layout['is_half'] else '<3f'
    stride = layout['stride']
    uv_off = layout['uv_offset']
    start = layout['start']

    for i in range(vtx_cnt):
        base = start + i * stride
        vx, vy, vz = struct.unpack_from(fmt, vb, base)
        if convert_z_up:
            verts.append((vx, vz, vy))
        else:
            verts.append((vx, vy, vz))

        if uv_off != -1:
            tu, tv = struct.unpack_from('<2e', vb, base + uv_off)
            uvs.append((tu, 1.0 - tv))
        else:
            uvs.append((0.0, 0.0))

    return verts, uvs, layout


def locate_and_read_faces(ib: bytes, start_pos: int, vtx_cnt: int, hdr: dict, convert_z_up: bool):
    tri_counts = hdr['tri_counts']
    raw_sum = sum(tri_counts)
    has_alt = (raw_sum % 3 == 0) and all(tc % 3 == 0 for tc in tri_counts)
    alt_tris = [tc // 3 for tc in tri_counts] if has_alt else list()
    alt_sum = sum(alt_tris)

    target_sums = sorted(list({raw_sum, alt_sum} if has_alt else {raw_sum}), reverse=True)
    data_len = len(ib)
    ib_off = None
    active_tris = None

    for t_cnt in target_sums:
        target_bytes = t_cnt * 6
        needle = struct.pack('<I', target_bytes)
        s_pos = start_pos
        while s_pos + 4 <= data_len:
            idx = ib.find(needle, s_pos)
            if idx == -1:
                break
            for delta in range(4, 160):
                cand = idx + delta
                if cand + target_bytes <= data_len:
                    step = max(1, t_cnt // 40)
                    ok = True
                    for t in range(0, t_cnt, step):
                        i0, i1, i2 = struct.unpack_from('<3H', ib, cand + t * 6)
                        if max(i0, i1, i2) >= vtx_cnt:
                            ok = False
                            break
                    if ok:
                        ib_off = cand
                        active_tris = alt_tris if (has_alt and t_cnt == alt_sum) else tri_counts
                        break
            if ib_off is not None:
                break
            s_pos = idx + 4
        if ib_off is not None:
            break

    if ib_off is None:
        if data_len < 6:
            return None
        pos = start_pos
        limit = data_len - 6
        best_off = None
        best_max = 0

        while pos <= limit:
            a, b, c = struct.unpack_from('<3H', ib, pos)
            if a != b and b != c and a != c and max(a, b, c) < vtx_cnt:
                cur = pos
                while cur <= limit:
                    i0, i1, i2 = struct.unpack_from('<3H', ib, cur)
                    if max(i0, i1, i2) >= vtx_cnt:
                        break
                    cur += 6
                tc = (cur - pos) // 6
                if tc in target_sums:
                    ib_off = pos
                    active_tris = alt_tris if (has_alt and tc == alt_sum) else tri_counts
                    break
                if tc > best_max and tc >= 1:
                    best_max = tc
                    best_off = pos
                    pos = cur
                    continue
            pos += 1

        if ib_off is None:
            if best_off is None:
                return None
            chosen = alt_tris if (has_alt and best_max == alt_sum) else tri_counts
            active_tris = list()
            acc = 0
            for tc in chosen:
                if acc + tc <= best_max:
                    active_tris.append(tc)
                    acc += tc
                else:
                    break
            if not active_tris:
                active_tris.append(best_max)
            elif acc < best_max:
                active_tris.append(best_max - acc)
            ib_off = best_off

    faces = list()
    mat_slots = list()
    cur_tri = 0
    max_possible = (data_len - ib_off) // 6
    peak_vtx = 0

    for sub_i, tc in enumerate(active_tris):
        m_slot = hdr['mat_indices'][sub_i] if sub_i < len(hdr['mat_indices']) else 0
        safe_tc = min(tc, max_possible - cur_tri)
        for t in range(safe_tc):
            i0, i1, i2 = struct.unpack_from('<3H', ib, ib_off + (cur_tri + t) * 6)
            hi = max(i0, i1, i2)
            if hi >= vtx_cnt:
                break
            if peak_vtx > 500 and hi < 32 and t > 0:
                break
            if i0 != i1 and i1 != i2 and i0 != i2:
                faces.append((i0, i2, i1) if convert_z_up else (i0, i1, i2))
                mat_slots.append(m_slot)
                if hi > peak_vtx:
                    peak_vtx = hi
        cur_tri += tc

    return faces, mat_slots


def _create_single_mesh(name: str, verts: list, uvs: list, faces: list, has_uv: bool, mat_names: list, mat_slots: list):
    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(verts, list(), faces)

    max_slot = max(mat_slots) if mat_slots else 0
    for idx in range(max_slot + 1):
        m_name = mat_names[idx] if idx < len(mat_names) else f"{name}_mat_{idx}"
        mat = bpy.data.materials.get(m_name) or bpy.data.materials.new(name=m_name)
        mesh.materials.append(mat)

    for poly, slot in zip(mesh.polygons, mat_slots):
        poly.material_index = slot

    if has_uv and uvs:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        for loop in mesh.loops:
            uv_layer.data[loop.index].uv = uvs[loop.vertex_index]

    mesh.update()
    return mesh


def build_blender_meshes(bundle: dict, convert_z_up: bool = True, split_by_material: bool = True):
    hdr = dl1_parser.read_submesh_table(bundle['hdr'])
    if not hdr:
        raise RuntimeError("Не найдена таблица сабмешей (12x 0xFF)")

    verts, uvs, layout = read_vertices(bundle['vb'], hdr['vtx_count'], bundle['is_parts'], convert_z_up)
    if not verts:
        raise RuntimeError("Не удалось определить формат вершинного буфера")

    ib_start = 0 if bundle['is_parts'] else (layout['start'] + hdr['vtx_count'] * layout['stride'])
    res = locate_and_read_faces(bundle['ib'], ib_start, hdr['vtx_count'], hdr, convert_z_up)
    if not res:
        raise RuntimeError("Ошибка чтения индексного буфера")

    faces, mat_slots = res
    mat_names = dl1_parser.extract_material_names(bundle['mat'])
    has_uv = (layout['uv_offset'] != -1)

    if not split_by_material:
        mesh = _create_single_mesh(bundle['name'], verts, uvs, faces, has_uv, mat_names, mat_slots)
        return [mesh], layout

    grouped_faces = dict()
    for face, slot in zip(faces, mat_slots):
        grouped_faces.setdefault(slot, list()).append(face)

    built_meshes = list()
    for slot in sorted(grouped_faces.keys()):
        sub_faces_raw = grouped_faces[slot]
        if not sub_faces_raw:
            continue

        remap = dict()
        sub_verts = list()
        sub_uvs = list()
        sub_faces = list()

        for f in sub_faces_raw:
            new_f = list()
            for old_idx in f:
                new_idx = remap.get(old_idx)
                if new_idx is None:
                    new_idx = len(sub_verts)
                    remap[old_idx] = new_idx
                    sub_verts.append(verts[old_idx])
                    if has_uv:
                        sub_uvs.append(uvs[old_idx])
                new_f.append(new_idx)
            sub_faces.append(tuple(new_f))

        raw_mat_name = mat_names[slot] if slot < len(mat_names) else f"mat_{slot}"
        clean_mat_name = os.path.splitext(raw_mat_name)[0]
        part_name = f"{bundle['name']}_sub{slot}_{clean_mat_name}"
        zero_slots = [0 for _ in range(len(sub_faces))]

        mesh = _create_single_mesh(
            name=part_name,
            verts=sub_verts,
            uvs=sub_uvs,
            faces=sub_faces,
            has_uv=has_uv,
            mat_names=[raw_mat_name],
            mat_slots=zero_slots,
        )
        built_meshes.append(mesh)

    return built_meshes, layout
