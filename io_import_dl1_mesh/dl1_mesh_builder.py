import os
import struct
import bpy
from . import dl1_parser

def _align_up(val: int, align: int) -> int:
    return (val + align - 1) & ~(align - 1)

def _read_vertex_slice(vb: bytes, offset: int, count: int, layout: dict, convert_z_up: bool):
    verts = list()
    uvs = list()
    fmt = '<3e' if layout['is_half'] else '<3f'
    stride = layout['stride']
    uv_off = layout['uv_offset']

    for i in range(count):
        base = offset + i * stride
        if base + stride > len(vb):
            break
        vx, vy, vz = struct.unpack_from(fmt, vb, base)
        if convert_z_up:
            verts.append((vx, vz, vy))
        else:
            verts.append((vx, vy, vz))

        if uv_off != -1 and base + uv_off + 4 <= len(vb):
            tu, tv = struct.unpack_from('<2e', vb, base + uv_off)
            uvs.append((tu, 1.0 - tv))
        else:
            uvs.append((0.0, 0.0))

    return verts, uvs

def _decode_indices_chunk(ib: bytes, ib_off: int, idx_count: int, vtx_cnt: int, convert_z_up: bool):
    end_off = min(len(ib), ib_off + idx_count * 2)
    num_u16 = (end_off - ib_off) // 2
    if num_u16 < 3:
        return list()

    raw_u16 = struct.unpack_from(f'<{num_u16}H', ib, ib_off)
    faces = list()

    if 0xFFFF in raw_u16:
        strip = list()
        for idx in raw_u16:
            if idx == 0xFFFF or idx >= vtx_cnt:
                strip.clear()
                continue
            strip.append(idx)
            if len(strip) >= 3:
                a, b, c = strip[-3], strip[-2], strip[-1]
                if a != b and b != c and a != c:
                    if (len(strip) % 2) == 1:
                        tri = (a, c, b) if convert_z_up else (a, b, c)
                    else:
                        tri = (a, b, c) if convert_z_up else (a, c, b)
                    faces.append(tri)
    else:
        for t in range(num_u16 // 3):
            i0, i1, i2 = raw_u16[t * 3], raw_u16[t * 3 + 1], raw_u16[t * 3 + 2]
            if max(i0, i1, i2) >= vtx_cnt:
                break
            if i0 != i1 and i1 != i2 and i0 != i2:
                faces.append((i0, i2, i1) if convert_z_up else (i0, i1, i2))

    return faces

def _read_faces_block(ib: bytes, ib_off: int, vtx_cnt: int, tri_counts: list[int], mat_indices: list[int], is_index_count: bool, convert_z_up: bool):
    faces = list()
    mat_slots = list()
    cur_byte = ib_off

    for sub_i, tc in enumerate(tri_counts):
        m_slot = mat_indices[sub_i] if sub_i < len(mat_indices) else 0
        u16_count = tc if is_index_count else (tc * 3)
        chunk_bytes = u16_count * 2
        if cur_byte + 6 > len(ib):
            break

        sub_faces = _decode_indices_chunk(ib, cur_byte, u16_count, vtx_cnt, convert_z_up)
        faces.extend(sub_faces)
        mat_slots.extend([m_slot] * len(sub_faces))
        cur_byte += chunk_bytes

    return faces, mat_slots, (cur_byte - ib_off)

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
    tables = dl1_parser.read_all_submesh_tables(bundle['hdr'])
    if not tables:
        raise RuntimeError("Не найдена таблица сабмешей (12x 0xFF)")

    vb = bundle['vb']
    ib = bundle['ib']
    mat_names = dl1_parser.extract_material_names(bundle['mat'])

    all_blocks = list()
    last_layout = None

    sorted_tables = sorted(tables, key=lambda x: x.get('vb_offset', 0))
    has_distinct_vb_offsets = (
        len(sorted_tables) > 1
        and all(
            sorted_tables[i].get('vb_offset', 0) > sorted_tables[i - 1].get('vb_offset', 0)
            for i in range(1, len(sorted_tables))
        )
    )

    if bundle['is_parts'] and has_distinct_vb_offsets:
        total_u16_if_indices = sum(sum(t['tri_counts']) for t in sorted_tables)
        is_index_count = (total_u16_if_indices * 2 <= len(ib)) and (total_u16_if_indices * 6 > len(ib))

        ib_cursor = 0

        for t_idx, tbl in enumerate(sorted_tables):
            v_cnt = tbl['vtx_count']
            vb_off = tbl.get('vb_offset', 0)
            if vb_off >= len(vb) or ib_cursor >= len(ib):
                break

            next_vb_off = sorted_tables[t_idx + 1].get('vb_offset', len(vb)) if (t_idx + 1 < len(sorted_tables)) else len(vb)
            seg_bytes = max(0, next_vb_off - vb_off)

            # Вычисляем точный Stride по размеру сегмента с учётом выравнивания до 128 байт
            layout = None
            if v_cnt > 0 and seg_bytes > 0:
                for cand_s in (40, 36, 32, 28, 26):
                    aligned_size = _align_up(v_cnt * cand_s, 128)
                    if aligned_size == seg_bytes or (v_cnt * cand_s <= seg_bytes < aligned_size + 128 and (seg_bytes - v_cnt * cand_s) < 128):
                        uv_o = dl1_parser.find_best_uv_offset(vb, vb_off, min(v_cnt, 60), 1, cand_s, 12)
                        if uv_o == -1:
                            uv_o = 28 if cand_s == 40 else (24 if cand_s == 36 else 20)
                        layout = {'stride': cand_s, 'is_half': False, 'uv_offset': uv_o, 'start': vb_off}
                        break

            if not layout:
                layout = dl1_parser.detect_from_offset(vb, vb_off, v_cnt, True)
            if not layout:
                layout = {'stride': 32, 'is_half': False, 'uv_offset': 20, 'start': vb_off}

            last_layout = layout
            verts, uvs = _read_vertex_slice(vb, vb_off, v_cnt, layout, convert_z_up)

            faces, mat_slots, ib_used = _read_faces_block(
                ib, ib_cursor, len(verts), tbl['tri_counts'], tbl['mat_indices'], is_index_count, convert_z_up
            )
            ib_cursor = _align_up(ib_cursor + ib_used, 4)

            if faces:
                all_blocks.append((f"{bundle['name']}_part{t_idx}", verts, uvs, faces, mat_slots, layout))

    # Режим одиночной модели без раздельных vb_offset
    if not all_blocks:
        hdr = max(tables, key=lambda x: x['vtx_count'])
        v_cnt = hdr['vtx_count']
        layout = dl1_parser.detect_from_offset(vb, 0, v_cnt, bundle['is_parts'])
        if not layout and not bundle['is_parts']:
            dk1 = vb.find(b'dkcP')
            c_start = dk1 if dk1 != -1 else 0
            for off in range(c_start + 16, min(len(vb) - 32, c_start + 16384), 4):
                layout = dl1_parser.detect_from_offset(vb, off, v_cnt, False)
                if layout:
                    break
        if not layout:
            raise RuntimeError("Не удалось определить формат вершинного буфера")
        last_layout = layout
        verts, uvs = _read_vertex_slice(vb, layout['start'], v_cnt, layout, convert_z_up)
        ib_start = 0 if bundle['is_parts'] else (layout['start'] + v_cnt * layout['stride'])
        is_idx_cnt = (ib_start + sum(hdr['tri_counts']) * 6 > len(ib))
        faces, mat_slots, _ = _read_faces_block(
            ib, ib_start, len(verts), hdr['tri_counts'], hdr['mat_indices'], is_idx_cnt, convert_z_up
        )
        if not faces:
            raise RuntimeError("Ошибка чтения индексного буфера")
        all_blocks.append((bundle['name'], verts, uvs, faces, mat_slots, layout))

    built_meshes = list()
    for blk_name, verts, uvs, faces, mat_slots, layout in all_blocks:
        has_uv = (layout['uv_offset'] != -1)
        if not split_by_material:
            mesh = _create_single_mesh(blk_name, verts, uvs, faces, has_uv, mat_names, mat_slots)
            built_meshes.append(mesh)
            continue

        grouped_faces = dict()
        for face, slot in zip(faces, mat_slots):
            grouped_faces.setdefault(slot, list()).append(face)

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
            clean_mat_name = os.path.splitext(raw_mat_name)
            part_name = f"{blk_name}_sub{slot}_{clean_mat_name}"
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

    return built_meshes, last_layout
