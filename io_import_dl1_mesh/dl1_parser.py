import os
import re
import struct


def valid_float(v: float, limit: float = 65504.0) -> bool:
    return (-limit <= v <= limit) and (v == v)


def extract_material_names(mat_data: bytes) -> list[str]:
    text = mat_data.decode('latin1', errors='ignore')
    matches = re.findall(r'[a-zA-Z0-9_./\\-]+\.mat', text, flags=re.IGNORECASE)
    seen = set()
    materials = []
    for m in matches:
        base = os.path.basename(m.replace('\\', '/'))
        if base and base not in seen:
            seen.add(base)
            materials.append(base)
    return materials


def read_submesh_table(hdr_data: bytes):
    marker = b'\xFF' * 12
    limit = len(hdr_data)
    pos = 0
    best_info = None

    while pos + 28 <= limit:
        found = hdr_data.find(marker, pos)
        if found == -1:
            break

        base = found + 12
        if base + 16 > limit:
            break

        vtx_cnt, _, sub_cnt, _ = struct.unpack_from('<4I', hdr_data, base)
        if 3 <= vtx_cnt <= 1_000_000 and 1 <= sub_cnt <= 256:
            tri_table_off = base + 16
            mat_table_off = tri_table_off + sub_cnt * 4
            if mat_table_off + sub_cnt * 2 <= limit:
                tris = struct.unpack_from(f'<{sub_cnt}I', hdr_data, tri_table_off)
                if all(0 < t <= 2_000_000 for t in tris):
                    if best_info is None or vtx_cnt > best_info['vtx_count']:
                        mats = struct.unpack_from(f'<{sub_cnt}H', hdr_data, mat_table_off)
                        best_info = {
                            'vtx_count': vtx_cnt,
                            'sub_count': sub_cnt,
                            'tri_counts': list(tris),
                            'mat_indices': list(mats),
                            'header_offset': base,
                        }
        pos = found + 4

    return best_info


def find_best_uv_offset(vb: bytes, off: int, vtx_cnt: int, step: int, stride: int, pos_bytes: int) -> int:
    best_uo = -1
    best_score = -1
    offsets = []
    if pos_bytes == 12 and stride == 32:
        offsets.append(20)
    for uo in range(pos_bytes, stride - 3, 2):
        if uo not in offsets:
            offsets.append(uo)

    for uo in offsets:
        score = 0
        valid = True
        for i in range(0, vtx_cnt, step):
            u, v = struct.unpack_from('<2e', vb, off + i * stride + uo)
            if not valid_float(u, 500.0) or not valid_float(v, 500.0):
                valid = False
                break
            if -10.0 <= u <= 10.0 and -10.0 <= v <= 10.0:
                score += 2
        if valid and score > best_score:
            best_score = score
            best_uo = uo
    return best_uo


def detect_from_offset(vb: bytes, off: int, vtx_cnt: int, is_part: bool):
    avail = len(vb) - off if len(vb) > off else 0
    if avail < vtx_cnt * 12:
        return None

    eff_len = avail
    while eff_len > 0 and vb[off + eff_len - 1] == 0:
        eff_len -= 1
    if eff_len == 0:
        eff_len = avail

    step = max(1, vtx_cnt // 60)
    checked = (vtx_cnt + step - 1) // step
    min_nz = min(2, checked)

    # 1. Float32
    f32_strides = [32, 28, 36, 24, 40, 48]
    if is_part and vtx_cnt > 0:
        exact = eff_len // vtx_cnt
        if 16 <= exact <= 64 and (exact % 4 == 0):
            if exact in f32_strides:
                f32_strides.remove(exact)
            f32_strides.insert(0, exact)

    for s in f32_strides:
        if vtx_cnt * s > avail:
            continue
        ok = True
        nz = 0
        ff_tail = 0
        min_x = min_y = min_z = 1e9
        max_x = max_y = max_z = -1e9

        for i in range(0, vtx_cnt, step):
            b = off + i * s
            x, y, z = struct.unpack_from('<3f', vb, b)
            if not (valid_float(x, 20000.0) and valid_float(y, 20000.0) and valid_float(z, 20000.0)):
                ok = False
                break
            if (0 < abs(x) < 1e-6) or (0 < abs(y) < 1e-6) or (0 < abs(z) < 1e-6):
                ok = False
                break
            if abs(x) > 1e-4 or abs(y) > 1e-4 or abs(z) > 1e-4:
                nz += 1
            if s == 32 and vb[b + 31] == 0xFF:
                ff_tail += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            min_z, max_z = min(min_z, z), max(max_z, z)

        if max(max_x - min_x, max_y - min_y, max_z - min_z) < 0.0005:
            ok = False

        if ok and nz >= min_nz:
            best_uv = find_best_uv_offset(vb, off, vtx_cnt, step, s, 12)
            if best_uv != -1 or ff_tail > checked // 2:
                return {'stride': s, 'is_half': False, 'uv_offset': best_uv if best_uv != -1 else 20, 'start': off}

    # 2. Half-Float
    hf_strides = [20, 16, 24, 28, 32, 12, 14, 18, 22]
    if is_part and vtx_cnt > 0:
        exact = eff_len // vtx_cnt
        if 8 <= exact <= 32 and (exact % 2 == 0):
            if exact in hf_strides:
                hf_strides.remove(exact)
            hf_strides.insert(0, exact)

    for s in hf_strides:
        if vtx_cnt * s > avail:
            continue
        ok = True
        nz = 0
        min_x = min_y = min_z = 1e9
        max_x = max_y = max_z = -1e9

        for i in range(0, vtx_cnt, step):
            b = off + i * s
            x, y, z = struct.unpack_from('<3e', vb, b)
            if not (valid_float(x, 20000.0) and valid_float(y, 20000.0) and valid_float(z, 20000.0)):
                ok = False
                break
            if abs(x) > 1e-4 or abs(y) > 1e-4 or abs(z) > 1e-4:
                nz += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            min_z, max_z = min(min_z, z), max(max_z, z)

        if max(max_x - min_x, max_y - min_y, max_z - min_z) < 0.0005:
            ok = False

        if ok and nz >= min_nz:
            best_uv = find_best_uv_offset(vb, off, vtx_cnt, step, s, 6)
            if best_uv != -1 or s <= 12:
                return {'stride': s, 'is_half': True, 'uv_offset': best_uv, 'start': off}

    return None


def load_dl1_bundle(filepath: str) -> dict:
    folder = os.path.dirname(filepath)
    fname = os.path.basename(filepath)
    m = re.match(r'^(.+)_part_?(\d+)\.bin$', fname, flags=re.IGNORECASE)

    if m:
        base_name = m.group(1)
        parts = {}
        for entry in os.listdir(folder):
            pm = re.match(r'^(.+)_part_?(\d+)\.bin$', entry, flags=re.IGNORECASE)
            if pm and pm.group(1).lower() == base_name.lower():
                full_p = os.path.join(folder, entry)
                if os.path.isfile(full_p):
                    with open(full_p, 'rb') as f:
                        parts[int(pm.group(2))] = f.read()

        if 3 not in parts or 4 not in parts:
            raise RuntimeError("Отсутствуют файлы _part_03 (вершины) или _part_04 (фейсы).")

        meta = b''.join(parts[k] for k in (0, 1, 2) if k in parts)
        return {
            'name': base_name,
            'is_parts': True,
            'vb': parts[3],
            'ib': parts[4],
            'mat': meta if meta else parts[4],
            'hdr': meta + parts[4],
        }
    else:
        with open(filepath, 'rb') as f:
            data = f.read()
        dk1 = data.find(b'dkcP')
        dk2 = data.find(b'dkcP', dk1 + 4) if dk1 != -1 else -1
        if dk2 != -1:
            data = data[:dk2]
        return {
            'name': os.path.splitext(fname)[0],
            'is_parts': False,
            'vb': data,
            'ib': data,
            'mat': data,
            'hdr': data,
        }
