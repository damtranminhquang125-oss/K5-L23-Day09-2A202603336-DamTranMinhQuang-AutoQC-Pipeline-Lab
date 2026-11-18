"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    fixes = []
    drop_lines = defaultdict(set)
    clip_lines = defaultdict(set)
    removed_stems = set()
    needs_label_stems = set()
    rename_classes_done = False

    removable_codes = {
        "DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST",
        "CORRUPT_IMAGE", "BAD_SIZE",
    }
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}

    for issue in issues:
        if issue.severity != ERROR:
            continue

        if issue.code in removable_codes:
            removed_stems.add(issue.image)
        elif issue.code in drop_codes:
            if issue.line is not None:
                drop_lines[issue.image].add(issue.line)
        elif issue.code == "BOX_OUT_OF_IMAGE":
            if issue.line is not None:
                clip_lines[issue.image].add(issue.line)

    for stem in sorted(removed_stems):
        image_path = ds.image_path(stem)
        label_path = ds.label_path(stem)
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        if label_path.exists():
            label_path.unlink()
        fixes.append({
            "image": stem,
            "line": "",
            "action": "remove_image",
            "before": "ảnh và/hoặc nhãn",
            "after": "đã xoá",
        })
        drop_lines.pop(stem, None)
        clip_lines.pop(stem, None)

    for stem in sorted(set(drop_lines) | set(clip_lines)):
        label_path = ds.label_path(stem)
        if stem in removed_stems or not label_path.exists():
            continue

        raw = label_path.read_text(encoding="utf-8", errors="replace")
        had_final_newline = raw.endswith("\n")
        lines = raw.splitlines()
        drops = drop_lines.get(stem, set())
        clips = clip_lines.get(stem, set())

        for line_no in sorted(drops | clips):
            index = line_no - 1
            if index < 0 or index >= len(lines):
                continue

            before = lines[index]
            if line_no in drops:
                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": before,
                    "after": "(đã xoá dòng)",
                })
                lines[index] = None
                continue

            try:
                class_id, cx, cy, w, h = parse_yolo_line(before)
            except (ValueError, OverflowError):
                continue

            left = max(0.0, cx - w / 2)
            right = min(1.0, cx + w / 2)
            top = max(0.0, cy - h / 2)
            bottom = min(1.0, cy + h / 2)

            if right <= left or bottom <= top:
                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": before,
                    "after": "(đã xoá dòng: box không còn diện tích)",
                })
                lines[index] = None
            else:
                new_cx = (left + right) / 2
                new_cy = (top + bottom) / 2
                new_w = right - left
                new_h = bottom - top
                after = f"{class_id} {new_cx:.6g} {new_cy:.6g} {new_w:.6g} {new_h:.6g}"
                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "clip_box",
                    "before": before,
                    "after": after,
                })
                lines[index] = after

        kept_lines = [line for line in lines if line is not None]
        new_raw = "\n".join(kept_lines)
        if had_final_newline:
            new_raw += "\n"
        label_path.write_text(new_raw, encoding="utf-8")

    for issue in issues:
        if issue.severity != ERROR:
            continue

        if issue.code == "MISSING_LABEL_FILE":
            if issue.image not in removed_stems:
                needs_label_stems.add(issue.image)
        elif issue.code == "BAD_CLASS_NAME":
            rename_classes_done = True

    for stem in sorted(needs_label_stems):
        fixes.append({
            "image": stem,
            "line": "",
            "action": "needs_label",
            "before": "(không có file)",
            "after": "chuyển người gán",
        })

    if rename_classes_done:
        if ds.yaml_path.exists():
            data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            data["names"] = list(cfg["classes"])
            ds.yaml_path.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            fixes.append({
                "image": "data.yaml",
                "line": "",
                "action": "rename_classes",
                "before": "tên/id lớp sai",
                "after": cfg["classes"],
            })

    return fixes
