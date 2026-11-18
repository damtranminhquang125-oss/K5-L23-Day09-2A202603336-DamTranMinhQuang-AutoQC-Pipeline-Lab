"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    issues = []
    fmt = cfg["label_format"]
    overflow_tol = fmt["overflow_tol"]
    min_box_px = fmt["min_box_px"]
    max_box_frac = fmt["max_box_frac"]
    dup_box_iou = fmt["dup_box_iou"]

    for label_path in ds.label_files():
        stem = label_path.stem
        previous_boxes = []
        size = ds.image_size(stem)

        for line_no, text in ds.read_label_lines(stem) or []:
            try:
                class_id, cx, cy, w, h = parse_yolo_line(text)
            except (ValueError, OverflowError):
                issues.append(Issue(
                    "label_format", ERROR, "BAD_FORMAT", stem,
                    "dòng nhãn không đúng 5 cột số",
                    line=line_no,
                ))
                continue

            if class_id < 0 or class_id >= len(ds.classes):
                issues.append(Issue(
                    "label_format", ERROR, "BAD_CLASS_ID", stem,
                    f"id lớp {class_id} nằm ngoài 0..{len(ds.classes) - 1}",
                    line=line_no,
                    extra={"class_id": class_id},
                ))
                continue

            if w <= 0 or h <= 0:
                issues.append(Issue(
                    "label_format", ERROR, "BOX_ZERO", stem,
                    "chiều rộng hoặc chiều cao box phải lớn hơn 0",
                    line=line_no,
                ))
                continue

            box = (cx, cy, w, h)

            left = cx - w / 2
            right = cx + w / 2
            top = cy - h / 2
            bottom = cy + h / 2
            overflow = max(0.0, -left, right - 1.0, -top, bottom - 1.0)
            if overflow > overflow_tol:
                issues.append(Issue(
                    "label_format", ERROR, "BOX_OUT_OF_IMAGE", stem,
                    f"box tràn ảnh {overflow:.4f}",
                    line=line_no,
                    extra={"overflow": overflow},
                ))

            if any(
                previous_class == class_id
                and iou_xywh(box, previous_box) > dup_box_iou
                for previous_class, previous_box in previous_boxes
            ):
                issues.append(Issue(
                    "label_format", ERROR, "DUP_BOX", stem,
                    "box trùng với box cùng lớp ở dòng trước",
                    line=line_no,
                ))

            if size is not None:
                width_px, height_px = size
                short_side_px = min(w * width_px, h * height_px)
                if short_side_px < min_box_px:
                    issues.append(Issue(
                        "label_format", WARN, "BOX_TINY", stem,
                        f"cạnh ngắn của box chỉ khoảng {short_side_px:.1f}px",
                        line=line_no,
                        extra={"short_side_px": short_side_px},
                    ))

            if w * h > max_box_frac:
                issues.append(Issue(
                    "label_format", WARN, "BOX_HUGE", stem,
                    f"box chiếm {w * h:.3f} diện tích ảnh",
                    line=line_no,
                    extra={"box_fraction": w * h},
                ))

            previous_boxes.append((class_id, box))

    return issues
