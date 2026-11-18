"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    sop = cfg["second_opinion"]
    conf_min = sop["conf"]
    min_area_px = sop["min_area_px"]
    fp_conf = sop["fp_conf"]
    iou_missing = sop["iou_missing"]
    iou_match = sop["iou_match"]
    iou_loose = sop["iou_loose"]
    issues = []

    for image_path in ds.image_files():
        stem = image_path.stem
        if stem not in preds or not ds.label_path(stem).exists():
            continue

        size = ds.image_size(stem)
        if size is None:
            continue
        width_px, height_px = size

        labels = []
        for line_no, text in ds.read_label_lines(stem) or []:
            try:
                box = parse_yolo_line(text)
            except (ValueError, OverflowError):
                continue
            class_id = box[0]
            if 0 <= class_id < len(ds.classes):
                labels.append((line_no, box))

        p_filtered = []
        p_fp = []
        for prediction in preds.get(stem, []):
            if len(prediction) != 6:
                continue
            class_id, conf, cx, cy, w, h = prediction
            pred_box = (cx, cy, w, h)
            if conf >= fp_conf:
                p_fp.append((class_id, conf, pred_box))
            if conf >= conf_min:
                area_px = w * width_px * h * height_px
                if area_px >= min_area_px:
                    p_filtered.append((class_id, conf, pred_box))

        for pred_class, conf, pred_box in p_filtered:
            cands = []
            for _, label in labels:
                iou = iou_xywh(pred_box, label[1:])
                if iou >= iou_missing:
                    cands.append((label[0], iou))

            if not cands:
                issues.append(Issue(
                    "second_opinion", WARN, "POSSIBLE_MISSING", stem,
                    "model lớn không tìm thấy nhãn trùng vật",
                    extra={"conf": conf},
                ))
                continue

            same = []
            other = []
            for _, label in labels:
                iou = iou_xywh(pred_box, label[1:])
                if iou >= iou_missing:
                    if label[0] == pred_class:
                        same.append(iou)
                    else:
                        other.append(iou)

            if same:
                best_same = max(same)
                if best_same < iou_loose:
                    issues.append(Issue(
                        "second_opinion", WARN, "POSSIBLE_LOOSE_BOX", stem,
                        "đúng vật nhưng box nhãn lệch so với dự đoán",
                        extra={"conf": conf, "iou": best_same},
                    ))
                continue

            if other:
                best_other = max(other)
                if best_other >= iou_match:
                    issues.append(Issue(
                        "second_opinion", WARN, "POSSIBLE_WRONG_CLASS", stem,
                        "box trùng vật nhưng lớp nhãn có thể sai",
                        extra={"conf": conf, "iou": best_other},
                    ))

        for line_no, label in labels:
            max_iou = 0.0
            for _, _, pred_box in p_fp:
                max_iou = max(max_iou, iou_xywh(label[1:], pred_box))
            if max_iou < iou_missing:
                issues.append(Issue(
                    "second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                    "nhãn không khớp dự đoán nào của model",
                    line=line_no,
                    extra={"iou": max_iou},
                ))

    return issues
