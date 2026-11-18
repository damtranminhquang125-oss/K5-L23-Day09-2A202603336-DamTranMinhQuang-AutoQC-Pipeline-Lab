# Báo cáo Day 09 — <Họ và tên>, <MSHV>

> Mỗi ý trả lời bằng **số lấy từ `submission/reports/`**, kèm mẫu số. Cả báo cáo ~1–2 trang.

## 1. Kết luận trước
Pipeline kết luận **Hold**. Intake quét 224 ảnh, nhận 193 ảnh (86,16%) và loại 31 ảnh. Người duyệt mô phỏng đã xem 80/142 ảnh trong hàng đợi; vòng QC cuối không còn lỗi bắt buộc nhưng còn cảnh báo trên 57/193 ảnh chưa duyệt (29,53%), vượt ngưỡng Hold 10%. `clean_v1` được xuất dưới dạng bản nháp, chưa được phát hành.

## 2. Lỗi và phân việc
Lấy từ `02_error_list_v1` (gom theo mã lỗi). Cột "ai sửa": **máy** (autofix) hay **người** (hàng đợi); chỉ ghi những mã có mặt trong lô của bạn.

| Mã lỗi | Số ảnh/dòng/cảnh báo | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| BAD_CLASS_ID | 6 dòng nhãn | Máy | Xoá dòng có class ID ngoài 0–3. |
| BAD_CLASS_NAME | 2 lỗi trong `data.yaml` | Máy | Chuẩn hoá tên lớp theo cấu hình. |
| BAD_FORMAT | 3 dòng nhãn | Máy | Xoá dòng sai định dạng YOLO. |
| BAD_SIZE | 3 ảnh | Máy | Loại ảnh có cạnh ngắn dưới mức tối thiểu. |
| BOX_OUT_OF_IMAGE | 8 dòng nhãn | Máy | Clip box vượt biên. |
| BOX_ZERO | 4 dòng nhãn | Máy | Xoá box có chiều rộng hoặc cao bằng 0. |
| CORRUPT_IMAGE | 2 ảnh | Máy | Loại ảnh không đọc được. |
| DUP_BOX | 7 dòng nhãn | Máy | Xoá box trùng. |
| DUP_IMAGE_EXACT | 8 ảnh | Máy | Loại ảnh trùng byte. |
| DUP_IMAGE_NEAR | 7 ảnh | Máy | Loại ảnh trùng gần theo pHash. |
| LEAK_TEST | 11 ảnh | Máy | Loại ảnh trùng kho test/val. |
| MISSING_LABEL_FILE | 6 ảnh | Người | Chuyển người gán nhãn; máy không tự tạo nhãn. |
| BOX_HUGE | 3 cảnh báo / 3 ảnh | Người | Xác minh box chiếm gần hết ảnh. |
| BOX_TINY | 6 cảnh báo / 4 ảnh | Người | Xác minh box nhỏ; không tự xoá vì có thể là vật thật. |
| CLASS_RARE | 4 cảnh báo / 1 ảnh | Người | Kiểm tra lớp hiếm/bất thường. |
| EMPTY_LABEL | 11 ảnh | Người | Xác nhận ảnh nền thật hay thiếu nhãn. |
| POSSIBLE_FALSE_POS | 28 cảnh báo / 26 ảnh | Người | Model không thấy vật tương ứng với box nhãn. |
| POSSIBLE_LOOSE_BOX | 56 cảnh báo / 43 ảnh | Người | Xem lại box lệch/lỏng. |
| POSSIBLE_MISSING | 256 cảnh báo / 95 ảnh | Người | Xác minh vật model thấy nhưng nhãn chưa có box. |
| POSSIBLE_WRONG_CLASS | 21 cảnh báo / 20 ảnh | Người | Xác minh lớp nhãn có thể sai. |

`02_error_list_v1.csv` có 452 issue records: 67 ERROR (31 từ intake và 36 lỗi QC vòng 1) cùng 385 WARN. Intake loại 31/224 ảnh: 3 `BAD_SIZE`, 2 `CORRUPT_IMAGE`, 8 `DUP_IMAGE_EXACT`, 7 `DUP_IMAGE_NEAR`, 11 `LEAK_TEST`. `03_fixes.csv` ghi 35 thao tác: 20 `drop_line`, 8 `clip_box`, 6 `needs_label`, 1 `rename_classes`. Hàng đợi có 142 ảnh; ảnh đầu `img_0160` đạt priority 109.727, cao nhất, do cảnh báo `POSSIBLE_MISSING` và `POSSIBLE_WRONG_CLASS` được cộng theo trọng số/confidence. Ngân sách bao phủ 80/142 ảnh; `06_review_log.csv` xác nhận cả 80 ảnh được mô phỏng duyệt bằng nhãn người. Cảnh báo trên ảnh chưa duyệt vẫn được tính vào gate.

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 16 | NEAR: P 1.00→1.00 / R 0.50→0.83; EXACT: P/R 1.00→1.00 | Ngưỡng 16 bắt 5/6 near-duplicate thay vì 3/6 tại 2, không có FP trên bộ calib. Ở 20, recall NEAR giảm còn 0.50 và EXACT còn 0.33. |
| `second_opinion.conf` | 0.25 | 0.1 | WRONG_CLASS: P 0.56→0.56 / R 0.83→0.83; MISSING: P 0.50→0.40 / R 1.00→1.00; LOOSE_BOX: P 0.29→0.38 / R 0.40→0.60; FALSE_POS: P/R 0.31/1.00→0.31/1.00 | Chọn 0.1 để ưu tiên recall cảnh báo, nhất là LOOSE_BOX; đánh đổi bằng precision MISSING thấp hơn. FALSE_POS không đổi vì dùng `fp_conf` cố định. |

Calibration có 39 ảnh; NEAR có 6 lỗi chèn (mỗi ca tương đương 16,7 điểm recall), còn các mã second-opinion có 5–8 lỗi chèn (12,5–20 điểm mỗi ca). Ở `phash_dist=16`, NEAR bắt 5/6; ở `conf=0.1`, LOOSE_BOX bắt 3/5. Đây là bằng chứng trên calib, không đảm bảo tương tự trên raw. Kết quả EXACT cũng bị ảnh hưởng bởi thứ tự ưu tiên `LEAK_TEST > DUP_IMAGE_EXACT > DUP_IMAGE_NEAR`; bộ calib không có ca LEAK_TEST, nên không thể coi precision trên calib là bảo đảm không gộp nhầm trên lô thật.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`: intake nhận 193/224 ảnh; không có ảnh được duyệt. Vòng cuối còn 6/193 lỗi `MISSING_LABEL_FILE` (`img_0031`, `img_0053`, `img_0126`, `img_0135`, `img_0170`, `img_0219`), nên gate kết luận **Rework** (mã thoát 20). `work/rep_nb/08_summary.json` ghi `n_errors=6`, `reviewed=0`, `exported=null`; dataset không được xuất.

Hai điểm mù có số đo: ở `phash_dist=16`, checker bỏ sót 1/6 near-duplicate; ở `second_opinion.conf=0.1`, bỏ sót 2/5 lỗi LOOSE_BOX. Ngoài ra, 57/193 ảnh còn cảnh báo sau 80 lượt duyệt cho thấy ngân sách có hạn không bảo đảm mọi nhãn nháp đều được người xem. Model đối chiếu có thể bỏ sót lỗi hoặc cảnh báo sai; calib nhỏ không chứng minh độ bao phủ trên raw.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
`python scripts/summarize_results.py` chỉ tổng hợp các JSON seed đang có, không chạy train. Bảng dưới phản ánh đúng `comparison.md` hiện tại: `yolo26n.pt`, 20 epoch, batch 16, 2 seed mỗi biến thể. Các JSON seed hiện khớp hash với config/export hiện tại sau khi chuẩn hoá CRLF/LF.

| Biến thể | Ảnh train | Số seed | mAP50 (TB ± độ lệch) | mAP50-95 (TB ± độ lệch) | Precision (TB ± độ lệch) | Recall (TB ± độ lệch) |
|---|---:|---:|---:|---:|---:|---:|
| raw | 224 | 2 | 0.726 ± 0.000 | 0.512 ± 0.004 | 0.764 ± 0.061 | 0.649 ± 0.028 |
| clean | 193 | 2 | 0.712 ± 0.004 | 0.501 ± 0.005 | 0.761 ± 0.032 | 0.616 ± 0.038 |

Chênh mAP50 (clean − raw) = **−0.014**. Cả hai seed đều thấp hơn ở clean: −0.0163 (seed 0) và −0.0112 (seed 1). Mức giảm lớn hơn độ lệch chuẩn mAP50 quan sát được (raw 0.0002; clean 0.0038), nên kết quả hai seed nghiêng về raw; tuy vậy, chỉ 2 seed và tập test 100 ảnh chưa đủ để kết luận chắc chắn về khả năng tổng quát. Cần thêm seed và đánh giá trên tập độc lập lớn hơn.

Hai giả thuyết cần kiểm: (1) raw có trùng/rò rỉ dữ liệu—intake hiện loại 8 ảnh `DUP_IMAGE_EXACT`, 7 `DUP_IMAGE_NEAR` và 11 `LEAK_TEST`; kiểm tra trên test độc lập không trùng/rò rỉ và so các seed tương ứng. (2) clean vẫn có thể còn nhãn sai hoặc thiếu—sau ngân sách duyệt 80 ảnh, 57/193 ảnh vẫn có cảnh báo; audit mù một mẫu ảnh clean và đối chiếu với nhãn người độc lập. Kết quả cũng có thể biến động do test chỉ 100 ảnh.

## 6. Quyết định và ghi chú AI
(a) Chưa có kết luận phát hành cuối cho cấu hình hiện tại: lần chạy kiểm tra `--budget 0` trả **Rework**, còn 6/193 ảnh thiếu file nhãn và 0 ảnh được duyệt; không tạo export. Để có thể Release, người cần gán nhãn cho img_0031, img_0053, img_0126, img_0135, img_0170, img_0219, rồi chạy pipeline với ngân sách duyệt bình thường và xử lý các cảnh báo còn lại. Chỉ Release khi không còn lỗi bắt buộc và tỉ lệ ảnh còn cảnh báo dưới 10%. Kết quả Hold 60/205 từ lần chạy trước dùng cấu hình cũ, không phải kết luận cuối hiện tại.

(b) Dùng AI viết code? Ghi 3 dòng: 
Đọc mọi tài liệu để nắm rõ bài toán để khi code, nắm chắc những gì mình viết.
Nhờ AI chạy lại code một lượt.
Không.

*(Thưởng +2, tùy chọn)* CVAT: lọc `width < 20`, ảnh/số box khớp, đối chiếu `BOX_TINY` (xem docs/CVAT_FILTER.md), 1 câu về `can_review`.
