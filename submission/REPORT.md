# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Thế Khôi  **MSSV**: 2A202601439  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Colab T4 16 GB


## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024; p95 đo được 98, gợi ý 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** Có. `template_check.json` báo reasoning được bảo toàn; tuy nhiên dữ liệu ticket không chứa reasoning trace thật và `valid_trace_rate` ở NB5 là 0.0.

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn supervised bắt đầu sau prompt và chứa câu trả lời:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Hai assert trong `mask_proof.json` đều xanh. Fraction 41.49% cho thấy loss không bị tính trên toàn bộ câu hỏi.

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7500 | 0.0000 | 3304.4 |
| (b) base + optimized prompt | 0.6875 | 0.7500 | 1.0000 | 955.9 |
| (c) LoRA fine-tune | 0.9375 | 0.5000 | 1.0000 | 1595.2 |

Baseline (b) mạnh hơn (a): target tăng từ 0 lên 0.6875, format từ 0 lên 1.0 và latency giảm từ 3304.4 xuống 955.9 ms. `OPTIMIZED_PROMPT` không bị sửa; SHA được lưu là `719e74d3b6232053`. Fine-tune đạt target cao hơn (b) trên smoke slice nhưng regression giảm từ 0.75 xuống 0.50.

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | target (NB5) | format | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6260 | 0.9375 | 1.0000 | 12.01 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 1e-4 | 0.5369 | 0.9375 | 1.0000 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.0000 | 0.0000 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.8438 | 1.0000 | 7.09 |

### 4.1 — Vị trí và rank

`attn_only` dùng 32,456,704 tham số, lệch khoảng 0.025% so với `correct`, nên đây là đối chứng công bằng về ngân sách. Trên target, hai run hòa ở 0.9375; loss attn-only thấp hơn (0.5369 so với 0.6260). Smoke slice chưa cho bằng chứng text-linear thắng q,v khi ngân sách đã được ghép; loss huấn luyện cũng không tự quyết định năng lực downstream.

### 4.2 — Learning rate

`wrong_lr` chỉ đổi learning rate từ 1e-4 xuống 1e-5 nhưng final loss tăng lên 1.5704, so với 0.6260 của `correct`. Target và format đều rơi về 0.0, latency tăng lên 4939.7 ms vì mô hình không tạo JSON dừng đúng lúc. Nếu chỉ nhìn số step mà không biết LR, có thể nhầm rằng LoRA đã học; thực tế LR là đòn bẩy lớn trong cấu hình này.

### 4.3 — QLoRA

QLoRA giảm peak VRAM từ 12.01 xuống 7.09 GB, tiết kiệm 4.92 GB (khoảng 41%). Đổi lại, target giảm từ 0.9375 xuống 0.8438 và loss cao hơn một chút (0.7058). Kết quả ủng hộ cảnh báo thận trọng về Qwen3.5, nhưng cần full eval trước khi kết luận tuyệt đối vì hiện mới có 8 mẫu.

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.250` · `regression Δ = -0.250` · `valid_trace_rate = 0.00`

Fine-tune vượt baseline (b) trên target smoke slice (0.9375 so với 0.6875), tăng 0.25 điểm và giữ format 1.0. Tuy nhiên cổng yêu cầu đồng thời không phá năng lực chung. Regression giảm từ 0.75 xuống 0.50, trong khi tolerance chỉ là 0.02, nên verdict phải là FAILED. Đây là failure có ý nghĩa: dữ liệu ticket và 30 step đã đẩy model mạnh về triage nhưng không có replay data để bảo vệ khả năng tổng quát. Kết quả cũng chưa đủ để kết luận vì EVAL_LIMIT=8 làm giảm độ tin cậy của phần trăm. Bước tiếp theo là chạy full eval, thêm 1–5% replay data hoặc giảm ngân sách huấn luyện, rồi đo lại cả target và regression thay vì nới lỏng gate.

## 6. Định tính — có cả ca chưa hoàn hảo

`qualitative.json` chỉ lưu dự đoán fine-tune, không lưu dự đoán từng mẫu của baseline b; vì vậy cột (b) ghi rõ điểm aggregate thay vì bịa output.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---:|---|---|---|---:|---|
| 1 | Chuột không dây — trả lại, gấp | `doi_tra, cao, chuột không dây, tich_cuc` | aggregate 0.6875 | 1.00 | FT đúng 4/4 |
| 2 | Ốp lưng — hoàn tiền, sớm, bực mình | `hoan_tien, trung_binh, ốp lưng, tieu_cuc` | aggregate 0.6875 | 1.00 | FT đúng 4/4 |
| 3 | Bình giữ nhiệt — chưa thấy tiền | `hoan_tien, thap, bình giữ nhiệt, tich_cuc` | aggregate 0.6875 | 0.75 | FT sai 1/4; ca thua |
| 4 | Nồi chiên — thiếu phụ kiện | `san_pham_loi, thap, nồi chiên, trung_tinh` | aggregate 0.6875 | 0.75 | FT sai 1/4; ca thua |
| 5 | Đèn LED — vỡ khi nhận, gấp | `san_pham_loi, cao, đèn LED, trung_tinh` | aggregate 0.6875 | 1.00 | FT đúng 4/4 |

Hai ca điểm 0.75 đều liên quan tới suy luận urgency/sentiment từ marker tiếng Việt; model nhận diện intent và product nhưng còn nhầm một trường phụ. Đây là lý do nên báo cáo field-level accuracy thay vì chỉ exact-match.

## 7. Kết luận & điều tôi học được

Không nên deploy adapter này dựa trên run hiện tại. Về tác vụ hẹp, fine-tune có tín hiệu tốt: target 0.9375, format 1.0 và latency 1595.2 ms, tốt hơn baseline naive và target cũng cao hơn prompt tối ưu. Nhưng deployment không chỉ là tối đa hóa target. Regression giảm 25 điểm phần trăm, vượt xa tolerance 2 điểm phần trăm; một hệ thống CSKH tốt hơn ở
ticket nhưng kém đi ở câu hỏi phổ thông vẫn là rủi ro vận hành. Hơn nữa, số đo hiện chỉ dựa trên 8 mẫu mỗi nhóm, chưa đủ đại diện cho full eval. NB4 cho thấy learning rate là đòn bẩy rõ nhất: sai LR làm target và format về 0, trong khi đổi vị trí với matched rank cho kết quả ngang `correct`. Mask là điều kiện nền tảng vì proof chứng minh loss chỉ áp
vào answer. Bước tiếp theo là chạy full eval, bổ sung replay data 1–5%, rồi kiểm tra lại regression gate trước khi cân nhắc deploy.

**Ba điều tôi học được**

1. Prompt tốt có thể tạo baseline 0.6875 và format 1.0 mà không cần train.
2. Matched-rank attention-only hòa target với all-linear; không thể suy luận từ rank hay train loss đơn lẻ.
3. Fine-tune đạt target cao vẫn phải bị từ chối nếu regression gate không qua.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** unset `EVAL_LIMIT` chạy full 50/15 mẫu; thêm replay
data 1–5%; sau đó so sánh lại `correct` với run ít step hơn để tìm điểm cân bằng.

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse (dữ liệu không có trace thật; `valid_trace_rate=0.0`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
