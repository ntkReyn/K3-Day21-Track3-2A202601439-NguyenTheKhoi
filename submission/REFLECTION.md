# Reflection — Lab 21

**1. Điều gì làm bạn ngạc nhiên nhất?**

Fine-tune đạt target 0.9375, cao hơn baseline prompt tối ưu 0.6875, nhưng vẫn bị FAILED vì regression giảm từ 0.75 xuống 0.50. “Đúng bài” và “an toàn để deploy” là hai
tiêu chí khác nhau.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Phần tốn thời gian nhất là chạy các contrast NB4 và chờ sinh kết quả trên T4. Mình dự đoán huấn luyện sẽ là phần nặng nhất, nhưng chạy lại nhiều adapter và đánh giá từng nhóm
cũng đáng kể không kém.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Mình không còn tin loss thấp hoặc target cao tự động có nghĩa là model tốt hơn. `attn_only` có loss thấp hơn `correct` nhưng target chỉ hòa; `wrong_lr` cho thấy một thay đổi learning rate có thể phá hỏng hoàn toàn kết quả.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

AI assistant giúp đọc cấu trúc repo, kiểm tra artefact, đối chiếu số liệu và soạn report. Nó không được phép bịa baseline theo từng mẫu: `qualitative.json` không lưu các dự đoán đó, nên report ghi rõ dữ liệu còn thiếu thay vì suy đoán.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Mình sẽ đóng băng một eval set đại diện và đo baseline prompt tốt trước khi train. Sau đó đặt regression, format và latency gate ngay từ đầu, để cải thiện target không thể che giấu việc model quên năng lực chung.
