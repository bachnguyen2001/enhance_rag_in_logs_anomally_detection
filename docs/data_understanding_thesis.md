# Data Understanding — nội dung tóm tắt cho luận văn

## Dataset và đơn vị phân tích

Nghiên cứu sử dụng dữ liệu HDFS_v1 đã được tiền xử lý. Theo README đi kèm dataset,
log được nhóm theo BlockId và nhãn được gán ở cấp BlockId. Vì vậy, một mẫu trong
nghiên cứu là chuỗi sự kiện có thứ tự của một BlockId. Nhãn Anomaly của mẫu không
đồng nghĩa mọi sự kiện trong chuỗi đều là sự kiện bất thường.

Các nguồn chính gồm `anomaly_label.csv` cung cấp nhãn Normal/Anomaly,
`Event_traces.csv` cung cấp ordered EventId sequences và `HDFS.log_templates.csv`
ánh xạ EventId sang template text. `Event_occurrence_matrix.csv` được dùng để đối
chiếu số lần xuất hiện sự kiện. Ma trận số đếm không giữ thông tin thứ tự, vì vậy
chuỗi EventId gốc vẫn được bảo toàn khi tạo đầu vào nghiên cứu.

## Tính nhất quán và phân bố dữ liệu

Dữ liệu được kiểm tra trên toàn bộ 575.061 BlockIds. Các file nhãn, event traces,
templates và occurrence matrix nhất quán theo BlockId, label và số lần xuất hiện
của từng event. Không phát hiện trace rỗng, EventId không xác định hoặc sai lệch
số đếm giữa trace và occurrence matrix. Trường Type bị loại khỏi model input vì
trạng thái tồn tại của nó mã hóa trực tiếp nhãn Anomaly. Các indicator suy ra từ
trạng thái missing của Type cũng không được sử dụng.

| Thống kê | Giá trị |
| --- | ---: |
| Tổng BlockIds | 575.061 |
| Normal | 558.223 |
| Anomaly | 16.838 |
| Tỷ lệ Anomaly | 2,93% |
| Event templates | 29 |
| Unique ordered EventId sequences | 18.373 |
| Sequences có cả hai nhãn | 10 |
| BlockIds thuộc các sequences xung đột | 46 |

Dataset mất cân bằng rõ rệt. Mô hình luôn dự đoán Normal có Accuracy khoảng 97,07%,
nhưng Recall và F1 của lớp Anomaly bằng 0. Do đó nghiên cứu dùng anomaly-class F1
làm metric chính, đồng thời báo cáo Precision, Recall và confusion matrix.

Có 564.097 BlockIds thuộc các nhóm exact sequence xuất hiện nhiều lần, tương đương
98,09% tổng số BlockIds. Nhiều BlockId khác nhau vì vậy có cùng execution pattern.
Mười nhóm sequence có cả hai nhãn cho thấy biểu diễn EventId không đủ để tái tạo
hoàn toàn nhãn quan sát của dataset. Các mẫu này được giữ nguyên, không tự sửa nhãn
hoặc loại bỏ để cải thiện kết quả. Nguyên nhân của xung đột chưa được xác định.

## Quyết định preprocessing và split chính

Một BlockId được giữ thành một sample, bảo toàn thứ tự EventId và ánh xạ template.
Nhãn được giữ riêng để đánh giá, không đưa vào query text. Type, thông tin thời gian
và NPZ không được sử dụng làm đầu vào chính. Đơn vị timing và nội dung NPZ so với
CSV chưa được xác minh; giới hạn này không ngăn cản phạm vi chỉ dùng EventId/template.

Nghiên cứu chọn Group-by-trace split v2 làm protocol chính. Các BlockIds có cùng
exact ordered EventId sequence được đặt trong cùng một tập. Thuật toán phân bổ
nguyên nhóm hướng đến tỷ lệ 70/15/15 theo số BlockIds và giữ tỷ lệ Anomaly gần nhau,
với seed cố định 42. Kết quả như sau:

| Tập | Tổng BlockIds | Normal | Anomaly |
| --- | ---: | ---: | ---: |
| Train | 402.543 | 390.757 | 11.786 |
| Validation | 86.259 | 83.733 | 2.526 |
| Test | 86.259 | 83.733 | 2.526 |

Các tập không chồng BlockId hoặc exact sequence. Việc chọn protocol này dựa trên
phát hiện rằng 97,68% test BlockIds trong random split có sequence đã xuất hiện ở
train. Random split được giữ trong EDA để giải thích lựa chọn protocol; các
experiment chính chỉ dùng Group-by-trace v2. Không có exact-sequence overlap không
có nghĩa mọi template hoặc subsequence trong test đều chưa được quan sát.

## Knowledge base và khả năng tái sử dụng

Knowledge base chỉ lấy Normal thuộc train, sau đó canonicalize theo exact ordered
sequence: 390.757 Normal train BlockIds được biểu diễn bằng 4.759 canonical traces.
Mỗi trace giữ occurrence_count để thống kê, nhưng giá trị này không được đưa vào
embedding hoặc prompt. M2 và M3 dùng cùng knowledge base; canonicalization là bước
chuẩn bị dữ liệu chung, không phải đóng góp riêng của Adaptive RAG.

Manifest, cấu hình preprocessing và các artifact được lưu cố định để M0–M3 dùng
chung dữ liệu. Notebook và CSV là phụ lục/tài liệu tái lập, không cần trình bày toàn
bộ mã kiểm tra trong chương chính. Thử nghiệm LLM sử dụng cùng một subset cố định
500 test BlockIds gồm 400 Normal và 100 Anomaly cho cả bốn phương pháp. Precision
và F1 đo trên subset có 20% Anomaly này không đại diện trực tiếp cho prevalence
2,93% của toàn dataset; quy trình chọn subset được ghi trong cấu hình experiment.

Nguồn mô tả dataset và thông tin trích dẫn gốc: README.md đi kèm HDFS_v1.
Số liệu trong mục này được tính từ các CSV và artifact đã được kiểm chứng của dự án.
