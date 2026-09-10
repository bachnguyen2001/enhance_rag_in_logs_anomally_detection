# Preprocessing policy — version 2

Data Understanding quyết định cách xử lý dựa trên chất lượng, cấu trúc và rủi ro dữ liệu;
không lựa chọn preprocessing theo F1 test. `processed/` là v1, `processed_v2/` là v2.
Không trộn manifests, KB hoặc kết quả của hai phiên bản.

## Nguồn và input

Một BlockId là một sample. Dùng anomaly_label.csv làm nhãn tham chiếu, Event_traces.csv
làm ordered EventIds, HDFS.log_templates.csv làm template gốc; đối chiếu counts bằng matrix.
Giữ nguyên thứ tự/lặp sự kiện và text/wildcards gốc. JSON arrays trong CSV đọc bằng json.loads.

Type.notna() khớp nhãn Anomaly trên toàn bộ 575.061 mẫu trong cả matrix và traces.
Cấm Type, indicator missingness và các feature suy ra từ Type làm input: chúng mã hóa target.
Ý nghĩa mã Type chưa xác minh, chỉ dùng phân tích hậu nghiệm sau khi có tài liệu giải nghĩa.
Không dùng NPZ hoặc timing làm nguồn/feature chính. Raw log chưa được audit;
NPZ content và đơn vị timing chưa xác minh. Kiểm tra tổng timing dùng rtol=0, atol=1e-6.

## Split trước KB

Cả hai protocol dùng seed 42, mục tiêu 70/15/15 theo BlockId và tỷ lệ Anomaly:

- Random BlockId: shuffle đã stratify theo nhãn, giữ thuật toán v1 để đối chiếu.
- Group trace: nhóm theo SHA256 compact JSON ordered sequence. Shuffle bằng seed để xử lý tie,
  ưu tiên nhóm lớn theo max(blocks/total_blocks, anomalies/total_anomalies), phân bổ nguyên nhóm
  theo mức giảm objective tổng bình phương sai lệch BlockId và Anomaly so với target.
  Tối đa 5 lượt di chuyển nguyên nhóm nếu objective giảm; không làm mất lớp đã có.
  Đây là heuristic, không khẳng định optimum. Thuật toán không dùng model score hoặc tìm seed theo F1.

Group v2 đạt 402.543/86.259/86.259 BlockIds (train/validation/test), không có exact-sequence overlap.
Anomaly counts là 11.786/2.526/2.526. Config và source hashes nằm trong processed_v2/preprocessing_config.json.
Experiment chính M0–M3 chỉ dùng Group-by-trace v2. Random split giữ làm tài liệu EDA;
không chạy bộ LLM experiment trên cả hai protocol. Không so model trực tiếp giữa v1/v2.

## Normal-only knowledge base

Chỉ Normal thuộc train, canonicalize exact ordered sequence và giữ occurrence_count từ Normal train.
Giữ representative_block_id để truy nguyên. M2–M3 dùng chung KB 4.759 canonical traces của Group-by-trace v2; occurrence_count
chỉ để thống kê, không đưa vào embedding/prompt. Dedup không phải đóng góp adaptive.
Không có exact match không chứng minh Anomaly; có exact match trong Normal KB không chứng minh Normal.
Chưa tạo embedding/index. Mức giảm số record được đo, lợi ích latency/index bytes chưa được đo.

## Nhãn và metadata đánh giá

Không xóa hoặc đổi nhãn 46 BlockId thuộc 10 nhóm xung đột. conflicting_trace suy từ nhãn toàn dataset
chỉ phục vụ hậu nghiệm; không dùng chọn split, lọc mẫu, build KB, route query hoặc làm model feature.
Query chỉ dùng event_ids/template_sequence, không chứa label, Type, BlockId hoặc các cờ đánh giá.

Báo cáo overlap theo query label cho validation/test: seen/unseen trong train cả hai lớp và exact match
trong Normal KB. Anomaly-class F1 là metric chính; thêm Precision, Recall, confusion matrix và class counts.
Giữ manifest test đầy đủ cố định. Experiment dùng chung subset 400 Normal/100 Anomaly,
seed 42, được lưu tại experiments/artifacts/queries/test_queries.csv. Precision/F1 được diễn giải
trên subset 20% Anomaly này; ngưỡng và mọi điều chỉnh dựa trên validation.

## Verification và khả năng tái lập

scripts/prepare_data.py fail-fast khi nguồn không hợp lệ, xuất config với policy_version, seed, source SHA256
và phiên bản thư viện. Script từ chối ghi đè output đã tồn tại.

scripts/verify_prepared.py và cell 12 trong notebook đọc toàn bộ block_sequences, nguồn CSV, manifests và KB,
recompute invariants mà không import producer hoặc tin JSON report của producer. Kiểm tra coverage,
khóa, labels, ordered sequences, template text, group separation, KB counts/payload/representatives
và annotation correctness. Kết quả kiểm chứng lưu trong outputs/data_understanding/prepared_verification.json.

Thiết kế bốn phương pháp M0–M3 được chốt trong docs/research_design.md; không thêm
hybrid, structural reranking, router, baseline hoặc ablation frequency. EDA chưa chứng minh representation hoặc pipeline nào tốt hơn.

Data Understanding đã hoàn tất; không mở rộng phân tích hoặc chạy lại kiểm tra chỉ để viết chương luận văn.
