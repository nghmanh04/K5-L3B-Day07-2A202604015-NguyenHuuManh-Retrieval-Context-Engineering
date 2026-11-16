# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).

## Day 7 NIST benchmark files

- day6_nist_questions.json is the 20-question benchmark copied from the Day 6 project. It keeps the Day 6 field names (gold_source_id, gold_page, evidence_quote); src/day6_loader.py converts those labels to the Day 7 format at runtime.
- nist_unanswerable_cases.json adds two dev and two test questions that should be refused. Each includes an absence reason for manual source checking.
- nist_ai_rmf_manifest.json records the NIST AI RMF 1.0 source, rights note, 34 printed content pages, and the printed-to-physical PDF page mapping.

The PDF itself is fetched from the public NIST URL by the Day 6 preparation pipeline; it is not copied into this repository.