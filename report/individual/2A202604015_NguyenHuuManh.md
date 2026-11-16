# Báo cáo cá nhân Day 7 — 2A202604015 Nguyen Huu Manh

## Dữ liệu và benchmark

- Nguồn: Tabassi (2023), Artificial Intelligence Risk Management Framework (AI RMF 1.0), NIST AI 100-1, DOI: https://doi.org/10.6028/NIST.AI.100-1. Corpus gồm 34 trang in (trang PDF vật lý 6–39); manifest ghi nguồn và yêu cầu rà soát riêng nội dung bên thứ ba.
- Benchmark có 24 câu: 20 câu answerable (dev 15, test 5) lấy từ Day 6 và 4 câu unanswerable (dev 2, test 2). Gold evidence chỉ dùng để chấm, không đưa vào retriever hoặc prompt.
- Vector store: PostgreSQL 16 + pgvector, cosine distance exact scan; embedding local paraphrase-multilingual-MiniLM-L12-v2 (384 chiều). Sau khi volume Docker cũ bị thiếu, đã chạy lại ingestion từ corpus có sẵn. Bảng rag_chunks_384: strategy A 194 chunks, B 122, C 122. Retrieval dense dùng strategy C.
- Model embedding đã có trong cache. Chat model dùng endpoint hosted cấu hình qua .env; không tải model chat local.

## Retrieval và thí nghiệm

Hit@1 đo evidence đúng ở kết quả đầu; Evidence Hit@3 đo có evidence trong ba kết quả đầu; Context Evidence Rate đo evidence còn nằm trong context sau khi đóng gói theo budget.

| Cấu hình | Phương pháp, top-k, budget | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Retrieval trung bình |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Dense Day 6 C, top-3, 800 chars | dev | 7/15 (46.7%) | 11/15 (73.3%) | 5/15 (33.3%) | 56.8 ms |
| Lexical | TF-IDF, top-3, 800 chars | dev | 9/15 (60.0%) | 13/15 (86.7%) | 4/15 (26.7%) | 0.4 ms |
| Hybrid | RRF dense + TF-IDF, top-3, 800 chars | dev | 9/15 (60.0%) | 13/15 (86.7%) | 6/15 (40.0%) | 42.1 ms |
| Top-k ablation | Dense Day 6 C, top-5, 800 chars | dev | 7/15 (46.7%) | 11/15 (73.3%) | 5/15 (33.3%) | 38.5 ms |
| Context ablation / selected | Hybrid RRF, top-3, 1,200 chars | dev | 9/15 (60.0%) | 13/15 (86.7%) | 9/15 (60.0%) | 42.3 ms |
| Frozen configuration | Hybrid RRF, top-3, 1,200 chars | test | 3/5 (60.0%) | 3/5 (60.0%) | 3/5 (60.0%) | 72.8 ms |

Tăng context budget từ 800 lên 1.200 chars giúp evidence còn lại sau đóng gói tăng từ 6/15 lên 9/15, không đổi Hit@1/Evidence Hit@3. Tăng dense top-k từ 3 lên 5 trong budget 800 chars không cải thiện metric. Vì vậy đã chốt hybrid RRF, top-3, 1.200 chars từ dev trước khi chạy test.

### Lỗi retrieval trên test

- q16 (MEASURE): gold ở trang in 28; top-3 trả về trang 29, 29 và 28 nhưng không chứa đoạn evidence được gán nhãn.
- q18 (MANAGE): gold ở trang in 31; top-3 trả về trang 33, 32 và 32 nên bỏ lỡ evidence.
- Không đổi cấu hình sau khi xem test.

## Agent và đánh giá

- Agent hỗ trợ tối đa 4 lượt search, chọn search/answer qua JSON, giới hạn context budget, chỉ chấp nhận citation chunk ID hợp lệ và abstain khi câu trả lời không được hỗ trợ.
- Dev benchmark (17 cases) with requested gpt-6-luna: 0 answered, 0 abstain, 1 provider error, and 16 pending. q01 returned rate_limit_exceeded. API response headers showed request limit 50, remaining 0, reset about 23h56m; token capacity still had 70,953 remaining.
- Test benchmark (7 cases) stopped on the first request: 0 answered, 0 abstain, 1 provider error, and 6 pending. q16 returned rate_limit_exceeded under the same exhausted request limit. Human review is pending because no test answers were generated.
- The current key remains only in local .env, which Git ignores; .env.example contains no key. The client now honors Retry-After and the reset header for the exhausted request limit, paces calls, adds jitter, and bounds retry time. It cannot raise the API project/org limit; resume after reset or after the API limit is increased.

## Chạy lại

Từ root Day 7, bật PostgreSQL Day 6 trước và dùng Python virtualenv Day 6:

    $python = "..\K5-L3B-Day06-NguyenHuuManh-2A202604015-DataIngestionEngineering\starter\.venv\Scripts\python.exe"
    & $python -m src.run_retrieval --split dev
    & $python -m src.run_retrieval --split test --config hybrid_top3_context1200
    & $python -m src.run_agent --split dev --model gpt-6-luna --output results/agent/gpt-6-luna
    & $python -m src.run_agent --split test --model gpt-6-luna --output results/agent/gpt-6-luna

Runner lưu checkpoint theo từng câu. --resume giữ trace answered/abstain và thử lại error/pending. Agent cần kết nối chat provider; .env.example là mẫu trống, còn .env thực không được commit.

## Giới hạn

Test retrieval chỉ có 5 câu answerable nên mỗi câu tương đương 20 điểm phần trăm. Kết quả agent chưa đủ để chấm grounded answer, citation support hay abstention. Thử thách tài liệu nâng cao (tài liệu từ 100 trang có bảng/ảnh/biểu đồ) chưa thực hiện.
