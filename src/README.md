# Day 7 implementation

## Connect Day 6

Day 7 reuses the Day 6 corpus, local embedding model, and PostgreSQL + pgvector index. The Day 6 project is expected in a sibling folder named K5-L3B-Day06-NguyenHuuManh-2A202604015-DataIngestionEngineering; use --day6-root to select another folder. The Day 6 starter/.env must define DATABASE_URL. The database must contain strategy C. The loader does not ingest, delete, or modify Day 6 rows.

Start the Day 6 database from its starter folder:

    docker compose -p day6_student_starter up -d --wait

Keep the Day 6 .env local. Do not copy it into this repo or print its values.

## Compare retrieval

From the Day 7 repository root, use the Python environment from Day 6:

    $python = "..\K5-L3B-Day06-NguyenHuuManh-2A202604015-DataIngestionEngineering\starter\.venv\Scripts\python.exe"
    & $python -m src.run_retrieval --split dev

This compares dense Day 6 strategy C, TF-IDF, and hybrid reciprocal-rank fusion, plus controlled top-k and context-budget variants. Full traces and metrics are saved under results/retrieval/dev/.

Current dev results (15 answerable questions): dense C reaches Hit@1 7/15 and Evidence Hit@3 11/15. TF-IDF and hybrid RRF each reach 9/15 and 13/15. With the hybrid context budget increased to 1,200 characters, context evidence is retained for 9/15. The agent defaults to hybrid RRF with a 1,200-character context budget.

After freezing that configuration on dev, evaluate only the selected configuration on test:

    & $python -m src.run_retrieval --split test --config hybrid_top3_context1200

The selected test result is Hit@1 3/5, Evidence Hit@3 3/5, and context evidence rate 3/5. This is a small test set of five answerable questions; see the report and saved traces before drawing conclusions.

## Run the agent with a hosted chat API

Day 6's Sentence Transformers model creates embeddings; Day 7's agent also needs a chat model. To avoid downloading a chat model, configure a hosted provider with an OpenAI-compatible /v1/chat/completions endpoint. The default endpoint is OpenAI; other compatible providers can be selected with CHAT_BASE_URL.

The Day 7 root .env is loaded automatically by the runner and is ignored by Git. ChatGPT Plus does not automatically include API usage or an API key; ChatGPT and API billing are separate. In CHAT_MODEL, use an API model ID available to your API project. For this run, gpt-6-luna is selected as requested. Model access and request limits are enforced by the API project. Change CHAT_BASE_URL for another compatible provider. Keep the key only in this local file; never commit it or send it in chat:

    CHAT_BASE_URL=https://api.openai.com/v1
    CHAT_MODEL=gpt-6-luna
    CHAT_API_KEY=your-secret-key

The .env.example file is a template with an empty key. Hosted chat sends each question and retrieved NIST excerpts to that provider; gold answers and evidence labels are not sent. Start a fresh run with: python -m src.run_agent --split dev --model gpt-6-luna --output results/agent/gpt-6-luna. Use the same output root for test after dev is frozen.

Review every test answer and citation against the NIST source. Fill human review labels before using starter/agent_metrics.py; that script aggregates supplied labels and does not judge answers itself.