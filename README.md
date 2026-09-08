# RAG Reliability Lab

> Repository history notice: this repository was initialized for public
> presentation on 2026-09-08. Any Git history starts from this cleanup point and
> does not represent the project's full development timeline.

A local reliability workbench for policy-document RAG. The project focuses on
a practical failure mode: an answer can sound plausible even when the correct
policy evidence was never retrieved, was lost during fusion, or was removed by
reranking.

The lab exposes retrieval stages, validates model-selected citation IDs against
server-owned source metadata, and abstains when the answer model does not provide
a valid evidence citation.

## Scope

Implemented:

- text-based PDF ingestion with 1-based source-page metadata;
- page-aware, sentence-boundary chunking and repeated margin-line cleanup;
- local dense retrieval with Chroma and `all-MiniLM-L6-v2`;
- BM25 + dense weighted fusion followed by a MiniLM cross-encoder reranker;
- side-by-side dense, hybrid-candidate, and reranked traces;
- grounded answer JSON contract, validated citations, and explicit abstention;
- deterministic Recall@4 / MRR@4 retrieval evaluation;
- development and frozen test datasets with evidence-page labels;
- FastAPI demo UI and API, plus regression and live smoke checks.

Not implemented:

- OCR or image understanding for scanned PDFs;
- semantic entailment verification between every answer sentence and citation;
- authentication, tenant isolation, production monitoring, or user analytics;
- Docker packaging, CI/CD, AWS App Runner, or any verified cloud deployment.

## Architecture

```text
Policy PDF
  -> page text extraction + cleanup
  -> page-aware chunks with {source, page}
  -> Chroma dense index --------------------------+
  -> in-memory BM25 index ------------------------+-> weighted fusion
                                                       -> cross-encoder top 4
                                                          +-> retrieval trace
                                                          +-> answer model
                                                               -> citation-ID validation
                                                               -> answer or abstention
```

The persisted Chroma index has a fingerprint covering the source PDF, embedding
model, chunk settings, and table-extraction setting. A configuration or document
change rebuilds the index instead of silently reusing incompatible vectors.

## Demo corpus and data provenance

The local corpus is the 20-page February 2026 Apple Business Conduct Policy,
downloaded from Apple's public compliance site:

`https://www.apple.com/compliance/pdfs/Business-Conduct-Policy.pdf`

The exact local file hash and dataset methodology are recorded in
`evaluation_manifest.json`. The third-party PDF is intentionally not committed.
Use the supplied download script to fetch it from the official URL and verify the
expected SHA-256 before indexing.

- `eval_dataset.json`: 20 hand-authored and manually page-verified development
  questions. This set was inspected during development and is not held out.
- `eval_dataset_test.json`: 8 new hand-authored, manually page-verified questions
  frozen after the retrieval configuration was selected.
- `demo_cases.json`: one normal, one unanswerable, and one cross-page demo contract.

These are curated evaluation cases, not production traffic or real-user queries.

## Setup

Python 3.12 is the maintained local environment.

```powershell
python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\scripts\download_sample_policy.ps1
```

Edit `.env` and replace `OPENAI_API_KEY` only if generated answers are needed.
Retrieval evaluation uses the local embedding and reranker models and does not
call the chat API. Model weights are downloaded on first use and cached under
`.model_cache/`.

Important settings:

- `RAG_EMBEDDING_BACKEND=hf`
- `HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2`
- `RAG_CHUNK_SIZE=900`
- `RAG_CHUNK_OVERLAP=80`
- `RAG_RECALL_K=10`
- `RAG_EXTRACT_TABLES=false`

Table detection is optional because it adds substantial PDF parsing time. The
included policy produced no detected tables, so table preservation is not part
of the verified result for this corpus.

## Run

Start the FastAPI workbench:

```powershell
.\run_lab.ps1
```

Then open:

- workbench: `http://127.0.0.1:8000`
- interactive API documentation: `http://127.0.0.1:8000/docs`
- health check: `http://127.0.0.1:8000/api/health`

The maintained demo entry is the FastAPI workbench. `app.py` and `run_app.ps1`
retain the older Streamlit conversation UI for compatibility, but it is not
required for the main demo.

## Three representative demonstrations

Open **Query inspector**, paste a question, and use **Generate cited answer**.

1. Normal answer
   - `Can an employee accept a $50 gift card from a supplier if the employee is under the $150 gift rule?`
   - Expected: answer **No**, with page 14 cited.
2. Evidence insufficiency
   - `What is the password for the employee cafeteria Wi-Fi?`
   - Expected: `I cannot answer this from the available policy evidence.`
3. Cross-page evidence
   - `Compare the rule for accepting a supplier gift card with the rule for providing something of value to a healthcare provider.`
   - Expected: cite pages 14 and 16.

To reproduce these three live contracts with the configured chat API:

```powershell
.\.venv312\Scripts\python.exe run_demo_checks.py
```

The latest observed run passed 3/3 contracts. This is a smoke-check result, not
an answer-accuracy metric.

A short capture of the same three flows is available at
[`docs/demo/rag-reliability-lab-demo.gif`](docs/demo/rag-reliability-lab-demo.gif).
The GIF was recorded from the local FastAPI workbench; it is presentation evidence,
not an additional evaluation run.

## Retrieval evaluation

Run:

```powershell
$env:HF_HUB_OFFLINE="1"  # after the two local models have been cached
.\.venv312\Scripts\python.exe retrieval_eval.py
```

Metric definition: a case is a Recall@4 hit when at least one human-labelled
evidence page appears among the first four returned chunks. MRR@4 uses the rank
of the first matching page. It measures retrieval only, not generated-answer
correctness.

Verified local result after rebuilding the fingerprinted index:

- Development (`n=20`): Dense Recall@4 `100.0%`, MRR@4 `0.904`, median
  `23 ms`; Hybrid + rerank Recall@4 `100.0%`, MRR@4 `0.975`, median `472 ms`.
- Frozen test (`n=8`): Dense Recall@4 `100.0%`, MRR@4 `0.938`, median
  `22 ms`; Hybrid + rerank Recall@4 `100.0%`, MRR@4 `1.000`, median `450 ms`.

The same configuration was run twice without tuning against the frozen test and
produced identical Recall@4 and MRR@4 values; latency varied slightly. On this
small corpus the advanced pipeline improved ranking, not recall, and added about
20x median latency. The old `85% -> 95% Recall@4` result came from an earlier
persisted-index state and must not be used as the current project result.

The full configuration, per-case rankings, candidate pages, and failure stages
are saved in `evaluation_runs/latest.json`.

## Tests

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s tests -v
```

The regression suite covers retrieval metric calculation, dataset splits,
failure classification, citation-ID validation, abstention behavior, API assets,
and route availability. Passing tests are regression evidence, not model accuracy.

## Known limitations

- The frozen test contains only 8 questions from one 20-page English policy.
- The included questions were authored from the document, not collected from users.
- Page-level labels do not prove that every returned chunk fully entails an answer.
- Citation IDs and source pages are server-validated, but semantic entailment is
  still delegated to the answer model.
- Unanswerable and cross-page behavior is represented by three live demo contracts;
  there is not yet a statistically meaningful answer-level benchmark.
- Cold start includes loading two local transformer models and can take tens of seconds.

## Project files

- `rag_engine.py`: PDF ingestion, chunking, vector index, BM25 fusion, reranking.
- `grounded_answer.py`: answer contract, citation validation, and abstention.
- `lab_api.py`: FastAPI workbench and API routes.
- `retrieval_eval.py`: deterministic retrieval benchmark.
- `run_demo_checks.py`: live normal/unanswerable/cross-page checks.
- `scripts/download_sample_policy.ps1`: official-source download and hash check.
- `web/`: existing workbench interface.
- `docs/DEMO_SCRIPT.md`: short recording script.
- `docs/THREE_MINUTE_PITCH.md`: English interview explanation.
