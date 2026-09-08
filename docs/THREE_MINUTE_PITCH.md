# Three-minute English project pitch

## 0:00-0:35 - Problem

I built RAG Reliability Lab for policy question answering. The problem I wanted
to address was not simply generating an answer from a PDF. A RAG answer can sound
correct even when the relevant evidence was missed during retrieval or removed
during reranking. In a policy setting, that makes the system difficult to audit
and unsafe to trust.

## 0:35-1:15 - System design

The pipeline extracts the PDF page by page and preserves one-based source-page
metadata. It creates page-aware chunks, removes repeated margin text, and stores
dense embeddings in Chroma. I kept a dense top-four retriever as the baseline.
The advanced path combines dense and BM25 candidates and applies a cross-encoder
to select the final four passages.

The interface exposes all three stages: dense results, hybrid candidates, and
reranked evidence. Given a labelled evidence page, it can distinguish a recall
failure from a fusion or reranking failure instead of reporting only a wrong answer.

## 1:15-2:00 - Grounding and citations

For answer generation, the model does not generate page numbers. It receives
server-numbered passages and must return a JSON decision with citation IDs. The
backend validates those IDs and maps them to source and page metadata that the
model cannot alter. If the model abstains, returns no answer, or supplies no valid
citation, the API returns a fixed insufficient-evidence response.

This does not prove semantic entailment, so I describe it as citation validation,
not hallucination elimination. I demonstrate the boundary with a normal policy
question, an unanswerable Wi-Fi-password question, and a question requiring pages
14 and 16.

## 2:00-2:40 - Evaluation

I separated the original 20 development questions from eight new frozen test
questions. All are hand-authored and manually labelled with evidence pages from
one 20-page policy. On the frozen test, both dense and hybrid pipelines reached
100 percent Recall at four. The cross-encoder improved MRR at four from 0.938 to
1.000, but median latency increased from about 22 to 450 milliseconds.

The important result is therefore better ranking, not better recall. The sample
is small and is not production traffic, so I keep that limitation explicit.

## 2:40-3:00 - Engineering decisions and next step

The FastAPI service caches the retrieval runtime and fingerprints the persisted
index against document and model settings. Regression tests cover metrics,
failure attribution, citation validation, and abstention. My next step would be
an answer-level benchmark with more unanswerable, paraphrased, and multi-document
cases, followed by an entailment checker and production authentication.
