const viewMeta = {
  overview: ["Workspace / Overview", "Reliability before answers."],
  inspector: ["Workspace / Query inspector", "See exactly where retrieval changes."],
  corpus: ["Workspace / Corpus", "Know what entered the index."],
  evaluations: ["Workspace / Evaluations", "No score without reproducible evidence."],
};

let overviewData = null;

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function showView(name) {
  document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
  document.getElementById(`${name}-view`).classList.add("active");
  document.querySelector(`[data-view="${name}"]`).classList.add("active");
  document.getElementById("view-kicker").textContent = viewMeta[name][0];
  document.getElementById("view-title").textContent = viewMeta[name][1];
  document.getElementById("open-inspector").hidden = name === "inspector";
  window.location.hash = name;
}

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.setTimeout(() => toast.classList.remove("visible"), 6000);
}

function applyOverview(data) {
  overviewData = data;
  const corpus = data.corpus;
  const evaluation = data.evaluation;
  const config = data.configuration;
  document.getElementById("case-count").textContent = evaluation.case_count;
  document.getElementById("labelled-count").textContent = `${evaluation.labelled_evidence_count} with evidence labels`;
  document.getElementById("pipeline-count").textContent = data.pipelines.length;
  document.getElementById("corpus-status").textContent = corpus.available ? "Ready" : "Missing";
  document.getElementById("corpus-size").textContent = corpus.available ? `${corpus.name} · ${corpus.size_mb} MB` : corpus.name;
  document.getElementById("index-status").textContent = corpus.index_present ? "Built" : "Pending";
  document.getElementById("chunk-size").textContent = config.chunk_size;
  document.getElementById("chunk-overlap").textContent = config.chunk_overlap;
  document.getElementById("recall-k").textContent = config.recall_k;
  document.getElementById("reranker").textContent = config.reranker;
  document.getElementById("corpus-name").textContent = corpus.name;
  document.getElementById("corpus-available").textContent = corpus.available ? "Ready" : "Missing";
  document.getElementById("corpus-index").textContent = corpus.index_present ? "Available" : "Not built";
  document.getElementById("corpus-detail-size").textContent = corpus.size_mb == null ? "—" : `${corpus.size_mb} MB`;
  document.getElementById("evaluation-total").textContent = evaluation.case_count;
}

function renderResults(targetId, results) {
  const target = document.getElementById(targetId);
  target.replaceChildren();
  if (!results.length) {
    target.innerHTML = '<p class="empty-copy">No passages returned.</p>';
    return;
  }
  results.forEach((result) => {
    const article = document.createElement("article");
    article.className = "trace-item";
    const meta = document.createElement("div");
    meta.className = "trace-meta";
    const rank = document.createElement("span");
    rank.textContent = `Rank ${result.rank}`;
    const page = document.createElement("span");
    page.textContent = result.page ? `Page ${result.page}` : "Page unknown";
    const content = document.createElement("p");
    content.textContent = result.content;
    meta.append(rank, page);
    article.append(meta, content);
    target.append(article);
  });
}

async function runComparison(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type='submit']");
  const query = document.getElementById("query").value.trim();
  const expectedPages = document.getElementById("expected-pages").value
    .split(",")
    .map((value) => Number.parseInt(value.trim(), 10))
    .filter(Number.isInteger);
  button.disabled = true;
  button.textContent = "Running pipelines…";
  try {
    const response = await fetch("/api/retrieval/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, expected_pages: expectedPages }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The experiment failed.");
    const dense = data.pipelines.find((pipeline) => pipeline.id === "dense");
    const hybrid = data.pipelines.find((pipeline) => pipeline.id === "hybrid_recall");
    const advanced = data.pipelines.find((pipeline) => pipeline.id === "hybrid_rerank");
    document.getElementById("dense-latency").textContent = `${dense.latency_ms} ms`;
    document.getElementById("hybrid-latency").textContent = `${hybrid.latency_ms} ms`;
    document.getElementById("advanced-latency").textContent = `${advanced.latency_ms} ms total`;
    renderResults("dense-results", dense.results);
    renderResults("hybrid-results", hybrid.results);
    renderResults("advanced-results", advanced.results);
    const card = document.getElementById("diagnosis-card");
    card.dataset.stage = data.diagnosis.stage;
    document.getElementById("diagnosis-stage").textContent = data.diagnosis.stage.replace("_", " ");
    document.getElementById("diagnosis-summary").textContent = data.diagnosis.summary;
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Run comparison";
  }
}

async function generateAnswer() {
  const button = document.getElementById("generate-answer");
  const question = document.getElementById("query").value.trim();
  if (question.length < 3) return;
  button.disabled = true;
  button.textContent = "Generating…";
  try {
    const response = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The answer pipeline failed.");
    const panel = document.getElementById("grounded-answer");
    panel.hidden = false;
    document.getElementById("answer-state").textContent = data.answerable ? "Evidence validated" : "Insufficient evidence";
    document.getElementById("answer-latency").textContent = `${data.retrieval_ms + data.generation_ms} ms total`;
    document.getElementById("answer-text").textContent = data.answer;
    const citations = document.getElementById("answer-citations");
    citations.replaceChildren();
    data.citations.forEach((citation) => {
      const item = document.createElement("article");
      item.className = "citation-item";
      const label = document.createElement("strong");
      label.textContent = `${citation.passage_id} · ${citation.source} · Page ${citation.page}`;
      const evidence = document.createElement("p");
      evidence.textContent = citation.evidence;
      item.append(label, evidence);
      citations.append(item);
    });
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Generate cited answer";
  }
}

function renderEvaluation(report) {
  const generated = new Date(report.generated_at);
  document.getElementById("evaluation-status").textContent =
    `Saved ${generated.toLocaleString()} · K=${report.configuration.k} · ${report.configuration.embedding_model}`;
  const grid = document.getElementById("evaluation-grid");
  grid.replaceChildren();
  report.pipelines.forEach((pipeline) => {
    const metrics = pipeline.metrics;
    const card = document.createElement("article");
    card.className = `evaluation-card ${pipeline.id === "hybrid_rerank" ? "advanced" : ""}`;
    const name = document.createElement("span");
    name.textContent = pipeline.name;
    const score = document.createElement("strong");
    score.textContent = formatPercent(metrics.recall_at_k);
    const row = document.createElement("div");
    row.className = "metric-row";
    row.innerHTML = `<span>MRR@4 <b>${metrics.mrr_at_k.toFixed(3)}</b></span><span>Median <b>${metrics.median_latency_ms} ms</b></span>`;
    const detail = document.createElement("small");
    detail.textContent = `${metrics.hit_count}/${metrics.case_count} evidence-page hits`;
    card.append(name, score, row, detail);
    grid.append(card);
  });

  const misses = document.getElementById("evaluation-misses");
  misses.replaceChildren();
  const advanced = report.pipelines.find((pipeline) => pipeline.id === "hybrid_rerank");
  const failedCases = advanced ? advanced.cases.filter((item) => item.first_relevant_rank == null) : [];
  if (!failedCases.length) {
    misses.innerHTML = '<p class="empty-copy">No evidence-page misses in the advanced pipeline.</p>';
    return;
  }
  failedCases.forEach((item) => {
    const row = document.createElement("article");
    row.className = "miss-item";
    const id = document.createElement("span");
    id.textContent = `#${String(item.id).padStart(2, "0")}`;
    const question = document.createElement("p");
    question.textContent = item.question;
    const pages = document.createElement("small");
    pages.textContent = `${item.failure_stage || "miss"} · expected p.${item.evidence_pages.join(", ")} · got ${item.retrieved_pages.join(", ")}`;
    row.append(id, question, pages);
    misses.append(row);
  });
}

async function loadLatestEvaluation() {
  const response = await fetch("/api/evaluations/latest");
  if (response.status === 404) {
    document.getElementById("evaluation-status").textContent = "No saved benchmark yet.";
    return;
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Could not load the saved benchmark.");
  renderEvaluation(data);
}

async function runEvaluation() {
  const button = document.getElementById("run-evaluation");
  button.disabled = true;
  button.textContent = "Running 40 retrievals…";
  document.getElementById("evaluation-status").textContent = "Evaluating two pipelines across 20 labelled questions…";
  try {
    const response = await fetch("/api/evaluations/run-retrieval", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The benchmark failed.");
    renderEvaluation(data);
  } catch (error) {
    showToast(error.message);
    document.getElementById("evaluation-status").textContent = "The latest run did not complete.";
  } finally {
    button.disabled = false;
    button.textContent = "Run full benchmark";
  }
}

document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
document.querySelectorAll("[data-go]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.go)));
document.getElementById("open-inspector").addEventListener("click", () => showView("inspector"));
document.getElementById("query-form").addEventListener("submit", runComparison);
document.getElementById("generate-answer").addEventListener("click", generateAnswer);
document.getElementById("run-evaluation").addEventListener("click", runEvaluation);

fetch("/api/overview")
  .then((response) => response.ok ? response.json() : Promise.reject(new Error("Workspace overview is unavailable.")))
  .then(applyOverview)
  .catch((error) => showToast(error.message));

loadLatestEvaluation().catch((error) => showToast(error.message));

const initialView = window.location.hash.slice(1);
if (viewMeta[initialView]) showView(initialView);
