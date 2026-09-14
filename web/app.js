const viewMeta = {
  inspector: ["Workspace / Documents", "Ask your documents."],
  evaluations: ["Workspace / Evaluations", "No score without reproducible evidence."],
};

let overviewData = null;
let libraryBusy = false;
let libraryReady = false;
let sourceRevision = 0;

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
  document.getElementById("evaluation-total").textContent = data.evaluation.case_count;
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
    if (result.document_id) {
      const source = document.createElement("a");
      source.href = `/api/documents/${encodeURIComponent(result.document_id)}/pages/${Number(result.page)}`;
      source.target = "_blank";
      source.rel = "noopener";
      source.textContent = `${result.source} · v${result.version}`;
      article.append(source);
    }
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
  const corpus = document.getElementById("query-corpus").value;
  if (libraryBusy || (corpus === "library" && !libraryReady)) return;
  const revision = sourceRevision;
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
      body: JSON.stringify({ query, expected_pages: corpus === "library" ? [] : expectedPages, corpus }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The experiment failed.");
    if (sourceRevision !== revision || document.getElementById("query-corpus").value !== corpus) return;
    document.getElementById("retrieval-details").open = true;
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
    syncLibraryControls();
  }
}

async function generateAnswer() {
  const button = document.getElementById("generate-answer");
  const question = document.getElementById("query").value.trim();
  const corpus = document.getElementById("query-corpus").value;
  if (libraryBusy || (corpus === "library" && !libraryReady)) return;
  const revision = sourceRevision;
  if (question.length < 3) return;
  button.disabled = true;
  button.textContent = "Generating…";
  try {
    const response = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, corpus }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The answer pipeline failed.");
    if (sourceRevision !== revision || document.getElementById("query-corpus").value !== corpus) return;
    const panel = document.getElementById("grounded-answer");
    panel.hidden = false;
    document.getElementById("answer-state").textContent = data.answerable ? "Citation IDs validated" : "Insufficient evidence";
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
      if (citation.document_id) {
        const link = document.createElement("a");
        link.href = `/api/documents/${encodeURIComponent(citation.document_id)}/pages/${Number(citation.page)}`;
        link.textContent = `Read source · version ${citation.version}`;
        link.target = "_blank";
        link.rel = "noopener";
        item.append(link);
      }
      citations.append(item);
    });
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Ask question";
    syncLibraryControls();
  }
}

function renderEvaluation(report) {
  const generated = new Date(report.generated_at);
  document.getElementById("evaluation-status").textContent =
    `Saved ${generated.toLocaleString()} · K=${report.configuration.k} · ${report.configuration.embedding_model} · ${report.configuration.warmup === true ? 'Warm-up excluded' : 'Legacy or un-warmed run'}`;
  const grid = document.getElementById("evaluation-grid");
  grid.replaceChildren();
  report.pipelines.forEach((pipeline) => {
    const metrics = pipeline.metrics;
    const card = document.createElement("article");
    card.className = `evaluation-card ${pipeline.id === "hybrid_rerank" ? "advanced" : ""}`;
    const name = document.createElement("span");
    name.textContent = pipeline.name;
    const score = document.createElement("strong");
    score.textContent = formatPercent(metrics.evidence_page_hit_rate_at_k);
    const row = document.createElement("div");
    row.className = "metric-row";
    row.innerHTML = `<span>MRR@${Number(report.configuration.k)} <b>${metrics.mrr_at_k.toFixed(3)}</b></span><span>Median <b>${metrics.median_latency_ms} ms</b></span>`;
    const detail = document.createElement("small");
    detail.textContent = `${metrics.hit_count}/${metrics.case_count} evidence-page hits`;
    card.append(name, score, row, detail);
    Object.entries(pipeline.metrics_by_split || {}).forEach(([split, values]) => {
      const line = document.createElement("small");
      line.textContent = `${split}: ${values.hit_count}/${values.case_count} hits · MRR ${values.mrr_at_k.toFixed(3)}`;
      card.append(line);
    });
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
  button.textContent = "Running benchmark…";
  document.getElementById("evaluation-status").textContent = "Comparing dense, BM25, hybrid and hybrid + rerank on development and frozen test cases…";
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
showView(viewMeta[initialView] ? initialView : "inspector");
window.addEventListener("hashchange", () => {
  const name = window.location.hash.slice(1);
  if (viewMeta[name]) showView(name);
  else showView("inspector");
});

// Managed library: original benchmark assets are never changed by uploads.
async function libraryRequest(url, body) {
  const response = await fetch(url, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Library request failed');
  return data;
}
async function loadLibrary() {
  const data = await libraryRequest('/api/documents');
  libraryReady = data.index_status === 'ready';
  document.getElementById('library-status').textContent = libraryReady ? 'Documents ready. You can ask a question below.' : data.index_status === 'empty' ? 'Upload a document to begin.' : 'Documents need preparation before you can ask questions.';
  document.getElementById('library-retry').hidden = libraryReady || data.index_status === 'empty';
  const list = document.getElementById('library-documents'); list.replaceChildren();
  const select = document.getElementById('library-replaces'); select.replaceChildren(new Option('Add a new document',''));
  const newest = new Set();
  data.documents.forEach(doc => {
    if (!newest.has(doc.logical_id)) { select.add(new Option(`${doc.name} · v${doc.version}`,doc.id)); newest.add(doc.logical_id); }
    const row = document.createElement('article'); row.className='library-row';
    const label=document.createElement('p'); label.textContent=`${doc.name} · v${doc.version} · ${doc.page_count} pages · ${doc.active?'Active':'Inactive'}`;
    const original=document.createElement('a'); original.href=`/api/documents/${doc.id}/original`; original.textContent='Download original';
    const toggle=document.createElement('button'); toggle.type='button'; toggle.className='secondary-action'; toggle.textContent=doc.active?'Deactivate':'Activate this version';
    toggle.onclick=async()=>{
      if(libraryBusy) return;
      setLibraryBusy(true);
      try {
        await libraryRequest(`/api/documents/${doc.id}/activation`,{active:!doc.active});
        await prepareLibrary();
      } catch(e) { libraryFailure(e); }
      finally { setLibraryBusy(false); }
    };
    row.append(label,original,toggle); list.append(row);
  });
  if(!data.documents.length) list.textContent='No documents yet. Upload a text PDF or TXT file to begin.';
  syncLibraryControls();
  return data;
}
function syncLibraryControls() {
  document.querySelectorAll('#library-upload input, #library-upload select, #library-upload button, #library-documents button, #library-refresh, #library-retry, #query-corpus').forEach(el=>el.disabled=libraryBusy);
  const blocked=libraryBusy || (document.getElementById('query-corpus').value==='library' && !libraryReady);
  document.querySelectorAll('#query-form button').forEach(el=>{
    const running=el.textContent==='Generating…' || el.textContent==='Running pipelines…';
    el.disabled=blocked || running;
  });
}
function setLibraryBusy(value) {
  libraryBusy=value;
  if(value) {
    sourceRevision++;
    libraryReady=false;
    document.getElementById('grounded-answer').hidden=true;
    ['dense-results','hybrid-results','advanced-results'].forEach(id=>document.getElementById(id).replaceChildren());
    document.getElementById('diagnosis-stage').textContent='Waiting for a trace';
    document.getElementById('diagnosis-summary').textContent='Documents are changing. Run a new query when they are ready.';
    document.getElementById('library-status').textContent='Preparing documents… This may take a moment on the first upload.';
  }
  syncLibraryControls();
}
function libraryFailure(error) {
  document.getElementById('library-status').textContent=`Could not finish preparing documents: ${error.message}. Saved documents are retained. Retry preparation to continue.`;
  document.getElementById('library-retry').hidden=false;
  showToast(error.message);
}
async function prepareLibrary() {
  const data=await libraryRequest('/api/documents');
  if(data.documents.some(doc=>doc.active)) await libraryRequest('/api/documents/build-index',{});
  await loadLibrary();
}
document.getElementById('library-upload').onsubmit=async event=>{
  event.preventDefault();
  if(libraryBusy) return;
  const file=document.getElementById('library-file').files[0];
  const replaces=document.getElementById('library-replaces').value||null;
  if(!file || file.size>5*1024*1024) return showToast('Choose a file up to 5 MB');
  setLibraryBusy(true);
  try {
    const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onerror=()=>reject(new Error('Could not read file'));reader.onload=()=>resolve(reader.result.split(',')[1]);reader.readAsDataURL(file);});
    await libraryRequest('/api/documents',{name:file.name,data,replaces});
    document.getElementById('library-file').value='';
    document.getElementById('query-corpus').value='library';
    document.getElementById('query-corpus').dispatchEvent(new Event('change'));
    await prepareLibrary();
    showToast('Documents ready. Ask your question below.');
    document.getElementById('query').focus();
  } catch(e){libraryFailure(e);} finally{setLibraryBusy(false);}
};
document.getElementById('library-retry').onclick=async()=>{
  if(libraryBusy) return;
  setLibraryBusy(true);
  try{await prepareLibrary();}catch(e){libraryFailure(e);}finally{setLibraryBusy(false);}
};


document.getElementById('query-corpus').onchange=event=>{
  sourceRevision++;
  const library=event.target.value==='library'; document.getElementById('expected-pages').disabled=library;
  document.getElementById('grounded-answer').hidden=true;
  ['dense-results','hybrid-results','advanced-results'].forEach(id=>document.getElementById(id).replaceChildren());
  document.getElementById('diagnosis-summary').textContent=library?'Library traces are unlabelled. Page-only benchmark labels apply to the fixed demo corpus.':'Run a query to inspect evidence.';
  syncLibraryControls();
};
(async()=>{
  setLibraryBusy(true);
  try {
    const data=await loadLibrary();
    if(data.documents.some(doc=>doc.active)) {
      document.getElementById('query-corpus').value='library';
      document.getElementById('query-corpus').dispatchEvent(new Event('change'));
      document.getElementById('library-status').textContent='Preparing your saved documents…';
      if(!libraryReady) await prepareLibrary();
      else document.getElementById('library-status').textContent='Documents ready. You can ask a question below.';
    }
  } catch(e){libraryFailure(e);} finally{setLibraryBusy(false);}
})();
