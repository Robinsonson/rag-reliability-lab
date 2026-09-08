"""
Streamlit enterprise RAG chat UI with conversational retrieval and source debug.

Uses LangChain's history-aware retriever plus a retrieval chain (the supported
pattern replacing legacy ConversationalRetrievalChain for LCEL-style apps).
"""

from __future__ import annotations

# Must run before ``import streamlit`` so the server does not walk transformers
# vision submodules (zoedepth → torchvision) during hot-reload introspection.
import os

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import asyncio
from collections.abc import AsyncIterator, Generator, Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from kb_prompts import PROMPT_VERSION, STRICT_GROUNDED_QA_SYSTEM
from rag_engine import (
    _CHROMA_DIR,
    AuditReport,
    build_compression_retriever,
    build_vectorstore,
    chroma_index_fingerprint,
    chroma_index_is_stale,
    generate_audit_checklist,
    get_embeddings,
    load_and_chunk_policy,
    policy_bm25_cache_key,
    reranker_model_name,
)

_BASE_DIR = Path(__file__).resolve().parent
_COLLECTION_NAME = "company_policy"

# Distinct chat avatars (emoji work across platforms without image assets).
_AVATAR_USER = "🧑‍💼"
_AVATAR_ASSISTANT = "🏢"


def _optional_str(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _load_application_env() -> None:
    """Load .env from project root so API keys are available before any LLM call."""
    load_dotenv(_BASE_DIR / ".env")


def _make_llm() -> ChatOpenAI:
    """
    Chat model for both (1) history-aware query rewriting and (2) grounded answers.

    Default model name targets OpenAI-compatible APIs such as DeepSeek.
    """
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "deepseek-chat"),
        temperature=0,
        streaming=True,
        api_key=_optional_str("OPENAI_API_KEY"),
        base_url=_optional_str("OPENAI_BASE_URL"),
    )


def _inject_custom_css() -> None:
    """Inject global styles for a cleaner, enterprise-style shell."""
    st.markdown(
        """
        <style>
            /* Hide default Streamlit menu and footer */
            #MainMenu {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            div[data-testid="stToolbar"] {visibility: hidden !important;}

            /* Main canvas */
            .block-container {
                padding-top: 1.25rem;
                /* Leave room for the fixed chat input bar at the bottom */
                padding-bottom: 6rem;
                max-width: 1080px;
            }

            /* Softer chat rows */
            div[data-testid="stChatMessage"] {
                background: linear-gradient(
                    180deg,
                    rgba(248, 250, 252, 0.92) 0%,
                    rgba(241, 245, 249, 0.96) 100%
                );
                border: 1px solid rgba(15, 23, 42, 0.07);
                border-radius: 14px;
                padding: 0.4rem 0.85rem 0.75rem 0.85rem;
                margin-bottom: 0.65rem;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }

            /* Sidebar: light enterprise panel */
            section[data-testid="stSidebar"] > div {
                background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
                border-right: 1px solid #e2e8f0;
            }

            /* Primary action emphasis in sidebar */
            div[data-testid="stSidebar"] button[kind="primary"] {
                width: 100%;
                font-weight: 600;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_retrieved_chunks_debug(docs: list[Document]) -> None:
    """Render retrieved passages in a structured, audit-friendly layout."""
    if not docs:
        st.info("No passages were returned for this turn.")
        return

    st.caption(
        f"{len(docs)} passage(s) after hybrid recall (BM25 + dense) + "
        f"cross-encoder rerank ({reranker_model_name()})."
    )
    for idx, doc in enumerate(docs, start=1):
        st.divider()
        with st.container(border=True):
            st.markdown(f"##### Passage {idx}")
            st.markdown("**Content**")
            st.text(doc.page_content)
            if doc.metadata:
                st.markdown("**Metadata**")
                st.json(doc.metadata)


@st.cache_resource(show_spinner="Indexing policy PDF into Chroma…")
def get_vectorstore(index_fingerprint: str) -> Chroma:
    """
    Return a persisted Chroma store aligned with the current policy PDF.

    ``index_fingerprint`` is a cache-bust key; when the PDF or embedding settings
    change, Streamlit rebuilds the index instead of reusing stale vectors.
    """
    del index_fingerprint
    _load_application_env()
    embeddings = get_embeddings()
    if chroma_index_is_stale():
        chunks = load_and_chunk_policy()
        return build_vectorstore(chunks, reset_persist=True)
    return Chroma(
        persist_directory=str(_CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=_COLLECTION_NAME,
    )


@st.cache_resource(show_spinner="Loading policy chunks for BM25 index...")
def get_policy_chunks() -> list[Document]:
    """Policy chunks shared by Chroma indexing and the in-memory BM25 sparse index."""
    _load_application_env()
    return load_and_chunk_policy()


def _build_english_query_retriever(llm: ChatOpenAI, compression_retriever: Any):
    """
    Always rewrite the user question into an English search query before retrieval.

    The Apple Business Conduct PDF is English; Chinese questions must not be sent
    verbatim to BM25 / English-biased embeddings (first-turn history is empty).
    """
    search_query_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You prepare search queries for an English Business Conduct policy "
                "knowledge base. Given the user question (Chinese or English) and "
                "optional chat history, output ONE concise English search query using "
                "policy vocabulary (gifts, entertainment, suppliers, bribery, conflict "
                "of interest, confidentiality, etc.). Do not answer the question—query "
                "text only.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def _retrieve(inputs: dict[str, Any]) -> list[Document]:
        search_query = (search_query_prompt | llm | StrOutputParser()).invoke(
            {
                "input": inputs["input"],
                "chat_history": inputs.get("chat_history") or [],
            }
        )
        return compression_retriever.invoke(search_query.strip())

    return RunnableLambda(_retrieve)


def _build_conversational_rag_chain(
    vectorstore: Chroma,
    chunks: list[Document],
):
    """
    Wire history-aware retrieval + document stuffing + final generation.

    Memory model (how follow-ups work):
    - `chat_history` is a list of prior HumanMessage / AIMessage objects passed
      into the chain on every turn. The model can therefore resolve pronouns and
      elliptical questions (e.g. follow-ups about gifts, conflicts of interest).
    - An English search-query rewriter runs before every retrieval so Chinese
      questions still match the English PDF (not only when chat history exists).
    - `create_retrieval_chain` attaches retrieved documents under the `context`
      key, then runs `create_stuff_documents_chain` to pack those documents into
      the system prompt before the chat history and the latest user message.

    Note: LangChain does not ship `create_conversational_retrieval_chain` as a
    single symbol in current releases; this pair is the documented replacement
    for conversational RAG with explicit control over prompts.
    """
    llm = _make_llm()
    compression_retriever = build_compression_retriever(vectorstore, chunks)

    history_aware_retriever = _build_english_query_retriever(
        llm, compression_retriever
    )

    # Shared strict-grounding rules (see kb_prompts.py and docs/EVOLUTION.md).
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", STRICT_GROUNDED_QA_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


@st.cache_resource(show_spinner="Starting retrieval pipeline (first run only)…")
def get_cached_rag_chain(
    prompt_version: str,
    policy_key: str,
    reranker_model: str,
    index_fingerprint: str,
) -> Any:
    """
    Build the full chain once per Streamlit server process.

    Subsequent reruns and warm-up clicks reuse the in-memory BM25 index and
    cross-encoder weights instead of rebuilding from scratch.
    """
    del prompt_version, policy_key, reranker_model
    vectorstore = get_vectorstore(index_fingerprint)
    policy_chunks = get_policy_chunks()
    return _build_conversational_rag_chain(vectorstore, policy_chunks)


def _ensure_rag_chain() -> None:
    """
    Lazily build the RAG chain on first question (not at page load).

    Loading Chroma, BM25, and BGE-reranker before ``st.chat_input`` blocks the
    whole script and the chat bar never appears on screen.
    """
    index_fp = chroma_index_fingerprint()
    if (
        st.session_state.get("rag_chain") is not None
        and st.session_state.get("rag_prompt_version") == PROMPT_VERSION
        and st.session_state.get("rag_index_fingerprint") == index_fp
    ):
        return

    model = reranker_model_name()
    with st.status("Starting retrieval pipeline…", expanded=True) as status:
        st.caption(
            f"Reranker: `{model}`. "
            "First download can take a few minutes; later loads are much faster."
        )
        st.session_state.rag_chain = get_cached_rag_chain(
            PROMPT_VERSION,
            policy_bm25_cache_key(),
            model,
            index_fp,
        )
        st.session_state.rag_prompt_version = PROMPT_VERSION
        st.session_state.rag_index_fingerprint = index_fp
        status.update(label="Retrieval pipeline ready", state="complete")


def _documents_from_result(result: dict[str, Any]) -> list[Document]:
    raw = result.get("context")
    if raw is None:
        return []
    if isinstance(raw, list) and all(isinstance(d, Document) for d in raw):
        return raw
    return []


def _documents_from_context(raw: Any) -> list[Document]:
    if isinstance(raw, list):
        return [d for d in raw if isinstance(d, Document)]
    return []


def _walk_stream_payload(payload: Any) -> Iterable[tuple[str, Any]]:
    """Yield (context|answer, value) pairs from nested LCEL stream update dicts."""
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if key in ("context", "answer"):
            yield key, value
        elif isinstance(value, dict):
            yield from _walk_stream_payload(value)


def _answer_text(value: Any) -> str:
    """Normalize answer field from stream chunks (str, AIMessage, etc.)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "content"):
        return _message_chunk_text(value)
    return str(value)


def _message_chunk_text(chunk: Any) -> str:
    """Extract printable text from an AIMessageChunk (str or block list)."""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def _is_final_answer_stream(metadata: dict[str, Any]) -> bool:
    """
    Ignore tokens from the history-aware query rewriter LLM.

    Only stream chunks from the grounded answer step (stuff-documents chain).
    """
    if metadata.get("langgraph_node") in ("contextualize", "retrieve"):
        return False
    if metadata.get("langgraph_node") == "generate":
        return True
    tags = metadata.get("tags") or ()
    if "generate" in tags or "combine" in tags:
        return True
    run_name = (metadata.get("run_name") or metadata.get("name") or "").lower()
    if any(token in run_name for token in ("contextual", "rewrite", "history")):
        return False
    if any(token in run_name for token in ("stuff", "combine", "retrieval")):
        return True
    # Without explicit combine/generate markers, rely on stream_mode="updates".
    return False


class _RagStreamAccumulator:
    """Collect answer deltas and retrieved docs while the LCEL chain streams."""

    def __init__(self) -> None:
        self.docs: list[Document] = []
        self.answer: str = ""
        self._prev_answer: str = ""

    def absorb_update(self, update: Any) -> str:
        delta_parts: list[str] = []
        for key, value in _walk_stream_payload(update):
            if key == "context":
                docs = _documents_from_context(value)
                if docs:
                    self.docs = docs
            elif key == "answer":
                text = _answer_text(value)
                if text:
                    delta_parts.append(self._consume_answer_piece(text))
        return "".join(delta_parts)

    def absorb_message(self, message_chunk: Any, metadata: dict[str, Any]) -> str:
        if not _is_final_answer_stream(metadata):
            return ""
        text = _message_chunk_text(message_chunk)
        if not text:
            return ""
        return self._consume_answer_piece(text)

    def absorb_context_only(self, state: Any) -> None:
        """Merge retrieved docs from a stream chunk without touching answer text."""
        if not isinstance(state, dict):
            return
        for key, value in _walk_stream_payload(state):
            if key == "context":
                docs = _documents_from_context(value)
                if docs:
                    self.docs = docs

    def absorb_values(self, state: Any) -> None:
        """Final invoke payload: set docs and answer only if streaming did not already."""
        if not isinstance(state, dict):
            return
        self.absorb_context_only(state)
        if self.answer.strip():
            return
        text = _answer_text(state.get("answer"))
        if text:
            self._consume_answer_piece(text)

    def _consume_answer_piece(self, piece: str) -> str:
        if piece.startswith(self._prev_answer):
            delta = piece[len(self._prev_answer) :]
            self._prev_answer = piece
        else:
            delta = piece
            self._prev_answer += piece
        self.answer = self._prev_answer
        return delta


async def _async_answer_token_events(
    chain: Any,
    inputs: dict[str, Any],
    acc: _RagStreamAccumulator,
    *,
    has_chat_history: bool,
) -> AsyncIterator[str]:
    """
    Token stream via astream_events (langchain_classic chains reject stream_mode).
    """
    llm_stream_calls = 0
    async for event in chain.astream_events(inputs, version="v2"):
        kind = event.get("event")
        if kind == "on_retriever_end":
            docs = _documents_from_context(event.get("data", {}).get("output"))
            if docs:
                acc.docs = docs
        elif kind == "on_chat_model_stream":
            llm_stream_calls += 1
            if has_chat_history and llm_stream_calls == 1:
                continue
            chunk = event.get("data", {}).get("chunk")
            text = _message_chunk_text(chunk)
            if text:
                delta = acc._consume_answer_piece(text)
                if delta:
                    yield delta
        elif kind == "on_chain_end":
            output = event.get("data", {}).get("output")
            if isinstance(output, dict):
                acc.absorb_context_only(output)


def _sync_iterate(async_gen: AsyncIterator[str]) -> Generator[str, None, None]:
    """Bridge async LangChain event stream to a sync generator for Streamlit."""
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(async_gen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()


def _stream_answer_tokens(
    chain: Any,
    inputs: dict[str, Any],
    acc: _RagStreamAccumulator,
    *,
    has_chat_history: bool = False,
) -> Generator[str, None, None]:
    """
    Yield answer token deltas from create_retrieval_chain.

    langchain_classic retrieval chains do not support the ``stream_mode`` kwarg on
    ``.stream()`` (it breaks inside RunnableParallel). We use default ``.stream()``
    first, then ``astream_events`` for true token-by-token output.
    """
    try:
        for chunk in chain.stream(inputs):
            delta = acc.absorb_update(chunk)
            if isinstance(chunk, dict):
                acc.absorb_context_only(chunk)
            if delta:
                yield delta
    except TypeError:
        pass

    if acc.answer.strip():
        return

    yield from _sync_iterate(
        _async_answer_token_events(
            chain, inputs, acc, has_chat_history=has_chat_history
        )
    )


def _init_session_state() -> None:
    if "turn_log" not in st.session_state:
        # Each entry is either a user text or an assistant payload with sources.
        st.session_state.turn_log = []
    if "audit_report" not in st.session_state:
        st.session_state.audit_report = None
    if "audit_topic" not in st.session_state:
        st.session_state.audit_topic = ""
    if "audit_error" not in st.session_state:
        st.session_state.audit_error = ""


def _render_sidebar(
    model_label: str,
    embedding_label: str,
    *,
    vectorstore: Chroma,
    policy_chunks: list[Document],
) -> None:
    """Professional sidebar: branding, status, and session controls."""
    st.sidebar.title("Enterprise Knowledge Base")
    st.sidebar.caption("Grounded policy Q&A · Retrieval audit")

    st.sidebar.markdown("---")
    st.sidebar.subheader("System status")
    st.sidebar.success("Runtime: **Online**")
    st.sidebar.info("Vector store: **ChromaDB** (local persist)")
    st.sidebar.info(f"Chat model: **{model_label}**")
    st.sidebar.info(f"Embeddings: **{embedding_label}**")
    st.sidebar.info(f"Reranker: **{reranker_model_name()}**")

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.markdown(
        "Answers are generated only from retrieved policy document passages. "
        "Expand **Retrieval trace** under any reply to review raw chunks for "
        "compliance and debugging."
    )

    st.sidebar.markdown("---")
    if st.sidebar.button(
        "Warm up retrieval pipeline",
        use_container_width=True,
        help="Preload Chroma, BM25, and BGE reranker before the first question.",
    ):
        _ensure_rag_chain()
        st.sidebar.success("Pipeline is ready.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("💼 Agentic Audit (Beta)")
    audit_topics = [
        "Gifts & Entertainment (礼品与招待)",
        "Conflicts of Interest (利益冲突)",
        "Insider Trading (内幕交易)",
    ]
    selected_topic = st.sidebar.selectbox(
        "选择审计主题",
        audit_topics,
        index=0,
    )
    if st.sidebar.button(
        "📊 一键生成结构化审计清单",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Agent 正在深度扫描合规文档并提取结构化规则..."):
                report: AuditReport = generate_audit_checklist(
                    vectorstore=vectorstore,
                    chunks=policy_chunks,
                    topic=selected_topic,
                )
            st.session_state.audit_report = report
            st.session_state.audit_topic = selected_topic
            st.session_state.audit_error = ""
            st.sidebar.success("结构化审计清单生成完成。")
        except ValueError as exc:
            st.session_state.audit_report = None
            st.session_state.audit_error = str(exc)
            st.sidebar.error("结构化解析失败：模型返回格式异常，请重试或切换主题。")
        except Exception as exc:
            st.session_state.audit_report = None
            st.session_state.audit_error = str(exc)
            st.sidebar.error("生成失败：请检查模型配置/网络后重试。")

    st.sidebar.markdown("---")
    if st.sidebar.button(
        "Clear conversation",
        type="primary",
        use_container_width=True,
        help="Reset chat history in this browser session.",
    ):
        st.session_state.turn_log = []
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Enterprise Knowledge Base Agent",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_custom_css()

    _load_application_env()
    _init_session_state()

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "deepseek-chat") or "deepseek-chat"
    emb_backend = (os.getenv("RAG_EMBEDDING_BACKEND") or "openai").strip().lower()
    embedding_label = (
        "HuggingFace (local)"
        if emb_backend in ("hf", "huggingface")
        else "OpenAI-compatible API"
    )

    if not _optional_str("OPENAI_API_KEY"):
        st.error("Missing `OPENAI_API_KEY` in `.env`. Add your key and reload the app.")
        st.stop()

    index_fp = chroma_index_fingerprint()
    if chroma_index_is_stale():
        st.warning(
            "检测到政策 PDF 或嵌入配置已更新，正在自动重建向量库（首次约 1–3 分钟）…"
        )
        for key in ("rag_chain", "rag_prompt_version", "rag_index_fingerprint"):
            st.session_state.pop(key, None)
        get_vectorstore.clear()
        get_policy_chunks.clear()
        get_cached_rag_chain.clear()

    vectorstore = get_vectorstore(index_fp)
    policy_chunks = get_policy_chunks()
    _render_sidebar(
        model_label=chat_model,
        embedding_label=embedding_label,
        vectorstore=vectorstore,
        policy_chunks=policy_chunks,
    )

    st.title("Enterprise Knowledge Base Agent")
    st.caption(
        "Ask anything about the business conduct policy (PDF knowledge base). Responses are grounded in retrieved "
        "documents; use the retrieval trace to verify evidence for each answer."
    )

    audit_report: AuditReport | None = st.session_state.get("audit_report")
    audit_topic: str = st.session_state.get("audit_topic", "")
    audit_error: str = st.session_state.get("audit_error", "")
    if audit_error:
        st.warning("Agentic Audit 生成未成功。可重试，或切换审计主题后再次生成。")
        with st.expander("查看错误详情（调试）", expanded=False):
            st.code(audit_error)

    if audit_report is not None:
        st.markdown("### Agentic Audit Checklist")
        st.caption(f"Topic: {audit_topic}")
        records_payload = [record.model_dump() for record in audit_report.records]
        audit_df = pd.DataFrame(
            records_payload,
            columns=[
                "policy_topic",
                "rule_summary",
                "amount_limit",
                "exceptions",
                "source_page",
            ],
        )
        st.dataframe(audit_df, use_container_width=True)
        csv_data = audit_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 下载审计清单 CSV",
            data=csv_data,
            file_name="agentic_audit_checklist.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("---")

    if (
        "rag_chain" not in st.session_state
        or st.session_state.get("rag_prompt_version") != PROMPT_VERSION
        or st.session_state.get("rag_index_fingerprint") != index_fp
    ):
        st.session_state.rag_chain = _build_conversational_rag_chain(
            vectorstore, policy_chunks
        )
        st.session_state.rag_prompt_version = PROMPT_VERSION
        st.session_state.rag_index_fingerprint = index_fp

    # Prior turns (wide layout uses horizontal space naturally).
    for entry in st.session_state.turn_log:
        if entry["role"] == "user":
            with st.chat_message("user", avatar=_AVATAR_USER):
                st.write(entry["content"])
        else:
            with st.chat_message("assistant", avatar=_AVATAR_ASSISTANT):
                st.write(entry["content"])
                docs: list[Document] = entry.get("source_documents") or []
                with st.expander("Retrieval trace (source passages)", expanded=False):
                    _render_retrieved_chunks_debug(docs)

    if st.session_state.get("rag_chain") is None:
        st.info(
            "Use the **message box at the bottom of the page** to ask a question. "
            "The **first** startup downloads the reranker model once (~30s with the default "
            "MiniLM model; longer if you set `RAG_RERANKER_MODEL=BAAI/bge-reranker-base`). "
            "After that, reloads are cached. Optional: **Warm up retrieval pipeline** in the sidebar."
        )

    # Render chat input before any heavy model / index work so the bar is always visible.
    user_text = st.chat_input("Message the policy assistant…")
    if not user_text:
        return

    _ensure_rag_chain()

    with st.chat_message("user", avatar=_AVATAR_USER):
        st.write(user_text)

    chat_history: list[BaseMessage] = []
    for entry in st.session_state.turn_log:
        if entry["role"] == "user":
            chat_history.append(HumanMessage(content=entry["content"]))
        else:
            chat_history.append(AIMessage(content=entry["content"]))

    chain_inputs = {"input": user_text, "chat_history": chat_history}
    stream_acc = _RagStreamAccumulator()

    with st.chat_message("assistant", avatar=_AVATAR_ASSISTANT):
        status = st.empty()
        status.caption("Retrieving evidence…")

        def _live_tokens() -> Generator[str, None, None]:
            first_token = True
            for delta in _stream_answer_tokens(
                st.session_state.rag_chain,
                chain_inputs,
                stream_acc,
                has_chat_history=bool(chat_history),
            ):
                if first_token:
                    status.empty()
                    first_token = False
                yield delta

        st.write_stream(_live_tokens())
        answer = stream_acc.answer.strip()
        docs = stream_acc.docs

        if not answer:
            status.caption("Finalizing answer…")
            final_result = st.session_state.rag_chain.invoke(chain_inputs)
            stream_acc.absorb_values(final_result)
            answer = stream_acc.answer.strip()
            docs = stream_acc.docs or _documents_from_result(final_result)
            status.empty()
            if answer:
                st.markdown(answer)
        elif not docs:
            final_result = st.session_state.rag_chain.invoke(chain_inputs)
            docs = _documents_from_result(final_result)

        if not answer:
            st.warning("The model returned an empty answer. Check API key, balance, and logs.")

        with st.expander("Retrieval trace (source passages)", expanded=False):
            _render_retrieved_chunks_debug(docs)

    # Freeze this turn so later reruns render the same text (no re-generation).
    st.session_state.turn_log.append({"role": "user", "content": user_text})
    st.session_state.turn_log.append(
        {
            "role": "assistant",
            "content": answer,
            "source_documents": list(docs),
        }
    )


if __name__ == "__main__":
    main()
