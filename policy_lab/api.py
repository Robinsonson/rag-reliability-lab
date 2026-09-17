"""Authenticated v2 API. The original unauthenticated demo is not mounted."""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
import uuid
from http.cookies import SimpleCookie
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field, StrictBool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.concurrency import run_in_threadpool

from .auth import LoginLimiter, csrf_token, hash_password, public_user, token_hash, verify_password
from .config import Settings
from .runtime import ThreadBusyError, make_checkpointer, thread_execution

LOG = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parents[1] / "web_v2"
COOKIE = "policy_session"


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=256)


class UserCreate(Credentials):
    role: Literal["reader", "manager", "admin"] = "reader"
    scopes: list[str] = Field(default_factory=lambda: ["general"], max_length=100)


class UserUpdate(BaseModel):
    role: Literal["reader", "manager", "admin"]
    scopes: list[str] = Field(max_length=100)
    active: bool


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=4000)


class Change(BaseModel):
    instruction: str = Field(min_length=3, max_length=4000)


class Resume(BaseModel):
    approve: StrictBool | None = None
    choice_id: str | None = Field(default=None, max_length=160)
    clarification: str | None = Field(default=None, max_length=2000)


class Rollback(BaseModel):
    revision_id: str = Field(max_length=64)
    base_revision_id: str = Field(max_length=64)


class UploadLimitMiddleware:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/api/v2/uploads":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        cookies = SimpleCookie()
        try:
            cookies.load(headers.get(b"cookie", b"").decode("latin1"))
            token = cookies[COOKIE].value if COOKIE in cookies else ""
        except Exception:
            token = ""
        repo = scope["app"].state.repo
        user = await run_in_threadpool(repo.get_session, token_hash(token)) if token else None
        if not user:
            return await JSONResponse({"detail": "Sign in before uploading"}, 401)(scope, receive, send)
        if user["role"] not in ("manager", "admin") or not hmac.compare_digest(
                headers.get(b"x-csrf-token", b"").decode("latin1"), csrf_token(token)):
            return await JSONResponse({"detail": "Upload permission or session verification failed"}, 403)(scope, receive, send)
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = self.limit + 1
        if declared > self.limit:
            return await JSONResponse({"detail": "Upload exceeds configured request size"}, 413)(scope, receive, send)
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > self.limit:
                raise StarletteHTTPException(413, "Upload exceeds configured request size")
            return message

        await self.app(scope, limited_receive, send)


def create_app(settings=None, repository=None, storage=None, checkpointer=None, llm=None):
    if settings is None:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        settings = Settings()
    limiter = LoginLimiter()

    @asynccontextmanager
    async def lifespan(application):
        from .graphs import build_graphs
        from .repository import Repository
        from .storage import make_storage
        with ExitStack() as stack:
            repo = repository or Repository(settings)
            repo.initialize()
            if not repo.list_users():
                if not settings.bootstrap_password:
                    raise RuntimeError("Set BOOTSTRAP_PASSWORD (12+ characters) for the first startup")
                repo.create_user(settings.bootstrap_username, hash_password(settings.bootstrap_password), "admin", ["general"])
            store = storage or make_storage(settings)
            store.ensure_bucket()
            saver = checkpointer or make_checkpointer(settings, stack)
            application.state.repo, application.state.storage = repo, store
            application.state.settings = settings
            application.state.answer_graph, application.state.update_graph = build_graphs(repo, saver, llm)
            yield
            if repository is None:
                repo.engine.dispose()

    app = FastAPI(title="Policy Reliability Workbench", version="2.0.0", lifespan=lifespan)
    app.add_middleware(UploadLimitMiddleware, limit=settings.max_upload_bytes + 1024 * 1024)
    # check_dir=False permits tests to import the API before frontend artifacts exist.
    app.mount("/static/v2", StaticFiles(directory=WEB, check_dir=False), name="v2-static")

    @app.middleware("http")
    async def response_headers(request, call_next):
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin writes are not allowed"}, 403)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        LOG.info("request id=%s method=%s path=%s status=%s elapsed_ms=%.1f", request_id,
                 request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(PermissionError)
    async def denied(request, exc):
        return JSONResponse({"detail": "You do not have permission for this action"}, 403)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Resource not found"}, 404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, 400)

    @app.exception_handler(ThreadBusyError)
    async def busy(request, exc):
        return JSONResponse({"detail": str(exc)}, 409)

    from .llm import ModelUnavailable
    @app.exception_handler(ModelUnavailable)
    async def model_unavailable(request, exc):
        return JSONResponse({"detail": str(exc)}, 503)

    # ConflictError is imported after the repository module becomes available.
    from .repository import ConflictError
    @app.exception_handler(ConflictError)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, 409)

    def actor(request: Request):
        token = request.cookies.get(COOKIE, "")
        user = request.app.state.repo.get_session(token_hash(token)) if token else None
        if not user or not user.get("active", True):
            raise HTTPException(401, "Sign in to continue")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            supplied = request.headers.get("X-CSRF-Token", "")
            if not supplied or not hmac.compare_digest(supplied, csrf_token(token)):
                raise HTTPException(403, "Session verification failed; refresh and retry")
        return user

    def admin(user=Depends(actor)):
        if user["role"] != "admin":
            raise HTTPException(403, "Administrator role required")
        return user

    def manager(user=Depends(actor)):
        if user["role"] not in ("manager", "admin"):
            raise HTTPException(403, "Policy manager role required")
        return user

    def session_result(user, token):
        return {"user": public_user(user), "csrf_token": csrf_token(token), "limits": {
            "max_upload_bytes": settings.max_upload_bytes, "max_documents": settings.max_documents,
            "max_pages": settings.max_pages, "formats": ["pdf", "docx", "pptx", "xlsx", "html", "md", "txt"],
            "capacity_verified": False,
        }}

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/v2/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/v2/login")
    def login(body: Credentials, request: Request):
        key = request.client.host if request.client else "unknown"
        if not limiter.attempt(key):
            raise HTTPException(429, "Too many login attempts; retry in five minutes")
        user = request.app.state.repo.find_user(body.username.strip())
        if not user or not user.get("active", True) or not verify_password(body.password, user["password_hash"]):
            raise HTTPException(401, "Invalid username or password")
        token = secrets.token_urlsafe(40)
        request.app.state.repo.create_session(token_hash(token), user["id"], time.time() + settings.session_hours * 3600)
        response = JSONResponse(session_result(user, token))
        response.set_cookie(COOKIE, token, httponly=True, secure=settings.secure_cookies, samesite="strict", max_age=settings.session_hours * 3600)
        return response

    @app.get("/api/v2/session")
    def session(request: Request, user=Depends(actor)):
        return session_result(user, request.cookies[COOKIE])

    @app.post("/api/v2/logout")
    def logout(request: Request, user=Depends(actor)):
        request.app.state.repo.delete_session(token_hash(request.cookies[COOKIE]))
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE)
        return response

    @app.get("/api/v2/users")
    def users(request: Request, user=Depends(admin)):
        return {"users": [public_user(u) for u in request.app.state.repo.list_users()]}

    @app.post("/api/v2/users", status_code=201)
    def create_user(body: UserCreate, request: Request, user=Depends(admin)):
        result = request.app.state.repo.create_user(body.username.strip(), hash_password(body.password), body.role, body.scopes)
        return {"user": public_user(result)}

    @app.patch("/api/v2/users/{identifier}")
    def update_user(identifier: str, body: UserUpdate, request: Request, user=Depends(admin)):
        if identifier == user["id"] and (not body.active or body.role != "admin"):
            raise HTTPException(400, "Use another administrator to change your own administrator access")
        result = request.app.state.repo.update_user(identifier, body.role, body.scopes, body.active)
        return {"user": public_user(result)}

    @app.get("/api/v2/policies")
    def policies(request: Request, limit: int = 50, offset: int = 0, user=Depends(actor)):
        return {"policies": request.app.state.repo.list_policies(user, limit=min(max(limit, 1), 100), offset=max(offset, 0))}

    @app.post("/api/v2/uploads", status_code=202)
    def upload(request: Request, file: UploadFile = File(...), title: str = Form(..., max_length=200),
               scope: str = Form("general", max_length=100), replaces: str | None = Form(None), user=Depends(manager)):
        title, scope = title.strip(), scope.strip()
        if not title or not scope or (user["role"] != "admin" and scope not in user["scopes"]):
            raise HTTPException(403, "Choose an authorized policy scope")
        name = (file.filename or "document").replace("\\", "/").rsplit("/", 1)[-1][:200]
        if Path(name).suffix.lower() not in (".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"):
            raise HTTPException(400, "Unsupported document format")
        digest, size = hashlib.sha256(), 0
        try:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, "File exceeds the configured upload size")
                digest.update(chunk)
            if size == 0:
                raise HTTPException(400, "Choose a nonempty document")
            file.file.seek(0)
            key = "originals/" + uuid.uuid4().hex + Path(name).suffix.lower()
            request.app.state.storage.put_file(key, file.file)
            try:
                return request.app.state.repo.enqueue_upload(user, title, scope, key, name, digest.hexdigest(), size, replaces or None)
            except Exception:
                request.app.state.storage.delete(key)
                raise
        finally:
            file.file.close()

    @app.get("/api/v2/jobs")
    def jobs(request: Request, user=Depends(actor)):
        return {"jobs": request.app.state.repo.list_jobs(user)}

    @app.get("/api/v2/jobs/{identifier}")
    def job(identifier: str, request: Request, user=Depends(actor)):
        return request.app.state.repo.get_job(user, identifier)

    @app.post("/api/v2/jobs/{identifier}/retry", status_code=202)
    def retry(identifier: str, request: Request, user=Depends(manager)):
        return request.app.state.repo.retry_job(user, identifier)

    @app.get("/api/v2/policies/{identifier}/revisions")
    def revisions(identifier: str, request: Request, user=Depends(actor)):
        return {"revisions": request.app.state.repo.list_revisions(user, identifier)}

    @app.get("/api/v2/revisions/{identifier}")
    def revision(identifier: str, request: Request, user=Depends(actor)):
        return request.app.state.repo.get_revision(user, identifier)

    @app.get("/api/v2/revisions/{identifier}/original")
    def original(identifier: str, request: Request, user=Depends(actor)):
        rev = request.app.state.repo.get_revision(user, identifier)
        key = rev.get("original_key")
        if not key:
            raise HTTPException(404, "No original attachment is associated with this revision")
        suffix = Path(rev.get("original_name", "document.bin")).suffix.lower()
        if suffix not in (".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"):
            suffix = ".bin"
        def content():
            with request.app.state.storage.open(key) as stream:
                while data := stream.read(1024 * 1024):
                    yield data
        return StreamingResponse(content(), media_type="application/octet-stream", headers={
            "Content-Disposition": f'attachment; filename="original{suffix}"'})

    @app.get("/api/v2/revisions/{identifier}/export")
    def export(identifier: str, request: Request, user=Depends(actor)):
        rev = request.app.state.repo.get_revision(user, identifier)
        policy = request.app.state.repo.get_policy(user, rev["policy_id"])
        lines = [f"# {policy['title']}", "", f"Revision: {rev.get('number', '')} · ID: {rev['id']}",
                 "", "Canonical policy text / 系统正式正文", ""]
        for block in rev["blocks"]:
            lines.extend([block["text"], ""])
        return PlainTextResponse("\n".join(lines), media_type="text/markdown", headers={
            "Content-Disposition": f'attachment; filename="policy-v{rev.get("number", 1)}.md"'})

    @app.post("/api/v2/policies/{identifier}/rollback", status_code=202)
    def rollback(identifier: str, body: Rollback, request: Request, user=Depends(manager)):
        return request.app.state.repo.rollback(user, identifier, body.revision_id, body.base_revision_id)

    @app.get("/api/v2/audit")
    def audit(request: Request, user=Depends(actor)):
        return {"events": request.app.state.repo.audit_events(user)}

    def graph_payload(graph, thread_id):
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        interrupts = [i.value for task in snapshot.tasks for i in task.interrupts]
        result = snapshot.values.get("result")
        recoverable = not interrupts and not result and set(snapshot.next) == {"publish"} and snapshot.values.get("approved") is True
        status = "interrupted" if interrupts else "recoverable" if recoverable else "failed" if snapshot.next else "completed"
        payload = {"thread_id": thread_id, "status": status, "interrupts": interrupts, "result": result}
        if recoverable:
            payload["recovery"] = {"type": "publish", "message": "Publication was interrupted after approval; resume processing safely."}
        return payload

    def authorize_saved_change(repo, user, snapshot):
        # Both graphs share the checkpoint database. An owned Q&A thread is not
        # a change request and must never be replayed through change endpoints.
        values = snapshot.values
        if not values.get("instruction") or values.get("question"):
            raise HTTPException(404, "Policy change workflow not found")
        if values.get("proposal_id"):
            repo.get_proposal(user, values["proposal_id"])
        for candidate in values.get("candidates", []):
            repo.get_policy(user, candidate["policy_id"])
        selected = values.get("selected")
        if selected:
            repo.get_policy(user, selected["policy_id"])

    def invoke_graph(request, graph, user, thread_id, value):
        repo = request.app.state.repo
        repo.authorize_thread(user, thread_id)
        with thread_execution(repo, thread_id):
            if isinstance(value, Command):
                snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
                authorize_saved_change(repo, user, snapshot)
                pending = [i.value for task in snapshot.tasks for i in task.interrupts]
                if not pending:
                    if snapshot.values.get("result"):
                        return graph_payload(graph, thread_id)
                    if set(snapshot.next) == {"publish"} and snapshot.values.get("approved") is True and value.resume == {"approve": True}:
                        # This continues a previously recorded approval. It does not
                        # treat a fresh approval as permission to skip a review.
                        value = None
                    else:
                        raise HTTPException(409, "This workflow is not waiting for a response; start a new request")
                else:
                    current, reply = pending[0], value.resume
                    if current.get("type") == "review":
                        repo.get_proposal(user, current["proposal"]["id"])
                        if set(reply) != {"approve"} or type(reply["approve"]) is not bool:
                            raise HTTPException(400, "Review requires an explicit approval or rejection")
                    elif current.get("type") == "clarification":
                        if set(reply) == {"choice_id"}:
                            if reply["choice_id"] not in {c["id"] for c in current.get("candidates", [])}:
                                raise HTTPException(400, "Choose one of the displayed candidates")
                        elif set(reply) != {"clarification"} or not reply["clarification"].strip():
                            raise HTTPException(400, "Provide a displayed choice or a business clarification")
            graph.invoke(value, {"configurable": {"thread_id": thread_id}, "recursion_limit": 32}, durability="sync")
            return graph_payload(graph, thread_id)

    @app.post("/api/v2/answers")
    def answer(body: Question, request: Request, user=Depends(actor)):
        thread_id = uuid.uuid4().hex
        request.app.state.repo.create_thread(user, thread_id)
        return invoke_graph(request, request.app.state.answer_graph, user, thread_id,
                            {"actor_id": user["id"], "question": body.question})

    @app.post("/api/v2/retrieval")
    def retrieval(body: Question, request: Request, user=Depends(actor)):
        repo = request.app.state.repo
        snapshot = repo.snapshot(user)
        started = time.perf_counter()
        evidence = repo.search(user, body.question, snapshot=snapshot)
        return {"evidence": evidence, "retrieval_ms": round((time.perf_counter() - started) * 1000, 2)}

    @app.post("/api/v2/changes")
    def change(body: Change, request: Request, user=Depends(manager)):
        thread_id = uuid.uuid4().hex
        request.app.state.repo.create_thread(user, thread_id)
        return invoke_graph(request, request.app.state.update_graph, user, thread_id,
                            {"actor_id": user["id"], "instruction": body.instruction})

    @app.get("/api/v2/changes/{thread_id}")
    def change_status(thread_id: str, request: Request, user=Depends(manager)):
        request.app.state.repo.authorize_thread(user, thread_id)
        snapshot = request.app.state.update_graph.get_state({"configurable": {"thread_id": thread_id}})
        authorize_saved_change(request.app.state.repo, user, snapshot)
        payload = graph_payload(request.app.state.update_graph, thread_id)
        # Revalidate the proposal's current ACL before replaying saved document text.
        for item in payload["interrupts"]:
            if item.get("type") == "review":
                request.app.state.repo.get_proposal(user, item["proposal"]["id"])
            if item.get("type") == "clarification":
                for evidence in item.get("candidates", []):
                    if evidence.get("policy_id"):
                        request.app.state.repo.get_policy(user, evidence["policy_id"])
        return payload

    @app.post("/api/v2/changes/{thread_id}/resume")
    def resume(thread_id: str, body: Resume, request: Request, user=Depends(manager)):
        data = body.model_dump(exclude_none=True)
        if len(data) != 1:
            raise HTTPException(400, "Provide one approval, choice, or clarification")
        return invoke_graph(request, request.app.state.update_graph, user, thread_id, Command(resume=data))

    return app


app = create_app()
