"""Explicit environment initialization and offline model preparation commands."""
from __future__ import annotations

import argparse
import secrets
import time
from contextlib import ExitStack
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "setup-env", "download-models"])
    args = parser.parse_args()
    if args.command == "setup-env":
        destination = Path(".env.v2")
        if destination.exists():
            raise SystemExit(".env.v2 already exists; it was not overwritten")
        template = Path(".env.v2.example").read_text(encoding="utf-8")
        for name in ["POSTGRES_PASSWORD", "S3_SECRET_KEY", "BOOTSTRAP_PASSWORD"]:
            template = template.replace(f"{name}=replace-with-a-long-random-password", f"{name}={secrets.token_urlsafe(32)}")
        destination.write_text(template, encoding="utf-8")
        try:
            destination.chmod(0o600)
        except OSError:
            pass
        print("Created private .env.v2. Open it locally to read the initial administrator password and configure the model key.")
        return

    from .config import Settings
    settings = Settings()
    if args.command == "download-models":
        from huggingface_hub import snapshot_download
        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(settings.embedding_model, cache_dir=str(settings.model_cache_dir))
        if settings.enable_reranker:
            snapshot_download(settings.reranker_model, cache_dir=str(settings.model_cache_dir))
        from docling.utils.model_downloader import download_models
        download_models(output_dir=settings.model_cache_dir / "docling", with_layout=True,
                        with_tableformer=True, with_code_formula=False,
                        with_picture_classifier=False, with_rapidocr=False)
        print("Model assets prepared. This command downloads model weights; it does not send policy documents.")
        return

    from .auth import hash_password
    from .repository import Repository
    from .runtime import make_checkpointer
    from .storage import make_storage
    repo = Repository(settings)
    repo.initialize()
    if not repo.list_users():
        if not settings.bootstrap_password:
            raise SystemExit("BOOTSTRAP_PASSWORD is required for first initialization")
        repo.create_user(settings.bootstrap_username, hash_password(settings.bootstrap_password), "admin", ["general"])
    with ExitStack() as stack:
        make_checkpointer(settings, stack)
    storage = make_storage(settings)
    for attempt in range(30):
        try:
            storage.ensure_bucket()
            break
        except Exception:
            if attempt == 29:
                raise
            time.sleep(2)
    repo.engine.dispose()
    print("Database, graph checkpoints, object bucket and initial account initialized.")


if __name__ == "__main__":
    main()
