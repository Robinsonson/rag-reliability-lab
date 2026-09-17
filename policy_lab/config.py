"""Environment configuration shared by API and ingestion workers."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./data/policy_lab.db"))
    checkpoint_url: str = field(default_factory=lambda: os.getenv("CHECKPOINT_URL", ""))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("POLICY_DATA_DIR", "./data")))
    storage_backend: str = field(default_factory=lambda: os.getenv("STORAGE_BACKEND", "local"))
    s3_endpoint: str = field(default_factory=lambda: os.getenv("S3_ENDPOINT", "http://object-store:8333"))
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", "policies"))
    s3_access_key: str = field(default_factory=lambda: os.getenv("S3_ACCESS_KEY", ""))
    s3_secret_key: str = field(default_factory=lambda: os.getenv("S3_SECRET_KEY", ""))
    s3_region: str = field(default_factory=lambda: os.getenv("S3_REGION", "us-east-1"))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024)
    max_documents: int = field(default_factory=lambda: int(os.getenv("MAX_DOCUMENTS", "10000")))
    max_pages: int = field(default_factory=lambda: int(os.getenv("MAX_DOCUMENT_PAGES", "2000")))
    max_characters: int = field(default_factory=lambda: int(os.getenv("MAX_EXTRACTED_CHARACTERS", "10000000")))
    parser_timeout: int = field(default_factory=lambda: int(os.getenv("PARSER_TIMEOUT_SECONDS", "900")))
    job_lease_seconds: int = field(default_factory=lambda: int(os.getenv("JOB_LEASE_SECONDS", "1800")))
    job_max_attempts: int = field(default_factory=lambda: int(os.getenv("JOB_MAX_ATTEMPTS", "3")))
    embedding_model: str = field(default_factory=lambda: os.getenv("POLICY_EMBEDDING_MODEL", "BAAI/bge-m3"))
    embedding_dimension: int = field(default_factory=lambda: int(os.getenv("POLICY_EMBEDDING_DIMENSION", "1024")))
    reranker_model: str = field(default_factory=lambda: os.getenv("POLICY_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    enable_reranker: bool = field(default_factory=lambda: os.getenv("POLICY_ENABLE_RERANKER", "true").lower() == "true")
    model_cache_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_CACHE_DIR", "./data/models")))
    chunk_size: int = 900
    chunk_overlap: int = 100
    search_candidates: int = 24
    session_hours: int = 8
    secure_cookies: bool = field(default_factory=lambda: os.getenv("SECURE_COOKIES", "false").lower() == "true")
    bootstrap_username: str = field(default_factory=lambda: os.getenv("BOOTSTRAP_USERNAME", "admin"))
    bootstrap_password: str = field(default_factory=lambda: os.getenv("BOOTSTRAP_PASSWORD", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "deepseek-chat"))
    chat_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"))
    chat_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    chat_timeout: int = 60
    chat_max_tokens: int = 1800

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.model_cache_dir = Path(self.model_cache_dir)
        if self.max_upload_bytes <= 0 or self.max_documents <= 0:
            raise ValueError("Upload and document limits must be positive")
