"""Immutable object keys, streaming S3 storage, and a local test adapter."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import shutil


def _safe_key(key):
    value = PurePosixPath(key)
    if not key or value.is_absolute() or ".." in value.parts or "\\" in key or ":" in key:
        raise ValueError("Invalid object key")
    return value


class LocalStorage:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _path(self, key):
        path = self.root.joinpath(*_safe_key(key).parts).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Invalid object key")
        return path

    def ensure_bucket(self):
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file(self, key, fileobj):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as output:
                shutil.copyfileobj(fileobj, output, length=1024*1024)
        except FileExistsError:
            raise ValueError("Object already exists") from None

    def open(self, key):
        return self._path(key).open("rb")

    def delete(self, key):
        self._path(key).unlink(missing_ok=True)


class S3Storage:
    def __init__(self, settings):
        import boto3
        self.bucket = settings.s3_bucket
        self.region = settings.s3_region
        self.client = boto3.client("s3", endpoint_url=settings.s3_endpoint or None,
            aws_access_key_id=settings.s3_access_key or None,
            aws_secret_access_key=settings.s3_secret_key or None,
            region_name=settings.s3_region)

    def ensure_bucket(self):
        from botocore.exceptions import ClientError
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) not in {"404", "NoSuchBucket"}:
                raise
            arguments = {"Bucket": self.bucket}
            if self.region != "us-east-1":
                arguments["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
            self.client.create_bucket(**arguments)

    def put_file(self, key, fileobj):
        _safe_key(key)
        # Keys are unguessable server-generated IDs; upload_fileobj streams multipart.
        self.client.upload_fileobj(fileobj, self.bucket, key)

    @contextmanager
    def open(self, key):
        _safe_key(key)
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        try:
            yield body
        finally:
            body.close()

    def delete(self, key):
        _safe_key(key)
        self.client.delete_object(Bucket=self.bucket, Key=key)


def make_storage(settings):
    if settings.storage_backend == "s3":
        return S3Storage(settings)
    if settings.storage_backend == "local":
        return LocalStorage(settings.data_dir / "objects")
    raise ValueError("Unsupported storage backend")
