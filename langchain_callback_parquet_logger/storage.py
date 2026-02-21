"""Storage backends for Parquet files."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq

from .config import S3Config


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def write(self, table: pa.Table, filepath: Path) -> None:
        """Write Parquet table to storage."""
        pass

    @abstractmethod
    def exists(self, filepath: Path) -> bool:
        """Check if file exists in storage."""
        pass

    @abstractmethod
    def list_files(self) -> List[str]:
        """Return list of relative parquet file paths under this storage root."""
        pass

    @abstractmethod
    def read_table(self, filepath: Path) -> pa.Table:
        """Read a parquet file and return as a PyArrow table."""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_dir: Path):
        """Initialize local storage with base directory."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, table: pa.Table, filepath: Path) -> None:
        """Write Parquet table to local filesystem."""
        full_path = self.base_dir / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, full_path, compression='snappy')

    def exists(self, filepath: Path) -> bool:
        """Check if file exists locally."""
        return (self.base_dir / filepath).exists()

    def list_files(self) -> List[str]:
        """List all parquet files under base_dir, returning relative paths."""
        return [
            str(p.relative_to(self.base_dir))
            for p in self.base_dir.rglob("*.parquet")
        ]

    def read_table(self, filepath: Path) -> pa.Table:
        """Read a parquet file from local filesystem."""
        return pq.read_table(self.base_dir / filepath)


class S3Storage(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(self, config: S3Config):
        """Initialize S3 storage with configuration."""
        self.config = config
        self._client = None

    @property
    def client(self):
        """Lazy load boto3 client with configured timeouts."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config as BotoConfig
                boto_config = BotoConfig(
                    connect_timeout=self.config.connect_timeout,
                    read_timeout=self.config.read_timeout,
                )
                self._client = boto3.client('s3', config=boto_config)
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3 support. "
                    "Install it with: pip install langchain-callback-parquet-logger[s3]"
                )
        return self._client

    def write(self, table: pa.Table, filepath: Path) -> None:
        """Write Parquet table to S3 with retry logic."""
        # Construct S3 key
        s3_key = f"{self.config.prefix}{filepath}"

        # Write table to buffer
        buffer = BytesIO()
        pq.write_table(table, buffer, compression='snappy')
        buffer.seek(0)

        # Upload with retries
        for attempt in range(self.config.retry_attempts):
            try:
                self.client.put_object(
                    Bucket=self.config.bucket,
                    Key=s3_key,
                    Body=buffer.getvalue()
                )
                return  # Success

            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    # Final attempt failed
                    error_msg = f"Failed to upload to S3 after {self.config.retry_attempts} attempts: {e}"
                    if self.config.on_failure == "error":
                        raise RuntimeError(error_msg) from e
                    else:
                        print(f"S3 upload failed (continuing): {error_msg}")
                        return

                # Exponential backoff before retry
                time.sleep(2 ** attempt)

    def exists(self, filepath: Path) -> bool:
        """Check if file exists in S3."""
        s3_key = f"{self.config.prefix}{filepath}"
        try:
            self.client.head_object(Bucket=self.config.bucket, Key=s3_key)
            return True
        except:
            return False

    def list_files(self) -> List[str]:
        """List all parquet files under the S3 prefix, returning relative paths."""
        paginator = self.client.get_paginator('list_objects_v2')
        files = []
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=self.config.prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.parquet'):
                    relative = key[len(self.config.prefix):]
                    if relative:
                        files.append(relative)
        return files

    def read_table(self, filepath: Path) -> pa.Table:
        """Read a parquet file from S3."""
        s3_key = f"{self.config.prefix}{filepath}"
        response = self.client.get_object(Bucket=self.config.bucket, Key=s3_key)
        data = response['Body'].read()
        return pq.read_table(pa.BufferReader(data))


class CompositeStorage(StorageBackend):
    """Composite storage that writes to multiple backends."""

    def __init__(self, backends: List[StorageBackend]):
        """Initialize with list of storage backends."""
        self.backends = backends

    def write(self, table: pa.Table, filepath: Path) -> None:
        """Write to all configured backends."""
        for backend in self.backends:
            backend.write(table, filepath)

    def exists(self, filepath: Path) -> bool:
        """Check if file exists in any backend."""
        return any(backend.exists(filepath) for backend in self.backends)

    def list_files(self) -> List[str]:
        """List files from all backends, deduplicated (preserving first-seen order)."""
        seen: set = set()
        result = []
        for backend in self.backends:
            try:
                for f in backend.list_files():
                    if f not in seen:
                        seen.add(f)
                        result.append(f)
            except Exception:
                pass
        return result

    def read_table(self, filepath: Path) -> pa.Table:
        """Read from the first backend that successfully returns the file."""
        last_error: Optional[Exception] = None
        for backend in self.backends:
            try:
                return backend.read_table(filepath)
            except Exception as e:
                last_error = e
        raise last_error or FileNotFoundError(f"File not found in any backend: {filepath}")


def create_storage(log_dir: str, s3_config: Optional[S3Config] = None) -> StorageBackend:
    """
    Create appropriate storage backend(s) based on configuration.

    Args:
        log_dir: Local directory for logs
        s3_config: Optional S3 configuration

    Returns:
        StorageBackend instance (may be composite)
    """
    backends = [LocalStorage(Path(log_dir))]

    if s3_config:
        backends.append(S3Storage(s3_config))

    if len(backends) == 1:
        return backends[0]
    else:
        return CompositeStorage(backends)