"""
Redis-based LangGraph Checkpointer for production conversation persistence.

This checkpointer stores LangGraph conversation state in Redis, ensuring
conversation context survives server restarts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, AsyncIterator

try:
    import orjson
    def _dumps(obj: Any) -> str:
        return orjson.dumps(obj, default=str).decode("utf-8")
    def _loads(s: str) -> Any:
        return orjson.loads(s)
except ImportError:
    import json
    def _dumps(obj: Any) -> str:
        return json.dumps(obj, default=str)
    def _loads(s: str) -> Any:
        return json.loads(s)

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .redis_utils import get_redis

logger = logging.getLogger(__name__)

# Default TTL for checkpoints (24 hours)
CHECKPOINT_TTL_SECONDS = 86400


class RedisCheckpointer(BaseCheckpointSaver):
    """
    Redis-based checkpoint saver for LangGraph.
    
    Stores conversation checkpoints in Redis with configurable TTL.
    This ensures conversation state persists across server restarts
    while automatically expiring old conversations.
    """

    def __init__(
        self,
        prefix: str = "agentic:checkpoint",
        ttl_seconds: int = CHECKPOINT_TTL_SECONDS,
    ):
        super().__init__(serde=JsonPlusSerializer())
        self._prefix = prefix
        self._ttl = ttl_seconds
        self._client = None

    def _get_client(self):
        """Lazy Redis client initialization."""
        if self._client is None:
            self._client = get_redis()
        return self._client

    def _checkpoint_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        """Generate Redis key for a checkpoint."""
        return f"{self._prefix}:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _metadata_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        """Generate Redis key for checkpoint metadata."""
        return f"{self._prefix}:meta:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _index_key(self, thread_id: str, checkpoint_ns: str) -> str:
        """Generate Redis key for checkpoint index (sorted set)."""
        return f"{self._prefix}:index:{thread_id}:{checkpoint_ns}"

    def _writes_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        """Generate Redis key for pending writes."""
        return f"{self._prefix}:writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _type_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        """Generate Redis key for checkpoint serialization type."""
        return f"{self._prefix}:type:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple by config."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        client = self._get_client()

        if checkpoint_id is None:
            # Get the latest checkpoint
            index_key = self._index_key(thread_id, checkpoint_ns)
            result = client.zrevrange(index_key, 0, 0)
            if not result:
                return None
            # Redis returns bytes if decode_responses=False
            checkpoint_id = result[0]
            if isinstance(checkpoint_id, bytes):
                checkpoint_id = checkpoint_id.decode("utf-8")

        checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        metadata_key = self._metadata_key(thread_id, checkpoint_ns, checkpoint_id)
        type_key = self._type_key(thread_id, checkpoint_ns, checkpoint_id)

        checkpoint_data = client.get(checkpoint_key)
        metadata_data = client.get(metadata_key)
        checkpoint_type = client.get(type_key)

        if not checkpoint_data:
            return None

        try:
            # Get the serialization type (json or bytes/msgpack)
            if checkpoint_type:
                if isinstance(checkpoint_type, bytes):
                    checkpoint_type = checkpoint_type.decode("utf-8")
            else:
                # Fallback: try to detect type from data
                # msgpack typically starts with bytes 0x80-0x9F, 0xA0-0xBF, 0xC0-0xDF, 0xDE, 0xDF, 0xDC, 0xDD
                # JSON typically starts with '{' (0x7B) or '[' (0x5B)
                first_byte = checkpoint_data[0] if isinstance(checkpoint_data, bytes) else ord(checkpoint_data[0])
                if first_byte in (0x7B, 0x5B):  # '{' or '['
                    checkpoint_type = "json"
                else:
                    checkpoint_type = "bytes"  # Assume msgpack/binary
            
            checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_data))
            
            # Handle metadata (JSON string, but comes as bytes when decode_responses=False)
            if metadata_data:
                if isinstance(metadata_data, bytes):
                    metadata_data = metadata_data.decode("utf-8")
                metadata = _loads(metadata_data)
            else:
                metadata = {}
        except Exception as e:
            logger.error("Failed to deserialize checkpoint: %s", e)
            return None

        # Get parent checkpoint id from metadata
        parent_checkpoint_id = metadata.get("parent_checkpoint_id")
        parent_config = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        # Get pending writes
        writes_key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)
        pending_writes_data = client.lrange(writes_key, 0, -1)
        pending_writes = []
        for write_data in pending_writes_data:
            try:
                # Handle bytes from Redis (decode_responses=False)
                if isinstance(write_data, bytes):
                    write_data = write_data.decode("utf-8")
                task_id, channel, value = _loads(write_data)
                pending_writes.append((task_id, channel, value))
            except Exception:
                pass

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread."""
        if config is None:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        client = self._get_client()
        index_key = self._index_key(thread_id, checkpoint_ns)

        # Get checkpoint IDs in reverse chronological order
        if before:
            before_id = before["configurable"]["checkpoint_id"]
            # Get rank of the before checkpoint
            rank = client.zrevrank(index_key, before_id)
            if rank is None:
                return
            start = rank + 1
        else:
            start = 0

        end = start + (limit - 1) if limit else -1
        checkpoint_ids = client.zrevrange(index_key, start, end)

        for checkpoint_id in checkpoint_ids:
            # Handle bytes from Redis (decode_responses=False)
            if isinstance(checkpoint_id, bytes):
                checkpoint_id = checkpoint_id.decode("utf-8")
            checkpoint_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }
            result = self.get_tuple(checkpoint_config)
            if result:
                yield result

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Store a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        client = self._get_client()

        # Serialize checkpoint - store both type and data
        checkpoint_type, checkpoint_data = self.serde.dumps_typed(checkpoint)
        
        # Add parent checkpoint id to metadata
        meta_to_store = dict(metadata) if metadata else {}
        if parent_checkpoint_id:
            meta_to_store["parent_checkpoint_id"] = parent_checkpoint_id
        metadata_data = _dumps(meta_to_store)

        # Store checkpoint, metadata, and type
        checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        metadata_key = self._metadata_key(thread_id, checkpoint_ns, checkpoint_id)
        type_key = self._type_key(thread_id, checkpoint_ns, checkpoint_id)
        index_key = self._index_key(thread_id, checkpoint_ns)

        # Use time-based score or similar.
        # LangGraph v0.2+ uses UUIDv7-like or random UUIDs for checkpoint_ids.
        # If they are UUIDs, converting to float fails if they contain a-f.
        # We need a score for zadd. 
        # We can use timestamp if available in metadata, or just use time.time().
        # Or better, since we want to order them, we can use a lexicographical trick 
        # but redis ZSET scores must be floats.
        
        # Strategy: Use current system timestamp as score.
        # This preserves insertion order, which is generally what we want for "latest".
        import time
        score = time.time()

        pipe = client.pipeline()
        pipe.set(checkpoint_key, checkpoint_data, ex=self._ttl)
        pipe.set(metadata_key, metadata_data, ex=self._ttl)
        pipe.set(type_key, checkpoint_type, ex=self._ttl)  # Store serialization type
        # Use current timestamp as score instead of parsing hex UUID
        pipe.zadd(index_key, {checkpoint_id: score})
        pipe.expire(index_key, self._ttl)
        pipe.execute()

        logger.debug("Stored checkpoint %s for thread %s", checkpoint_id, thread_id)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes for a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        client = self._get_client()
        writes_key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)

        pipe = client.pipeline()
        for channel, value in writes:
            write_data = _dumps([task_id, channel, value])
            pipe.rpush(writes_key, write_data)
        pipe.expire(writes_key, self._ttl)
        pipe.execute()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread."""
        client = self._get_client()
        
        # Handle both dict config and string thread_id
        if isinstance(thread_id, dict):
            thread_id = thread_id.get("configurable", {}).get("thread_id", str(thread_id))
        
        pattern = f"{self._prefix}:*:{thread_id}:*"
        keys = list(client.scan_iter(pattern))
        
        # Also delete index keys
        index_pattern = f"{self._prefix}:index:{thread_id}:*"
        keys.extend(list(client.scan_iter(index_pattern)))
        
        if keys:
            client.delete(*keys)
            logger.info("Deleted %d checkpoint keys for thread %s", len(keys), thread_id)

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Async get a checkpoint tuple by config."""
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async list checkpoints for a thread."""
        for cp in self.list(config, filter=filter, before=before, limit=limit):
            yield cp

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Async store a checkpoint."""
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Async store pending writes for a checkpoint."""
        return self.put_writes(config, writes, task_id)

