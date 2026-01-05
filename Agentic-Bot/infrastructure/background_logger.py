"""
Enterprise-grade background logging for MongoDB conversation history.

This module provides non-blocking, async conversation logging with:
- Bounded async queue to prevent memory issues under load
- Graceful shutdown with configurable drain timeout
- Automatic retry with exponential backoff
- Comprehensive metrics for observability
- Fallback to sync logging when queue is full (backpressure handling)
- Thread-safe singleton pattern

Usage:
    from agentic.infrastructure.background_logger import BackgroundLogger
    
    # Get singleton instance
    logger = BackgroundLogger.get_instance()
    
    # Start the worker (call during app startup)
    await logger.start()
    
    # Enqueue logs (non-blocking)
    await logger.enqueue(session_id, user_msg, bot_msg, metadata)
    
    # Shutdown gracefully (call during app shutdown)
    await logger.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
    SGT_TZ = ZoneInfo("Asia/Singapore")
except Exception:
    SGT_TZ = timezone(timedelta(hours=8))

from .mongo_history import log_history
from .metrics import (
    AGENTIC_LATENCY,
)

# Try to import prometheus metrics, create if not exist
try:
    from .metrics import Counter, Gauge, Histogram
    
    # Background logger specific metrics
    BG_LOG_QUEUE_SIZE = Gauge(
        "agentic_bg_log_queue_size",
        "Current number of items in background log queue"
    )
    BG_LOG_ENQUEUED_TOTAL = Counter(
        "agentic_bg_log_enqueued_total",
        "Total conversation logs enqueued",
        ["status"]  # success, dropped, fallback_sync
    )
    BG_LOG_PROCESSED_TOTAL = Counter(
        "agentic_bg_log_processed_total",
        "Total conversation logs processed by background worker",
        ["status"]  # success, error
    )
    BG_LOG_LATENCY = Histogram(
        "agentic_bg_log_latency_seconds",
        "Time from enqueue to successful write",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    BG_LOG_RETRY_TOTAL = Counter(
        "agentic_bg_log_retry_total",
        "Total retry attempts for failed log writes"
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_QUEUE_SIZE = 1000
DEFAULT_DRAIN_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_RETRY_DELAY_SECONDS = 0.1


@dataclass
class LogEntry:
    """Immutable log entry for queue processing."""
    session_id: str
    user_message: str
    assistant_message: str
    metadata: Optional[Dict[str, Any]]
    timestamp: datetime
    enqueue_time: float  # time.perf_counter() for latency tracking
    retry_count: int = 0


class BackgroundLogger:
    """
    Enterprise-grade background logger for MongoDB conversation history.
    
    Features:
    - Async queue-based processing to avoid blocking request handlers
    - Bounded queue with configurable size to prevent memory exhaustion
    - Graceful shutdown that drains pending logs
    - Automatic retry with exponential backoff for transient failures
    - Fallback to synchronous logging when queue is full
    - Prometheus metrics for observability
    
    Usage:
        # At startup (in main.py lifespan):
        bg_logger = BackgroundLogger()
        await bg_logger.start()
        
        # Store reference for enqueue_log to use
        set_background_logger(bg_logger)
        
        # Throughout the application:
        await enqueue_log(session_id, user_msg, bot_msg)
        
        # At shutdown:
        await bg_logger.stop()
    """
    
    def __init__(
        self,
        max_queue_size: int = DEFAULT_QUEUE_SIZE,
        drain_timeout_seconds: float = DEFAULT_DRAIN_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_retry_delay_seconds: float = DEFAULT_BASE_RETRY_DELAY_SECONDS,
    ):
        """
        Initialize BackgroundLogger.
        
        Args:
            max_queue_size: Maximum entries in queue before dropping/fallback
            drain_timeout_seconds: Max time to wait for queue drain on shutdown
            max_retries: Maximum retry attempts for failed writes
            base_retry_delay_seconds: Base delay for exponential backoff
        """
        self._queue: asyncio.Queue[Optional[LogEntry]] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._drain_timeout = drain_timeout_seconds
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay_seconds
        self._max_queue_size = max_queue_size
        
        logger.info(
            "BackgroundLogger initialized: max_queue=%d, drain_timeout=%.1fs, max_retries=%d",
            max_queue_size, drain_timeout_seconds, max_retries
        )
    
    async def start(self) -> None:
        """
        Start the background worker task.
        
        Safe to call multiple times - will only start one worker.
        Should be called during application startup.
        """
        if self._running:
            logger.debug("BackgroundLogger already running")
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(
            self._worker_loop(),
            name="background_logger_worker"
        )
        logger.info("BackgroundLogger worker started")
    
    async def stop(self) -> None:
        """
        Stop the background worker gracefully.
        
        Waits for pending logs to be processed up to drain_timeout_seconds.
        Should be called during application shutdown.
        """
        if not self._running:
            logger.debug("BackgroundLogger not running")
            return
        
        logger.info(
            "BackgroundLogger stopping, draining queue (size=%d, timeout=%.1fs)",
            self._queue.qsize(), self._drain_timeout
        )
        
        self._running = False
        
        # Signal worker to stop by sending None sentinel
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            # Queue is full, worker will see _running=False anyway
            pass
        
        # Wait for worker to finish with timeout
        if self._worker_task:
            try:
                await asyncio.wait_for(
                    self._worker_task,
                    timeout=self._drain_timeout
                )
                logger.info("BackgroundLogger worker stopped gracefully")
            except asyncio.TimeoutError:
                logger.warning(
                    "BackgroundLogger drain timeout, %d entries may be lost",
                    self._queue.qsize()
                )
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            finally:
                self._worker_task = None
    
    async def enqueue(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Enqueue a conversation log entry for background processing.
        
        Non-blocking. If queue is full, falls back to synchronous logging
        to ensure the log is not lost (with a warning).
        
        Args:
            session_id: Unique session identifier
            user_message: User's message
            assistant_message: Bot's response
            metadata: Optional metadata dict
            
        Returns:
            True if enqueued successfully, False if fallback was used
        """
        entry = LogEntry(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            metadata=metadata,
            timestamp=datetime.now(SGT_TZ),
            enqueue_time=time.perf_counter(),
        )
        
        try:
            # Non-blocking put
            self._queue.put_nowait(entry)
            
            if METRICS_AVAILABLE:
                BG_LOG_ENQUEUED_TOTAL.labels(status="success").inc()
                BG_LOG_QUEUE_SIZE.set(self._queue.qsize())
            
            logger.debug(
                "BackgroundLogger.enqueue: session=%s queue_size=%d",
                session_id, self._queue.qsize()
            )
            return True
            
        except asyncio.QueueFull:
            # Backpressure: queue is full, fall back to sync logging
            logger.warning(
                "BackgroundLogger queue full (size=%d), falling back to sync logging for session=%s",
                self._max_queue_size, session_id
            )
            
            if METRICS_AVAILABLE:
                BG_LOG_ENQUEUED_TOTAL.labels(status="fallback_sync").inc()
            
            # Run sync log in thread pool to avoid blocking event loop
            try:
                await asyncio.to_thread(
                    log_history,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    ts=entry.timestamp,
                    metadata=metadata,
                )
                logger.debug(
                    "BackgroundLogger.fallback_sync: session=%s completed",
                    session_id
                )
            except Exception as e:
                logger.error(
                    "BackgroundLogger.fallback_sync failed: session=%s error=%s",
                    session_id, str(e)
                )
            
            return False
    
    async def _worker_loop(self) -> None:
        """
        Background worker that processes log entries from the queue.
        
        Runs until stop() is called. Handles retries with exponential backoff.
        """
        logger.debug("BackgroundLogger worker loop starting")
        
        while self._running or not self._queue.empty():
            try:
                # Wait for entry with timeout to check _running periodically
                try:
                    entry = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # None sentinel signals shutdown
                if entry is None:
                    self._queue.task_done()
                    if not self._running:
                        break
                    continue
                
                # Process the entry
                await self._process_entry(entry)
                self._queue.task_done()
                
                if METRICS_AVAILABLE:
                    BG_LOG_QUEUE_SIZE.set(self._queue.qsize())
                    
            except asyncio.CancelledError:
                logger.debug("BackgroundLogger worker cancelled")
                raise
            except Exception as e:
                logger.error("BackgroundLogger worker error: %s", str(e))
                # Continue processing other entries
        
        logger.debug("BackgroundLogger worker loop exiting")
    
    async def _process_entry(self, entry: LogEntry) -> None:
        """
        Process a single log entry with retry logic.
        
        Uses exponential backoff for transient failures.
        """
        while entry.retry_count <= self._max_retries:
            try:
                # Run sync MongoDB operation in thread pool
                await asyncio.to_thread(
                    log_history,
                    session_id=entry.session_id,
                    user_message=entry.user_message,
                    assistant_message=entry.assistant_message,
                    ts=entry.timestamp,
                    metadata=entry.metadata,
                )
                
                # Success metrics
                if METRICS_AVAILABLE:
                    latency = time.perf_counter() - entry.enqueue_time
                    BG_LOG_LATENCY.observe(latency)
                    BG_LOG_PROCESSED_TOTAL.labels(status="success").inc()
                
                logger.debug(
                    "BackgroundLogger.processed: session=%s latency=%.3fs retries=%d",
                    entry.session_id,
                    time.perf_counter() - entry.enqueue_time,
                    entry.retry_count
                )
                return
                
            except Exception as e:
                entry.retry_count += 1
                
                if entry.retry_count > self._max_retries:
                    # Max retries exceeded, log and drop
                    logger.error(
                        "BackgroundLogger.max_retries_exceeded: session=%s error=%s",
                        entry.session_id, str(e)
                    )
                    if METRICS_AVAILABLE:
                        BG_LOG_PROCESSED_TOTAL.labels(status="error").inc()
                    return
                
                # Exponential backoff: 0.1s, 0.2s, 0.4s, ...
                delay = self._base_retry_delay * (2 ** (entry.retry_count - 1))
                
                logger.warning(
                    "BackgroundLogger.retry: session=%s attempt=%d/%d delay=%.2fs error=%s",
                    entry.session_id,
                    entry.retry_count,
                    self._max_retries,
                    delay,
                    str(e)
                )
                
                if METRICS_AVAILABLE:
                    BG_LOG_RETRY_TOTAL.inc()
                
                await asyncio.sleep(delay)
    
    @property
    def queue_size(self) -> int:
        """Current number of entries in the queue."""
        return self._queue.qsize()
    
    @property
    def is_running(self) -> bool:
        """Whether the background worker is running."""
        return self._running


# ============================================
# Module-level Reference (set at startup)
# ============================================

_background_logger: Optional[BackgroundLogger] = None


def set_background_logger(bg_logger: BackgroundLogger) -> None:
    """Set the module-level background logger instance.
    
    Called once at application startup after creating the BackgroundLogger.
    """
    global _background_logger
    _background_logger = bg_logger
    logger.debug("Background logger reference set")


def get_background_logger() -> Optional[BackgroundLogger]:
    """Get the module-level background logger instance."""
    return _background_logger


async def enqueue_log(
    session_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Enqueue a log entry for background processing.
    
    Uses the module-level BackgroundLogger set at startup.
    Falls back to sync logging if BackgroundLogger not available.
    
    Args:
        session_id: Unique session identifier
        user_message: User's message
        assistant_message: Bot's response
        metadata: Optional metadata dict
    """
    if _background_logger is None or not _background_logger.is_running:
        # BackgroundLogger not initialized or not running, use sync fallback
        logger.debug(
            "BackgroundLogger not available, using sync logging for session=%s",
            session_id
        )
        try:
            await asyncio.to_thread(
                log_history,
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(
                "Sync log_history failed: session=%s error=%s",
                session_id, str(e)
            )
        return
    
    await _background_logger.enqueue(
        session_id=session_id,
        user_message=user_message,
        assistant_message=assistant_message,
        metadata=metadata,
    )
