"""
Infrastructure module for production-grade agentic chatbot.
Provides Redis session management, MongoDB history persistence, metrics, LangGraph checkpointer,
LLM initialization, and vector store client.
"""

from .redis_utils import (
    get_redis,
    RedisLock,
    SessionCache,
    RateLimiter,
    Deduplicator,
    OrderGuard,
    session_lock_key,
)
from .session import SessionManager
from .mongo_history import log_history, get_history, clear_history
from .background_logger import BackgroundLogger, enqueue_log, set_background_logger, get_background_logger
from .redis_checkpointer import RedisCheckpointer
from .llm import (
    initialize_models,
    get_chat_llm,
    get_response_llm,
    get_embeddings,
    cleanup as llm_cleanup,
)
from .vector_store import initialize_weaviate, get_weaviate_client, close_weaviate_client, WEAVIATE_AVAILABLE
from .metrics import (
    AGENTIC_MESSAGES_TOTAL,
    AGENTIC_LATENCY,
    SESSION_CACHE_HITS,
    SESSION_CACHE_MISSES,
    WA_MESSAGES_PROCESSED_TOTAL,
    LIVE_AGENT_HANDOFFS,
    POLICY_VIOLATIONS,
    REDIS_LOCK_TIMEOUTS,
    # Memory management metrics
    MEMORY_SUMMARIZATION_TOTAL,
    MEMORY_SUMMARIZATION_LATENCY,
    MEMORY_TOKENS_BEFORE_TRIM,
    MEMORY_TOKENS_AFTER_TRIM,
    MEMORY_MESSAGES_PRUNED,
    MEMORY_SUMMARY_LENGTH,
    # Tool metrics
    TOOL_CALLS_TOTAL,
    TOOL_LATENCY,
    TOOL_ERRORS_TOTAL,
    # LLM metrics
    LLM_CALLS_TOTAL,
    LLM_LATENCY,
    # Weaviate metrics
    WEAVIATE_QUERIES_TOTAL,
    WEAVIATE_LATENCY,
    # Message handling metrics
    MESSAGE_CREATED_TOTAL,
    MESSAGE_REMOVED_TOTAL,
    MESSAGE_WITH_ID_TOTAL,
    MESSAGE_WITH_METADATA_TOTAL,
    LLM_INPUT_TOKENS_TOTAL,
    LLM_OUTPUT_TOKENS_TOTAL,
    LLM_TOKENS_PER_CALL,
    MESSAGE_HISTORY_SIZE,
    MESSAGE_HISTORY_TOKENS,
    # Autonomous routing metrics
    AUTONOMOUS_ROUTING_TOTAL,
    SELF_CORRECTION_TOTAL,
    REFLECTION_LATENCY,
    INTERRUPT_REQUESTS_TOTAL,
    ROUTING_DECISION_LATENCY,
    COMMAND_RETURNS_TOTAL,
    # Conversation flow metrics
    INTENT_CLASSIFICATION_TOTAL,
    RECOMMENDATION_GIVEN_TOTAL,
    PURCHASE_LINK_GENERATED_TOTAL,
    CONVERSATION_TURNS,
    # Multi-turn conversation metrics
    PHASE_TRANSITION_TOTAL,
    PHASE_DURATION_TURNS,
    PRONOUN_RESOLUTION_TOTAL,
    REFERENCE_CONTEXT_UPDATES,
    INTENT_WITH_SUMMARY_TOTAL,
    CONVERSATION_PHASE_CURRENT,
)

__all__ = [
    # Redis utilities
    "get_redis",
    "RedisLock",
    "SessionCache",
    "RateLimiter",
    "Deduplicator",
    "OrderGuard",
    "session_lock_key",
    # Session management
    "SessionManager",
    # MongoDB history
    "log_history",
    "get_history",
    "clear_history",
    # Background logger (non-blocking MongoDB)
    "BackgroundLogger",
    "enqueue_log",
    "set_background_logger",
    "get_background_logger",
    # LangGraph checkpointer
    "RedisCheckpointer",
    # LLM (thread-safe singletons)
    "initialize_models",
    "get_chat_llm",
    "get_response_llm",
    "get_embeddings",
    "llm_cleanup",
    # Vector store
    "initialize_weaviate",
    "get_weaviate_client",
    "close_weaviate_client",
    "WEAVIATE_AVAILABLE",
    # Metrics - General
    "AGENTIC_MESSAGES_TOTAL",
    "AGENTIC_LATENCY",
    "SESSION_CACHE_HITS",
    "SESSION_CACHE_MISSES",
    "WA_MESSAGES_PROCESSED_TOTAL",
    "LIVE_AGENT_HANDOFFS",
    "POLICY_VIOLATIONS",
    "REDIS_LOCK_TIMEOUTS",
    # Metrics - Memory management
    "MEMORY_SUMMARIZATION_TOTAL",
    "MEMORY_SUMMARIZATION_LATENCY",
    "MEMORY_TOKENS_BEFORE_TRIM",
    "MEMORY_TOKENS_AFTER_TRIM",
    "MEMORY_MESSAGES_PRUNED",
    "MEMORY_SUMMARY_LENGTH",
    # Metrics - Tools
    "TOOL_CALLS_TOTAL",
    "TOOL_LATENCY",
    "TOOL_ERRORS_TOTAL",
    # Metrics - LLM
    "LLM_CALLS_TOTAL",
    "LLM_LATENCY",
    # Metrics - Weaviate
    "WEAVIATE_QUERIES_TOTAL",
    "WEAVIATE_LATENCY",
    # Metrics - Message handling
    "MESSAGE_CREATED_TOTAL",
    "MESSAGE_REMOVED_TOTAL",
    "MESSAGE_WITH_ID_TOTAL",
    "MESSAGE_WITH_METADATA_TOTAL",
    "LLM_INPUT_TOKENS_TOTAL",
    "LLM_OUTPUT_TOKENS_TOTAL",
    "LLM_TOKENS_PER_CALL",
    "MESSAGE_HISTORY_SIZE",
    "MESSAGE_HISTORY_TOKENS",
    # Metrics - Autonomous routing
    "AUTONOMOUS_ROUTING_TOTAL",
    "SELF_CORRECTION_TOTAL",
    "REFLECTION_LATENCY",
    "INTERRUPT_REQUESTS_TOTAL",
    "ROUTING_DECISION_LATENCY",
    "COMMAND_RETURNS_TOTAL",
    # Metrics - Conversation flow
    "INTENT_CLASSIFICATION_TOTAL",
    "RECOMMENDATION_GIVEN_TOTAL",
    "PURCHASE_LINK_GENERATED_TOTAL",
    "CONVERSATION_TURNS",
    # Metrics - Multi-turn conversation
    "PHASE_TRANSITION_TOTAL",
    "PHASE_DURATION_TURNS",
    "PRONOUN_RESOLUTION_TOTAL",
    "REFERENCE_CONTEXT_UPDATES",
    "INTENT_WITH_SUMMARY_TOTAL",
    "CONVERSATION_PHASE_CURRENT",
]
