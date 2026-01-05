"""
LLM Configuration and Initialization for Agentic Chatbot
========================================================

Production-optimized LLM management with:
- Eager initialization at startup (fail fast)
- Connection pooling via httpx
- Multi-provider support (Azure, OpenRouter)

Supported Providers (set via LLM_PROVIDER env var):
- "azure" (default): Azure OpenAI
- "openrouter": OpenRouter API (supports many models including free ones)

Usage:
    # At application startup (in main.py lifespan):
    initialize_models()  # Raises on failure - fail fast
    
    # Throughout the application:
    llm = get_chat_llm()  # Returns initialized instance
"""

import os
import logging
from typing import Optional, Union

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import httpx

load_dotenv(find_dotenv(), override=True)
logger = logging.getLogger(__name__)

# ============================================
# Configuration (loaded once at module import)
# ============================================

# Provider toggle: "azure" or "openrouter"
LLM_PROVIDER = (os.environ.get("LLM_PROVIDER", "azure") or "azure").strip().lower()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")

# Embeddings Configuration
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT") or AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_EMBEDDING_API_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_API_KEY") or AZURE_OPENAI_API_KEY
AZURE_OPENAI_EMBEDDING_API_VERSION = os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION") or AZURE_OPENAI_API_VERSION
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")

# Response LLM Configuration
AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME", "gpt-4o-mini")

# Temperature settings
AZURE_OPENAI_TEMPERATURE = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0.2"))
AZURE_OPENAI_RESPONSE_TEMPERATURE = float(os.environ.get("AZURE_OPENAI_RESPONSE_TEMPERATURE", "0.3"))

# ============================================
# OpenRouter Configuration
# ============================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")  # Default model
OPENROUTER_RESPONSE_MODEL = os.environ.get("OPENROUTER_RESPONSE_MODEL") or OPENROUTER_MODEL
OPENROUTER_TEMPERATURE = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.2"))
OPENROUTER_RESPONSE_TEMPERATURE = float(os.environ.get("OPENROUTER_RESPONSE_TEMPERATURE", "0.3"))

# Connection pool settings
HTTP_POOL_SIZE = int(os.environ.get("AGENTIC_HTTP_POOL_SIZE", "100"))
HTTP_TIMEOUT = float(os.environ.get("AGENTIC_HTTP_TIMEOUT", "30.0"))



# ============================================
# Module-level State (Eager Initialization)
# ============================================

_chat_llm: Optional[Union[AzureChatOpenAI, ChatOpenAI]] = None
_response_llm: Optional[Union[AzureChatOpenAI, ChatOpenAI]] = None
_embeddings: Optional[AzureOpenAIEmbeddings] = None
_http_client: Optional[httpx.Client] = None
_async_http_client: Optional[httpx.AsyncClient] = None
_current_provider: Optional[str] = None  # Track which provider is active


def _get_http_client() -> httpx.Client:
    """Get shared sync HTTP client with connection pooling."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=HTTP_POOL_SIZE,
                max_keepalive_connections=HTTP_POOL_SIZE // 2,
            ),
            timeout=httpx.Timeout(HTTP_TIMEOUT),
        )
    return _http_client


def _get_async_http_client() -> httpx.AsyncClient:
    """Get shared async HTTP client with connection pooling."""
    global _async_http_client
    if _async_http_client is None:
        _async_http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=HTTP_POOL_SIZE,
                max_keepalive_connections=HTTP_POOL_SIZE // 2,
            ),
            timeout=httpx.Timeout(HTTP_TIMEOUT),
        )
    return _async_http_client


def _validate_config() -> None:
    """Validate required environment variables based on provider."""
    missing = []
    
    if LLM_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        if not OPENROUTER_MODEL:
            missing.append("OPENROUTER_MODEL")
    else:  # azure (default)
        if not AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not AZURE_OPENAI_CHAT_DEPLOYMENT_NAME:
            missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        if not AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME:
            missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    
    if missing:
        raise ValueError(f"Missing required environment variables for {LLM_PROVIDER}: {missing}")


def _initialize_azure() -> None:
    """Initialize Azure OpenAI models."""
    global _chat_llm, _response_llm, _embeddings
    
    # Chat LLM (for routing, intent detection)
    _chat_llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        temperature=AZURE_OPENAI_TEMPERATURE,
        max_retries=2,
        request_timeout=HTTP_TIMEOUT,
    )
    logger.info("Agentic Chat LLM initialized (Azure): %s", AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
    
    # Response LLM (for generating user-facing responses)
    _response_llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME,
        temperature=AZURE_OPENAI_RESPONSE_TEMPERATURE,
        max_retries=2,
        request_timeout=HTTP_TIMEOUT,
    )
    logger.info("Agentic Response LLM initialized (Azure): %s", AZURE_OPENAI_RESPONSE_DEPLOYMENT_NAME)
    
    # Embeddings with chunking for batch efficiency
    _embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
        api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
        api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        chunk_size=16,
    )
    logger.info("Agentic Embeddings initialized (Azure): %s", AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME)


def _initialize_openrouter() -> None:
    """Initialize OpenRouter models."""
    global _chat_llm, _response_llm, _embeddings
    
    # Chat LLM via OpenRouter (OpenAI-compatible API)
    _chat_llm = ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        temperature=OPENROUTER_TEMPERATURE,
        max_retries=2,
        request_timeout=HTTP_TIMEOUT,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://hlas.com"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "HLAS Agentic Chatbot"),
        },
    )
    logger.info("Agentic Chat LLM initialized (OpenRouter): %s", OPENROUTER_MODEL)
    
    # Response LLM via OpenRouter
    _response_llm = ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_RESPONSE_MODEL,
        temperature=OPENROUTER_RESPONSE_TEMPERATURE,
        max_retries=2,
        request_timeout=HTTP_TIMEOUT,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://hlas.com"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "HLAS Agentic Chatbot"),
        },
    )
    logger.info("Agentic Response LLM initialized (OpenRouter): %s", OPENROUTER_RESPONSE_MODEL)
    
    # Embeddings - OpenRouter doesn't support embeddings, fall back to Azure if configured
    if AZURE_OPENAI_EMBEDDING_ENDPOINT and AZURE_OPENAI_EMBEDDING_API_KEY:
        _embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
            api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
            api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            chunk_size=16,
        )
        logger.info("Agentic Embeddings initialized (Azure fallback): %s", AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME)
    else:
        logger.warning("Embeddings not available - OpenRouter doesn't support embeddings and Azure not configured")
        _embeddings = None


def initialize_models() -> None:
    """
    Initialize LLM and embedding models eagerly at startup.
    
    MUST be called once at application startup. Raises on failure (fail fast).
    Idempotent - safe to call multiple times.
    
    Provider is selected via LLM_PROVIDER env var:
    - "azure" (default): Azure OpenAI
    - "openrouter": OpenRouter API
    
    Raises:
        ValueError: If required environment variables are missing
        Exception: If model initialization fails
    """
    global _chat_llm, _response_llm, _embeddings, _current_provider
    
    # Idempotent: skip if already initialized
    if _chat_llm is not None:
        logger.debug("LLM models already initialized (provider: %s)", _current_provider)
        return
    
    _validate_config()
    
    # Initialize based on provider - no try/except, let errors propagate
    if LLM_PROVIDER == "openrouter":
        _initialize_openrouter()
        _current_provider = "openrouter"
    else:
        _initialize_azure()
        _current_provider = "azure"
    
    logger.info("All agentic LLM models initialized successfully (provider: %s)", _current_provider)


def get_chat_llm() -> Union[AzureChatOpenAI, ChatOpenAI]:
    """Get the chat LLM instance.
    
    Raises:
        RuntimeError: If initialize_models() was not called at startup
    """
    if _chat_llm is None:
        raise RuntimeError(
            "LLM not initialized. Call initialize_models() at application startup."
        )
    return _chat_llm


def get_response_llm() -> Union[AzureChatOpenAI, ChatOpenAI]:
    """Get the response LLM instance with auto-reinitialization support.
    
    Raises:
        RuntimeError: If initialize_models() was not called at startup and re-initialization fails
    """
    global _response_llm
    
    # Auto-reinitialize if None (happens in thread executors)
    if _response_llm is None:
        logger.warning("Response LLM was None, attempting re-initialization")
        try:
            initialize_models()
        except Exception as e:
            logger.error(f"Failed to re-initialize LLM models: {e}")
    
    # Check again after re-initialization attempt
    if _response_llm is None:
        raise RuntimeError(
            "LLM not initialized. Call initialize_models() at application startup."
        )
    return _response_llm


def get_embeddings() -> Optional[AzureOpenAIEmbeddings]:
    """Get the embeddings instance.
    
    Note: Returns None if using OpenRouter without Azure embeddings configured.
    
    Raises:
        RuntimeError: If initialize_models() was not called at startup
    """
    global _chat_llm, _embeddings
    if _chat_llm is None:  # Check chat_llm as proxy for initialization
        logger.warning("LLM models were None, attempting re-initialization")
        try:
            initialize_models()
        except Exception as e:
            logger.error(f"Failed to re-initialize LLM models: {e}")
            raise RuntimeError(
                "LLM not initialized. Call initialize_models() at application startup."
            )
    return _embeddings


def get_current_provider() -> Optional[str]:
    """Get the currently active LLM provider name."""
    return _current_provider


# ============================================
# Cleanup
# ============================================

def cleanup() -> None:
    """Cleanup resources on shutdown."""
    global _http_client, _async_http_client
    
    if _http_client:
        _http_client.close()
        _http_client = None
    
    if _async_http_client:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_async_http_client.aclose())
            else:
                loop.run_until_complete(_async_http_client.aclose())
        except Exception:
            pass
        _async_http_client = None
    
    logger.info("LLM resources cleaned up")


__all__ = [
    "initialize_models",
    "get_chat_llm",
    "get_response_llm", 
    "get_embeddings",
    "get_current_provider",
    "cleanup",
    "LLM_PROVIDER",
]
