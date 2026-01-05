"""
Vector Store (Weaviate) Client for Agentic Chatbot
==================================================

Provides Weaviate client for semantic search / RAG operations.
Uses eager initialization at startup for consistent latency.

Usage:
    # At application startup (in main.py lifespan):
    initialize_weaviate()  # Raises on failure - fail fast
    
    # Throughout the application:
    client = get_weaviate_client()  # Returns initialized instance
"""

import os
import logging
from typing import Optional
from urllib.parse import urlparse

try:
    import weaviate
    from weaviate.auth import AuthApiKey
    import weaviate.classes as wvc
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None
    AuthApiKey = None
    wvc = None

logger = logging.getLogger(__name__)

# Global Weaviate client instance
_weaviate_client = None


def initialize_weaviate() -> None:
    """
    Initialize Weaviate client eagerly at startup.
    
    MUST be called once at application startup. Raises on failure (fail fast).
    Idempotent - safe to call multiple times.
    
    Raises:
        RuntimeError: If Weaviate package not installed
        Exception: If connection fails
    """
    global _weaviate_client
    
    if not WEAVIATE_AVAILABLE:
        raise RuntimeError("Weaviate package not installed. Install with: pip install weaviate-client")
    
    # Idempotent: skip if already initialized
    if _weaviate_client is not None:
        logger.debug("Weaviate client already initialized")
        return
    
    # Suppress httpx INFO logs
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)
    
    # Disable version check
    os.environ["WEAVIATE_SKIP_INIT_CHECKS"] = "true"
    
    weaviate_url = os.getenv("WEAVIATE_URL") or os.getenv("WEAVIATE_ENDPOINT") or "http://localhost:8080"
    parsed_url = urlparse(weaviate_url)
    
    auth_credentials = None
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    if weaviate_api_key:
        auth_credentials = AuthApiKey(api_key=weaviate_api_key)
    
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    
    # No try/except - let errors propagate for fail-fast
    _weaviate_client = weaviate.connect_to_custom(
        http_host=parsed_url.hostname,
        http_port=parsed_url.port or 8080,
        http_secure=parsed_url.scheme == "https",
        grpc_host=parsed_url.hostname,
        grpc_port=grpc_port,
        grpc_secure=False,
        auth_credentials=auth_credentials,
        additional_config=wvc.init.AdditionalConfig(
            timeout=wvc.init.Timeout(init=30),
        ),
        skip_init_checks=True,
    )
    logger.info("Weaviate client initialized: %s", weaviate_url)


def get_weaviate_client():
    """
    Get the Weaviate client instance.
    
    Raises:
        RuntimeError: If initialize_weaviate() was not called at startup
    """
    global _weaviate_client
    if _weaviate_client is None:
        # Try to reinitialize if None (shouldn't happen, but safety net)
        logger.warning("Weaviate client was None, attempting re-initialization")
        try:
            initialize_weaviate()
        except Exception as e:
            logger.error(f"Failed to re-initialize Weaviate: {e}")
            raise RuntimeError(
                "Weaviate not initialized. Call initialize_weaviate() at application startup."
            )
    return _weaviate_client


def close_weaviate_client():
    """Close the Weaviate client connection."""
    global _weaviate_client
    if _weaviate_client is not None:
        try:
            _weaviate_client.close()
            logger.info("Weaviate client connection closed.")
        except Exception as e:
            logger.error("Error closing Weaviate client: %s", e)
        finally:
            _weaviate_client = None


__all__ = ["initialize_weaviate", "get_weaviate_client", "close_weaviate_client", "WEAVIATE_AVAILABLE"]
