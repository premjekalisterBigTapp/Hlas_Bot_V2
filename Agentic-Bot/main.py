#!/usr/bin/env python3
"""
HLAS Agentic Chatbot - Standalone FastAPI Application
=====================================================

Production-ready FastAPI server for the agentic chatbot.
Can be deployed independently from the legacy HLAS system.

Usage (from agentic directory):
    cd D:\\agentic
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
    
Or with auto-reload for development:
    python -m uvicorn main:app --reload --port 8000

Or run directly:
    python main.py
"""

import os
import sys
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

# =============================================================================
# PATH AND PACKAGE SETUP
# =============================================================================
# This enables running as standalone script while supporting relative imports
# in all submodules.

_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent

# Add parent directory to path so we can import from this directory
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Set package name for relative imports to work
# Map the actual folder name to 'agentic' module name
import importlib.util
_folder_name = _current_dir.name  # 'Agentic-Bot' or 'agentic'
spec = importlib.util.spec_from_file_location(_folder_name, _current_dir / "__init__.py")
_temp_module = importlib.util.module_from_spec(spec)
sys.modules['agentic'] = _temp_module
sys.modules[_folder_name] = _temp_module
spec.loader.exec_module(_temp_module)

# =============================================================================
# IMPORTS (using package-relative style after path setup)
# =============================================================================

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Now import from the agentic package
# This works because we set up the module alias above
import agentic
from agentic.infrastructure import (
    SessionManager,
    llm_cleanup,
    initialize_models,
    initialize_weaviate,
    BackgroundLogger,
    WEAVIATE_AVAILABLE,
)
from agentic.infrastructure.background_logger import set_background_logger
from agentic.infrastructure.metrics import AGENTIC_MESSAGES_TOTAL, AGENTIC_LATENCY

# Import handlers
from agentic.handlers import (
    handle_agentic_whatsapp_verification,
    handle_agentic_whatsapp_message,
    close_agentic_whatsapp_client,
    agentic_whatsapp_handler,
)
from agentic.infrastructure.idle_monitor import (
    idle_monitor_loop,
    set_whatsapp_handler,
    ENABLE_IDLE_FAREWELL,
)

# Get the main chat function
agentic_chat = agentic.agentic_chat

# Optional: Prometheus metrics - DISABLED due to duplicate registration issues
# TODO: Fix Prometheus metrics duplicate registration
PROMETHEUS_AVAILABLE = False
# try:
#     from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
#     PROMETHEUS_AVAILABLE = True
# except ImportError:
#     PROMETHEUS_AVAILABLE = False


# ============================================
# Lifespan Management
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events - fail fast on initialization errors."""
    # Startup - eager initialization, fail fast on errors
    logger.info("Starting HLAS Agentic Chatbot...")
    
    # Initialize LLM models - MUST succeed or app won't start
    initialize_models()
    logger.info("LLM models initialized")
    
    # Initialize Weaviate client for RAG - MUST succeed or app won't start
    if WEAVIATE_AVAILABLE:
        initialize_weaviate()
        logger.info("Weaviate client initialized")
    else:
        logger.warning("Weaviate not available - RAG features disabled")
    
    # Start background logger for non-blocking MongoDB writes
    bg_logger = BackgroundLogger()
    await bg_logger.start()
    set_background_logger(bg_logger)  # Set module-level reference for enqueue_log
    logger.info("Background logger started")
    
    # Register WhatsApp handler for idle monitor
    set_whatsapp_handler(agentic_whatsapp_handler)
    
    # Start idle monitor background task
    idle_monitor_task = None
    if ENABLE_IDLE_FAREWELL:
        idle_monitor_task = asyncio.create_task(idle_monitor_loop())
        logger.info("Idle monitor started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HLAS Agentic Chatbot...")
    
    # Cancel idle monitor
    if idle_monitor_task:
        idle_monitor_task.cancel()
        try:
            await idle_monitor_task
        except asyncio.CancelledError:
            pass
    
    # Stop background logger (drains pending logs)
    await bg_logger.stop()
    logger.info("Background logger stopped")
    
    await close_agentic_whatsapp_client()
    llm_cleanup()
    logger.info("Shutdown complete")


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="HLAS Agentic Chatbot",
    description="Production-ready LangGraph-based insurance chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Request/Response Models
# ============================================

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: Optional[str] = ""
    debug_state: Optional[dict] = None


# ============================================
# Health & Metrics Endpoints
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "hlas-agentic",
        "version": "1.0.0",
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check - verifies dependencies."""
    checks = {}
    
    # Check Redis
    try:
        from agentic.infrastructure import get_redis
        redis = get_redis()
        redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
    
    # Check if models can be initialized
    try:
        from agentic.infrastructure import get_chat_llm
        llm = get_chat_llm()
        checks["llm"] = "ok" if llm else "not initialized"
    except Exception as e:
        checks["llm"] = f"error: {e}"
    
    # Check Weaviate
    try:
        from agentic.infrastructure import get_weaviate_client, WEAVIATE_AVAILABLE
        if WEAVIATE_AVAILABLE:
            client = get_weaviate_client()
            checks["weaviate"] = "ok" if client else "not initialized"
        else:
            checks["weaviate"] = "not available"
    except Exception as e:
        checks["weaviate"] = f"error: {e}"
    
    all_ok = all(v == "ok" or v == "not available" for v in checks.values())
    
    return {
        "ready": all_ok,
        "checks": checks,
    }


if PROMETHEUS_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )


# ============================================
# Chat Endpoints
# ============================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for the agentic chatbot.
    
    Args:
        request: ChatRequest with session_id and message
        
    Returns:
        ChatResponse with bot response and debug info
    """
    import time
    start_time = time.time()
    
    try:
        result = await agentic_chat(request.session_id, request.message)
        
        # Record metrics
        latency = time.time() - start_time
        AGENTIC_LATENCY.labels(endpoint="chat").observe(latency)
        AGENTIC_MESSAGES_TOTAL.labels(
            result="ok",
            product=result.get("debug_state", {}).get("product") or "unknown"
        ).inc()
        
        return ChatResponse(
            response=result.get("response", ""),
            sources=result.get("sources", ""),
            debug_state=result.get("debug_state"),
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        AGENTIC_MESSAGES_TOTAL.labels(result="error", product="unknown").inc()
        return ChatResponse(
            response="I'm sorry, something went wrong. Please try again.",
            debug_state={"error": str(e)},
        )


@app.post("/agent-chat", response_model=ChatResponse)
async def agent_chat_endpoint(request: ChatRequest):
    """Alias for /chat endpoint (backwards compatibility)."""
    return await chat_endpoint(request)


# ============================================
# WhatsApp Webhook Endpoints
# ============================================

@app.get("/webhook/whatsapp")
async def whatsapp_verification(request: Request):
    """WhatsApp webhook verification (GET)."""
    return handle_agentic_whatsapp_verification(request)


@app.post("/webhook/whatsapp")
async def whatsapp_message(request: Request):
    """WhatsApp message handler (POST)."""
    return await handle_agentic_whatsapp_message(request)


# Alternative webhook paths (for flexibility)
@app.get("/agentic-webhook")
async def agentic_webhook_verification(request: Request):
    """Alternative WhatsApp webhook verification."""
    return handle_agentic_whatsapp_verification(request)


@app.post("/agentic-webhook")
async def agentic_webhook_message(request: Request):
    """Alternative WhatsApp message handler."""
    return await handle_agentic_whatsapp_message(request)


# Legacy path support (requested by logs)
@app.get("/agent-whatsapp")
async def agent_whatsapp_verification(request: Request):
    """WhatsApp webhook verification (GET) - Legacy path."""
    return handle_agentic_whatsapp_verification(request)


@app.post("/agent-whatsapp")
async def agent_whatsapp_message(request: Request):
    """WhatsApp message handler (POST) - Legacy path."""
    return await handle_agentic_whatsapp_message(request)



# ============================================
# Session Management Endpoints
# ============================================

@app.post("/session/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset a session to initial state."""
    try:
        session_manager = SessionManager()
        session_manager.reset_session(session_id)
        return {"status": "ok", "message": f"Session {session_id} reset"}
    except Exception as e:
        logger.error(f"Session reset error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session state (for debugging)."""
    try:
        session_manager = SessionManager()
        session = session_manager.get_session(session_id)
        return {"status": "ok", "session": session}
    except Exception as e:
        logger.error(f"Session get error: {e}")
        return {"status": "error", "message": str(e)}


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", "1")),
    )


### run error fixing : 1. Port 8000 Already in Use (Error 10048)
#ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000): only one usage of each socket address (protocol/network address/port) is normally permitted
#What this means: Another instance of the application was already running and listening on port 8000, preventing the new instance from starting.
#How I fixed it:
#Ran netstat -ano | findstr :8000 to identify which process was using port 8000
#Found that process ID (PID) 32512 was occupying the port
#Killed that process using: taskkill /PID 32512 /F
###

