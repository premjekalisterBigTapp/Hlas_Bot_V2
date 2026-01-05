from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# Ensure .env is loaded before reading any config
load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

# Provider toggle: "azure" or "openrouter"
LLM_PROVIDER = (os.environ.get("LLM_PROVIDER", "azure") or "azure").strip().lower()

# --- Paths ---
# Agentic layer uses its own configs folder to avoid modifying legacy configs
# config.py is in hlas/src/hlas/agentic/
# Path(__file__).parent is hlas/src/hlas/agentic/
AGENTIC_DIR = Path(__file__).resolve().parent
CONFIG_DIR = AGENTIC_DIR / "configs"

IR_RESPONSE_PATH = CONFIG_DIR / "ir_response.yaml"
SUMMARY_RESPONSE_PATH = CONFIG_DIR / "summary_response.yaml"
CMP_RESPONSE_PATH = CONFIG_DIR / "cmp_response.yaml"
RECOMMENDATION_RESPONSE_PATH = CONFIG_DIR / "recommendation_response.yaml"
SLOT_RULES_PATH = CONFIG_DIR / "slot_validation_rules.yaml"
KNOWLEDGE_BASE_PATH = CONFIG_DIR / "knowledge_base.txt"
LINKS_PATH = CONFIG_DIR / "links.yaml"

# --- Caches ---
_ir_templates_cache: Dict[str, Any] = {}
_summary_templates_cache: Dict[str, Any] = {}
_cmp_templates_cache: Dict[str, Any] = {}
_rec_templates_cache: Dict[str, Any] = {}
_slot_rules_cache: Dict[str, Any] = {}
_links_cache: Dict[str, str] = {}
_kb_text_cache: Optional[str] = None


def _build_router_model() -> Union[AzureChatOpenAI, ChatOpenAI]:
    """Create a chat model based on LLM_PROVIDER env var.
    
    Supports:
    - "azure" (default): Azure OpenAI
    - "openrouter": OpenRouter API
    """
    try:
        temperature = float(os.getenv("AGENTIC_ROUTER_TEMPERATURE", "0.1"))
    except ValueError:
        temperature = 0.1
    
    max_tokens = int(os.getenv("AGENTIC_ROUTER_MAX_TOKENS", "1024"))
    
    if LLM_PROVIDER == "openrouter":
        # OpenRouter configuration
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        
        if not api_key:
            raise RuntimeError("Agentic runtime: missing OPENROUTER_API_KEY")
        
        logger.info(
            "Agentic runtime: initializing OpenRouter ChatOpenAI (model=%s)",
            model,
        )
        
        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            default_headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://hlas.com"),
                "X-Title": os.environ.get("OPENROUTER_TITLE", "HLAS Agentic Chatbot"),
            },
        )
    else:
        # Azure OpenAI (default)
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        deployment = (
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        )

        missing = []
        if not endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not api_version:
            missing.append("AZURE_OPENAI_API_VERSION")
        if not deployment:
            missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

        if missing:
            raise RuntimeError(
                f"Agentic runtime: missing required Azure OpenAI env vars: {missing}"
            )

        logger.info(
            "Agentic runtime: initializing AzureChatOpenAI (deployment=%s, endpoint=%s)",
            deployment,
            endpoint,
        )

        return AzureChatOpenAI(
            azure_endpoint=endpoint.rstrip("/") if endpoint else None,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# Shared router model instance (respects LLM_PROVIDER)
_router_model: Union[AzureChatOpenAI, ChatOpenAI] = _build_router_model()


def _load_yaml_cached(path: Path, cache: Dict[str, Any]) -> Dict[str, Any]:
    if cache:
        return cache
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            cache.update({str(k).lower(): v for k, v in data.items()})
        else:
            cache.clear()
    except Exception as e:
        logger.warning("Agentic: failed to load YAML from %s - %s", path, e)
        cache.clear()
    return cache


def _load_ir_templates() -> Dict[str, Any]:
    return _load_yaml_cached(IR_RESPONSE_PATH, _ir_templates_cache)


def _load_summary_templates() -> Dict[str, Any]:
    return _load_yaml_cached(SUMMARY_RESPONSE_PATH, _summary_templates_cache)


def _load_cmp_templates() -> Dict[str, Any]:
    return _load_yaml_cached(CMP_RESPONSE_PATH, _cmp_templates_cache)


def _load_rec_templates() -> Dict[str, Any]:
    return _load_yaml_cached(RECOMMENDATION_RESPONSE_PATH, _rec_templates_cache)


def _load_slot_rules() -> Dict[str, Any]:
    if _slot_rules_cache:
        return _slot_rules_cache
    try:
        text = SLOT_RULES_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            _slot_rules_cache.update(data)
    except Exception as e:
        logger.warning("Agentic: failed to load slot_validation_rules.yaml - %s", e)
    return _slot_rules_cache


def _load_knowledge_base() -> str:
    global _kb_text_cache
    if isinstance(_kb_text_cache, str):
        return _kb_text_cache
    try:
        _kb_text_cache = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
    except Exception:
        _kb_text_cache = ""
    return _kb_text_cache


def _load_purchase_links() -> Dict[str, str]:
    if _links_cache:
        return _links_cache
    try:
        text = LINKS_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            _links_cache.update({str(k).lower(): str(v) for k, v in data.items()})
    except Exception as e:
        logger.warning("Agentic: failed to load links.yaml - %s", e)
    return _links_cache
