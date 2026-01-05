"""
Plan comparison tool for comparing different tiers/plans.

This tool:
- Compares plans using benefits data and templates
- Supports product-specific comparison logic
- Includes comprehensive logging and metrics
"""
from __future__ import annotations

import logging
import time
import traceback
from typing import List, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from ..infrastructure import get_response_llm
from ..infrastructure.metrics import LLM_CALLS_TOTAL, LLM_LATENCY
from .benefits import get_product_benefits
from ..config import _load_cmp_templates
from ..utils.slots import _normalize_product_key, _detect_product_llm

logger = logging.getLogger(__name__)


def _compare_tool(
    product: Optional[str], tiers: List[str], question: str
) -> Tuple[str, List[str]]:
    """
    Comparison tool: plan-only comparisons using benefits and templates.
    
    Args:
        product: Product name
        tiers: List of tier names to compare (optional)
        question: User's comparison question
        
    Returns:
        Tuple of (comparison_text, empty_sources_list)
        
    Raises:
        Exception: Re-raises exceptions for the caller to handle
    """
    start_time = time.time()
    
    raw_product = product
    prod = _normalize_product_key(product)
    
    # Attempt product detection from question if not provided
    if not prod:
        logger.debug(
            "Tool.compare.detecting_product: question='%s'",
            (question or "")[:100]
        )
        detected = _detect_product_llm(question)
        prod = _normalize_product_key(detected)
    
    logger.info(
        "Tool.compare.start: raw_product=%s resolved_product=%s tiers=%s question_len=%d",
        raw_product, prod, tiers, len(question or "")
    )
    
    if not prod:
        logger.warning("Tool.compare.no_product: could not determine product")
        return (
            "Which product would you like to compare plans for: Travel, Maid, Car, Personal Accident, "
            "Home, Early, Fraud or Hospital?",
            [],
        )

    # Get benefits text for the product
    benefits_text = ""
    benefits_start = time.time()
    try:
        benefits_text = get_product_benefits(prod)
        logger.debug(
            "Tool.compare.benefits_loaded: product=%s len=%d duration=%.3fs",
            prod, len(benefits_text), time.time() - benefits_start
        )
    except Exception as e:
        logger.warning(
            "Tool.compare.benefits_failed: product=%s error=%s",
            prod, str(e)
        )
        benefits_text = ""

    # Load comparison templates
    cmp_templates = _load_cmp_templates()
    tpl = cmp_templates.get(prod, {}) if cmp_templates else {}
    
    # Default system prompt with styling instructions (skips styler node for faster response)
    default_sys = (
        "You are HLAS Smart Bot comparing insurance plans.\n\n"
        "RESPONSE STYLE (WhatsApp-friendly):\n"
        "• Use • for bullet points, *asterisks* for plan names\n"
        "• Numbers as digits ($500,000) - NEVER use abbreviations like $500k or $1M\n"
        "• Clean line breaks between sections\n"
        "• NO headers (###), NO tables\n"
        "• Be warm and conversational, not robotic\n"
        "• Keep response concise but informative\n\n"
        "STRUCTURE:\n"
        "1. Brief intro acknowledging the comparison request\n"
        "2. Key differences between plans with specific amounts\n"
        "3. Simple recommendation based on coverage needs\n"
        "4. Optional: brief closing question about preference\n\n"
        "Compare the plans using only the provided context."
    )
    sys_t = tpl.get("system") or default_sys
    tiers_txt = ", ".join(tiers) if tiers else ""
    usr_t = (tpl.get("user") or "Product: {product}\nTiers: {tiers}\nQuestion: {question}\n\n[Context]\n{context}").format(
        product=prod,
        tiers=tiers_txt,
        question=question,
        context=benefits_text or "",
    )

    # Generate LLM response
    answer = ""
    llm_start = time.time()
    try:
        llm = get_response_llm()
        if llm:
            messages = [SystemMessage(content=sys_t), HumanMessage(content=usr_t)]
            response = llm.invoke(messages)
            answer = str(response.content).strip()
            
            llm_duration = time.time() - llm_start
            logger.info(
                "Tool.compare.llm_response: product=%s answer_len=%d duration=%.3fs",
                prod, len(answer), llm_duration
            )
            
            # Record metrics
            try:
                LLM_CALLS_TOTAL.labels(model="response_llm", status="success").inc()
                LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
            except Exception:
                pass
        else:
            logger.error("Tool.compare.llm_not_initialized")
    except Exception as e:
        llm_duration = time.time() - llm_start
        logger.error(
            "Tool.compare.llm_failed: product=%s duration=%.3fs error=%s\n%s",
            prod, llm_duration, str(e), traceback.format_exc()
        )
        try:
            LLM_CALLS_TOTAL.labels(model="response_llm", status="error").inc()
            LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
        except Exception:
            pass
        answer = ""

    # Fallback answer
    if not answer:
        answer = (
            f"Here is a high-level comparison of the available {prod.title()} plans. "
            "You can ask about a specific benefit if you need more detail."
        )

    total_duration = time.time() - start_time
    logger.info(
        "Tool.compare.completed: product=%s answer_len=%d total_duration=%.3fs",
        prod, len(answer), total_duration
    )
    
    return answer, []
