"""
Information retrieval tool using RAG over Weaviate.

This tool:
- Searches the HLAS knowledge base using hybrid search (vector + keyword)
- Uses product-specific templates for response generation
- Returns answers with source citations
- Includes comprehensive logging and metrics
"""
from __future__ import annotations

import logging
import time
import traceback
from typing import List, Optional, Tuple

from weaviate.classes.query import TargetVectors, Filter
from langchain_core.messages import SystemMessage, HumanMessage

from ..infrastructure import get_weaviate_client, get_embeddings, get_response_llm
from ..infrastructure.metrics import WEAVIATE_QUERIES_TOTAL, WEAVIATE_LATENCY, LLM_CALLS_TOTAL, LLM_LATENCY
from ..config import _load_ir_templates
from ..utils.slots import _normalize_product_key, _detect_product_llm

logger = logging.getLogger(__name__)


def _get_models():
    """Get LLM models from local infrastructure (thread-safe singletons)."""
    return get_embeddings(), get_response_llm()


def _info_tool(product: Optional[str], question: str, conversation_context: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Information tool: RAG over Weaviate using ir_response.yaml templates.
    
    Args:
        product: Product name to filter by (optional)
        question: User's question
        conversation_context: Last bot message for context (helps reformulate vague queries)
        
    Returns:
        Tuple of (answer_text, source_files)
        
    Raises:
        Exception: Re-raises exceptions for the caller to handle
    """
    start_time = time.time()
    
    raw_product = product
    prod = _normalize_product_key(product)
    
    # Attempt product detection from question if not provided
    if not prod:
        logger.debug(
            "Tool.info.detecting_product: question='%s'",
            (question or "")[:100]
        )
        detected = _detect_product_llm(question)
        prod = _normalize_product_key(detected)
    
    # Query reformulation: Let LLM decide if the query needs context-based reformulation
    # Note: Clarifying questions during slot collection are handled by rec_subgraph's side_info.
    # This is for general info queries that may reference previous conversation.
    original_question = question
    
    if conversation_context and len((question or "").split()) <= 6:
        # Short queries may need context - let LLM decide
        from ..config import _router_model
        try:
            reformulate_prompt = (
                f"CONTEXT: The user is asking for information about insurance.\n"
                f"Bot's last message: \"{conversation_context[:600]}\"\n"
                f"User's message: \"{question}\"\n\n"
                f"TASK: Determine if the user's message is a standalone question or refers to something in the bot's message.\n\n"
                f"If the user's message is vague, a pronoun reference (like 'that', 'it', 'those'), or an affirmation "
                f"(like 'yes', 'tell me more'), create a specific search query based on what the bot mentioned.\n\n"
                f"If the user's message is already a clear, specific question, return it unchanged.\n\n"
                f"Product context: {prod or 'insurance'}\n\n"
                f"Respond with ONLY the final search query, nothing else."
            )
            result = _router_model.invoke([HumanMessage(content=reformulate_prompt)])
            reformulated = str(getattr(result, "content", "") or "").strip()
            if reformulated and len(reformulated) > 5:
                question = reformulated
                if question != original_question:
                    logger.info(
                        "Tool.info.query_reformulated: original='%s' -> reformulated='%s'",
                        original_question, question[:100]
                    )
        except Exception as e:
            logger.warning("Tool.info.reformulation_failed: %s", str(e))
    
    logger.info(
        "Tool.info.start: raw_product=%s resolved_product=%s question_len=%d",
        raw_product, prod, len(question or "")
    )
    
    if not prod:
        logger.warning("Tool.info.no_product: could not determine product")
        return (
            "Which product would you like to ask about: Travel Protect360, Maid Protect360, Car Protect360, Personal Accident Protect360, "
            "Home Protect360, Early Critical Illness Protect360, Fraud Protect360 or Hospital Cash Protect360?",
            [],
        )

    # Initialize Weaviate client
    try:
        client = get_weaviate_client()
        collection = client.collections.get("Insurance_Knowledge_Base")
        logger.debug("Tool.info.weaviate_connected: collection=Insurance_Knowledge_Base")
    except Exception as e:
        logger.error(
            "Tool.info.weaviate_init_failed: error=%s\n%s",
            str(e), traceback.format_exc()
        )
        try:
            WEAVIATE_QUERIES_TOTAL.labels(status="error").inc()
        except Exception:
            pass
        raise  # Re-raise for caller to handle

    embeddings, llm = _get_models()

    # Generate embeddings for the question
    emb = None
    emb_start = time.time()
    try:
        if embeddings:
            emb = embeddings.embed_query(question)
            logger.debug(
                "Tool.info.embedding_generated: duration=%.3fs",
                time.time() - emb_start
            )
        else:
            logger.error("Tool.info.embeddings_not_initialized")
    except Exception as e:
        logger.error(
            "Tool.info.embedding_failed: error=%s\n%s",
            str(e), traceback.format_exc()
        )
        emb = None

    # Execute Weaviate hybrid search
    objects = []
    weaviate_start = time.time()
    if emb is not None:
        try:
            result = collection.query.hybrid(
                query=question,
                vector={
                    "content_vector": emb,
                    "questions_vector": emb,
                },
                target_vector=TargetVectors.average(["content_vector", "questions_vector"]),
                filters=Filter.by_property("product_name").equal(prod),
                limit=5,  # Reduced from 10 for faster queries - top 5 hits are usually sufficient
                alpha=0.7,
                return_properties=["content", "product_name", "doc_type", "source_file"],
            )
            objects = getattr(result, "objects", []) or []
            
            weaviate_duration = time.time() - weaviate_start
            logger.info(
                "Tool.info.weaviate_query: product=%s hits=%d duration=%.3fs",
                prod, len(objects), weaviate_duration
            )
            
            # Record metrics
            try:
                WEAVIATE_QUERIES_TOTAL.labels(status="success").inc()
                WEAVIATE_LATENCY.observe(weaviate_duration)
            except Exception:
                pass
                
        except Exception as e:
            weaviate_duration = time.time() - weaviate_start
            logger.error(
                "Tool.info.weaviate_query_failed: product=%s duration=%.3fs error=%s\n%s",
                prod, weaviate_duration, str(e), traceback.format_exc()
            )
            try:
                WEAVIATE_QUERIES_TOTAL.labels(status="error").inc()
                WEAVIATE_LATENCY.observe(weaviate_duration)
            except Exception:
                pass
            objects = []

    # Handle no results
    if not objects:
        question_preview = (question or "").replace("\n", " ")[:160]
        logger.warning(
            "Tool.info.no_results: product=%s question='%s'",
            prod, question_preview
        )
        return (
            f"I couldn't find detailed information for that in our {prod.title()} plans. "
            "Could you share a bit more so I can look up the exact coverage?",
            [],
        )

    # Build context from results
    context_str = "\n---\n".join(
        [str(obj.properties.get("content", "") or "") for obj in objects]
    )
    sources = sorted(
        {
            str(obj.properties.get("source_file", "") or "")
            for obj in objects
            if obj.properties.get("source_file")
        }
    )
    
    logger.debug(
        "Tool.info.context_built: product=%s context_len=%d sources=%d",
        prod, len(context_str), len(sources)
    )

    # Load templates and generate response
    ir_templates = _load_ir_templates()
    tpl = ir_templates.get(prod, {}) if ir_templates else {}

    # Base system prompt from templates (if any) or default.
    base_sys = tpl.get("system") or (
        "You are HLAS's digital insurance assistant answering information questions. "
        "Answer using only the provided context from our official knowledge base."
    )

    # Styling and flow rules inspired by the global styler, so we can safely
    # skip the styler node for pure information flows.
    style_suffix = (
        "\n\nRESPONSE STYLE (WhatsApp-friendly):\n"
        "• Use • for bullet points and *asterisks* for product or plan names where helpful\n"
        "• Keep answers clear, concise and friendly – avoid long paragraphs\n"
        "• Use digits for numbers and sums (e.g. $500,000)\n"
        "• No markdown headers (###) or tables\n"
        "• Be honest about any limits or exclusions in the context – do not invent details beyond it\n\n"
        "FLOW & NAMING RULES:\n"
        "1. Focus on directly answering the user's question first, then optionally add one short follow-up tip or clarification.\n"
        "2. Do not push recommendations or purchase links in pure information responses unless the user explicitly asks.\n"
        "3. Use the full official product names at least once when relevant: Travel Protect360, Maid Protect360, Car Protect360, "
        "Home Protect360, Personal Accident Protect360, Early Critical Illness Protect360, Fraud Protect360, Hospital Cash Protect360.\n"
        "4. You may shorten names (e.g. 'Travel plan') only after using the full name once in the answer.\n"
        "5. Do not mention phone numbers, emails or hotlines unless the user explicitly asks for contact details.\n"
        "6. Avoid starting every reply with the same phrase like 'Thanks for your question'; vary intros and keep them brief.\n"
        "7. Stay strictly within the provided context – if something is not covered, say so clearly.\n"
    )

    sys_t = base_sys + style_suffix
    usr_t = (tpl.get("user") or "Question: {question}\n\n[Context]\n{context}").format(
        question=question,
        context=context_str,
    )

    # Generate LLM response
    answer = ""
    llm_start = time.time()
    try:
        if llm:
            result = llm.invoke([
                SystemMessage(content=sys_t),
                HumanMessage(content=usr_t),
            ])
            answer = str(result.content).strip()
            
            llm_duration = time.time() - llm_start
            logger.info(
                "Tool.info.llm_response: product=%s answer_len=%d duration=%.3fs",
                prod, len(answer), llm_duration
            )
            
            # Record metrics
            try:
                LLM_CALLS_TOTAL.labels(model="response_llm", status="success").inc()
                LLM_LATENCY.labels(model="response_llm").observe(llm_duration)
            except Exception:
                pass
        else:
            logger.error("Tool.info.llm_not_initialized")
    except Exception as e:
        llm_duration = time.time() - llm_start
        logger.error(
            "Tool.info.llm_failed: product=%s duration=%.3fs error=%s\n%s",
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
        answer = "I couldn't find precise details. Could you clarify your question?"

    total_duration = time.time() - start_time
    logger.info(
        "Tool.info.completed: product=%s answer_len=%d sources=%d total_duration=%.3fs",
        prod, len(answer), len(sources), total_duration
    )
    
    return answer, sources
