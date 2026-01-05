from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from ..state import AgentState, IntentPrediction, ConversationPhase, ReferenceContext
from ..config import _router_model
from ..utils.slots import _detect_product_llm
from ..utils.products import get_product_names_str, get_product_aliases_prompt
from ..utils.memory import _build_history_context_from_messages, _get_last_user_message

logger = logging.getLogger(__name__)


# =============================================================================
# CACHED STRUCTURED OUTPUT WRAPPERS
# Cache these to avoid rebuilding on each call (~50-100ms saved per call)
# =============================================================================
_intent_classifier_cached = None

def _get_intent_classifier():
    """Get cached intent classifier with structured output."""
    global _intent_classifier_cached
    if _intent_classifier_cached is None:
        _intent_classifier_cached = _router_model.with_structured_output(IntentPrediction)
    return _intent_classifier_cached


# =============================================================================
# PRONOUN RESOLUTION
# =============================================================================

def _extract_reference_context(
    messages: List[BaseMessage],
    current_product: Optional[str] = None,
    current_slots: Optional[Dict[str, Any]] = None,
) -> ReferenceContext:
    """
    Extract reference context from recent messages for pronoun resolution.
    
    This addresses the "No Pronoun Resolution" issue by tracking:
    - Last mentioned product, tier, destination
    - Recently compared items
    - Last question asked by bot
    
    Args:
        messages: Recent conversation messages
        current_product: Current product from state
        current_slots: Current slots from state
        
    Returns:
        ReferenceContext with extracted references
    """
    context = ReferenceContext()
    
    if not messages:
        return context
    
    # Track last mentioned product
    context.last_mentioned_product = current_product
    
    # Extract destination from slots if available
    if current_slots:
        dest = current_slots.get("destination") or current_slots.get("travel_destination")
        if dest:
            context.last_mentioned_destination = str(dest)
    
    # Scan recent messages for references
    recent_messages = messages[-10:]  # Look at last 10 messages
    
    for msg in reversed(recent_messages):
        content = str(getattr(msg, "content", "") or "").lower()
        
        # Extract last bot question
        if isinstance(msg, AIMessage) and context.last_bot_question is None:
            if "?" in content:
                # Extract the question part
                question_start = content.rfind("?")
                # Find sentence start (look for period, newline, or start)
                sentence_start = max(
                    content.rfind(".", 0, question_start),
                    content.rfind("\n", 0, question_start),
                    0
                )
                context.last_bot_question = content[sentence_start:question_start + 1].strip()
        
        # Extract tier mentions from AI messages (for "it", "that plan")
        if isinstance(msg, AIMessage) and context.last_mentioned_tier is None:
            tier_keywords = ["gold", "silver", "platinum", "basic", "bronze", "essential", "premium"]
            for tier in tier_keywords:
                if tier in content:
                    context.last_mentioned_tier = tier.capitalize()
                    break
        
        # Extract compared items from comparison responses
        if isinstance(msg, AIMessage) and not context.compared_items:
            if "compare" in content or "vs" in content or "difference" in content:
                # Look for tier names in the comparison
                tier_keywords = ["gold", "silver", "platinum", "basic", "bronze", "essential", "premium"]
                found_tiers = [t.capitalize() for t in tier_keywords if t in content]
                if len(found_tiers) >= 2:
                    context.compared_items = found_tiers[:3]  # Max 3
    
    logger.debug(
        "Intent.reference_context: product=%s tier=%s dest=%s compared=%s question=%s",
        context.last_mentioned_product,
        context.last_mentioned_tier,
        context.last_mentioned_destination,
        context.compared_items,
        context.last_bot_question[:50] if context.last_bot_question else None,
    )
    
    return context


def _build_pronoun_resolution_prompt(
    reference_context: ReferenceContext,
    last_user_message: str,
) -> str:
    """
    Build pronoun resolution guidance for the intent classifier.
    
    This helps the classifier understand what pronouns refer to.
    """
    # Check if user message contains pronouns that need resolution
    pronoun_indicators = ["it", "that", "this", "them", "those", "there", "the one", "the first", "the second"]
    has_pronouns = any(p in last_user_message.lower().split() for p in pronoun_indicators)
    
    if not has_pronouns:
        return ""
    
    parts = ["PRONOUN RESOLUTION (user message contains references):"]
    
    ref_context_str = reference_context.to_prompt_context()
    if ref_context_str:
        parts.append(ref_context_str)
    
    parts.append(
        "RESOLUTION RULES:\n"
        "  - 'it', 'that', 'this' → likely refers to last_mentioned_tier or last_mentioned_product\n"
        "  - 'them', 'those' → likely refers to compared_items\n"
        "  - 'there' → likely refers to last_mentioned_destination\n"
        "  - 'the first one', 'the second' → refers to compared_items in order\n"
        "  - Short answers (yes, no, numbers, locations) → likely answers to last_bot_question"
    )
    
    return "\n".join(parts)


# =============================================================================
# ENHANCED INTENT CLASSIFICATION
# =============================================================================

def _classify_intent_from_messages(
    messages: List[BaseMessage], 
    known_product: Optional[str] = None,
    active_slot: Optional[str] = None,
    summary: Optional[str] = None,
    current_phase: Optional[str] = None,
    current_slots: Optional[Dict[str, Any]] = None,
    reference_context: Optional[ReferenceContext] = None,
    rec_given: bool = False,
) -> IntentPrediction:
    """
    Classify high-level intent + product from conversation with full context.

    Enhanced to address Multi-Turn Conversation Failures:
    1. Uses summary for long-term context (addresses "Intent Classification Doesn't Consider Full History")
    2. Uses reference_context for pronoun resolution (addresses "No Pronoun Resolution")
    3. Uses current_phase for phase-aware classification
    4. Uses rec_given to handle post-recommendation responses properly
    
    Args:
        messages: Recent conversation messages
        known_product: Current product from state
        active_slot: Slot currently being asked for
        summary: Conversation summary for long-term context
        current_phase: Current conversation phase
        current_slots: Current collected slots
        reference_context: Context for pronoun resolution
        rec_given: Whether a recommendation has already been given
    """
    start_time = time.perf_counter()

    if not messages:
        logger.debug("Intent.classify: no messages, returning info intent")
        return IntentPrediction(intent="info", product=known_product, reason="no_messages")

    # Build history context - use fewer messages since we have summary for long-term context
    history_window = 3 if summary else 5
    history_ctx = _build_history_context_from_messages(messages[-history_window:])
    last_user = _get_last_user_message(messages) or ""
    
    product_list = get_product_names_str()
    product_aliases = get_product_aliases_prompt()
    
    # Build reference context if not provided
    if reference_context is None:
        reference_context = _extract_reference_context(messages, known_product, current_slots)
    
    # Build context sections
    context_parts = []
    
    # 1. Summary context (addresses "Intent Classification Doesn't Consider Full History")
    if summary:
        context_parts.append(
            f"CONVERSATION SUMMARY (long-term context):\n{summary}\n"
            "Use this summary to understand the full conversation context, "
            "but prioritize recent messages for current intent."
        )
    
    # 2. Active slot context
    if active_slot:
        context_parts.append(
            f"[IMPORTANT CONTEXT]: The bot explicitly asked the user for the '{active_slot}' slot. "
            f"If the user's message '{last_user}' looks like an answer to this (e.g. a location, a number, a yes/no), "
            "you MUST classify this as 'recommend' to continue the form-filling flow. "
            "Only choose a different intent if they explicitly change the topic (e.g. 'stop', 'switch to car')."
        )
    
    # 3. Current phase context
    if current_phase:
        phase_guidance = {
            ConversationPhase.GREETING.value: "User is in greeting phase. Look for product interest or general questions.",
            ConversationPhase.PRODUCT_SELECTION.value: "User is exploring products. Look for product mentions or comparison requests.",
            ConversationPhase.SLOT_FILLING.value: "User is providing information for a recommendation. Short answers likely relate to pending questions.",
            ConversationPhase.RECOMMENDATION.value: "User received a recommendation. Look for purchase intent, comparison, or new questions.",
            ConversationPhase.COMPARISON.value: "User is comparing plans. Look for selection, more comparisons, or purchase intent.",
            ConversationPhase.PURCHASE.value: "User is in purchase flow. Look for confirmation or additional questions.",
            ConversationPhase.INFO_QUERY.value: "User is asking questions. Look for specific product/coverage questions.",
        }
        guidance = phase_guidance.get(current_phase, "")
        if guidance:
            context_parts.append(f"CURRENT PHASE: {current_phase}\n{guidance}")
    
    # 3.5. Incomplete recommendation flow context (when user might be resuming)
    # This helps classify short answers correctly even when pending_slot is None
    if known_product and not rec_given and not active_slot:
        context_parts.append(
            f"INCOMPLETE RECOMMENDATION FLOW:\n"
            f"The user has been working on a {known_product} recommendation but hasn't received one yet.\n"
            "If the user provides a short answer that looks like slot data (e.g., a number like '14' or '26', "
            "a duration like '14 months', a location, or 'yes'/'no'), classify as 'recommend' to continue the flow.\n"
            "This takes priority over 'chat' or 'other' for ambiguous short answers."
        )
    
    # 4. Post-recommendation context (prevents re-recommending after upsell offer)
    if rec_given:
        context_parts.append(
            "CRITICAL - POST-RECOMMENDATION STATE:\n"
            "A recommendation has ALREADY been given to this user. Do NOT classify as 'recommend' "
            "unless they explicitly ask for a NEW or DIFFERENT recommendation.\n"
            "- If user says 'yes', 'sure', 'tell me more', 'details' → classify as 'info' (they want more details about the mentioned tier/plan)\n"
            "- If user wants to buy, get quote, or proceed → classify as 'purchase'\n"
            "- If user asks to compare plans → classify as 'compare'\n"
            "- Only use 'recommend' if they say 'different plan', 'new recommendation', or switch products"
        )
    
    # 5. Pronoun resolution context (addresses "No Pronoun Resolution")
    pronoun_prompt = _build_pronoun_resolution_prompt(reference_context, last_user)
    if pronoun_prompt:
        context_parts.append(pronoun_prompt)
    
    # Build system message
    sys_msg = (
        "You are an intent classifier for the HLAS Smart Bot. "
        "Your job is to decide what the user is trying to do and which "
        "insurance product (if any) they are talking about.\n\n"
        "You MUST choose one of these intents exactly: "
        "'info', 'summary', 'compare', 'recommend', 'purchase', "
        "'capabilities', 'greet', 'chat', 'policy_service', 'other'.\n\n"
        "Guidelines:\n"
        "- info: asking about coverage, benefits, exclusions, scenarios, or 'tell me about X'.\n"
        "- summary: asking for a high-level overview of a product or tiers.\n"
        "- compare: asking for differences between plans/tiers.\n"
        "- recommend: wants a personalised plan suggestion, 'best plan', OR says 'I want X insurance'. "
        "INCLUDES answering slot-filling questions like 'Where are you traveling?'. "
        "Use 'recommend' when user expresses intent to GET a specific product (e.g., 'I want Fraud Protect360').\n"
        "- purchase: when user wants to buy, get a quote, asks for price, or wants a link. Also use this if they ask 'how much is it' or 'can I buy'.\n"
        "- policy_service: when user wants to check/manage EXISTING policies or claims. "
        "Examples: 'what is my policy status', 'check my claim', 'update my email', 'change my phone number', "
        "'where is my claim', 'list my policies', 'update my address', 'change payment info'. "
        "Use this for ANY request about existing account, policies, claims, or personal detail updates.\n"
        "- capabilities: asks what the bot can do or support.\n"
        "- greet: very short greetings like 'hi', 'hello', 'hey'.\n"
        "- chat: small-talk or open conversation without a clear insurance task yet.\n"
        "- other: anything else.\n\n"
        
        "CONTEXT-AWARE CLASSIFICATION (IMPORTANT):\n"
        "- If the user just asked an 'info' question (current phase is 'info_query') and now asks:\n"
        "  * 'what else', 'what about', 'tell me more', 'anything else', 'what other', 'more details', etc.\n"
        "  → Classify as 'info' (it's a follow-up info question, NOT recommend)\n"
        "- If the user is in 'info_query' phase and asks a question about coverage/benefits/features:\n"
        "  → Classify as 'info' (they're still exploring information)\n"
        "- Only classify as 'recommend' if the user explicitly wants a plan suggestion or says 'I want X insurance'\n"
        "- Pay attention to the current_phase - it provides crucial context for disambiguation!\n\n"
        
        f"Products available: {product_list}\n"
        "PRODUCT ALIAS MAPPING (Use these strict mappings):\n"
        f"{product_aliases}\n\n"
        "SESSION RESTART DETECTION (CRITICAL - HIGHEST PRIORITY):\n"
        "Set 'reset' to True if the user wants to restart the session or start fresh.\n"
        "This takes ABSOLUTE PRECEDENCE over all other intent classification.\n"
        "Examples that MUST trigger reset=True:\n"
        "- 'restart session', 'restart the session', 'restart', 'restart chat'\n"
        "- 'start over', 'start again', 'begin again', 'start from scratch'\n"
        "- 'reset', 'reset session', 'reset the chat', 'reset everything'\n"
        "- 'new conversation', 'fresh start', 'new session', 'start fresh'\n"
        "- 'I want to start fresh', 'let's start from the beginning', 'let's start fresh'\n"
        "- 'clear everything and start over', 'clear and restart'\n"
        "IMPORTANT: If you detect ANY variation of restart/reset/start fresh intent, "
        "you MUST set reset=True regardless of what else the message contains.\n"
        "Use your understanding of natural language - if the user clearly wants to abandon the current "
        "conversation and start fresh, set reset=True. This is NOT keyword matching - understand the intent.\n\n"
    )
    
    # Add context sections
    if context_parts:
        sys_msg += "ADDITIONAL CONTEXT:\n" + "\n\n".join(context_parts)

    user_ctx = (
        f"Current conversation phase: {current_phase or 'unknown'}\n"
        f"Known product from history: {known_product or 'None'}\n\n"
        f"Recent conversation (most recent last):\n{history_ctx}\n\n"
        f"Latest user message:\n{last_user}"
    )

    try:
        structured = _get_intent_classifier()  # Use cached classifier
        result = structured.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=user_ctx),
        ])

        duration = time.perf_counter() - start_time

        if isinstance(result, IntentPrediction):
            # Log reset flag detection
            if getattr(result, "reset", False):
                logger.warning(
                    "Intent.classify: SESSION RESTART DETECTED (reset=True) intent=%s",
                    result.intent
                )
            
            # Explicitly prioritize strong product signals from detection
            if result.product and result.product.lower() != (known_product or "").lower():
                logger.info(
                    "Intent.classify: product switch detected: %s -> %s", 
                    known_product, result.product
                )
            
            logger.info(
                "Intent.classify: intent=%s product=%s reset=%s reason=%s phase=%s duration=%.3fs",
                result.intent, result.product, getattr(result, "reset", False), 
                result.reason, current_phase, duration
            )
            return result
        return IntentPrediction.model_validate(result)
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.error(
            "Intent.classify.FAILED: error=%s duration=%.3fs",
            str(e), duration,
            exc_info=True
        )
        return IntentPrediction(
            intent="info", product=known_product, reason="classification_failed"
        )

def detect_product_node(state: AgentState) -> AgentState:
    """Parallel Product Detection Node.

    Runs concurrently with Master Agent.
    Detects if the user has switched product context and updates the state.
    This ensures the NEXT turn has the correct product context.
    """
    messages = list(state.get("messages", []) or [])
    if not messages:
        return {}

    # Get the latest user message
    last_user_full = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_full = str(getattr(m, "content", "") or "")
            break
            
    if not last_user_full:
        return {}

    current_product = state.get("product")
    
    try:
        # Use pure LLM detection with context awareness
        # This runs in parallel so latency is hidden
        detected_product = _detect_product_llm(last_user_full, current_product=current_product)
        
        # Use case-insensitive comparison to avoid false positives (Travel vs travel)
        if detected_product and current_product and detected_product.lower() != current_product.lower():
            # Product change detected - flag it for supervisor to decide based on state
            # NO HEURISTIC CHECKS - let supervisor use conversation state to decide
            logger.warning(
                "Agentic.detect_node: PRODUCT SWITCH ATTEMPT DETECTED %s -> %s (setting flag for supervisor)",
                current_product,
                detected_product,
            )
            # Set flag for supervisor to handle based on conversation state
            return {
                "product": current_product,  # Keep current product for now
                "product_switch_attempted": detected_product  # Flag for supervisor
            }
        
        # No product yet - set initial product
        if detected_product and not current_product:
            logger.info("Agentic.detect_node: initial product detected: %s", detected_product)
            return {"product": detected_product}
        
        # Add debug log for detection result even if no change
        logger.debug(
            "Agentic.detect_node: detection run. current=%s detected=%s",
            current_product, detected_product
        )
            
    except Exception as e:
        logger.warning("Agentic.detect_node: detection failed: %s", e)

    return {}
