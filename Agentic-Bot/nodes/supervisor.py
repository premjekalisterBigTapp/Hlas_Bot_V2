"""
Supervisor Node - Intent classification, phase tracking, and autonomous routing.

This module implements the supervisor with:
- Explicit conversation phase tracking (ConversationPhase enum)
- Phase history for debugging and analytics
- Pronoun/reference context extraction and passing
- Summary-aware intent classification
- Autonomous Command-based routing

Multi-Turn Conversation Fixes:
1. Intent classification now uses summary for full history context
2. Pronoun resolution via ReferenceContext
3. Explicit phase tracking with phase_history
4. Phase-aware intent classification

The supervisor uses Command(update={...}, goto="node") to:
- Update state AND route in a single operation
- Track conversation phase transitions
- Route to self_correction on repeated tool errors
- Route to live_agent_handoff when detected
- Handle negative feedback with reflection
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Union, Literal, List

from langchain_core.messages import AIMessage
from langgraph.types import Command

from ..state import AgentState, ConversationPhase, ReferenceContext
from .feedback import _classify_feedback_from_messages, _self_critique_and_rewrite_from_messages
from .intent import _classify_intent_from_messages, _extract_reference_context
from .autonomous_routing import (
    analyze_routing_context,
    AUTONOMOUS_ROUTING_TOTAL,
    SELF_CORRECTION_TOTAL,
)
from ..infrastructure.metrics import INTENT_CLASSIFICATION_TOTAL

logger = logging.getLogger(__name__)


# =============================================================================
# PHASE TRACKING METRICS (imported from centralized metrics)
# =============================================================================

from ..infrastructure.metrics import PHASE_TRANSITION_TOTAL, PHASE_DURATION_TURNS as PHASE_DURATION


# Valid target nodes for supervisor routing
SupervisorTargets = Literal[
    "greet_agent",
    "capabilities_agent", 
    "chat_agent",
    "info_agent",
    "summary_agent",
    "compare_agent",
    "purchase_agent",
    "recommendation",
    "service_flow",
    "self_correction",
    "live_agent_handoff",
    "styler",
]


def _compute_new_phase(
    intent: str,
    product: Optional[str],
    rec_given: bool,
    purchase_offered: bool,
    live_agent_requested: bool,
) -> ConversationPhase:
    """
    Compute the new conversation phase based on intent and state.
    
    This provides a deterministic mapping from intent + state to phase,
    ensuring consistent phase tracking across the system.
    """
    if live_agent_requested:
        return ConversationPhase.ESCALATION
    
    return ConversationPhase.from_intent(
        intent=intent,
        has_product=bool(product),
        rec_given=rec_given,
        purchase_offered=purchase_offered,
    )


def _update_phase_history(
    current_history: List[str],
    new_phase: ConversationPhase,
    max_history: int = 20,
) -> List[str]:
    """
    Update phase history with new phase, maintaining max length.
    
    Args:
        current_history: Existing phase history
        new_phase: New phase to add
        max_history: Maximum history length to maintain
        
    Returns:
        Updated phase history
    """
    history = list(current_history) if current_history else []
    history.append(new_phase.value)
    
    # Trim to max length
    if len(history) > max_history:
        history = history[-max_history:]
    
    return history


async def _supervisor_node(state: AgentState) -> Union[Command[SupervisorTargets], Dict[str, Any]]:
    """
    Supervisor node with phase tracking, pronoun resolution, and autonomous routing.
    
    This node addresses Multi-Turn Conversation Failures:
    1. Uses summary for intent classification (full history context)
    2. Extracts and passes reference context for pronoun resolution
    3. Tracks explicit conversation phase with phase_history
    4. Routes autonomously based on state and intent
    
    Autonomous routing features:
    - Routes to self_correction on repeated tool errors
    - Routes to live_agent_handoff when detected
    - Clears slots on product switch or reset
    - Updates conversation phase on every turn
    
    Performance optimization:
    - Runs feedback and intent classification in PARALLEL using asyncio.gather
    - Saves ~1-2s per turn by eliminating sequential LLM calls
    
    Returns:
        Command with update and goto
    """
    start_time = time.perf_counter()
    
    messages = list(state.get("messages", []) or [])
    if not messages:
        logger.warning("Supervisor.no_messages: routing to chat_agent")
        return Command(
            update={
                "phase": ConversationPhase.GREETING.value,
                "phase_history": [ConversationPhase.GREETING.value],
            },
            goto="chat_agent"
        )

    known_product = state.get("product")
    turn_count = state.get("turn_count", 0)
    current_phase = state.get("phase")
    current_slots = state.get("slots") or {}
    summary = state.get("summary", "")
    rec_given = state.get("rec_given", False)
    purchase_offered = state.get("purchase_offered", False)
    phase_history = state.get("phase_history") or []
    product_switch_attempted = state.get("product_switch_attempted")
    
    # CRITICAL: Check for product switch BEFORE running intent classification
    # This prevents the intent classifier from overwriting the product
    if product_switch_attempted and known_product:
        logger.warning(
            "Supervisor.PRODUCT_SWITCH_REJECTED: current=%s attempted=%s turn=%d -> BLOCKING and asking user to restart",
            known_product, product_switch_attempted, turn_count
        )
        logger.info(
            "Supervisor.PRODUCT_SWITCH_REJECTED: Keeping product=%s, clearing flag, routing to styler",
            known_product
        )
        return Command(
            update={
                "messages": [AIMessage(
                    content=f"I'm sorry, but I cannot switch to {product_switch_attempted} insurance during our current conversation. "
                           f"To explore {product_switch_attempted} insurance, please say 'Restart Session' or 'Start Over' to begin fresh. "
                           f"For now, let's continue with {known_product} insurance. Would you like to proceed with {known_product}?"
                )],
                "product": known_product,  # Keep current product - DO NOT SWITCH
                "product_switch_attempted": None,  # Clear flag
                "phase": ConversationPhase.PRODUCT_SELECTION.value,
                "phase_history": _update_phase_history(phase_history, ConversationPhase.PRODUCT_SELECTION),
            },
            goto="styler"
        )
    
    # Build routing context for autonomous decisions
    routing_context = analyze_routing_context(state)
    
    # Extract reference context for pronoun resolution
    reference_context = _extract_reference_context(messages, known_product, current_slots)
    
    logger.debug(
        "Supervisor.context: turn=%d phase=%s tool_errors=%d live_agent=%s product=%s",
        turn_count,
        current_phase,
        routing_context.tool_error_count,
        routing_context.live_agent_requested,
        known_product,
    )
    
    # Priority 1: Check for live agent escalation
    if routing_context.live_agent_requested:
        new_phase = ConversationPhase.ESCALATION
        
        logger.info(
            "Supervisor.live_agent_detected: turn=%d phase=%s->%s routing to live_agent_handoff",
            turn_count, current_phase, new_phase.value
        )
        
        AUTONOMOUS_ROUTING_TOTAL.labels(
            source_node="supervisor",
            target_node="live_agent_handoff",
            reason="live_agent_requested"
        ).inc()
        PHASE_TRANSITION_TOTAL.labels(
            from_phase=current_phase or "unknown",
            to_phase=new_phase.value,
            trigger="live_agent_request"
        ).inc()
        
        return Command(
            update={
                "intent": "live_agent",
                "phase": new_phase.value,
                "phase_history": _update_phase_history(phase_history, new_phase),
            },
            goto="live_agent_handoff",
        )
    
    # Priority 2: Check for self-correction need (repeated tool errors)
    if routing_context.tool_error_count >= 2:
        logger.warning(
            "Supervisor.tool_errors_detected: turn=%d errors=%d routing to self_correction",
            turn_count, routing_context.tool_error_count
        )
        AUTONOMOUS_ROUTING_TOTAL.labels(
            source_node="supervisor",
            target_node="self_correction",
            reason="repeated_tool_errors"
        ).inc()
        SELF_CORRECTION_TOTAL.labels(
            trigger="tool_error",
            outcome="routing_to_correction"
        ).inc()
        return Command(
            update={"intent": "self_correct"},
            goto="self_correction",
        )

    # ======================================================================
    # SERVICE FLOW GUARD: while in policy/claim service flow, do not
    # re-interpret user messages as new top-level intents.
    #
    # This prevents short replies like NRIC fragments or initials from being
    # classified as 'other' and routed to the generic chat agent, which the
    # user experiences as hallucination. As long as we are mid service flow
    # (not yet validated, collecting credentials, or executing an action),
    # always route back to the service_flow subgraph.
    # ======================================================================
    if current_phase == ConversationPhase.SERVICE_FLOW.value:
        customer_validated = state.get("customer_validated", False)
        service_action = state.get("service_action")
        service_pending_slot = state.get("service_pending_slot")

        in_active_service_flow = (
            not customer_validated
            or bool(service_pending_slot)
            or bool(service_action)
        )

        if in_active_service_flow:
            new_phase = ConversationPhase.SERVICE_FLOW
            logger.info(
                "Supervisor.service_flow_guard: turn=%d phase=%s validated=%s action=%s pending_slot=%s -> service_flow",
                turn_count,
                current_phase,
                customer_validated,
                service_action,
                service_pending_slot,
            )

            return Command(
                update={
                    "intent": "policy_service",
                    "phase": new_phase.value,
                    "phase_history": _update_phase_history(phase_history, new_phase),
                },
                goto="service_flow",
            )
    
    # ==========================================================================
    # SLOT COLLECTION PATH: When a slot question is pending, always route to
    # the recommendation subgraph. The LLM-based slot extractor will intelligently
    # decide if the user's message is:
    # - A slot answer → extract and continue
    # - A side question → detect via side_question field and answer it
    # - Something else → handle appropriately
    # 
    # This is simpler and more robust than pattern matching.
    # 
    # EXCEPTION: If user explicitly asks to "compare", route to comparison
    # even if there's a pending slot. User is changing their mind.
    # ==========================================================================
    pending_slot = state.get("pending_slot")
    
    # Extract last user message
    last_user_msg = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            last_user_msg = (getattr(msg, "content", "") or "").strip()
            break
    
    # ==========================================================================
    # PARALLEL CLASSIFICATION: Run feedback and intent classification concurrently
    # This saves ~1-2s per turn by eliminating sequential LLM calls
    # IMPORTANT: We MUST run intent classification even if pending_slot is set
    # to detect product switches and session restarts
    # ==========================================================================
    
    # Define the classification tasks
    async def classify_feedback():
        return await asyncio.to_thread(_classify_feedback_from_messages, messages)
    
    async def classify_intent():
        return await asyncio.to_thread(
            _classify_intent_from_messages,
            messages=messages,
            known_product=known_product,
            active_slot=pending_slot,
            summary=summary,
            current_phase=current_phase,
            current_slots=current_slots,
            reference_context=reference_context,
            rec_given=rec_given,  # Pass rec_given to prevent re-recommending after upsell
        )
    
    # Run both classifiers in parallel
    parallel_start = time.perf_counter()
    feedback, intent_pred = await asyncio.gather(
        classify_feedback(),
        classify_intent(),
    )
    parallel_duration = time.perf_counter() - parallel_start
    logger.debug(
        "Supervisor.parallel_classification: duration=%.3fs",
        parallel_duration
    )
    
    # Priority 3: Session Restart Detection (LLM-Driven, Not Heuristic)
    # Check if user wants to restart the session (e.g., "restart session", "start over")
    if intent_pred and getattr(intent_pred, "reset", False):
        logger.warning(
            "Supervisor.SESSION_RESTART_TRIGGERED: reset=True detected at turn=%d -> CLEARING ALL STATE",
            turn_count
        )
        logger.info(
            "Supervisor.SESSION_RESTART: Clearing product, slots, phase, and all conversation state"
        )
        return Command(
            update={
                "messages": [AIMessage(
                    content="Sure! Let's start fresh. How can I help you today?"
                )],
                # Clear all state to restart session
                "product": None,
                "product_switch_attempted": None,
                "intent": "greet",
                "phase": ConversationPhase.GREETING.value,
                "phase_history": [ConversationPhase.GREETING.value],
                "slots": {},
                "pending_slot": None,
                "rec_ready": False,
                "rec_given": False,
                "purchase_offered": False,
                "sources": [],
                "tiers": [],
                "summary": "",
            },
            goto="chat_agent",
        )
    
    # Priority 4: Negative feedback handling / reflection
    pending_slot = state.get("pending_slot")
    if feedback and feedback.category == "negative_feedback":
        revised = _self_critique_and_rewrite_from_messages(
            messages, 
            pending_slot=pending_slot,
            product=known_product
        )
        if revised:
            logger.info(
                "Supervisor.negative_feedback: turn=%d triggering self-critique (slot_mode=%s)",
                turn_count, bool(pending_slot)
            )
            AUTONOMOUS_ROUTING_TOTAL.labels(
                source_node="supervisor",
                target_node="styler",
                reason="negative_feedback_reflection"
            ).inc()
            
            # If we're in slot collection mode, keep pending_slot so flow continues
            # Otherwise clear it as the context has changed
            slot_updates = {}
            if pending_slot:
                # Keep the slot context - user should answer the re-asked question
                slot_updates["pending_slot"] = pending_slot
                slot_updates["is_slot_reask"] = True
            else:
                slot_updates["pending_slot"] = None
                slot_updates["is_slot_reask"] = None
            
            return Command(
                update={
                    "messages": [AIMessage(content=revised)],
                    "feedback": "negative_feedback",
                    "sources": [],
                    "intent": "reflect_done",
                    **slot_updates,
                },
                goto="styler",
            )

    # Priority 4: Detect product switch from intent classification
    # If intent_pred detected a NEW product different from known_product, block it
    # Use case-insensitive comparison to avoid false positives (Travel vs travel)
    if intent_pred.product and known_product and intent_pred.product.lower() != known_product.lower():
        logger.warning(
            "Supervisor.PRODUCT_SWITCH_FROM_INTENT: current=%s detected=%s turn=%d -> BLOCKING",
            known_product, intent_pred.product, turn_count
        )
        return Command(
            update={
                "messages": [AIMessage(
                    content=f"I'm sorry, but I cannot switch to {intent_pred.product} insurance during our current conversation. "
                           f"To explore {intent_pred.product} insurance, please say 'Restart Session' or 'Start Over' to begin fresh. "
                           f"For now, let's continue with {known_product} insurance. Would you like to proceed with {known_product}?"
                )],
                "product": known_product,  # Keep current product - DO NOT SWITCH
                "phase": ConversationPhase.PRODUCT_SELECTION.value,
                "phase_history": _update_phase_history(phase_history, ConversationPhase.PRODUCT_SELECTION),
            },
            goto="styler"
        )
    
    # Priority 5: Slot collection shortcut (if pending_slot and no product switch detected)
    # This allows the recommendation flow to continue collecting slots
    if pending_slot and not rec_given and known_product:
        logger.info(
            "Supervisor.slot_collection_path: pending_slot=%s msg='%s' -> routing to recommendation subgraph",
            pending_slot, last_user_msg[:50]
        )
        new_phase = ConversationPhase.SLOT_FILLING
        return Command(
            update={
                "intent": "recommend",
                "product": known_product,
                "phase": new_phase.value,
                "phase_history": _update_phase_history(phase_history, new_phase),
            },
            goto="recommendation",
        )
    
    # Priority 6: Use intent classification result (already computed in parallel)
    
    raw_intent = (intent_pred.intent or "").strip().lower()
    
    # Map intent to target node
    intent_to_node = {
        "info": "info_agent",
        "summary": "summary_agent",
        "compare": "compare_agent",
        "recommend": "recommendation",
        "purchase": "purchase_agent",
        "capabilities": "capabilities_agent",
        "greet": "greet_agent",
        "chat": "chat_agent",
        "policy_service": "service_flow",
        "other": "chat_agent",
    }
    
    # Normalize intent and determine target
    if raw_intent not in intent_to_node:
        normalized_intent = "chat"
        target_node = "chat_agent"
        logger.debug(
            "Supervisor.unknown_intent: raw=%s normalized to chat",
            raw_intent
        )
    else:
        normalized_intent = raw_intent
        target_node = intent_to_node[raw_intent]

    product = intent_pred.product or known_product
    
    # Compute new conversation phase
    new_phase = _compute_new_phase(
        intent=normalized_intent,
        product=product,
        rec_given=rec_given,
        purchase_offered=purchase_offered,
        live_agent_requested=False,
    )
    
    # Track phase transition
    if current_phase and current_phase != new_phase.value:
        PHASE_TRANSITION_TOTAL.labels(
            from_phase=current_phase,
            to_phase=new_phase.value,
            trigger=f"intent_{normalized_intent}"
        ).inc()
        logger.info(
            "Supervisor.phase_transition: %s -> %s (intent=%s)",
            current_phase, new_phase.value, normalized_intent
        )
    
    # Build state updates
    updates: Dict[str, Any] = {
        "intent": normalized_intent,
        "product": product,
        "phase": new_phase.value,
        "phase_history": _update_phase_history(phase_history, new_phase),
        # Store reference context for downstream use
        "reference_context": {
            "last_mentioned_product": reference_context.last_mentioned_product,
            "last_mentioned_tier": reference_context.last_mentioned_tier,
            "last_mentioned_destination": reference_context.last_mentioned_destination,
            "compared_items": reference_context.compared_items,
            "last_bot_question": reference_context.last_bot_question,
            "last_updated_turn": turn_count,
        },
    }
    
    # Intelligent State Management:
    # Clear slots on product switch, explicit reset, OR greeting (fresh start)
    is_reset = getattr(intent_pred, "reset", False)
    is_product_switch = (product and known_product and product != known_product)
    is_greeting = (normalized_intent == "greet")  # User says hi/hello = fresh start
    
    if is_product_switch or is_reset or is_greeting:
        reason = "greeting" if is_greeting else ("product_switch" if is_product_switch else "explicit_reset")
        logger.info(
            "Supervisor.clearing_state: reason=%s old_product=%s",
            reason, known_product
        )
        # Clear recommendation-related state
        updates["slots"] = {}
        updates["rec_ready"] = False
        updates["rec_given"] = False
        updates["pending_slot"] = None
        updates["slot_validation_errors"] = {}
        updates["side_info"] = None
        updates["pending_side_question"] = None
        updates["is_slot_reask"] = None
        updates["product"] = None  # Clear product on greeting
        
        # Clear service flow state (policy/claim services)
        updates["service_action"] = None
        updates["service_slots"] = {}
        updates["service_pending_slot"] = None
        updates["customer_validated"] = False
        updates["customer_nric"] = None
        updates["customer_data"] = None
        
        # Reset phase
        updates["phase"] = ConversationPhase.GREETING.value
        updates["phase_history"] = _update_phase_history(
            updates["phase_history"],
            ConversationPhase.GREETING
        )
    
    duration = time.perf_counter() - start_time
    
    logger.info(
        "Supervisor.routing: turn=%d intent=%s product=%s phase=%s -> %s duration=%.3fs",
        turn_count, normalized_intent, product, new_phase.value, target_node, duration
    )
    
    # Record metrics
    INTENT_CLASSIFICATION_TOTAL.labels(
        intent=normalized_intent,
        product=product or "unknown"
    ).inc()
    AUTONOMOUS_ROUTING_TOTAL.labels(
        source_node="supervisor",
        target_node=target_node,
        reason=f"intent_{normalized_intent}"
    ).inc()

    return Command(update=updates, goto=target_node)
