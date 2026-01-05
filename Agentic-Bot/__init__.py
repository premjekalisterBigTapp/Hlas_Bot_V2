"""
HLAS Agentic Chatbot - Main Entry Point

This module provides the main `agentic_chat` function that orchestrates:
- LangGraph-based conversation flow
- Token-aware memory management
- MongoDB conversation history persistence
- Live agent handoff detection

Key features:
- Session-based conversation persistence via Redis checkpointer
- Rolling memory compression with summarization
- Parallel policy validation (when enabled)
- Proper message ID and metadata tracking for debugging
- Comprehensive logging for debugging
- Runtime context passing (session_id, channel) for middleware

Message Handling:
- All messages created with unique IDs for targeted RemoveMessage operations
- Metadata includes session_id, turn_count, timestamp, and channel
- Usage metadata extracted from AI responses for cost tracking

Context Engineering:
- AgentContext is passed to the graph for middleware access
- Middleware can access session_id, channel via runtime.context
- Enables dynamic prompts and tool filtering based on context
"""
from __future__ import annotations

import logging
import time
import asyncio
from typing import Any, Dict

from langchain_core.messages import HumanMessage, AIMessage

from .state import AgentState
from .graph import get_agent_graph, _memory_saver
from .middleware import AgentContext  # Context schema for runtime context
from .infrastructure.background_logger import enqueue_log
from .infrastructure.metrics import AGENTIC_MESSAGES_TOTAL, AGENTIC_LATENCY
from .utils.messages import (
    create_human_message,
    log_usage_metadata,
    get_message_id,
    log_messages_summary,
)
from .utils.pii_masker import get_pii_masker

# Local greeting (avoid cross-module dependency)
GREETING_MESSAGE = (
    "Hello! 👋 I’m the HLAS Smart Bot. I’m here to guide you through our insurance products and services, "
    "answer your questions instantly, and make things easier for you. How can I help you today?"
)

from .utils.slots import _detect_product_llm
# Policy validator disabled for now - uncomment when needed
# from .nodes.policy_validator import check_policy

logger = logging.getLogger(__name__)

# Response pattern that indicates bot is handing off to live agent
# The master prompt instructs the bot to say this exact phrase
LIVE_AGENT_HANDOFF_PHRASE = "connect you with a live agent"

def _is_live_agent_response(response: str) -> bool:
    """Check if bot response indicates live agent handoff."""
    if not response:
        return False
    return LIVE_AGENT_HANDOFF_PHRASE in response.lower()

async def agentic_chat(
    session_id: str,
    message: str,
    channel: str = "api",
) -> Dict[str, Any]:
    """
    Entrypoint for the LangGraph-based agentic chatbot.

    This function:
    1. Handles "hi" greeting resets (clears LangGraph checkpoints)
    2. Invokes the LangGraph agent with the user's message
    3. Logs conversation history to MongoDB
    4. Detects live agent handoff requests
    
    The MongoDB logging ensures conversation history is preserved for:
    - Analytics and reporting
    - Debugging conversation issues
    - Compliance and audit trails
    
    Note: WhatsApp handler also calls this function, so MongoDB logging
    happens for all channels through this single entry point.

    Args:
        session_id: Unique session/thread identifier
        message: User's message text
        channel: Communication channel ("api", "whatsapp", "web")
        
    Returns:
        Dict with 'response', 'sources', and 'debug_state' keys
    """
    start_time = time.perf_counter()

    # Entry log with truncated message preview
    msg_preview = (message or "").replace("\n", " ")[:160]
    logger.info(
        "Agentic.chat.start: session=%s msg='%s'",
        session_id,
        msg_preview,
    )

    # Special-case "hi" greeting to reset conversation state, mirroring /chat.
    if (message or "").strip().lower() == "hi":
        # Reset logic
        try:
            # Try standard clear if supported
            # If delete_thread exists, it likely takes config. But if it fails with unhashable dict,
            # it implies implementation differences. We'll try update_state to clear messages instead.
            
            # Attempt to clear history by updating state with empty messages list
            # This is the standard LangGraph way to "reset" or modify past state.
            # Note: This appends/overwrites depending on reducer. For 'messages' (add_messages), 
            # we usually can't easily clear without a custom reducer or checkpoint hack.
            # But for this specific user request ("reset"), we can just generate a new thread ID 
            # effectively by handling it at the client/session manager level (which we do in /chat).
            # Here in /agent-chat, we rely on session_id.
            
            # Let's try passing the string session_id directly to delete_thread if config dict failed.
            if hasattr(_memory_saver, "delete_thread"):
                 try:
                     _memory_saver.delete_thread({"configurable": {"thread_id": session_id}})
                 except TypeError:
                     # Fallback: try passing session_id as string
                     _memory_saver.delete_thread(session_id)
            else:
                # If we can't delete, we just proceed. The ReAct agent handles context windowing anyway.
                pass
        except Exception as e:
            logger.warning("Agentic.chat: failed to reset LangGraph thread for 'hi' - %s", e)
        
        logger.info("Agentic.chat.hi_reset: session=%s", session_id)
        return {"response": GREETING_MESSAGE, "sources": "", "debug_state": {}}

    # Run Policy Check AND Graph Execution in Parallel
    config = {"configurable": {"thread_id": session_id}}
    agent_graph = get_agent_graph()
    
    # Create runtime context for middleware access
    # This provides session_id and channel to middleware (dynamic_prompt, tool filtering)
    runtime_context = AgentContext(
        session_id=session_id,
        channel=channel,
    )
    
    try:
        # Get history for context (optional, but helpful for policy check)
        history_snapshot = await agent_graph.aget_state(config)
        history_msgs = history_snapshot.values.get("messages", [])
        historical_product = history_snapshot.values.get("product")

        # Prefer product detected from the current message; fall back to history if needed.
        # OPTIMIZATION: Use historical product for Policy Check to avoid blocking LLM call.
        # The Graph will run product detection in parallel for the NEXT turn.
        product = historical_product

        logger.debug(
            "Agentic.chat.history: session=%s messages=%d product=%s",
            session_id,
            len(history_msgs or []),
            product,
        )
        # Simple string history for query generator
        history_str = "\n".join([f"{m.type}: {m.content}" for m in history_msgs[-4:]])
        
        # Get current turn count from state
        turn_count = history_snapshot.values.get("turn_count", 0)

        # =================================================================
        # PII MASKING - Mask sensitive data before ANY LLM processing
        # =================================================================
        pii_masker = get_pii_masker()
        masked_message, new_pii_mapping = pii_masker.mask(message, session_id)
        
        if new_pii_mapping:
            logger.info(
                "Agentic.chat.pii_masked: session=%s types=%s count=%d",
                session_id,
                list(set(p.split('_')[0].strip('[]') for p in new_pii_mapping.keys())),
                len(new_pii_mapping),
            )
        
        # Get accumulated PII mapping for this session
        session_pii_mapping = pii_masker.get_session_mapping(session_id)

        # Prepare graph inputs with properly constructed HumanMessage
        # Message has unique ID and metadata for tracking and targeted removal
        # NOTE: We pass the MASKED message to the graph - PII never reaches LLM
        human_msg = create_human_message(
            content=masked_message,  # MASKED message
            session_id=session_id,
            turn_count=turn_count,
            channel=channel,
            product=product,
        )
        
        logger.debug(
            "Agentic.chat.message_created: id=%s session=%s turn=%d",
            get_message_id(human_msg),
            session_id,
            turn_count,
        )
        
        # Include PII mapping in graph inputs for service flow to use
        graph_inputs = {
            "messages": [human_msg],
            "pii_mapping": session_pii_mapping,  # Full session PII mapping
        }

        # Launch both tasks concurrently
        # policy_task = asyncio.create_task(check_policy(message, history_str, product))
        # graph_task = asyncio.create_task(agent_graph.ainvoke(
        #     graph_inputs,
        #     config=config,
        #     context=runtime_context,  # Pass context for middleware
        # ))
        
        # TRUE PARALLELISM with early return:
        # - If policy finds violation FIRST → return rejection immediately (cancel graph)
        # - If graph finishes FIRST → wait for policy, then decide
        # - If policy is CLEAN → return graph result
        # result = None
        # is_violation = False
        # policy_reply = ""
        
        # pending = {policy_task, graph_task}
        # while pending:
        #     done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            
        #     for task in done:
        #         if task is policy_task:
        #             is_violation, policy_reply, policy_reason = task.result()
        #             if is_violation:
        #                 # VIOLATION: Cancel graph and return immediately
        #                 # Don't await - let it finish/timeout in background to avoid blocking
        #                 graph_task.cancel()
                        
        #                 reply_preview = (policy_reply or "").replace("\n", " ")[:200]
        #                 logger.warning(
        #                     "Agentic.chat.policy_block: session=%s reason='%s' preview='%s'",
        #                     session_id,
        #                     policy_reason,
        #                     reply_preview,
        #                 )
                        
        #                 # Persist rejection to history (this overwrites any concurrent graph state)
        #                 try:
        #                     await agent_graph.aupdate_state(
        #                         config,
        #                         {"messages": [AIMessage(content=policy_reply)]},
        #                         as_node="master_agent",
        #                     )
        #                 except Exception as e:
        #                     logger.warning("Agentic.chat: failed to persist policy rejection: %s", e)
                        
        #                 return {"response": policy_reply, "sources": "", "debug_state": {"violation": True, "reason": policy_reason}}
                
        #         elif task is graph_task:
        #             result = task.result()
        
        # Both tasks completed, policy was CLEAN
        # if result is None:
        #     raise RuntimeError("Graph task did not produce a result")

        # Invoke graph with runtime context
        # Context is accessible to middleware via runtime.context
        logger.debug(
            "Agentic.chat.invoking: session=%s channel=%s",
            runtime_context.session_id,
            runtime_context.channel,
        )
        result = await agent_graph.ainvoke(
            graph_inputs,
            config=config,
            context=runtime_context,  # Pass context for middleware access
        )
        
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("Agentic.chat: graph invocation failed, returning fallback")
        fallback = (
            "Something went wrong while processing your request. "
            "Please try rephrasing your question or ask about a specific "
            "HLAS product such as Travel, Maid, Car, Personal Accident, "
            "Home, Fraud, Hospital or Early."
        )
        return {"response": fallback, "sources": "", "debug_state": {}}
    
    # ... existing result processing
    


    messages = result.get("messages", []) or []
    
    # Log message summary for debugging
    log_messages_summary(messages, context=f"session={session_id}")
    
    reply = ""
    last_ai_msg = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            last_ai_msg = m
            reply = str(getattr(m, "content", "") or "").strip()
            if reply:
                break
    
    # Extract and log usage metadata from AI response for cost tracking
    if last_ai_msg:
        try:
            log_usage_metadata(last_ai_msg, session_id=session_id)
        except Exception as e:
            logger.debug("Agentic.chat.usage_metadata_error: %s", e)
    
    if not reply:
        reply = (
            "I'm not sure I understood that. Could you clarify what you'd like "
            "to know about our insurance plans?"
        )

    sources_val = result.get("sources") or []
    if isinstance(sources_val, str):
        sources_str = sources_val
    else:
        sources_str = "\n".join(str(s) for s in sources_val if s)

    # Detect live agent handoff from EITHER:
    # 1. State flag set by autonomous routing (live_agent_handoff_node)
    # 2. Bot response content (legacy detection)
    live_agent_from_state = result.get("live_agent_requested", False)
    live_agent_from_response = _is_live_agent_response(reply)
    live_agent_requested = live_agent_from_state or live_agent_from_response
    
    if live_agent_requested:
        logger.info(
            "Agentic.chat.live_agent_detected: session=%s from_state=%s from_response=%s",
            session_id, live_agent_from_state, live_agent_from_response
        )
    
    debug_state = {
        "intent": result.get("intent"),
        "product": result.get("product"),
        "rec_ready": result.get("rec_ready", False),
        "live_agent_requested": live_agent_requested,
        # Slot-filling state
        "slots": result.get("slots"),
        "pending_slot": result.get("pending_slot"),
        # Conversation phase
        "phase": result.get("phase"),
        # Additional autonomous routing info for debugging
        "last_routing_decision": result.get("last_routing_decision"),
        "self_correction_count": result.get("self_correction_count", 0),
    }

    # Calculate duration
    duration = time.perf_counter() - start_time
    
    logger.info(
        "Agentic.chat.completed: session=%s intent=%s product=%s live_agent=%s reply_len=%d duration=%.3fs",
        session_id,
        debug_state.get("intent"),
        debug_state.get("product"),
        live_agent_requested,
        len(reply),
        duration,
    )
    
    # Log debug info at debug level (full reply)
    logger.debug(
        "Agentic.chat.reply: session=%s reply='%s'",
        session_id,
        reply.replace("\n", " ")[:500],
    )

    # Record metrics
    try:
        AGENTIC_LATENCY.labels(endpoint="agentic_chat").observe(duration)
        AGENTIC_MESSAGES_TOTAL.labels(
            result="success" if reply else "empty",
            product=debug_state.get("product") or "unknown"
        ).inc()
    except Exception as e:
        logger.warning("Agentic.chat.metrics_error: %s", e)

    # Log conversation to MongoDB (non-blocking background task)
    # This is the SINGLE point of MongoDB logging for all channels
    # (HTTP /agent-chat and WhatsApp both go through this function)
    # 
    # Uses BackgroundLogger for:
    # - Non-blocking writes (doesn't add latency to response)
    # - Automatic retry with exponential backoff
    # - Graceful degradation (sync fallback if queue full)
    # - Proper shutdown handling (drains pending logs)
    await enqueue_log(
        session_id=session_id,
        user_message=message,
        assistant_message=reply,
        metadata=debug_state,
    )
    logger.debug("Agentic.chat.mongo_enqueued: session=%s", session_id)

    return {"response": reply, "sources": sources_str, "debug_state": debug_state}

__all__ = ["agentic_chat"]
