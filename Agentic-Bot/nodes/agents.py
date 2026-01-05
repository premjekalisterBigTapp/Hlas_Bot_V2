from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from ..state import AgentState
from ..utils.memory import _get_last_user_message
from ..tools.info import _info_tool
from ..tools.summary import _summary_tool
from ..tools.compare import _compare_tool
from ..tools.purchase import _purchase_tool
from ..tools.capabilities import _capabilities_tool

logger = logging.getLogger(__name__)


def _get_continuation_for_incomplete_rec(state: AgentState) -> Tuple[Optional[str], Optional[str]]:
    """
    Check if we're in an incomplete recommendation flow and return continuation prompt.
    
    Returns:
        Tuple of (continuation_text, missing_slot_name) or (None, None) if not applicable
    """
    product = state.get("product")
    rec_given = state.get("rec_given", False)
    slots = state.get("slots") or {}
    
    if not product or rec_given:
        return None, None
    
    # Import here to avoid circular imports
    from ..utils.products import PRODUCT_DEFINITIONS
    from ..utils.slots import _normalize_product_key
    
    prod_key = _normalize_product_key(product)
    if not prod_key or prod_key not in PRODUCT_DEFINITIONS:
        return None, None
    
    prod_def = PRODUCT_DEFINITIONS[prod_key]
    required_slots = prod_def.required_slots
    
    # Find the first missing slot
    missing_slot = None
    for slot in required_slots:
        if not slots.get(slot):
            missing_slot = slot
            break
    
    if not missing_slot:
        return None, None
    
    # Get the slot question if available
    slot_config = prod_def.slot_config.get(missing_slot)
    if slot_config and slot_config.question:
        continuation = f"\n\nNow, back to your recommendation — {slot_config.question}"
    else:
        # Build a friendly prompt based on slot name
        slot_label = missing_slot.replace("_", " ")
        continuation = f"\n\nNow, back to your recommendation — could you please share your {slot_label}?"
    
    return continuation, missing_slot


def _greet_agent_node(state: AgentState) -> AgentState:
    logger.info("Agentic.agents: executing greet_agent")
    reply = (
        "Hello! 👋 I’m the HLAS Smart Bot. I’m here to guide you through our insurance products and services, "
        "answer your questions instantly, and make things easier for you. How can I help you today?"
    )
    return {"messages": [AIMessage(content=reply)], "sources": []}


def _capabilities_agent_node(state: AgentState) -> AgentState:
    user_text = _get_last_user_message(state.get("messages", []) or [])
    logger.info("Agentic.agents: executing capabilities_agent for query: %s", user_text)
    reply = _capabilities_tool(user_text)
    return {"messages": [AIMessage(content=reply)], "sources": []}


def _chat_agent_node(state: AgentState) -> AgentState:
    """General conversation agent with guardrails.

    Handles greetings and farewells naturally, but politely redirects
    out-of-scope questions back to insurance topics.
    """

    user_text = _get_last_user_message(state.get("messages", []) or [])
    logger.info("Agentic.agents: executing chat_agent for query: %s", user_text)

    system_prompt = """You are HLAS's friendly digital insurance assistant.

IMPORTANT GUARDRAILS - You must follow these strictly:

1. OUT-OF-SCOPE QUESTIONS: If the user asks about topics unrelated to insurance 
   (weather, sports, news, coding, recipes, general knowledge, etc.), politely 
   decline and redirect:
   - "I'm your insurance assistant, so I can't help with that. But I'd love to 
     help you with travel insurance, motor insurance, or any other coverage needs!"
   - "That's outside my expertise! I specialize in insurance - would you like 
     help finding the right coverage for you?"

2. GREETINGS & FAREWELLS: Respond naturally to "hi", "hello", "how are you", 
   "thanks", "bye" etc. Keep it brief and warm.

3. INSURANCE-ADJACENT TOPICS: If the user mentions something that COULD relate 
   to insurance (travel plans, new car, health concerns), acknowledge it warmly 
   and offer relevant insurance help.

4. NEVER engage with or answer:
   - Weather questions
   - General knowledge questions
   - Requests for information outside insurance
   - Controversial topics (politics, religion, etc.)
   - Personal advice unrelated to insurance

Keep replies concise (1-2 sentences max for redirects).
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{user_text}"),
        ]
    )

    # Reuse the router model for chat-style responses to keep configuration
    # simple and fully LLM-driven.
    from ..config import _router_model

    chain = prompt | _router_model
    ai_msg = chain.invoke({"user_text": user_text or ""})

    content = getattr(ai_msg, "content", None) or str(ai_msg)
    return {"messages": [AIMessage(content=content)], "sources": []}


def _info_agent_node(state: AgentState) -> AgentState:
    messages = state.get("messages", []) or []
    user_text = _get_last_user_message(messages)
    product = state.get("product")
    
    # Extract last bot message for context (helps reformulate vague queries like "yes please")
    last_bot_message = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            last_bot_message = (getattr(msg, "content", "") or "").strip()
            break
    
    logger.info("Agentic.agents: executing info_agent for product=%s query=%s", product, user_text)
    answer, srcs = _info_tool(product, user_text, conversation_context=last_bot_message)
    
    # Check if we're in an incomplete recommendation flow - guide user back
    continuation, missing_slot = _get_continuation_for_incomplete_rec(state)
    if continuation and missing_slot:
        answer = answer.rstrip() + continuation
        logger.info("Agentic.agents: info_agent appended continuation for slot=%s", missing_slot)
        return {
            "messages": [AIMessage(content=answer)], 
            "sources": srcs,
            "pending_slot": missing_slot,
        }
    
    return {"messages": [AIMessage(content=answer)], "sources": srcs}


def _summary_agent_node(state: AgentState) -> AgentState:
    user_text = _get_last_user_message(state.get("messages", []) or [])
    product = state.get("product")
    logger.info("Agentic.agents: executing summary_agent for product=%s", product)
    answer, srcs = _summary_tool(product, state.get("tiers") or [], user_text)
    
    # Check if we're in an incomplete recommendation flow - guide user back
    continuation, missing_slot = _get_continuation_for_incomplete_rec(state)
    if continuation and missing_slot:
        answer = answer.rstrip() + continuation
        logger.info("Agentic.agents: summary_agent appended continuation for slot=%s", missing_slot)
        return {
            "messages": [AIMessage(content=answer)], 
            "sources": srcs,
            "pending_slot": missing_slot,
        }
    
    return {"messages": [AIMessage(content=answer)], "sources": srcs}


def _compare_agent_node(state: AgentState) -> AgentState:
    user_text = _get_last_user_message(state.get("messages", []) or [])
    product = state.get("product")
    logger.info("Agentic.agents: executing compare_agent for product=%s", product)
    
    # Check if user is trying to compare different products (cross-product comparison)
    all_products = ["travel", "car", "maid", "home", "personal accident", "early", "fraud", "hospital"]
    user_lower = user_text.lower()
    mentioned_products = []
    
    for prod in all_products:
        if prod in user_lower:
            mentioned_products.append(prod.title())
    
    # If multiple products mentioned, explain limitation
    if len(mentioned_products) > 1:
        logger.info(
            "CompareAgent: Cross-product comparison requested: %s - explaining limitation",
            mentioned_products
        )
        return {
            "messages": [AIMessage(
                content=f"I can compare different coverage tiers within a single product, "
                        f"but I can't directly compare {mentioned_products[0]} vs {mentioned_products[1]} "
                        f"insurance as they serve different purposes and have different coverage types.\n\n"
                        f"Which product would you like to explore: {' or '.join(mentioned_products)}?"
            )],
            "sources": []
        }
    
    # Normal tier comparison
    answer, srcs = _compare_tool(product, state.get("tiers") or [], user_text)
    
    # Check if we're in an incomplete recommendation flow - guide user back
    continuation, missing_slot = _get_continuation_for_incomplete_rec(state)
    if continuation and missing_slot:
        answer = answer.rstrip() + continuation
        logger.info("Agentic.agents: compare_agent appended continuation for slot=%s", missing_slot)
        return {
            "messages": [AIMessage(content=answer)], 
            "sources": srcs,
            "pending_slot": missing_slot,
        }
    
    return {"messages": [AIMessage(content=answer)], "sources": srcs}


def _purchase_agent_node(state: AgentState) -> AgentState:
    product = state.get("product")
    logger.info("Agentic.agents: executing purchase_agent for product=%s", product)
    reply = _purchase_tool(product)
    return {"messages": [AIMessage(content=reply)], "sources": []}
