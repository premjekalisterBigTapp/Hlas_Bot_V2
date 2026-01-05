from __future__ import annotations

import logging
from typing import Tuple, Optional

from langchain_core.messages import SystemMessage
from weaviate.classes.query import TargetVectors, Filter

from ..config import _router_model
from ..infrastructure import get_weaviate_client, get_embeddings

logger = logging.getLogger(__name__)

# 1. Query Generation Prompt
# Generates a search query to find relevant policy documents, or SKIP for generic messages
QUERY_GEN_PROMPT = """Decide if this message needs a policy exclusion check.

Message: "{message}"

SKIP if the message is:
- A simple response: "yes", "no", "ok", "sure", "thanks"
- A number: "14", "26", "500", "18"
- A generic term: "family", "individual", "group"
- A product name: "travel", "maid", "car"
- A question about features or benefits

CHECK if the message contains:
- A specific COUNTRY name (e.g., "Sudan", "Libya", "India", "Japan")
- A specific ACTIVITY (e.g., "skydiving", "scuba diving", "racing")
- A specific MEDICAL CONDITION

Output ONLY one of:
- "SKIP" - for generic messages
- "excluded countries list" - if message contains a country name
- "excluded activities list" - if message contains an activity
- "excluded medical conditions" - if message contains a medical condition
"""

# 2. Validation Prompt
# Makes final decision based on retrieved documents
VALIDATION_PROMPT = """Check if "{message}" appears in any exclusion list below.

Policy Documents:
{context}

Task: Find if "{message}" is EXPLICITLY WRITTEN in any exclusion list.

Rules:
1. Search for lists of excluded countries/territories/activities in the documents.
2. VIOLATION only if "{message}" (or its common name) is literally in the list.
3. Generic clauses like "sanctions may apply" without listing specific names = CLEAN.
4. If "{message}" is NOT in any list = CLEAN.

Output (one word or line):
- "VIOLATION: [quote the exclusion list containing {message}]"
- "CLEAN"
"""

def _get_embeddings():
    """Get embeddings from local infrastructure (thread-safe singleton)."""
    return get_embeddings()


async def _rag_policy_check(query: str, message: str, product: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    Perform RAG check against Weaviate to find policy violations.
    
    Filters by product when available to get relevant policy documents.
    
    Returns:
        Tuple of (is_violation: bool, rejection_message: str, reason: str)
    """
    try:
        client = get_weaviate_client()
        collection = client.collections.get("Insurance_Knowledge_Base")
        embeddings = _get_embeddings()
        
        if not embeddings:
            logger.error("PolicyRAG: Embeddings not initialized")
            return False, "", ""
            
        emb = embeddings.embed_query(query)
        
        # Filter by product if available
        filters = None
        if product:
            filters = Filter.by_property("product_name").equal(product.lower())
        
        # Log the query being used for retrieval
        logger.info("PolicyRAG.query: '%s' (product_filter=%s)", query, product or "None")
        
        result = collection.query.hybrid(
            query=query,
            vector={
                "content_vector": emb,
                "questions_vector": emb,
            },
            target_vector=TargetVectors.average(["content_vector", "questions_vector"]),
            filters=filters,
            limit=3,
            alpha=0.7,
            return_properties=["content", "product_name"]
        )
        
        objects = getattr(result, "objects", []) or []
        if not objects:
            logger.debug("PolicyRAG: No documents found for query")
            return False, "", "No relevant policy documents found"
        
        # Build context from retrieved docs
        context_str = "\n---\n".join(
            [str(obj.properties.get("content", "") or "") for obj in objects]
        )
        logger.info(
            "PolicyRAG.retrieved: docs=%d context_len=%d preview='%s...'",
            len(objects),
            len(context_str),
            context_str[:150].replace("\n", " ")
        )
        
        # LLM validation based on retrieved docs
        validation_res = await _router_model.ainvoke([
            SystemMessage(content=VALIDATION_PROMPT.format(context=context_str, message=message)),
        ])
        
        decision = str(validation_res.content).strip()
        logger.info("PolicyRAG.decision: %s", decision)
        logger.info("PolicyRAG.full_response: %s", decision.replace("\n", " "))
        
        if decision.upper().startswith("VIOLATION"):
            reason = decision.split(":", 1)[1].strip() if ":" in decision else "Policy Restriction"
            rejection = (
                f"I apologize, but I cannot assist with this request because our policy restricts coverage regarding {reason}. "
                "However, I'd be happy to help you with other destinations or insurance needs! Is there anything else I can do for you?"
            )
            return True, rejection, reason
        
        # For CLEAN decisions, extract any reason if provided
        reason = "No policy violation found"
        if ":" in decision:
            reason = decision.split(":", 1)[1].strip()
            
    except Exception as e:
        error_msg = str(e).lower()
        # Handle Azure Content Filter triggers (fail safe -> violation)
        if "content management policy" in error_msg or "filtered" in error_msg:
            logger.warning("PolicyRAG.content_filter: Azure content filter triggered. Defaulting to VIOLATION.")
            rejection = (
                "I apologize, but I cannot assist with this specific request due to content safety restrictions. "
                "However, I'd be happy to help you with other questions about our insurance products!"
            )
            return True, rejection, "Azure Content Filter Triggered"
            
        logger.error("PolicyRAG.error: %s", e)
        return False, "", f"Error: {e}"
        
    return False, "", reason

async def check_policy(message: str, history_str: str = "", product: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    Policy compliance checker for user messages.
    
    Flow:
    1. Generate a search query (or SKIP for generic messages)
    2. RAG retrieval from Weaviate knowledge base
    3. LLM validation against retrieved policy documents
    
    Returns:
        Tuple of (is_violation: bool, rejection_message: str, reason: str)
    """
    if not message:
        return False, "", ""

    msg_preview = (message or "").replace("\n", " ")[:100]
    logger.info("PolicyCheck.start: product=%s msg='%s'", product, msg_preview)

    try:
        # Stage 1: Decide if check is needed
        query_res = await _router_model.ainvoke([
            SystemMessage(content=QUERY_GEN_PROMPT.format(message=message))
        ])
        check_type = str(query_res.content).strip().lower()
        
        # Skip generic messages that have nothing specific to check
        if "skip" in check_type:
            logger.info("PolicyCheck.skipped: no specific entities to check")
            return False, "", "No specific entities to check"

        logger.info("PolicyCheck.check_type: '%s' for message='%s'", check_type, message[:50])
        
        # Stage 2: RAG check against knowledge base
        # Use the check_type as the search query
        is_violation, reply, reason = await _rag_policy_check(check_type, message, product)
        
        logger.info("PolicyCheck.result: violation=%s reason='%s'", is_violation, reason[:100] if reason else "")
        return is_violation, reply, reason
            
    except Exception as e:
        logger.warning("PolicyCheck.error: %s", e)
        return False, "", f"Error: {e}"


