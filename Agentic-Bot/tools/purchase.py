"""
Purchase link generation tool.

This tool:
- Returns purchase links for products
- Uses product-specific link configuration
- Includes comprehensive logging and metrics
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from ..config import _load_purchase_links
from ..utils.slots import _normalize_product_key
from ..infrastructure.metrics import PURCHASE_LINK_GENERATED_TOTAL

logger = logging.getLogger(__name__)


# Friendly product names for user-facing messages
FRIENDLY_NAMES = {
    "travel": "Travel",
    "maid": "Maid",
    "car": "Car",
    "personalaccident": "Personal Accident",
    "home": "Home",
    "early": "Early Critical Illness",
    "fraud": "Fraud Protect360",
    "hospital": "Hospital Protect360",
}


def _purchase_tool(product: Optional[str]) -> str:
    """
    Purchase tool: returns purchase link or friendly fallback.
    
    Args:
        product: Product name
        
    Returns:
        Purchase link message or fallback message
    """
    start_time = time.time()
    
    raw_product = product
    prod = _normalize_product_key(product)
    
    logger.info(
        "Tool.purchase.start: raw_product=%s resolved_product=%s",
        raw_product, prod
    )
    
    if not prod:
        logger.warning("Tool.purchase.no_product: could not determine product")
        return (
            "Which product would you like to buy? Available options: Travel Protect360, Maid Protect360, Car Protect360, Personal Accident Protect360, "
            "Home Protect360, Fraud Protect360, Early Critical Illness Protect360, Hospital Cash Protect360."
        )

    # Load purchase links
    links = _load_purchase_links()
    link = links.get(prod)
    friendly = FRIENDLY_NAMES.get(prod, product or "this")
    
    if link:
        # Record metric
        try:
            PURCHASE_LINK_GENERATED_TOTAL.labels(product=prod).inc()
        except Exception:
            pass
        
        duration = time.time() - start_time
        logger.info(
            "Tool.purchase.completed: product=%s has_link=True duration=%.3fs",
            prod, duration
        )
        
        return (
            f"Great! You can visit this link and enter the details to get your quote for {friendly} insurance: {link}\n\n"
            "Make sure to select your preferred plan and add-ons!"
        )
    
    duration = time.time() - start_time
    logger.warning(
        "Tool.purchase.no_link: product=%s duration=%.3fs",
        prod, duration
    )
    
    return (
        f"I don't have a direct purchase link for the {friendly} plan right now. "
        "Please let me know if you'd like me to connect you with a specialist."
    )
