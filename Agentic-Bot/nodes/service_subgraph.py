"""
Policy Service Subgraph for HLAS Agentic Chatbot
=================================================

Handles policy/claim status checks and customer updates with secure PII handling.
This subgraph ensures:
- Customer validation before any sensitive operations
- PII never sent to LLM (uses placeholders)
- LLM-based intent detection (not keyword-based)
- Proper error handling with user-friendly messages

Supported Actions:
- claim_status: Check status of claims
- policy_status: Check status of a specific policy
- update_email: Update email address
- update_mobile: Update mobile number
- update_address: Update mailing address
- update_payment: Update payment information
- update_insured_address: Update Home Protect insured address
"""

from __future__ import annotations

import logging
import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Literal, Tuple
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from ..state import AgentState
from ..config import _router_model
from ..utils.pii_masker import get_pii_masker
from ..utils.memory import _get_last_user_message, _build_history_context_from_messages
from ..integrations.hlas_api import get_hlas_api_client, HLASApiClient

logger = logging.getLogger("agentic.service_flow")


# =============================================================================
# PYDANTIC MODELS FOR LLM-BASED CLASSIFICATION
# =============================================================================

class ServiceActionDetection(BaseModel):
    """LLM-based detection of what service action the user wants."""
    
    action: Literal[
        "claim_status",
        "policy_status", 
        "update_email",
        "update_mobile",
        "update_address",
        "update_payment",
        "update_insured_address",
        "unclear",
    ] = Field(
        description=(
            "The service action user wants to perform: "
            "'claim_status' - check claim status, "
            "'policy_status' - check specific policy status, "
            "'update_email' - change email address, "
            "'update_mobile' - change phone number, "
            "'update_address' - change mailing address, "
            "'update_payment' - change payment info, "
            "'update_insured_address' - change Home Protect insured address, "
            "'unclear' - cannot determine what user wants"
        )
    )
    
    policy_no: Optional[str] = Field(
        default=None,
        description="Policy number if user mentioned one (use placeholder like [POLICY_1] if masked)"
    )
    
    reason: str = Field(
        default="",
        description="Brief explanation of why this action was detected"
    )


class CredentialExtraction(BaseModel):
    """Extract validation credentials from user message (using placeholders)."""
    
    nric_placeholder: Optional[str] = Field(
        default=None,
        description="NRIC placeholder like [NRIC_1] if user provided NRIC"
    )
    
    first_name: Optional[str] = Field(
        default=None,
        description="User's first name if provided"
    )
    
    last_name: Optional[str] = Field(
        default=None,
        description="User's last name if provided"
    )
    
    mobile_placeholder: Optional[str] = Field(
        default=None,
        description="Mobile placeholder like [MOBILE_1] if user provided mobile"
    )
    
    policy_placeholder: Optional[str] = Field(
        default=None,
        description="Policy placeholder like [POLICY_1] if user provided policy number"
    )
    
    email_placeholder: Optional[str] = Field(
        default=None,
        description="Email placeholder like [EMAIL_1] if user provided email"
    )
    
    postal_placeholder: Optional[str] = Field(
        default=None,
        description="Postal code placeholder like [POSTAL_1] if user provided postal code"
    )


# Cache structured output models
_action_detector = None
_credential_extractor = None


def _get_action_detector():
    """Get cached action detection model."""
    global _action_detector
    if _action_detector is None:
        _action_detector = _router_model.with_structured_output(ServiceActionDetection)
    return _action_detector


def _get_credential_extractor():
    """Get cached credential extraction model."""
    global _credential_extractor
    if _credential_extractor is None:
        _credential_extractor = _router_model.with_structured_output(CredentialExtraction)
    return _credential_extractor


# =============================================================================
# VALIDATION CREDENTIAL SLOTS
# =============================================================================

VALIDATION_SLOTS = {
    "nric": {
        "question": "Please provide your NRIC/FIN number for verification.",
        "placeholder_prefix": "NRIC",
    },
    "first_name": {
        "question": "What is your first name as registered with us?",
        "placeholder_prefix": None,  # Not masked
    },
    "last_name": {
        "question": "What is your last name as registered with us?",
        "placeholder_prefix": None,  # Not masked
    },
    "mobile": {
        "question": "Please provide your registered mobile number.",
        "placeholder_prefix": "MOBILE",
    },
    "policy_no": {
        "question": "Please provide any policy number you have with us.",
        "placeholder_prefix": "POLICY",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_date(date_str: Optional[str]) -> str:
    """Format ISO date string to user-friendly format."""
    if not date_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y")
    except Exception:
        return date_str.split("T")[0] if "T" in str(date_str) else str(date_str)


def _format_policy_list(policies: List[Dict], max_display: int = 10) -> str:
    """Format policy list for user display."""
    if not policies:
        return "You don't have any policies on record."
    
    # Group by status
    active = [p for p in policies if p.get("status", "").lower() in ("active", "pending new business")]
    lapsed = [p for p in policies if p.get("status", "").lower() == "lapsed"]
    other = [p for p in policies if p not in active and p not in lapsed]
    
    lines = ["Here are your policies:\n"]
    
    def format_policy(p):
        status_emoji = "✅" if p in active else "⏸️" if p in lapsed else "📋"
        end_date = _format_date(p.get("policyEndDate"))
        return f"{status_emoji} *{p.get('policyNo', 'N/A')}* - {p.get('productName', 'Unknown')}\n   Status: {p.get('status', 'Unknown')} | Ends: {end_date}"
    
    if active:
        lines.append("*Active Policies:*")
        for p in active[:max_display]:
            lines.append(format_policy(p))
        lines.append("")
    
    if lapsed and len(active) < max_display:
        remaining = max_display - len(active)
        lines.append("*Lapsed Policies:*")
        for p in lapsed[:remaining]:
            lines.append(format_policy(p))
    
    total = len(policies)
    displayed = min(total, max_display)
    if total > displayed:
        lines.append(f"\n_Showing {displayed} of {total} policies._")
    
    return "\n".join(lines)


def _format_claim_list(claims: List[Dict]) -> str:
    """Format claim list for user display."""
    if not claims:
        return "You don't have any claims on record."
    
    lines = ["Here are your claims:\n"]
    
    for c in claims:
        status = c.get("status", "Unknown")
        status_emoji = "⏳" if status.lower() == "processing" else "✅" if status.lower() == "approved" else "❌" if status.lower() == "rejected" else "📋"
        lines.append(f"{status_emoji} Policy *{c.get('policyNo', 'N/A')}* - Status: {status}")
    
    return "\n".join(lines)


# =============================================================================
# INPUT VALIDATION FUNCTIONS (No PII sent to LLM - all local validation)
# =============================================================================

def _validate_nric(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Singapore NRIC/FIN format.
    Returns (is_valid, error_message).
    Error messages are generic and don't expose the actual value.
    """
    if not value:
        return False, None
    
    value = value.strip().upper()
    
    # Basic format check: S/T/F/G/M + 7 digits + letter
    nric_pattern = r'^[STFGM]\d{7}[A-Z]$'
    if not re.match(nric_pattern, value):
        # Provide helpful feedback without exposing the actual value
        if len(value) < 9:
            return False, "The NRIC/FIN seems too short. It should be 9 characters (e.g., S1234567A)."
        elif len(value) > 9:
            return False, "The NRIC/FIN seems too long. It should be 9 characters (e.g., S1234567A)."
        elif not value[0] in "STFGM":
            return False, "NRIC/FIN should start with S, T, F, G, or M."
        else:
            return False, "Please enter a valid NRIC/FIN in the format S1234567A."
    
    return True, None


def _validate_mobile(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Singapore mobile number format.
    Returns (is_valid, error_message).
    """
    if not value:
        return False, None
    
    # Clean the value
    cleaned = re.sub(r'[^\d+]', '', value)
    
    # Remove +65 prefix if present
    if cleaned.startswith('+65'):
        cleaned = cleaned[3:]
    elif cleaned.startswith('65') and len(cleaned) > 10:
        cleaned = cleaned[2:]
    
    # Should be 8 digits starting with 6, 8, or 9
    if len(cleaned) != 8:
        return False, "Mobile number should be 8 digits (e.g., 91234567 or +65 91234567)."
    
    if cleaned[0] not in '689':
        return False, "Singapore mobile numbers start with 6, 8, or 9."
    
    if not cleaned.isdigit():
        return False, "Mobile number should contain only digits."
    
    return True, None


def _validate_policy_no(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate HLAS policy number format.
    Returns (is_valid, error_message).
    """
    if not value:
        return False, None
    
    value = value.strip().upper()
    
    # HLAS format: 2 letters + 6 digits (e.g., DY300318, HC123456)
    policy_pattern = r'^[A-Z]{2}\d{6}$'
    if not re.match(policy_pattern, value):
        if len(value) < 8:
            return False, "Policy number seems too short. It should be 8 characters (e.g., DY300318)."
        elif len(value) > 8:
            return False, "Policy number seems too long. It should be 8 characters (e.g., DY300318)."
        else:
            return False, "Policy number should be 2 letters followed by 6 digits (e.g., DY300318)."
    
    return True, None


def _validate_name(value: str, field_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate name field.
    Returns (is_valid, error_message).
    """
    if not value:
        return False, None
    
    value = value.strip()
    
    if len(value) < 1:
        return False, f"Please enter your {field_name}."
    
    # Check for obviously invalid characters
    if any(c.isdigit() for c in value):
        return False, f"Your {field_name} should not contain numbers."
    
    # Check for special characters (allow hyphens, apostrophes, spaces for names like O'Brien, Mary-Jane)
    if re.search(r'[^a-zA-Z\s\'\-]', value):
        return False, f"Your {field_name} contains invalid characters."
    
    return True, None


def _get_latest_from_pii_mapping(pii_mapping: Dict[str, str], prefix: str) -> Optional[str]:
    """
    Get the LATEST value from pii_mapping for a given placeholder prefix.
    
    pii_mapping contains entries like [POLICY_1], [POLICY_2], etc.
    This function finds the highest numbered placeholder (most recent user input)
    and returns its value.
    
    Args:
        pii_mapping: Dict mapping placeholders to original values
        prefix: The prefix to look for, e.g., "[POLICY_", "[EMAIL_", "[POSTAL_"
        
    Returns:
        The original value for the highest numbered placeholder, or None if not found
    """
    latest_placeholder = None
    latest_num = -1
    
    for placeholder in pii_mapping.keys():
        if placeholder.startswith(prefix):
            try:
                # Extract number from placeholder like [POLICY_42] -> 42
                num_str = placeholder.replace(prefix, "").replace("]", "")
                num = int(num_str)
                if num > latest_num:
                    latest_num = num
                    latest_placeholder = placeholder
            except ValueError:
                pass
    
    if latest_placeholder:
        return pii_mapping[latest_placeholder]
    return None


# =============================================================================
# SERVICE SUBGRAPH NODES
# =============================================================================

async def _service_detect_action(state: AgentState) -> Dict[str, Any]:
    """
    Detect what service action the user wants using LLM.
    
    This uses LLM-based classification, NOT keyword matching.
    """
    messages = list(state.get("messages", []) or [])
    if not messages:
        return {"service_action": "unclear"}
    
    # Get last user message (already masked)
    user_msg = _get_last_user_message(messages)
    if not user_msg:
        return {"service_action": "unclear"}
    
    # Build context from recent history
    history_ctx = _build_history_context_from_messages(messages, max_pairs=3)
    
    # Check if we already have an action from previous turn
    existing_action = state.get("service_action")
    if existing_action and existing_action != "unclear":
        logger.debug("ServiceFlow.detect_action: using existing action=%s", existing_action)
        return {}  # Keep existing action
    
    # Skip action detection if we're in credential collection phase
    # This prevents the LLM from misinterpreting credential inputs as action requests
    # e.g., user providing mobile number for verification being detected as "update_mobile"
    service_pending_slot = state.get("service_pending_slot")
    credential_slots = {"nric", "first_name", "last_name", "mobile", "policy_no"}
    if service_pending_slot in credential_slots:
        logger.debug(
            "ServiceFlow.detect_action: skip detection during credential collection pending_slot=%s",
            service_pending_slot
        )
        return {}  # Keep existing action, don't re-detect
    
    sys_prompt = """You are detecting what policy service action the user wants to perform.

Based on the conversation, determine the most likely action:
- claim_status: User asking about claim status, where is my claim, claim update
- policy_status: User asking about a specific policy's status or details
- update_email: User wants to change their email address
- update_mobile: User wants to change their phone/mobile number
- update_address: User wants to change their mailing/correspondence address
- update_payment: User wants to update payment/credit card information
- update_insured_address: User wants to change the insured property address (Home Protect only)
- unclear: Cannot determine what user wants

If the user mentioned a policy number (shown as [POLICY_X] placeholder), extract it.

Be liberal in detecting service actions - if user mentions anything about existing policies, claims, or account updates, detect the appropriate action."""

    user_prompt = f"""Recent conversation:
{history_ctx}

Latest message: {user_msg}

What service action does the user want?"""

    try:
        detector = _get_action_detector()
        result = detector.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt),
        ])
        
        logger.info(
            "ServiceFlow.detect_action: action=%s policy=%s reason='%s'",
            result.action, result.policy_no, result.reason[:50] if result.reason else ""
        )
        
        update = {"service_action": result.action}
        if result.policy_no:
            service_slots = dict(state.get("service_slots") or {})
            service_slots["policy_no_placeholder"] = result.policy_no
            update["service_slots"] = service_slots
        
        return update
        
    except Exception as e:
        logger.error("ServiceFlow.detect_action.failed: %s", e)
        return {"service_action": "unclear"}


def _service_check_validated(state: AgentState) -> Literal["validated", "not_validated", "ask_credentials"]:
    """
    Check if customer is validated and route accordingly.
    """
    is_validated = state.get("customer_validated", False)
    service_action = state.get("service_action")
    
    logger.debug(
        "ServiceFlow.check_validated: validated=%s action=%s",
        is_validated, service_action
    )
    
    if is_validated:
        return "validated"
    
    # Check if we have enough credentials to attempt validation
    service_slots = state.get("service_slots") or {}
    has_nric = bool(service_slots.get("nric"))
    has_name = bool(service_slots.get("first_name") and service_slots.get("last_name"))
    has_mobile = bool(service_slots.get("mobile"))
    has_policy = bool(service_slots.get("policy_no"))
    
    if has_nric and has_name and has_mobile and has_policy:
        return "not_validated"  # Have credentials, try to validate
    
    return "ask_credentials"  # Need to collect credentials


async def _service_collect_credentials(state: AgentState) -> Dict[str, Any]:
    """
    Collect validation credentials from user, extracting from PII mapping.
    """
    messages = list(state.get("messages", []) or [])
    pii_mapping = state.get("pii_mapping") or {}
    service_slots = dict(state.get("service_slots") or {})
    
    # Get last user message
    user_msg = _get_last_user_message(messages)

    # Try to extract credentials using LLM
    if user_msg:
        try:
            extractor = _get_credential_extractor()

            sys_prompt = """Extract validation credentials from the user's message.

PII values are masked with placeholders like [NRIC_1], [MOBILE_1], [POLICY_1], [EMAIL_1], [POSTAL_1].
If you see these placeholders, extract them.
Names (first_name, last_name) are NOT masked - extract the actual names."""

            result = extractor.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=f"User message: {user_msg}"),
            ])

            logger.debug(
                "ServiceFlow.extract_credentials: nric=%s mobile=%s policy=%s name=%s %s",
                result.nric_placeholder,
                result.mobile_placeholder,
                result.policy_placeholder,
                result.first_name,
                result.last_name,
            )

            # Map placeholders to real values and store
            if result.nric_placeholder and result.nric_placeholder in pii_mapping:
                service_slots["nric"] = pii_mapping[result.nric_placeholder]

            if result.mobile_placeholder and result.mobile_placeholder in pii_mapping:
                service_slots["mobile"] = pii_mapping[result.mobile_placeholder]

            if result.policy_placeholder and result.policy_placeholder in pii_mapping:
                service_slots["policy_no"] = pii_mapping[result.policy_placeholder]

            if result.email_placeholder and result.email_placeholder in pii_mapping:
                service_slots["email"] = pii_mapping[result.email_placeholder]

            if result.postal_placeholder and result.postal_placeholder in pii_mapping:
                service_slots["postal_code"] = pii_mapping[result.postal_placeholder]

            # Names are not masked
            if result.first_name:
                service_slots["first_name"] = result.first_name
            if result.last_name:
                service_slots["last_name"] = result.last_name

        except Exception as e:
            logger.warning("ServiceFlow.extract_credentials.failed: %s", e)

        # ------------------------------------------------------------------
        # Manual fallback for the currently pending credential slot.
        # This prevents the bot from repeatedly asking the same question
        # when the user provides short answers like initials (e.g. "WL").
        # ------------------------------------------------------------------
        pending_slot = state.get("service_pending_slot")
        validation_error = None  # Track validation errors
        
        if pending_slot and not service_slots.get(pending_slot):
            text = (user_msg or "").strip()

            if pending_slot == "nric":
                # Get the LATEST NRIC from pii_mapping
                nric = _get_latest_from_pii_mapping(pii_mapping, "[NRIC_")
                if nric:
                    # Validate the NRIC before accepting it
                    is_valid, error_msg = _validate_nric(nric)
                    if is_valid:
                        service_slots["nric"] = nric
                        logger.debug("ServiceFlow.fallback: filled nric from pii_mapping")
                    else:
                        validation_error = error_msg
                        logger.debug("ServiceFlow.validation_failed: nric invalid - %s", error_msg)
                
                # If no NRIC found in PII mapping, check if user typed something that looks like an attempt
                if not service_slots.get("nric") and not validation_error and text:
                    # User typed something but it wasn't recognized as NRIC
                    if len(text) >= 5:  # Looks like they tried to enter something
                        validation_error = "That doesn't look like a valid NRIC/FIN. Please enter in format S1234567A."

            elif pending_slot == "mobile":
                # Get the LATEST mobile from pii_mapping
                latest_mobile = _get_latest_from_pii_mapping(pii_mapping, "[MOBILE_")
                if latest_mobile:
                    # Validate the mobile before accepting it
                    is_valid, error_msg = _validate_mobile(latest_mobile)
                    if is_valid:
                        service_slots["mobile"] = latest_mobile
                        logger.debug("ServiceFlow.fallback: filled mobile from latest pii_mapping")
                    else:
                        validation_error = error_msg
                        logger.debug("ServiceFlow.validation_failed: mobile invalid - %s", error_msg)
                
                # If no mobile found, check if user typed something
                if not service_slots.get("mobile") and not validation_error and text:
                    if any(c.isdigit() for c in text):  # Looks like they tried to enter a number
                        validation_error = "Please enter a valid Singapore mobile number (e.g., 91234567 or +65 91234567)."

            elif pending_slot == "policy_no":
                # Get the LATEST policy from pii_mapping
                policy = _get_latest_from_pii_mapping(pii_mapping, "[POLICY_")
                if policy:
                    # Validate the policy number before accepting it
                    is_valid, error_msg = _validate_policy_no(policy)
                    if is_valid:
                        service_slots["policy_no"] = policy
                        logger.debug("ServiceFlow.fallback: filled policy_no from pii_mapping")
                    else:
                        validation_error = error_msg
                        logger.debug("ServiceFlow.validation_failed: policy_no invalid - %s", error_msg)
                
                # If no policy found, check if user typed something
                if not service_slots.get("policy_no") and not validation_error and text:
                    if len(text) >= 4:  # Looks like they tried to enter something
                        validation_error = "That doesn't look like a valid policy number. Please enter in format DY300318 (2 letters + 6 digits)."

            elif pending_slot in ("first_name", "last_name") and text:
                # Validate the name before accepting it
                field_display = "first name" if pending_slot == "first_name" else "last name"
                is_valid, error_msg = _validate_name(text, field_display)
                if is_valid:
                    service_slots[pending_slot] = text
                    logger.debug("ServiceFlow.fallback: filled %s from raw message", pending_slot)
                else:
                    validation_error = error_msg
                    logger.debug("ServiceFlow.validation_failed: %s invalid - %s", pending_slot, error_msg)
    
    # Determine which credential to ask for next
    missing_slots = []
    for slot_name in ["nric", "first_name", "last_name", "mobile", "policy_no"]:
        if not service_slots.get(slot_name):
            missing_slots.append(slot_name)
    
    if missing_slots:
        next_slot = missing_slots[0]
        question = VALIDATION_SLOTS[next_slot]["question"]

        # Create intro message if this is the first credential request
        if len(missing_slots) == 5:  # All slots missing = first time
            intro = "To help you with your request, I'll need to verify your identity first.\n\n"
        else:
            intro = ""

        # If there was a validation error for the current slot, show it
        if validation_error and pending_slot == next_slot:
            full_question = f"⚠️ {validation_error}\n\n{question}"
            logger.info(
                "ServiceFlow.collect_credentials: validation_error for %s, re-asking",
                next_slot
            )
        else:
            full_question = f"{intro}{question}"
            logger.info(
                "ServiceFlow.collect_credentials: asking for %s (have: %s)",
                next_slot,
                [s for s in ["nric", "first_name", "last_name", "mobile", "policy_no"] if service_slots.get(s)],
            )

        return {
            "service_slots": service_slots,
            "service_pending_slot": next_slot,
            "messages": [AIMessage(content=full_question)],
        }
    
    # All credentials collected; clear any pending slot marker.
    return {
        "service_slots": service_slots,
        "service_pending_slot": None,
    }


async def _service_validate_customer(state: AgentState) -> Dict[str, Any]:
    """
    Call the validation API with collected credentials.
    """
    service_slots = state.get("service_slots") or {}
    
    nric = service_slots.get("nric")
    first_name = service_slots.get("first_name")
    last_name = service_slots.get("last_name")
    mobile = service_slots.get("mobile")
    policy_no = service_slots.get("policy_no")
    
    if not all([nric, first_name, last_name, mobile, policy_no]):
        logger.error("ServiceFlow.validate_customer: missing required fields")
        return {
            "messages": [AIMessage(content="I'm missing some information to verify your identity. Let me ask again.")],
            "service_slots": {},  # Clear and start over
            "service_pending_slot": None,
        }
    
    logger.info(
        "ServiceFlow.validate_customer: attempting validation nric=%s...%s policy=%s",
        nric[:3], nric[-1:], policy_no
    )
    
    try:
        client = get_hlas_api_client()
        result = await client.validate_customer(
            nric=nric,
            first_name=first_name,
            last_name=last_name,
            mobile=mobile,
            policy_no=policy_no,
        )

        # Log full API result (structure + values) at INFO level for debugging,
        # but run it through the PII masker so NRIC, emails, mobiles, policy
        # numbers and other identifiers are replaced with placeholders.
        try:
            result_json = json.dumps(result, default=str)
            pii_masker = get_pii_masker()
            masked_json, _ = pii_masker.mask(result_json, session_id="service_api_log")
            logger.info("ServiceFlow.validate_customer.api_raw=%s", masked_json)
        except Exception as log_err:
            logger.warning("ServiceFlow.validate_customer.api_log_failed: %s", log_err)
        
        if result.get("success"):
            customer_data = result.get("data", {})

            # Extract name for greeting
            given_name = customer_data.get("givenName", first_name)

            success_msg = f"Thanks {given_name}! I've verified your identity. How can I help you today?"
            logger.info(
                "ServiceFlow.validate_customer: SUCCESS for %s reply='%s'",
                given_name,
                success_msg.replace("\n", " ")[:200],
            )

            return {
                "customer_validated": True,
                "customer_nric": nric,
                "customer_data": customer_data,
                "service_pending_slot": None,
                "messages": [AIMessage(content=success_msg)],
            }
        else:
            error_msg = result.get("error", "The details provided don't match our records.")
            final_msg = (
                "❌ *Verification Failed*\n\n"
                f"{error_msg}\n\n"
                "Please double-check your details:\n"
                "• NRIC/FIN number\n"
                "• First and last name (as registered)\n"
                "• Mobile number\n"
                "• Policy number\n\n"
                "Would you like to try again? Just type your NRIC to start."
            )

            logger.warning(
                "ServiceFlow.validate_customer: FAILED - %s reply='%s'",
                error_msg,
                final_msg.replace("\n", " ")[:200],
            )

            return {
                "customer_validated": False,
                "service_slots": {},  # Clear credentials for retry
                "service_pending_slot": None,
                # NOTE: Do NOT clear service_action here!
                # We want to preserve the user's original intent (policy_status, claim_status, etc.)
                # so when they retry, we don't re-detect the action and possibly get it wrong.
                "messages": [AIMessage(content=final_msg)],
            }
            
    except Exception as e:
        logger.exception("ServiceFlow.validate_customer: exception")
        return {
            "messages": [AIMessage(content="I encountered an error while verifying your identity. Please try again later.")],
        }


async def _service_execute_action(state: AgentState) -> Dict[str, Any]:
    """
    Execute the service action after customer is validated.
    """
    action = state.get("service_action")
    customer_nric = state.get("customer_nric")
    customer_data = state.get("customer_data") or {}
    service_slots = state.get("service_slots") or {}
    pii_mapping = state.get("pii_mapping") or {}
    
    if not customer_nric:
        logger.error("ServiceFlow.execute_action: no customer_nric")
        return {"messages": [AIMessage(content="Please verify your identity first.")]}
    
    logger.info("ServiceFlow.execute_action: action=%s", action)
    
    client = get_hlas_api_client()
    
    try:
        # =================================================================
        # CLAIM STATUS
        # =================================================================
        if action == "claim_status":
            result = await client.get_claims(customer_nric)
            
            if result.get("success"):
                claims = result.get("data", [])
                response = _format_claim_list(claims)
            else:
                response = f"I couldn't retrieve your claims. {result.get('error', '')}"
            
            # Single-turn action: clear service_action so future messages
            # can trigger a new detection (e.g. switch to policy_status).
            return {
                "messages": [AIMessage(content=response)],
                "service_action": None,
            }
        
        # =================================================================
        # POLICY STATUS
        # =================================================================
        elif action == "policy_status":
            # policy_status is driven off the chatbot policies payload, which
            # has this shape per policy:
            # {"policyNo", "productName", "status", "commencementDate", "policyEndDate", ...}

            policies = customer_data.get("policies", [])
            if not policies:
                # Fallback: fetch via chatbot policies endpoint
                result = await client.get_policies(customer_nric)
                if result.get("success"):
                    policies = result.get("data", [])

            specific_policy: Optional[str] = None

            # Prefer the explicit policy_no we used during validation
            if service_slots.get("policy_no"):
                specific_policy = service_slots["policy_no"]
            else:
                # Fallback to placeholder mapping if available
                policy_placeholder = service_slots.get("policy_no_placeholder")
                if policy_placeholder:
                    specific_policy = pii_mapping.get(policy_placeholder)

            matched_policy = None
            if specific_policy and policies:
                for p in policies:
                    if p.get("policyNo") == specific_policy:
                        matched_policy = p
                        break

            if matched_policy:
                product_name = matched_policy.get("productName") or matched_policy.get("product_name") or "N/A"
                policy_status = matched_policy.get("status") or matched_policy.get("policy_status") or "N/A"

                commencement_raw = matched_policy.get("commencementDate") or matched_policy.get("commencement_date")
                end_raw = matched_policy.get("policyEndDate") or matched_policy.get("policy_end_date")
                commencement = _format_date(commencement_raw) if commencement_raw else "N/A"
                end_date = _format_date(end_raw) if end_raw else "N/A"

                # Status emoji
                status_lower = policy_status.lower()
                if status_lower in ("active", "pending new business"):
                    status_emoji = "✅"
                elif status_lower == "lapsed":
                    status_emoji = "⏸️"
                else:
                    status_emoji = "📋"

                response_lines = [
                    f"📋 *Policy {matched_policy.get('policyNo', specific_policy)}*\n",
                    f"• *Product:* {product_name}",
                    f"• *Status:* {status_emoji} {policy_status}",
                    f"• *Start Date:* {commencement}",
                    f"• *End Date:* {end_date}",
                ]

                # Include any other relevant fields
                excluded_keys = {
                    "policyNo",
                    "productName",
                    "product_name",
                    "status",
                    "policy_status",
                    "commencementDate",
                    "commencement_date",
                    "policyEndDate",
                    "policy_end_date",
                }
                extra_items = {
                    k: v for k, v in matched_policy.items() if k not in excluded_keys and v
                }
                if extra_items:
                    response_lines.append("")
                    for k in sorted(extra_items.keys()):
                        # Format key nicely
                        formatted_key = k.replace("_", " ").title()
                        response_lines.append(f"• *{formatted_key}:* {extra_items[k]}")

                response_lines.append("\nIs there anything else you'd like to know about this policy?")
                response = "\n".join(response_lines)
            else:
                response = (
                    f"I couldn't find details for policy {specific_policy or ''}. "
                    "Please check the policy number and try again."
                )

            # After serving policy_status, clear service_action so the user
            # can ask for a different service action (e.g. claim_status).
            return {
                "messages": [AIMessage(content=response)],
                "service_action": None,
            }
        
        # =================================================================
        # UPDATE EMAIL
        # =================================================================
        elif action == "update_email":
            new_email = None
            
            service_pending_slot = state.get("service_pending_slot")
            
            if service_pending_slot == "new_email":
                # We already asked for the new email – use the LATEST email from pii_mapping
                new_email = _get_latest_from_pii_mapping(pii_mapping, "[EMAIL_")
            # else: First time entering update_email - we MUST ask for the new email.
            
            if not new_email:
                return {
                    "service_pending_slot": "new_email",
                    "messages": [AIMessage(content="What would you like your new email address to be?")],
                }
            
            # Log masked email going to the API for debugging without exposing PII
            try:
                masker = get_pii_masker()
                masked_email, _ = masker.mask(new_email, session_id="service_debug")
                logger.info("ServiceFlow.update_email.request email=%s", masked_email)
            except Exception as log_err:
                logger.warning("ServiceFlow.update_email.log_failed: %s", log_err)

            result = await client.update_email(customer_nric, new_email)
            
            if result.get("success"):
                response = f"✅ Your email has been updated successfully!\n\nIs there anything else I can help you with?"
            else:
                response = f"❌ I couldn't update your email. {result.get('error', '')}\n\nWould you like to try again?"
            
            return {
                "service_action": None,  # Clear action after completion
                "service_pending_slot": None,
                "messages": [AIMessage(content=response)],
            }
        
        # =================================================================
        # UPDATE MOBILE
        # =================================================================
        elif action == "update_mobile":
            new_mobile = None
            
            service_pending_slot = state.get("service_pending_slot")

            if service_pending_slot == "new_mobile":
                # Use the LATEST mobile from pii_mapping
                new_mobile = _get_latest_from_pii_mapping(pii_mapping, "[MOBILE_")

                # Fallback: check if user typed a number directly
                if not new_mobile:
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        text = last_msg.strip()
                        if any(ch.isdigit() for ch in text):
                            new_mobile = text
            # else: First time entering update_mobile - we MUST ask for the new number.
            # We should NOT use the validation mobile. The user needs to provide a NEW mobile.
            
            if not new_mobile:
                return {
                    "service_pending_slot": "new_mobile",
                    "messages": [AIMessage(content="What would you like your new mobile number to be?")],
                }
            
            # Log masked mobile going to the API for debugging without exposing PII
            try:
                masker = get_pii_masker()
                masked_mobile, _ = masker.mask(new_mobile, session_id="service_debug")
                logger.info("ServiceFlow.update_mobile.request mobile=%s", masked_mobile)
            except Exception as log_err:
                logger.warning("ServiceFlow.update_mobile.log_failed: %s", log_err)

            result = await client.update_mobile(customer_nric, new_mobile)
            
            if result.get("success"):
                response = f"✅ Your mobile number has been updated successfully!\n\nIs there anything else I can help you with?"
            else:
                response = f"❌ I couldn't update your mobile number. {result.get('error', '')}\n\nWould you like to try again?"
            
            return {
                "service_action": None,
                "service_pending_slot": None,
                "messages": [AIMessage(content=response)],
            }
        
        # =================================================================
        # UPDATE ADDRESS
        # =================================================================
        elif action == "update_address":
            service_pending_slot = state.get("service_pending_slot")
            
            # Step 1: Collect postal code
            postal_code = service_slots.get("postal_code")
            if not postal_code:
                if service_pending_slot == "postal_code":
                    # Get the LATEST postal code from pii_mapping
                    postal_code = _get_latest_from_pii_mapping(pii_mapping, "[POSTAL_")
            
            if not postal_code:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "postal_code",
                    "messages": [AIMessage(content="What is your new postal code?")],
                }
            
            # Validate postal code using the API (only for validation, not auto-fill)
            if not service_slots.get("postal_validated"):
                postal_result = await client.get_postal_code_info(postal_code)
                if postal_result.get("success"):
                    service_slots["postal_code"] = postal_code
                    service_slots["postal_validated"] = True
                    logger.info("ServiceFlow.update_address: postal code %s is valid", postal_code)
                else:
                    # Postal code validation failed
                    error_msg = postal_result.get("error", "We couldn't validate that postal code.")
                    logger.warning(
                        "ServiceFlow.update_address: postal validation failed code=%s error=%s",
                        postal_code, error_msg
                    )
                    return {
                        "service_slots": service_slots,
                        "service_pending_slot": "postal_code",
                        "messages": [AIMessage(
                            content=f"⚠️ {error_msg}\n\nPlease enter a valid 6-digit Singapore postal code."
                        )],
                    }
            
            # Step 2: Collect block/house number
            house_no = service_slots.get("house_no")
            if not house_no:
                if service_pending_slot == "house_no":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        house_no = last_msg.strip()
                        service_slots["house_no"] = house_no
            
            if not house_no:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "house_no",
                    "messages": [AIMessage(content="What is your block or house number? (e.g., BLK 123 or 45)")],
                }
            
            # Step 3: Collect street name
            street_name = service_slots.get("street_name")
            if not street_name:
                if service_pending_slot == "street_name":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        street_name = last_msg.strip()
                        service_slots["street_name"] = street_name
            
            if not street_name:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "street_name",
                    "messages": [AIMessage(content="What is your street name?")],
                }
            
            # Step 4: Collect unit number
            unit_no = service_slots.get("unit_no")
            if not unit_no:
                if service_pending_slot == "unit_no":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        unit_no = last_msg.strip()
                        service_slots["unit_no"] = unit_no
            
            if not unit_no:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "unit_no",
                    "messages": [AIMessage(content="What is your unit number? (e.g., #10-10)")],
                }
            
            # Step 5: Building name is optional - ask if not provided
            building_name = service_slots.get("building_name", "")
            
            # All required fields collected, call the update API
            result = await client.update_address(
                nric=customer_nric,
                postal_code=postal_code,
                unit_no=unit_no,
                house_no=house_no,
                street_name=street_name,
                building_name=building_name,
            )
            
            if result.get("success"):
                response = "✅ Your address has been updated successfully!\n\nIs there anything else I can help you with?"
            else:
                response = f"❌ I couldn't update your address. {result.get('error', '')}\n\nWould you like to try again?"
            
            return {
                "service_action": None,
                "service_slots": {},
                "service_pending_slot": None,
                "messages": [AIMessage(content=response)],
            }
        
        # =================================================================
        # UPDATE INSURED ADDRESS (Home Protect)
        # =================================================================
        elif action == "update_insured_address":
            service_pending_slot = state.get("service_pending_slot")
            
            # Step 1: Select which Home Protect policy to update
            policy_no = service_slots.get("insured_policy_no")
            if not policy_no:
                if service_pending_slot == "insured_policy_no":
                    # Get the LATEST policy number from pii_mapping
                    policy_no = _get_latest_from_pii_mapping(pii_mapping, "[POLICY_")
                    if policy_no:
                        service_slots["insured_policy_no"] = policy_no
            
            if not policy_no:
                # Find Home Protect policies from customer data
                policies = customer_data.get("policies", [])
                home_policies = [p for p in policies if p.get("productName", "").lower().startswith("home")]
                
                if not home_policies:
                    return {
                        "messages": [AIMessage(content="I couldn't find any Home Protect policies on your account. Insured address updates are only available for Home Protect policies.")],
                        "service_action": None,
                    }
                
                # Always show list for user to choose
                policy_list = "\n".join([
                    f"• {p.get('policyNo', 'N/A')} - {p.get('productName', 'N/A')}"
                    for p in home_policies
                ])
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "insured_policy_no",
                    "messages": [AIMessage(
                        content=f"Which Home Protect policy would you like to update the insured address for?\n\n{policy_list}\n\nPlease enter the policy number."
                    )],
                }
            
            # Verify it's a Home Protect policy
            if not policy_no.upper().startswith("HC") and not policy_no.upper().startswith("HP") and not policy_no.upper().startswith("HA"):
                return {
                    "messages": [AIMessage(content="Insured address updates are only available for Home Protect policies.")],
                    "service_action": None,
                    "service_slots": {},
                }
            
            # Step 2: Collect postal code
            postal_code = service_slots.get("postal_code")
            if not postal_code:
                if service_pending_slot == "postal_code":
                    # Get the LATEST postal code from pii_mapping
                    postal_code = _get_latest_from_pii_mapping(pii_mapping, "[POSTAL_")
            
            if not postal_code:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "postal_code",
                    "messages": [AIMessage(content="What is the new postal code for the insured property?")],
                }
            
            # TODO: Re-enable postal code validation when ready
            # Skipping postal code validation for now - just store the value
            service_slots["postal_code"] = postal_code
            # # Validate postal code using the API
            # if not service_slots.get("postal_validated"):
            #     postal_result = await client.get_postal_code_info(postal_code)
            #     if postal_result.get("success"):
            #         service_slots["postal_code"] = postal_code
            #         service_slots["postal_validated"] = True
            #         logger.info("ServiceFlow.update_insured_address: postal code %s is valid", postal_code)
            #     else:
            #         error_msg = postal_result.get("error", "We couldn't validate that postal code.")
            #         logger.warning("ServiceFlow.update_insured_address: postal validation failed code=%s", postal_code)
            #         return {
            #             "service_slots": service_slots,
            #             "service_pending_slot": "postal_code",
            #             "messages": [AIMessage(content=f"⚠️ {error_msg}\n\nPlease enter a valid 6-digit Singapore postal code.")],
            #         }
            
            # Step 3: Collect block/house number
            house_no = service_slots.get("house_no")
            if not house_no:
                if service_pending_slot == "house_no":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        house_no = last_msg.strip()
                        service_slots["house_no"] = house_no
            
            if not house_no:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "house_no",
                    "messages": [AIMessage(content="What is the block or house number? (e.g., BLK 123 or 45)")],
                }
            
            # Step 4: Collect street name
            street_name = service_slots.get("street_name")
            if not street_name:
                if service_pending_slot == "street_name":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        street_name = last_msg.strip()
                        service_slots["street_name"] = street_name
            
            if not street_name:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "street_name",
                    "messages": [AIMessage(content="What is the street name?")],
                }
            
            # Step 5: Collect unit number
            unit_no = service_slots.get("unit_no")
            if not unit_no:
                if service_pending_slot == "unit_no":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        unit_no = last_msg.strip()
                        service_slots["unit_no"] = unit_no
            
            if not unit_no:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "unit_no",
                    "messages": [AIMessage(content="What is the unit number? (e.g., #10-10)")],
                }
            
            # Building name is optional
            building_name = service_slots.get("building_name", "")
            
            # Log the request
            logger.info(
                "ServiceFlow.update_insured_address: calling API policy=%s postal=%s",
                policy_no, postal_code
            )
            
            result = await client.update_insured_address(
                policy_no=policy_no,
                postal_code=postal_code,
                unit_no=unit_no,
                house_no=house_no,
                street_name=street_name,
                building_name=building_name,
            )
            
            if result.get("success"):
                response = f"✅ The insured address for policy {policy_no} has been updated successfully!\n\nIs there anything else I can help you with?"
            else:
                response = f"❌ I couldn't update the insured address. {result.get('error', '')}\n\nWould you like to try again?"
            
            return {
                "service_action": None,
                "service_slots": {},
                "service_pending_slot": None,
                "messages": [AIMessage(content=response)],
            }
        
        # =================================================================
        # UPDATE PAYMENT
        # =================================================================
        elif action == "update_payment":
            service_pending_slot = state.get("service_pending_slot")
            
            # Step 1: Select which policy to update payment for
            policy_no = service_slots.get("payment_policy_no")
            if not policy_no:
                if service_pending_slot == "payment_policy_no":
                    # Get the LATEST policy number from pii_mapping
                    policy_no = _get_latest_from_pii_mapping(pii_mapping, "[POLICY_")
                    if policy_no:
                        service_slots["payment_policy_no"] = policy_no
            
            if not policy_no:
                # Show list of policies for user to choose
                policies = customer_data.get("policies", [])
                if policies:
                    policy_list = "\n".join([
                        f"• {p.get('policyNo', 'N/A')} - {p.get('productName', 'N/A')}"
                        for p in policies[:10]  # Limit to 10
                    ])
                    return {
                        "service_slots": service_slots,
                        "service_pending_slot": "payment_policy_no",
                        "messages": [AIMessage(
                            content=f"Which policy would you like to update payment for?\n\n{policy_list}\n\nPlease enter the policy number."
                        )],
                    }
                else:
                    return {
                        "service_slots": service_slots,
                        "service_pending_slot": "payment_policy_no",
                        "messages": [AIMessage(content="Please enter the policy number you want to update payment for.")],
                    }
            
            # Step 2: Collect card number
            card_no = service_slots.get("card_no")
            if not card_no:
                if service_pending_slot == "card_no":
                    # Get the LATEST card number from pii_mapping
                    card_no = _get_latest_from_pii_mapping(pii_mapping, "[CARD_")
                    if card_no:
                        card_no = card_no.replace(" ", "").replace("-", "")
                        service_slots["card_no"] = card_no
            
            if not card_no:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "card_no",
                    "messages": [AIMessage(content="Please enter your new credit/debit card number.")],
                }
            
            # Step 3: Collect card expiry date
            card_expiry = service_slots.get("card_expiry")
            if not card_expiry:
                if service_pending_slot == "card_expiry":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        card_expiry = last_msg.strip()
                        service_slots["card_expiry"] = card_expiry
            
            if not card_expiry:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "card_expiry",
                    "messages": [AIMessage(content="What is the card expiry date? (e.g., 12/2028 or 01/10/2029)")],
                }
            
            # Step 4: Collect card type
            card_type = service_slots.get("card_type")
            if not card_type:
                if service_pending_slot == "card_type":
                    last_msg = _get_last_user_message(state.get("messages", []) or [])
                    if last_msg:
                        card_type = last_msg.strip().upper()
                        # Normalize card type
                        if "VISA" in card_type:
                            card_type = "VISA"
                        elif "MASTER" in card_type:
                            card_type = "MASTERCARD"
                        elif "AMEX" in card_type or "AMERICAN" in card_type:
                            card_type = "AMEX"
                        service_slots["card_type"] = card_type
            
            if not card_type:
                return {
                    "service_slots": service_slots,
                    "service_pending_slot": "card_type",
                    "messages": [AIMessage(content="What type of card is this? (VISA, Mastercard, or AMEX)")],
                }
            
            # Get payer details from customer data (already validated)
            payer_surname = customer_data.get("surname", "")
            payer_given_name = customer_data.get("givenName", "")
            payer_nric = customer_nric
            
            # Log all parameters before API call
            logger.info(
                "ServiceFlow.update_payment: calling API policy=%s card_type=%s expiry=%s payer=%s %s",
                policy_no, card_type, card_expiry, payer_given_name, payer_surname
            )
            
            # Call the API to update payment info
            result = await client.update_payment_info(
                nric=customer_nric,
                card_no=card_no,
                card_expire=card_expiry,
                credit_card_type=card_type,
                policy_no=policy_no,
                payer_surname=payer_surname,
                payer_given_name=payer_given_name,
                payer_nric=payer_nric,
            )
            
            if result.get("success"):
                # Mask card number for display (show last 4 digits only)
                masked_card = f"****{card_no[-4:]}" if len(card_no) >= 4 else "****"
                response = (
                    f"✅ Your payment information has been updated successfully!\n\n"
                    f"• Policy: {policy_no}\n"
                    f"• Card: {masked_card} ({card_type})\n"
                    f"• Expiry: {card_expiry}\n\n"
                    f"Is there anything else I can help you with?"
                )
            else:
                response = f"❌ I couldn't update your payment information. {result.get('error', '')}\n\nWould you like to try again?"
            
            return {
                "service_action": None,
                "service_slots": {},
                "service_pending_slot": None,
                "messages": [AIMessage(content=response)],
            }
        
        # =================================================================
        # UNCLEAR ACTION
        # =================================================================
        else:
            return {
                "messages": [AIMessage(
                    content="I'm not sure what you'd like to do. I can help you with:\n\n"
                    "• Check claim status\n"
                    "• Check policy status\n"
                    "• Update your email address\n"
                    "• Update your mobile number\n"
                    "• Update your mailing address\n"
                    "• Update payment information\n"
                    "• Update insured address (Home Protect)\n\n"
                    "What would you like to do?"
                )],
            }
            
    except Exception as e:
        logger.exception("ServiceFlow.execute_action: exception for action=%s", action)
        return {
            "messages": [AIMessage(content=f"I encountered an error while processing your request. Please try again later.")],
        }


# =============================================================================
# SUBGRAPH CONSTRUCTION
# =============================================================================

def _route_after_validation_check(state: AgentState) -> str:
    """Route based on validation check result."""
    # _service_check_validated returns one of:
    #   "validated", "not_validated", "ask_credentials"
    # These values are used directly as keys in the
    # add_conditional_edges mapping for "detect_action".
    return _service_check_validated(state)


def _route_after_validation(state: AgentState) -> str:
    """Route after validation attempt."""
    if state.get("customer_validated"):
        return "execute_action"
    else:
        return "end"  # Return message is already set


def _route_after_credentials(state: AgentState) -> str:
    """Route after credential collection."""
    service_slots = state.get("service_slots") or {}
    
    # Check if we have all required credentials
    required = ["nric", "first_name", "last_name", "mobile", "policy_no"]
    if all(service_slots.get(s) for s in required):
        return "validate_customer"
    else:
        return "end"  # Still collecting, return question message


# Build the subgraph
_service_builder = StateGraph(AgentState)

# Add nodes
_service_builder.add_node("detect_action", _service_detect_action)
_service_builder.add_node("collect_credentials", _service_collect_credentials)
_service_builder.add_node("validate_customer", _service_validate_customer)
_service_builder.add_node("execute_action", _service_execute_action)

# Entry point - detect action first
_service_builder.set_entry_point("detect_action")

# After action detection, check validation status
_service_builder.add_conditional_edges(
    "detect_action",
    _route_after_validation_check,
    {
        "validated": "execute_action",
        "not_validated": "validate_customer",
        "ask_credentials": "collect_credentials",
    }
)

# After credential collection
_service_builder.add_conditional_edges(
    "collect_credentials",
    _route_after_credentials,
    {
        "validate_customer": "validate_customer",
        "end": END,
    }
)

# After validation attempt
_service_builder.add_conditional_edges(
    "validate_customer",
    _route_after_validation,
    {
        "execute_action": "execute_action",
        "end": END,
    }
)

# Execute action always ends
_service_builder.add_edge("execute_action", END)

# Compile the subgraph
service_subgraph = _service_builder.compile()

logger.info("ServiceSubgraph: compiled with 4 nodes")
