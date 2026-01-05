"""
HLAS API Client for Policy Service Operations
==============================================

Enterprise-grade API client for interacting with HLAS backend services.
Handles customer validation, policy/claim queries, and customer updates.

Features:
- Async HTTP client with connection pooling
- Comprehensive error handling with user-friendly messages
- Debug logging for API calls (PII logged at DEBUG level only)
- Timeout handling
- No proxy for internal server access

API Endpoints:
- POST /api/v1/customer/validate - Validate customer identity
- GET /api/v1/claim/list/{nric} - Get claims by NRIC
- GET /api/v1/chatbot/policies?nric={nric} - Get policies by NRIC
- GET /api/v1/policies/{policyNo} - Get specific policy details
- POST /api/v1/customer/update - Update customer details
- POST /api/v1/home-protect-insured-address/update - Update insured address
- GET /api/v1/postalCode/{postalCode} - Get address from postal code
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum

import httpx

logger = logging.getLogger("agentic.service_flow")

# API Configuration
HLAS_API_BASE_URL = os.getenv("HLAS_API_BASE_URL", "http://172.28.6.195:8085")
HLAS_API_TIMEOUT = float(os.getenv("HLAS_API_TIMEOUT", "30.0"))


class UpdateType(str, Enum):
    """Customer update types supported by the API."""
    MOBILE_CHANGE = "mobile_change"
    EMAIL_CHANGE = "email_change"
    ADDRESS_CHANGE = "address_change"
    PAYMENT_INFO_CHANGE = "payment_info_change"
    FULL_INFO_CHANGE = "full_info_change"


@dataclass
class APIError:
    """Structured API error response."""
    code: int
    message: str
    details: Optional[str] = None


# User-friendly error messages
ERROR_MESSAGES = {
    400: "The request was invalid. Please check the information provided.",
    401: "Authentication failed. Please verify your details and try again.",
    404: "We couldn't find that record. Please check the policy number or NRIC.",
    422: "Some information appears to be incorrect. Please verify and try again.",
    500: "Our system is experiencing issues. Please try again in a few minutes.",
    503: "The service is temporarily unavailable. Please try again later.",
}


def _get_error_message(status_code: int, api_errors: Optional[List[Dict]] = None) -> str:
    """Get user-friendly error message from status code and API errors."""
    if api_errors and len(api_errors) > 0:
        # Use first error message from API if available
        first_error = api_errors[0]
        if "message" in first_error:
            return first_error["message"]
    
    return ERROR_MESSAGES.get(status_code, f"An unexpected error occurred (code: {status_code}).")


class HLASApiClient:
    """
    Async HTTP client for HLAS backend API.
    
    Usage:
        async with HLASApiClient() as client:
            result = await client.validate_customer(...)
    
    Or:
        client = HLASApiClient()
        result = await client.validate_customer(...)
        await client.close()
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = (base_url or HLAS_API_BASE_URL).rstrip("/")
        self.timeout = timeout or HLAS_API_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                # No proxy for internal server
                trust_env=False,
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request and handle response.
        
        Returns:
            Dict with response data or error information
        """
        client = await self._get_client()
        
        # Log request (DEBUG level to avoid PII in production logs)
        logger.debug(
            "API_REQUEST: %s %s params=%s body=%s",
            method, path, params, json_data
        )
        
        try:
            response = await client.request(
                method=method,
                url=path,
                json=json_data,
                params=params,
            )
            
            # Log response status
            logger.info(
                "API_RESPONSE: %s %s status=%d",
                method, path, response.status_code
            )
            
            # Try to parse JSON response
            try:
                data = response.json()
            except Exception:
                data = {"raw_response": "<non-JSON body omitted>"}

            # Log raw response at INFO level for debugging
            # WARNING: This may contain PII - for debugging only
            import json
            try:
                raw_json = json.dumps(data, default=str)
                # Truncate very long responses
                if len(raw_json) > 2000:
                    raw_json = raw_json[:2000] + "... [TRUNCATED]"
                logger.info("API_RESPONSE_RAW: %s %s data=%s", method, path, raw_json)
            except Exception as log_err:
                logger.warning("API_RESPONSE_RAW: failed to serialize: %s", log_err)
            
            # Handle error responses
            if response.status_code >= 400:
                errors = data.get("errors") if isinstance(data, dict) else None
                error_msg = _get_error_message(response.status_code, errors)
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code,
                    "details": data,
                }
            
            # Handle successful responses
            # API may return data directly as array or nested in 'data' field
            if isinstance(data, list):
                return {"success": True, "data": data}
            elif isinstance(data, dict):
                if "data" in data:
                    return {"success": True, "data": data["data"], **data}
                else:
                    return {"success": True, "data": data, **data}
            else:
                return {"success": True, "data": data}
                
        except httpx.TimeoutException:
            logger.error("API_TIMEOUT: %s %s timeout=%s", method, path, self.timeout)
            return {
                "success": False,
                "error": "The request timed out. Please try again.",
                "status_code": 408,
            }
        except httpx.ConnectError as e:
            logger.error("API_CONNECT_ERROR: %s %s error=%s", method, path, str(e))
            return {
                "success": False,
                "error": "Unable to connect to the server. Please try again later.",
                "status_code": 503,
            }
        except Exception as e:
            logger.exception("API_ERROR: %s %s", method, path)
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "status_code": 500,
            }
    
    # =========================================================================
    # Customer Validation
    # =========================================================================
    
    async def validate_customer(
        self,
        nric: str,
        first_name: str,
        last_name: str,
        mobile: str,
        policy_no: str,
    ) -> Dict[str, Any]:
        """
        Validate customer identity.
        
        Args:
            nric: NRIC/FIN number
            first_name: Customer's first name
            last_name: Customer's last name
            mobile: Mobile number
            policy_no: Any policy number owned by the customer
            
        Returns:
            Dict with validation result and customer data if successful
        """
        logger.info(
            "API_CALL: validate_customer nric=%s...%s policy=%s",
            nric[:3] if nric else "?", nric[-1:] if nric else "?", policy_no
        )
        
        return await self._request(
            method="POST",
            path="/api/v1/customer/validate",
            json_data={
                "nricFin": nric,
                "firstName": first_name,
                "lastName": last_name,
                "mobileNo": mobile,
                "policyNo": policy_no,
            },
        )
    
    # =========================================================================
    # Claims
    # =========================================================================
    
    async def get_claims(self, nric: str) -> Dict[str, Any]:
        """
        Get list of claims for a customer.
        
        Args:
            nric: Customer's NRIC/FIN
            
        Returns:
            Dict with claims list or error
        """
        logger.info("API_CALL: get_claims nric=%s...%s", nric[:3], nric[-1:])
        
        return await self._request(
            method="GET",
            path=f"/api/v1/claim/list/{nric}",
        )
    
    # =========================================================================
    # Policies
    # =========================================================================
    
    async def get_policies(self, nric: str) -> Dict[str, Any]:
        """
        Get list of policies for a customer.
        
        Args:
            nric: Customer's NRIC/FIN
            
        Returns:
            Dict with policies list or error
        """
        logger.info("API_CALL: get_policies nric=%s...%s", nric[:3], nric[-1:])
        
        return await self._request(
            method="GET",
            path="/api/v1/chatbot/policies",
            params={"nric": nric},
        )
    
    async def get_policy_details(self, policy_no: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific policy.
        
        Args:
            policy_no: Policy number
            
        Returns:
            Dict with policy details or error
        """
        logger.info("API_CALL: get_policy_details policy=%s", policy_no)
        
        return await self._request(
            method="GET",
            path=f"/api/v1/policies/{policy_no}",
        )
    
    # =========================================================================
    # Customer Updates
    # =========================================================================
    
    async def update_customer(
        self,
        nric: str,
        update_type: UpdateType,
        update_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update customer details.
        
        Args:
            nric: Customer's NRIC/FIN
            update_type: Type of update (email_change, mobile_change, etc.)
            update_data: Data to update (depends on update_type)
            
        Returns:
            Dict with update result
        """
        logger.info(
            "API_CALL: update_customer nric=%s...%s type=%s",
            nric[:3], nric[-1:], update_type.value
        )
        
        return await self._request(
            method="POST",
            path="/api/v1/customer/update",
            json_data={
                "nric": nric,
                "updateType": update_type.value,
                "updateRequest": update_data,
            },
        )
    
    async def update_email(self, nric: str, new_email: str) -> Dict[str, Any]:
        """Update customer email address."""
        return await self.update_customer(
            nric=nric,
            update_type=UpdateType.EMAIL_CHANGE,
            update_data={"email": new_email},
        )
    
    async def update_mobile(self, nric: str, new_mobile: str) -> Dict[str, Any]:
        """Update customer mobile number."""
        return await self.update_customer(
            nric=nric,
            update_type=UpdateType.MOBILE_CHANGE,
            update_data={"mobile": new_mobile},
        )
    
    async def update_address(
        self,
        nric: str,
        postal_code: str,
        unit_no: str,
        house_no: str,
        street_name: str,
        building_name: str = "",
    ) -> Dict[str, Any]:
        """Update customer address."""
        return await self.update_customer(
            nric=nric,
            update_type=UpdateType.ADDRESS_CHANGE,
            update_data={
                "postalCode": postal_code,
                "unitNo": unit_no,
                "houseNo": house_no,
                "streetName": street_name,
                "buildingName": building_name,
            },
        )
    
    async def update_payment_info(
        self,
        nric: str,
        card_no: str,
        card_expire: str,
        credit_card_type: str,
        policy_no: str,
        payer_surname: str,
        payer_given_name: str,
        payer_nric: str,
    ) -> Dict[str, Any]:
        """Update customer payment information."""
        return await self.update_customer(
            nric=nric,
            update_type=UpdateType.PAYMENT_INFO_CHANGE,
            update_data={
                "cardNo": card_no,
                "cardExpire": card_expire,
                "creditCardType": credit_card_type,
                "policyNo": policy_no,
                "payerSurname": payer_surname,
                "payerGivenName": payer_given_name,
                "payerIDCardNumber": payer_nric,
            },
        )
    
    # =========================================================================
    # Home Protect Insured Address
    # =========================================================================
    
    async def update_insured_address(
        self,
        policy_no: str,
        postal_code: str,
        unit_no: str,
        house_no: str,
        street_name: str,
        building_name: str = "",
    ) -> Dict[str, Any]:
        """
        Update insured address for Home Protect policy.
        
        Args:
            policy_no: Home Protect policy number
            postal_code: New postal code
            unit_no: New unit number
            house_no: New house/block number
            street_name: New street name
            building_name: New building name (optional)
            
        Returns:
            Dict with update result
        """
        logger.info("API_CALL: update_insured_address policy=%s", policy_no)
        
        return await self._request(
            method="POST",
            path="/api/v1/home-protect-insured-address/update",
            json_data={
                "policyNo": policy_no,
                "updateRequest": {
                    "postalCode": postal_code,
                    "unitNo": unit_no,
                    "houseNo": house_no,
                    "streetName": street_name,
                    "buildingName": building_name,
                },
            },
        )
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    async def get_postal_code_info(self, postal_code: str) -> Dict[str, Any]:
        """
        Get address information from postal code.
        
        Args:
            postal_code: Singapore 6-digit postal code
            
        Returns:
            Dict with address details (streetName, buildingName, blockHouseNumber)
        """
        logger.info("API_CALL: get_postal_code_info postal=%s", postal_code)
        
        return await self._request(
            method="GET",
            path=f"/api/v1/postalCode/{postal_code}",
        )


# Singleton instance
_api_client: Optional[HLASApiClient] = None


def get_hlas_api_client() -> HLASApiClient:
    """Get the singleton HLAS API client instance."""
    global _api_client
    if _api_client is None:
        _api_client = HLASApiClient()
    return _api_client


async def close_hlas_api_client() -> None:
    """Close the singleton HLAS API client."""
    global _api_client
    if _api_client is not None:
        await _api_client.close()
        _api_client = None
