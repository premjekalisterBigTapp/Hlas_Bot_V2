"""
V3 Testing with Detailed Logs

Tests the key V3 features:
1. Product switching prevention (LLM-driven)
2. Session restart (intelligent)
3. Natural language understanding
"""

import requests
import json
import time
import uuid

BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BASE_URL}/chat"

def print_separator(title=""):
    print("\n" + "="*80)
    if title:
        print(f" {title}")
        print("="*80)

def print_log(label, content, color=""):
    """Print formatted log"""
    print(f"\n[{label}]")
    print(content)

def send_message(session_id, message):
    """Send message and return response with detailed logging"""
    print_log("USER", message)
    
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={
                "session_id": session_id,
                "message": message,
                "user_id": "test-user-v3"
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print_log("ERROR", f"HTTP {response.status_code}")
            return None
        
        data = response.json()
        bot_response = data.get('response', 'NO RESPONSE')
        
        print_log("BOT", bot_response)
        
        # Print debug info
        debug = data.get('debug', {})
        if debug:
            debug_info = []
            debug_info.append(f"Intent: {debug.get('intent', 'unknown')}")
            debug_info.append(f"Product: {debug.get('product', 'none')}")
            debug_info.append(f"Phase: {debug.get('phase', 'unknown')}")
            
            if debug.get('product_switch_attempted'):
                debug_info.append(f"[!] Product Switch Attempted: {debug['product_switch_attempted']}")
            
            if debug.get('reset'):
                debug_info.append(f"[!] Reset Flag: {debug['reset']}")
            
            if debug.get('pending_slot'):
                debug_info.append(f"Pending Slot: {debug['pending_slot']}")
            
            if debug.get('slots'):
                debug_info.append(f"Slots: {debug['slots']}")
            
            print_log("DEBUG", " | ".join(debug_info))
        
        return data
        
    except Exception as e:
        print_log("EXCEPTION", str(e))
        return None

def test_product_switching():
    """Test 1: Product Switching Prevention"""
    print_separator("TEST 1: PRODUCT SWITCHING PREVENTION")
    
    session_id = str(uuid.uuid4())
    
    print("\n>>> Scenario: User starts with Travel, then tries to switch to Car")
    
    # Start with Travel
    print("\n--- Step 1: Start Travel Insurance ---")
    send_message(session_id, "I want travel insurance")
    time.sleep(1)
    
    # Try to switch to Car
    print("\n--- Step 2: Try to Switch to Car ---")
    resp = send_message(session_id, "Actually, I want car insurance")
    time.sleep(1)
    
    # Check result
    if resp:
        bot_response = resp.get('response', '').lower()
        if 'sorry' in bot_response and 'cannot switch' in bot_response:
            print_log("RESULT", "[PASS] Product switching was REJECTED as expected")
        else:
            print_log("RESULT", "[FAIL] Product switching was NOT rejected properly")
    
    return session_id

def test_session_restart(session_id):
    """Test 2: Session Restart (Intelligent)"""
    print_separator("TEST 2: SESSION RESTART (INTELLIGENT)")
    
    print("\n>>> Scenario: User restarts session with natural language")
    
    # Try restart with "Restart session"
    print("\n--- Step 1: Restart with 'Restart session' ---")
    resp1 = send_message(session_id, "Restart session")
    time.sleep(1)
    
    # Check if restarted
    if resp1:
        bot_response = resp1.get('response', '').lower()
        if 'start fresh' in bot_response or 'how can i help' in bot_response:
            print_log("RESULT", "[PASS] Session restart detected with 'Restart session'")
        else:
            print_log("RESULT", "[FAIL] Session restart NOT detected")
    
    # Try another variation
    print("\n--- Step 2: Restart with 'Let's start fresh' ---")
    new_session = str(uuid.uuid4())
    send_message(new_session, "I want travel insurance")
    time.sleep(1)
    
    resp2 = send_message(new_session, "Let's start fresh")
    time.sleep(1)
    
    if resp2:
        bot_response = resp2.get('response', '').lower()
        if 'start fresh' in bot_response or 'how can i help' in bot_response:
            print_log("RESULT", "[PASS] Session restart detected with 'Let's start fresh'")
        else:
            print_log("RESULT", "[FAIL] Session restart NOT detected with variation")

def test_natural_language():
    """Test 3: Natural Language Understanding"""
    print_separator("TEST 3: NATURAL LANGUAGE UNDERSTANDING")
    
    session_id = str(uuid.uuid4())
    
    print("\n>>> Scenario: User provides vague phrases for slots")
    
    # Start travel insurance
    print("\n--- Step 1: Start Travel Insurance ---")
    send_message(session_id, "I want travel insurance")
    time.sleep(1)
    
    # Provide "Just me" for coverage scope
    print("\n--- Step 2: Say 'Just me' for coverage scope ---")
    resp1 = send_message(session_id, "Just me")
    time.sleep(1)
    
    if resp1:
        bot_response = resp1.get('response', '').lower()
        if 'destination' in bot_response or 'where' in bot_response or 'traveling' in bot_response:
            print_log("RESULT", "[PASS] 'Just me' understood as coverage_scope=self")
        else:
            print_log("RESULT", "[FAIL] 'Just me' NOT understood properly")
    
    # Provide destination
    print("\n--- Step 3: Provide destination ---")
    send_message(session_id, "Japan")
    time.sleep(1)
    
    # Provide "About a week" for duration
    print("\n--- Step 4: Say 'About a week' for duration ---")
    resp2 = send_message(session_id, "About a week")
    time.sleep(1)
    
    if resp2:
        bot_response = resp2.get('response', '').lower()
        if 'recommend' in bot_response or 'suggest' in bot_response or 'plan' in bot_response:
            print_log("RESULT", "[PASS] 'About a week' understood and recommendation provided")
        else:
            print_log("RESULT", "[PARTIAL] 'About a week' may need more processing")

def test_comparison_intent():
    """Test 4: Comparison Intent Detection"""
    print_separator("TEST 4: COMPARISON INTENT DETECTION")
    
    session_id = str(uuid.uuid4())
    
    print("\n>>> Scenario: User asks for comparison")
    
    # Start with product
    print("\n--- Step 1: Start Travel Insurance ---")
    send_message(session_id, "I want travel insurance")
    time.sleep(1)
    
    # Ask for comparison
    print("\n--- Step 2: Ask for comparison ---")
    resp = send_message(session_id, "Compare the Gold and Silver plans")
    time.sleep(1)
    
    if resp:
        debug = resp.get('debug', {})
        intent = debug.get('intent', '').lower()
        bot_response = resp.get('response', '').lower()
        
        if intent == 'compare' or ('gold' in bot_response and 'silver' in bot_response):
            print_log("RESULT", "[PASS] Comparison intent detected")
        else:
            print_log("RESULT", "[FAIL] Comparison intent NOT detected")

def test_complete_flow():
    """Test 5: Complete Flow - Product Switch + Restart + New Product"""
    print_separator("TEST 5: COMPLETE FLOW")
    
    session_id = str(uuid.uuid4())
    
    print("\n>>> Scenario: Switch attempt -> Restart -> New product")
    
    # Start with Travel
    print("\n--- Step 1: Start Travel Insurance ---")
    send_message(session_id, "I want travel insurance")
    time.sleep(1)
    
    # Try to switch
    print("\n--- Step 2: Try to switch to Car ---")
    send_message(session_id, "I want car insurance")
    time.sleep(1)
    
    # Restart
    print("\n--- Step 3: Restart session ---")
    send_message(session_id, "Restart session")
    time.sleep(1)
    
    # Ask for Car
    print("\n--- Step 4: Now ask for Car insurance ---")
    resp = send_message(session_id, "I want car insurance")
    time.sleep(1)
    
    if resp:
        debug = resp.get('debug', {})
        product = debug.get('product', '').lower()
        
        if 'car' in product:
            print_log("RESULT", "[PASS] After restart, Car insurance started successfully")
        else:
            print_log("RESULT", "[FAIL] Car insurance NOT started after restart")

def main():
    """Run all tests"""
    print_separator("V3 TESTING WITH DETAILED LOGS")
    print("\nTesting LLM-Driven Architecture Changes")
    print("- Product Switching Prevention")
    print("- Intelligent Session Restart")
    print("- Natural Language Understanding")
    print("- Comparison Intent Detection")
    
    # Check if chatbot is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print_log("STATUS", "Chatbot is running on http://localhost:8000")
    except:
        print_log("ERROR", "Chatbot is NOT running. Please start it first.")
        print("\nTo start chatbot:")
        print("  cd Agentic-Bot")
        print("  python main.py")
        return
    
    print("\nStarting tests in 2 seconds...")
    time.sleep(2)
    
    # Run tests
    session_id = test_product_switching()
    time.sleep(2)
    
    test_session_restart(session_id)
    time.sleep(2)
    
    test_natural_language()
    time.sleep(2)
    
    test_comparison_intent()
    time.sleep(2)
    
    test_complete_flow()
    
    print_separator("TESTING COMPLETE")
    print("\nAll tests finished!")
    print("\nKey Points to Verify:")
    print("1. Product switching should be REJECTED with apologetic message")
    print("2. 'Restart session' and variations should clear state")
    print("3. 'Just me' should be understood as coverage_scope=self")
    print("4. 'About a week' should be understood as duration")
    print("5. Comparison requests should be detected by LLM")
    print("\nCheck the logs above for [PASS] or [FAIL] markers")

if __name__ == "__main__":
    main()

