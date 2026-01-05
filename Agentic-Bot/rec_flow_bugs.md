# Recommendation Flow Bug Audit

## Flow Snapshot
- **Entry:** `supervisor` routes intent=`recommend` into `nodes/rec_subgraph.recommendation_subgraph` once the classifier believes the user needs a personalised plan.
- **Product gate:** `_rec_ensure_product` tries to detect a product from the latest user turn and either keeps the existing `state.product` or re-prompts.
- **Slot turn:** `_rec_extract_slots` runs the router LLM to fill the product-specific required slots, optionally spawning side-info lookups when the user asks clarifying questions.
- **Parallel branch:** `validate_slots` (currently a stub) and `side_info` are meant to run in parallel before `_rec_resolve_step` decides between `_rec_ask_next_slot` and `_rec_generate_recommendation`.
- **Question stage:** `_rec_ask_next_slot` uses either static questions from `utils/products.py` or a free-form LLM prompt to ask for the next missing slot and flags `pending_slot` for the rest of the system.
- **Recommendation stage:** `_rec_generate_recommendation` calls `tools/recommendation._generate_recommendation_text`, which picks a tier via `_select_tier` and feeds `configs/recommendation_response.yaml` plus `configs/benefits_raw.json` into the response LLM; the result is sent directly to the user because `styler` is skipped for intent=`recommend` with `rec_given=True`.

## Detailed Findings (No fixes included)

### 1. Validation pipeline is effectively disabled
- **Files:** `nodes/rec_subgraph.py` (`_rec_validate_slots`), `configs/slot_validation_rules.yaml`.
- **What happens:** `_rec_validate_slots` always returns `{}` and never loads `slot_validation_rules.yaml`, even though the comments state that a full implementation should enforce min/max ranges, enumerations, and “do not auto-correct” guidance.
- **Impact:** Obvious invalid answers (e.g., "260 months" for maid duration, "36" for daily hospital cash) are accepted silently, and the bot never tells users *why* their answer is unusable. When the user keeps guessing wrong values, the system just keeps re-asking the same question without new guidance, which looks like the bot is stuck.

### 2. Destination normalization contradicts compliance instructions
- **Files:** `nodes/rec_subgraph.py` (slot extraction system prompt), `configs/slot_validation_rules.yaml` (`travel.destination`).
- **What happens:** The extraction prompt explicitly says “normalize city names to the country name,” while the validation rules demand the opposite (“Do NOT infer the country from city/place names. Ask explicitly.”). Because validation never runs, the chatbot currently converts cities to countries on its own.
- **Impact:** Travel recommendations can be based on the wrong jurisdiction without ever confirming with the user, which is the exact scenario the validation rules were written to prevent. This also removes any opportunity to educate the user about why the country is required.

### 3. Side questions are dropped when slot collection finishes
- **Files:** `nodes/rec_subgraph.py` (`_rec_side_info`, `_rec_ask_next_slot`, `_rec_generate_recommendation`).
- **What happens:** The only place that sends `state["side_info"]` back to the user is `_rec_ask_next_slot`, where the explanation is prepended to the next question. If the user asks “What does coverage_scope mean?” while answering the *final* required slot, `_rec_generate_recommendation` runs immediately and `side_info` never gets surfaced, so the question simply disappears.
- **Impact:** Users who ask legitimate clarifications on the last slot never get an answer and instead receive a recommendation, which looks evasive and undermines trust.

### 4. The intended join between `validate_slots` and `side_info` never happens
- **Files:** `nodes/rec_subgraph.py` (comments around the `resolve_node` wiring).
- **What happens:** Both `validate_slots` and `side_info` point to the same `resolve_node`, but LangGraph executes that node once per incoming edge; there is no actual synchronization. As soon as `validate_slots` completes, `_rec_resolve_step` fires and routes to `_rec_ask_next_slot` or `_rec_generate_recommendation` even if the side-info branch is still running.
- **Impact:** Slot questions or recommendations can be emitted before the side-info lookup returns, so the explanation meant to precede the next question arrives one turn late (or triggers a duplicate question when the second invocation reaches `resolve_node`).

### 5. No strategy for repeated invalid answers
- **Files:** `nodes/rec_subgraph.py` (`_rec_ask_next_slot`).
- **What happens:** The only enhanced re-ask text is for `destination` and `maid_country`. Every other slot just repeats the same question verbatim regardless of how many times the user gave an off-topic or ambiguous answer, and there is no attempt counter, no fallback multiple-choice, and no guidance pulled from `slot_validation_rules.yaml`.
- **Impact:** When a user keeps supplying the wrong type of answer (e.g., "maybe" for `coverage_amount`), the bot just loops “Could you please provide…” forever. This is exactly the “I don't understand…” stagnation the user wanted to avoid.

### 6. Supervisor fast-path traps short intent changes
- **Files:** `nodes/supervisor.py` (fast-path logic around lines 223‑274).
- **What happens:** Any ≤3-word message while `pending_slot` is non-null is forced back into the recommendation subgraph unless it contains one of a small set of keywords. Common pivots like “new plan”, “different product pls”, or “reset” (without the literal word “reset”) are misclassified as slot answers.
- **Impact:** Users trying to change direction with a short utterance get dragged back into slot filling and may never exit the form unless they phrase the request in a longer sentence.

### 7. Product switches leave stale slot metadata
- **Files:** `nodes/supervisor.py` (product switch handling), `state.py` (`pending_slot` usage).
- **What happens:** When the supervisor clears `slots` because it detected a new product or a reset, it does **not** clear `pending_slot`, `pending_side_question`, or `side_info`. The next time `rec_subgraph` runs, it still thinks the old slot is pending and builds prompts around it.
- **Impact:** The first answer for the new product can be misinterpreted as a reply to the previous slot, triggering the fast-path and inaccurate logging. It also confuses the styler, which keeps enforcing the old slot-specific rules even though the flow restarted.

### 8. Collected slots rarely influence the final message
- **Files:** `configs/recommendation_response.yaml`.
- **What happens:** The templates only reference `{tier}`, `{destination}` (travel), and `{add_ons}` (maid). None of the other required slots—coverage_scope, duration, maid_country, risk_level, dependants, occupation, support, scam experience, etc.—are surfaced in the “user” prompt given to the response LLM.
- **Impact:** Users answer several detailed questions but the final recommendation ignores everything except the tier, so the chatbot feels scripted rather than intelligent. It also removes the justification auditors expect (“You said you shop daily, so I picked Platinum,” etc.).

### 9. Tier selection ignores most of the collected data
- **Files:** `tools/recommendation.py` (`_select_tier`).
- **What happens:** Travel is hard-coded to Gold every time; Car is hard-coded to “Standard”; Maid uses only `coverage_above_mom_minimum` (so Exclusive is never recommended); Fraud looks only at purchase frequency; Hospital reduces four slots (age, occupation, dependants, coverage) down to the numeric coverage figure; Personal Accident ignores risk level and coverage scope entirely.
- **Impact:** The user’s answers have almost no effect on plan selection, and in the Maid/Fraud cases some tiers are literally unreachable. The flow therefore cannot produce a “most intelligent chatbot” experience because personalization is fake.

### 10. Templates demand exact benefit amounts without providing them
- **Files:** `configs/recommendation_response.yaml`, `tools/recommendation.py` (`_generate_recommendation_text`).
- **What happens:** The templates instruct the LLM to print “EXACTLY these benefits with amounts” and to upsell using specific dollar figures, but the `user` prompt only passes the entire raw `benefits_text` block (containing every tier). The model has to guess which number matches the recommended tier.
- **Impact:** The assistant frequently hallucinates or mixes benefits from other tiers, producing inaccurate dollar figures that could mislead customers and trigger compliance issues.

### 11. Side-info answers lose their citations
- **Files:** `nodes/rec_subgraph.py` (`_rec_side_info`), `tools/info.py` (`_info_tool`).
- **What happens:** `_info_tool` returns both the answer and the source documents, but `_rec_side_info` discards the sources and only keeps the text.
- **Impact:** Unlike normal info responses, clarifications inside the recommendation flow cannot show references, so there is no audit trail for the explanation even though the knowledge base requires attribution.

### 12. Fraud Protect360 flow ignores user consent
- **Files:** `nodes/rec_subgraph.py` (Fraud-specific branch in `_rec_extract_slots` and `_rec_ask_next_slot`).
- **What happens:** If the user says “no” to learning more about Fraud Protect360, the code sets `fraud_intro_shown = "no"` and immediately moves on to `purchase_frequency` questions without checking `_fraud_rec_started`. The same happens if they decline the real-life example.
- **Impact:** Users who explicitly declined to continue still get interrogated for purchase behaviour, which violates the intended educational gating and feels spammy.
