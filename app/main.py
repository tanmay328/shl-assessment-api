from dotenv import load_dotenv
load_dotenv()

# ... rest of your imports (FastAPI, LLMGenerator, etc.)

# app/main.py - Complete FastAPI Application with LLM + RAG
import json
import os
import re
import sys
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from app.models import Message, ChatRequest, ChatResponse, Recommendation
from app.rag.vector_store import VectorStore
from app.llm.generator import LLMGenerator
from app.evaluator.metrics import Evaluator

# --- Create FastAPI App ---
app = FastAPI(
    title="SHL Assessment Recommender API",
    description="LLM + RAG powered assessment recommendation system",
    version="1.0.0"
)

print("=" * 60)
print(" SHL Assessment Recommender API (LLM + RAG)")
print("=" * 60)
print(" Note: 'Uvicorn running on http://0.0.0.0:...' below is the")
print(" container's internal bind address, not a URL you can open.")
print(" If running via `docker run -p 8001:8000 ...`, use instead:")
print("   http://localhost:8001/health")
print("=" * 60)

# --- Initialize Components ---
print("\n Initializing components...")

# 1. Vector Store (RAG)
try:
    vector_store = VectorStore(catalog_path="data/catalog.json")
    print(" Vector Store (RAG) initialized")
except Exception as e:
    print(f" Vector Store error: {e}")
    vector_store = None

# 2. LLM Generator
try:
    llm = LLMGenerator()
    print(" LLM Generator initialized")
except Exception as e:
    print(f" LLM error: {e}")
    llm = None

# 3. Evaluator (for metrics)
evaluator = Evaluator()

print("=" * 60)

# --- Helper Functions ---
def is_vague_query(query: str) -> bool:
    """Check if query is vague and needs clarification"""
    vague_patterns = [
        r'^(assessment|test|hire|need|looking for|recruiting|screen)(\?)?$',
        r'^(i need|i want|looking for|hiring for)\s+(an?\s+)?(assessment|test|solution|tool)(\?)?$',
        r'^(hello|hi|hey|help)(\?)?$',
        r'^(\?)?$',
    ]
    return any(re.search(p, query.lower()) for p in vague_patterns)

def is_off_topic(query: str) -> bool:
    """Check if query is off-topic (not about SHL assessments)"""
    off_topic_keywords = [
        "weather", "news", "sports", "movie", "recipe", "cooking", 
        "politics", "stock", "market", "bitcoin", "crypto", "music", 
        "game", "football", "cricket", "food", "restaurant"
    ]
    return any(kw in query.lower() for kw in off_topic_keywords)

def is_prompt_injection(query: str) -> bool:
    """Check for prompt injection attempts"""
    injection_patterns = [
        r'ignore previous',
        r'forget (the|your) instructions',
        r'disregard (the|your) rules',
        r'you are now (a|an)',
        r'pretend (you are|to be)',
        r'act as (a|an)',
        r'system prompt',
        r'override (the|your)',
        r'bypass (the|your)',
    ]
    return any(re.search(p, query.lower()) for p in injection_patterns)

def should_end_conversation(query: str) -> bool:
    """Check if user wants to end conversation"""
    end_phrases = ["goodbye", "bye", "thanks", "that's all", "finished", "exit", "done", "thank you"]
    return any(phrase in query.lower() for phrase in end_phrases)

def has_seniority_info(text: str) -> bool:
    """Check if seniority/experience level info is present anywhere in the text"""
    seniority_patterns = [
        r'\b(entry[\s-]?level|junior|jr\.?|associate)\b',
        r'\b(mid[\s-]?level|mid[\s-]?senior|intermediate)\b',
        r'\b(senior|sr\.?|lead|principal|staff)\b',
        r'\b(manager|director|executive|c-level|vp)\b',
        r'\b\d+\s*\+?\s*year',
        r'\bnew\s*grad',
        r'\bfresh(er|man)?\b',
    ]
    return any(re.search(p, text.lower()) for p in seniority_patterns)

def extract_shortlist_from_text(text: str) -> List[Dict]:
    """Parse the numbered shortlist out of a previous assistant reply, e.g.:
        1. **Java 8 (New)** (K)
           https://www.shl.com/products/product-catalog/view/java-8-new/
    This is how we reconstruct 'shortlist state' across turns despite the
    API being stateless (the full history only carries role/content strings).
    """
    items = []
    pattern = re.compile(
        r'\*\*(.+?)\*\*\s*\(([A-Za-z]+)\)\s*\n\s*(https?://\S+)'
    )
    for match in pattern.finditer(text):
        name, test_type, url = match.groups()
        items.append({"name": name.strip(), "url": url.strip(), "test_type": test_type.strip()})
    return items

def find_previous_shortlist(messages: List[Message]) -> List[Dict]:
    """Scan backwards through assistant turns for the most recent shortlist table."""
    for m in reversed(messages):
        if m.role == 'assistant':
            items = extract_shortlist_from_text(m.content)
            if items:
                return items
    return []

def is_confirmation(query: str) -> bool:
    """Detect if the user is accepting/finalizing the current shortlist"""
    confirm_phrases = [
        "confirmed", "that's good", "that works", "sounds good", "perfect",
        "locking it in", "keep the shortlist", "final list", "that's what we need",
        "that covers it", "good choice", "that's all", "looks good", "great, thanks",
        "keep it as is", "keep the list", "as-is", "confirm the shortlist",
    ]
    q = query.lower()
    return any(p in q for p in confirm_phrases)

def parse_refine_terms(query: str) -> Tuple[List[str], List[str]]:
    """Extract 'add X' and 'remove/drop Y' phrases from a refine message.
    Splits on conjunctions/punctuation FIRST so a single sentence containing
    both an add and a remove instruction (e.g. 'drop X and add Y') doesn't
    get captured as one greedy blob."""
    q = query.lower()
    add_terms, remove_terms = [], []
    clauses = re.split(r'\b(?:and|but|also)\b|[,.;]', q)
    filler_suffix = re.compile(r'\b(one|test|exam|assessment|option|please)\b\s*$')

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        m = re.match(r'^(?:actually\s*)?(?:please\s*)?add(?:ing)?\s+(?:an?\s+|the\s+)?(.+)$', clause)
        if m:
            term = filler_suffix.sub('', m.group(1)).strip()
            if term:
                add_terms.append(term)
            continue
        m = re.match(r'^(?:actually\s*)?(?:please\s*)?(?:remove|drop|exclude)\s+(?:the\s+)?(.+)$', clause)
        if m:
            term = filler_suffix.sub('', m.group(1)).strip()
            if term:
                remove_terms.append(term)

    return add_terms, remove_terms

def is_refine_query(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in ["add ", "adding ", "remove ", "drop ", "exclude ", "instead of", "replace "])

def format_shortlist_reply(items: List[Dict], intro: str) -> str:
    """Render a shortlist in the one format extract_shortlist_from_text can
    parse back out — used everywhere a shortlist is returned so state
    round-trips correctly across stateless turns."""
    reply = intro.rstrip() + "\n\n"
    for i, rec in enumerate(items[:10], 1):
        reply += f"{i}. **{rec['name']}** ({rec['test_type']})\n   {rec['url']}\n\n"
    return reply.rstrip()

SOFT_SKILL_PATTERNS = {
    "stakeholder": "stakeholder management communication skills",
    "communicat": "communication skills interpersonal",
    "leadership": "leadership potential management competency",
    "team": "teamwork collaboration interpersonal skills",
    "client": "client facing communication interpersonal skills",
    "customer": "customer service interpersonal skills",
    "personality": "personality behavior assessment",
    "culture": "cultural fit values behavior",
    "manage people": "people management leadership",
}

def detect_soft_skill_query(text: str) -> Optional[str]:
    """If the request mentions interpersonal/behavioral needs alongside a
    technical skill, return a supplementary search string for them. A single
    combined-text vector search tends to be dominated by whichever keyword
    (e.g. a tech stack name) appears most often in catalog descriptions,
    drowning out softer signals like 'works with stakeholders' — so we run
    a second, targeted search and merge results rather than relying on one
    query to represent multiple distinct needs."""
    text_lower = text.lower()
    for trigger, search_term in SOFT_SKILL_PATTERNS.items():
        if trigger in text_lower:
            return search_term
    return None

def is_comparison_query(query: str) -> bool:
    """Check if query is asking for comparison"""
    comparison_words = ["difference", "compare", "versus", "vs", "different", "better", "which is"]
    return any(word in query.lower() for word in comparison_words)

def extract_product_names(query: str) -> List[str]:
    """Extract product names from comparison query"""
    patterns = [
        r'difference between (.+?) and (.+?)(?:\?|$)',
        r'compare (.+?) and (.+?)(?:\?|$)',
        r'(.+?) vs (.+?)(?:\?|$)',
        r'(.+?) versus (.+?)(?:\?|$)',
        r'(.+?) and (.+?) difference',
        r'(.+?) or (.+?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return [match.group(1).strip(), match.group(2).strip()]
    return []

def get_test_type_from_keys(keys: List[str]) -> str:
    """Get test type code from keys list"""
    if "Personality & Behavior" in keys:
        return "P"
    elif "Ability & Aptitude" in keys:
        return "A"
    elif "Simulations" in keys:
        return "S"
    elif "Biodata & Situational Judgment" in keys:
        return "B"
    elif "Competencies" in keys:
        return "C"
    else:
        return "K"

# --- Comparison Function with LLM ---
def generate_comparison(item1: Dict, item2: Dict, query: str) -> str:
    """Generate comparison using LLM if available, else fallback"""
    
    # Try LLM first
    if llm and llm.loaded:
        try:
            prompt = f"""Compare these two SHL assessments based on the user's question.

Assessment 1: {item1.get('name', 'Unknown')}
Description: {item1.get('description', 'No description')[:200]}
Keys: {item1.get('keys', 'Not specified') or 'Not specified'}
Duration: {item1.get('duration', 'Not specified')}

Assessment 2: {item2.get('name', 'Unknown')}
Description: {item2.get('description', 'No description')[:200]}
Keys: {item2.get('keys', 'Not specified') or 'Not specified'}
Duration: {item2.get('duration', 'Not specified')}

User question: {query}

Provide a clear comparison highlighting:
1. What each assessment measures
2. Key differences in purpose and use cases
3. Which assessment might be better for different scenarios

Comparison:"""
            
            llm_response = llm.generate_response(prompt, max_length=400)
            if llm_response:
                return llm_response
        except Exception as e:
            print(f" LLM comparison error: {e}")
    
    # Fallback: Template comparison
    name1 = item1.get('name', 'Unknown')
    name2 = item2.get('name', 'Unknown')
    desc1 = item1.get('description', 'No description available.')[:200]
    desc2 = item2.get('description', 'No description available.')[:200]
    keys1 = item1.get('keys', 'Not specified') or 'Not specified'
    keys2 = item2.get('keys', 'Not specified') or 'Not specified'
    duration1 = item1.get('duration', 'Not specified')
    duration2 = item2.get('duration', 'Not specified')
    
    comparison = f"""**COMPARISON: {name1} vs {name2}**

**{name1}:**
- Description: {desc1}...
- Test Type: {keys1}
- Duration: {duration1}

**{name2}:**
- Description: {desc2}...
- Test Type: {keys2}
- Duration: {duration2}

**Key Differences:**
"""
    
    differences = []
    if keys1 != keys2:
        differences.append(f"- {name1} measures {keys1}, while {name2} measures {keys2}")
    if duration1 != duration2 and duration1 != 'Not specified' and duration2 != 'Not specified':
        differences.append(f"- {name1} takes {duration1}, while {name2} takes {duration2}")
    if "personality" in desc1.lower() and "personality" not in desc2.lower():
        differences.append(f"- {name1} focuses on personality assessment")
    elif "personality" in desc2.lower() and "personality" not in desc1.lower():
        differences.append(f"- {name2} focuses on personality assessment")
    if "skills" in desc1.lower() and "skills" not in desc2.lower():
        differences.append(f"- {name1} measures skills")
    elif "skills" in desc2.lower() and "skills" not in desc1.lower():
        differences.append(f"- {name2} measures skills")
    if "simulation" in desc1.lower() and "simulation" not in desc2.lower():
        differences.append(f"- {name1} includes simulations")
    elif "simulation" in desc2.lower() and "simulation" not in desc1.lower():
        differences.append(f"- {name2} includes simulations")
    
    if not differences:
        differences.append("- Both assessments serve different purposes. Choose based on your specific needs.")
    
    comparison += "\n".join(differences)
    comparison += "\n\nFor more details, please refer to the product pages below."
    
    return comparison

# --- Main Conversation Logic ---
def process_conversation(messages: List[Message]) -> ChatResponse:
    """Process conversation and generate response with LLM + RAG.

    Stateless per the API spec: every call receives the FULL history.
    Any shortlist 'state' is reconstructed by parsing the last assistant
    turn that contained a recommendation table.
    """

    user_msgs = [m for m in messages if m.role == 'user']
    assistant_msgs = [m for m in messages if m.role == 'assistant']

    if not user_msgs:
        return ChatResponse(
            reply="I'm here to help you find SHL assessments. Could you describe the role you're hiring for?",
            recommendations=[],
            end_of_conversation=False
        )

    last_query = user_msgs[-1].content
    previous_shortlist = find_previous_shortlist(messages)

    # --- Conversation end (explicit farewell) ---
    if should_end_conversation(last_query) and not previous_shortlist:
        return ChatResponse(
            reply="You're welcome! If you need further assistance, feel free to ask.",
            recommendations=[],
            end_of_conversation=True
        )

    # --- Off-topic ---
    if is_off_topic(last_query):
        return ChatResponse(
            reply="I'm specialized in SHL assessments and can only help with assessment selection. Could you tell me about the role you're hiring for?",
            recommendations=[],
            end_of_conversation=False
        )

    # --- Prompt injection ---
    if is_prompt_injection(last_query):
        return ChatResponse(
            reply="I can only help you find SHL assessments. Please describe the role you're hiring for.",
            recommendations=[],
            end_of_conversation=False
        )

    # --- Legal / general hiring advice out of scope ---
    legal_patterns = [r'legally required', r'\blaw\b', r'compliance obligation', r'lawsuit', r'sue\b', r'attorney', r'legal advice']
    if any(re.search(p, last_query.lower()) for p in legal_patterns):
        return ChatResponse(
            reply="That's a legal/compliance question outside what I can advise on - I can help you select assessments, but not interpret regulatory obligations. Your legal team is the right resource for that.",
            recommendations=[],
            end_of_conversation=False
        )

    if not vector_store:
        return ChatResponse(
            reply="I'm having trouble accessing the assessment catalog. Please try again later.",
            recommendations=[],
            end_of_conversation=True
        )

    combined_query = " ".join(m.content for m in user_msgs)

    # --- Comparison queries ---
    if is_comparison_query(last_query):
        product_names = extract_product_names(last_query)
        item1 = item2 = None
        # try explicit "X vs Y" extraction first
        if len(product_names) >= 2:
            for name in product_names:
                hits = vector_store.search(name, k=1)
                if hits:
                    full = vector_store.get_assessment(hits[0]['name'])
                    if item1 is None:
                        item1 = full
                    elif item2 is None:
                        item2 = full
        # else try matching against the current shortlist by substring
        if not (item1 and item2) and previous_shortlist:
            mentioned = [it for it in previous_shortlist if it['name'].lower() in last_query.lower()]
            if len(mentioned) >= 2:
                item1 = vector_store.get_assessment(mentioned[0]['name'])
                item2 = vector_store.get_assessment(mentioned[1]['name'])

        if item1 and item2:
            comparison = generate_comparison(item1, item2, last_query)
            return ChatResponse(
                reply=comparison,
                recommendations=[],
                end_of_conversation=False
            )
        # couldn't identify two items to compare
        return ChatResponse(
            reply="Could you specify which two assessments you'd like me to compare by name?",
            recommendations=[],
            end_of_conversation=False
        )

    # --- Refine: modify an existing shortlist ---
    if previous_shortlist and is_refine_query(last_query):
        add_terms, remove_terms = parse_refine_terms(last_query)

        def base_name(n: str) -> str:
            # Strip trailing catalog suffixes like "(New)" for looser matching
            return re.sub(r'\s*\([^)]*\)\s*$', '', n.lower()).strip()

        kept = list(previous_shortlist)
        for term in remove_terms:
            kept = [
                it for it in kept
                if not (term in base_name(it['name']) or base_name(it['name']) in term)
            ]

        new_adds = []
        for term in add_terms:
            hits = vector_store.search(term, k=3)
            for h in hits:
                already_present = any(
                    it['name'].lower() == h['name'].lower()
                    for it in kept + new_adds
                )
                if not already_present:
                    new_adds.append(h)
                    break

        # Cap at 10 total, but trim from the END of the kept originals
        # (lowest relevance) rather than discarding the items the user just
        # asked to add — a naive updated[:10] silently drops new adds if
        # the list was already at capacity.
        max_keep = max(0, 10 - len(new_adds))
        combined = kept[:max_keep] + new_adds
        if not combined:
            combined = previous_shortlist

        ends_now = is_confirmation(last_query)
        intro = "Updated and confirmed. Here's the final shortlist:" if ends_now else "Updated the shortlist based on your changes:"
        return ChatResponse(
            reply=format_shortlist_reply(combined, intro),
            recommendations=[Recommendation(**it) for it in combined],
            end_of_conversation=ends_now
        )

    # --- Confirmation: user accepts current shortlist as final ---
    if previous_shortlist and is_confirmation(last_query) and not is_refine_query(last_query):
        return ChatResponse(
            reply=format_shortlist_reply(previous_shortlist, "Great - shortlist confirmed:"),
            recommendations=[Recommendation(**it) for it in previous_shortlist],
            end_of_conversation=True
        )

    # --- Vague query on the first substantive turn: ask a clarifying question ---
    if is_vague_query(last_query) or (len(user_msgs) == 1 and len(last_query.split()) < 8):
        return ChatResponse(
            reply="Happy to help narrow that down. Who is this role for, and what should the assessment focus on (skills, personality, seniority)?",
            recommendations=[],
            end_of_conversation=False
        )

    # --- Otherwise: search and commit to a shortlist using full context ---
    try:
        primary = vector_store.search(combined_query, k=8)
        recommendations = list(primary)

        # A single combined-text search tends to be dominated by whichever
        # keyword repeats most across catalog descriptions (e.g. a tech
        # stack name), so behavioral/interpersonal needs mentioned alongside
        # a technical ask can get crowded out entirely. Run a second,
        # targeted search for that signal and merge in a couple of results
        # so the final shortlist actually reflects both needs.
        soft_skill_term = detect_soft_skill_query(combined_query)
        if soft_skill_term:
            soft_hits = vector_store.search(soft_skill_term, k=3)
            for h in soft_hits:
                if not any(r['name'].lower() == h['name'].lower() for r in recommendations):
                    recommendations.append(h)
                if len(recommendations) >= 10:
                    break

        recommendations = recommendations[:10]
    except Exception as e:
        print(f" Search error: {e}")
        recommendations = []

    if not recommendations:
        return ChatResponse(
            reply="I couldn't find specific SHL assessments matching your request. Could you provide more details about the role, such as the specific skills or level?",
            recommendations=[],
            end_of_conversation=False
        )
    
    # --- Generate Response ---
    # Deliberately NOT using the LLM here: its free-form phi-2 output can't
    # be reliably parsed back into shortlist state on the next turn (see
    # extract_shortlist_from_text), and the API is stateless so that parsing
    # is the only way we track an evolving shortlist across turns. The LLM
    # is still used for comparison narratives, which don't need to round-trip.
    reply = format_shortlist_reply(recommendations, "Based on your needs, I recommend the following assessments:")
    
    return ChatResponse(
        reply=reply,
        recommendations=[Recommendation(**r) for r in recommendations],
        end_of_conversation=False
    )

# --- API Endpoints ---
@app.get("/")
async def root():
    rag_status = "Ready" if vector_store else "Not Ready"
    llm_status = "Loaded" if (llm and llm.loaded) else "Fallback Mode"
    
    return HTMLResponse(f"""
    <html>
        <head>
            <title>SHL Assessment Recommender API</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .badge {{ display: inline-block; padding: 5px 10px; border-radius: 5px; font-size: 12px; font-weight: bold; }}
                .llm {{ background: #e74c3c; color: white; }}
                .rag {{ background: #3498db; color: white; }}
                .endpoint {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .method {{ display: inline-block; padding: 3px 10px; border-radius: 3px; font-weight: bold; }}
                .get {{ background: #61affe; color: white; }}
                .post {{ background: #49cc90; color: white; }}
                .status {{ color: #27ae60; font-weight: bold; }}
                .info {{ background: #eaf2f8; padding: 10px; border-left: 4px solid #3498db; margin: 10px 0; }}
                code {{ background: #e8e8e8; padding: 2px 6px; border-radius: 3px; }}
                pre {{ background: #2c3e50; color: white; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1> SHL Assessment Recommender API</h1>
                <p><span class="status">[OK]</span> API is running</p>
                <p>
                    <span class="badge llm">LLM</span>
                    <span class="badge rag">RAG</span>
                    <strong>Powered by:</strong> Phi-2 (LLM) + ChromaDB (RAG)
                </p>
                
                <div class="info">
                    <strong>Status:</strong><br>
                    LLM: {llm_status}<br>
                    RAG: {rag_status}
                </div>
                
                <h2>Endpoints</h2>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/health</strong><br>
                    <code>http://127.0.0.1:8000/health</code>
                </div>
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <strong>/chat</strong><br>
                    <code>http://127.0.0.1:8000/chat</code>
                </div>
                
                <h2>Test Command</h2>
                <pre>
curl -X POST "http://127.0.0.1:8000/chat" \\
  -H "Content-Type: application/json" \\
  -d '{{"messages": [{{"role": "user", "content": "I need to hire a Java developer"}}]}}'</pre>
            </div>
        </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        start_time = time.time()

        # Enforce turn cap (max 8): still compute a real response so
        # "final recommendations" isn't an empty promise, just force
        # end_of_conversation=True.
        if len(request.messages) > 8:
            response = process_conversation(request.messages)
            response.end_of_conversation = True
            if not response.recommendations:
                response.reply = "I've reached the conversation limit. " + response.reply
        else:
            response = process_conversation(request.messages)

        elapsed_ms = (time.time() - start_time) * 1000
        evaluator.log_metric("response_times", elapsed_ms)

        return response
    except Exception as e:
        print(f" Error: {e}")
        return ChatResponse(
            reply="I encountered an error processing your request. Please try again.",
            recommendations=[],
            end_of_conversation=True
        )

@app.get("/metrics")
async def get_metrics():
    return evaluator.get_average_metrics()