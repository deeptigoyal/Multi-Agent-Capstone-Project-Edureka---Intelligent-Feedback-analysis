import streamlit as st
import asyncio
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from typing import TypedDict, Optional, Dict, List
import re
from dotenv import load_dotenv
import os
import requests
import logging

# === CONFIG ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONSTANTS ===
INTENT_DETECTION_NODE = "Intent Detection"

# === STATE ===
################################Changed##############################

class FeedbackState(TypedDict):
    # === INPUT ===
    source_id: str                    # review_id or email_id
    source_type: str                  # "app_review" | "support_email"
    raw_text: str                     # review_text or email body/subject
    metadata: Dict[str, str]          # platform, rating, timestamp, app_version, sender, etc.

    # === FEEDBACK CLASSIFICATION ===
    category: Optional[str]           # Bug | Feature Request | Praise | Complaint | Spam
    confidence: Optional[float]       # Classification confidence score
    priority: Optional[str]           # Critical | High | Medium | Low

    # === BUG/Feature ANALYSIS OUTPUT ===
    technical_details: Optional[str]  # For bugs: device, OS, repro steps, severity
    feature_details: Optional[str]    # For features: description, impact, demand
    analysis_notes: Optional[str]     # Agent reasoning / observations

    # === TICKET CREATION ===
    ticket_title: Optional[str]       # Suggested ticket title
    ticket_description: Optional[str] # Structured ticket body
    ticket_metadata: Optional[Dict[str, str]]  # tags, component, version, etc.

    # === QUALITY CONTROL ===
    qc_status: Optional[str]          # Approved | Needs Review | Rejected
    qc_feedback: Optional[str]        # Critic agent comments

    # === CONTROL & TRACEABILITY ===
    manual_review_flag: Optional[bool]  # Human-in-the-loop trigger
    processing_log: Optional[List[str]] # Step-by-step processing history


# === LLM ===
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

# === USER PROFILE COLLECTION ===
# PATTERN: ReAct (LLM extraction = Reason → Update profile = Action)
# MODULE: Perception + Learning (extracts missing profile info)
async def csv_reader_agent(state: FeedbackState) -> FeedbackState:
    return state

#async def feedback_classifier_agent(state: FeedbackState) -> FeedbackState:
#    return state

# === FEEDBACK CLASSIFIER AGENT ===
async def feedback_classifier_agent(state: FeedbackState) -> FeedbackState:
    raw_text = state.get("raw_text", "")
    metadata = state.get("metadata", {})
    processing_log: List[str] = state.get("processing_log", [])

    # --- LLM Prompt ---
    prompt = (
        f"Classify the following user feedback into one of: "
        f"'Bug', 'Feature Request', 'Praise', 'Complaint', 'Spam'. "
        f"Provide a confidence score (0-1) and suggest an initial priority "
        f"(Critical, High, Medium, Low).\n\n"
        f"Feedback: {raw_text}\n"
        f"Metadata: {metadata}\n\n"
        f"Format: category: <category>\n"
        f"confidence: <0-1>\n"
        f"priority: <priority>\n"
    )

    # --- Call LLM ---
    response = await llm.ainvoke(prompt)
    message = response.content.strip()
    processing_log.append(f"Classifier prompt: {prompt}")
    processing_log.append(f"Classifier response: {message}")

    # --- Parse LLM response ---
    category_match = re.search(r"category:\s*(\w+)", message, re.IGNORECASE)
    confidence_match = re.search(r"confidence:\s*([0-1]\.?\d*)", message)
    priority_match = re.search(r"priority:\s*(\w+)", message, re.IGNORECASE)

    category = category_match.group(1) if category_match else "Unknown"
    confidence = float(confidence_match.group(1)) if confidence_match else None
    priority = priority_match.group(1) if priority_match else "Medium"

    # --- Update state ---
    state.update({
        "category": category,
        "confidence": confidence,
        "priority": priority,
        "processing_log": processing_log
    })

    return state

#async def bug_analysis_agent(state: FeedbackState) -> FeedbackState:
#    return state

async def bug_analysis_agent(state: FeedbackState) -> FeedbackState:
    if state.get("category") != "Bug":
        # Skip if not a bug
        return state

    raw_text = state.get("raw_text", "")
    metadata = state.get("metadata", {})
    processing_log: List[str] = state.get("processing_log", [])

    # --- LLM Prompt ---
    prompt = (
        f"Analyze this bug report and extract technical details:\n"
        f"- Device/OS information\n"
        f"- Steps to reproduce\n"
        f"- Severity (Critical/High/Medium/Low)\n"
        f"Provide the output in structured format.\n\n"
        f"Feedback: {raw_text}\n"
        f"Metadata: {metadata}\n"
        f"Format:\n"
        f"device_info: <device info>\n"
        f"os_version: <OS info>\n"
        f"steps_to_reproduce: <steps>\n"
        f"severity: <severity>\n"
    )

    response = await llm.ainvoke(prompt)
    message = response.content.strip()

    processing_log.append(f"Bug Analysis prompt: {prompt}")
    processing_log.append(f"Bug Analysis response: {message}")

    # --- Parse LLM response ---
    device_info_match = re.search(r"device_info:\s*(.*)", message)
    os_version_match = re.search(r"os_version:\s*(.*)", message)
    steps_match = re.search(r"steps_to_reproduce:\s*(.*)", message)
    severity_match = re.search(r"severity:\s*(.*)", message, re.IGNORECASE)

    technical_details = {
        "device_info": device_info_match.group(1) if device_info_match else "",
        "os_version": os_version_match.group(1) if os_version_match else "",
        "steps_to_reproduce": steps_match.group(1) if steps_match else "",
    }

    priority = severity_match.group(1) if severity_match else state.get("priority", "Medium")

    # --- Update state ---
    state.update({
        "technical_details": technical_details,
        "priority": priority,
        "processing_log": processing_log
    })

    return state


#async def feature_extraction_agent(state: FeedbackState) -> FeedbackState:
#    return state
async def feature_extraction_agent(state: FeedbackState) -> FeedbackState:
    if state.get("category") != "Feature Request":
        # Skip if not a feature request
        return state

    raw_text = state.get("raw_text", "")
    metadata = state.get("metadata", {})
    processing_log: List[str] = state.get("processing_log", [])

    # --- LLM Prompt ---
    prompt = (
        f"Analyze this feature request and extract actionable details:\n"
        f"- Feature description\n"
        f"- User intent / goal\n"
        f"- Estimated user impact / demand\n"
        f"Provide the output in structured format.\n\n"
        f"Feedback: {raw_text}\n"
        f"Metadata: {metadata}\n"
        f"Format:\n"
        f"feature_description: <description>\n"
        f"user_intent: <intent>\n"
        f"user_impact: <high/medium/low>\n"
    )

    # --- Call LLM ---
    response = await llm.ainvoke(prompt)
    message = response.content.strip()

    processing_log.append(f"Feature Extraction prompt: {prompt}")
    processing_log.append(f"Feature Extraction response: {message}")

    # --- Parse LLM response ---
    description_match = re.search(r"feature_description:\s*(.*)", message)
    intent_match = re.search(r"user_intent:\s*(.*)", message)
    impact_match = re.search(r"user_impact:\s*(.*)", message, re.IGNORECASE)

    feature_details = {
        "feature_description": description_match.group(1) if description_match else "",
        "user_intent": intent_match.group(1) if intent_match else "",
        "user_impact": impact_match.group(1) if impact_match else "Medium"
    }

    # --- Update state ---
    state.update({
        "feature_details": feature_details,
        "processing_log": processing_log
    })

    return state



#async def ticket_creation_agent(state: FeedbackState) -> FeedbackState:
#    return state

#ticket agent which combines outputs from Bug Analysis or Feature Extraction into structured tickets.
async def ticket_creation_agent(state: FeedbackState) -> FeedbackState:
    category = state.get("category", "")
    priority = state.get("priority", "Medium")
    technical_details = state.get("technical_details", {})
    feature_details = state.get("feature_details", {})
    processing_log: List[str] = state.get("processing_log", [])

    # --- Construct ticket ---
    if category == "Bug":
        ticket_title = f"[BUG] {technical_details.get('device_info', '')} - Issue"
        ticket_description = (
            f"Category: {category}\n"
            f"Priority: {priority}\n"
            f"Device Info: {technical_details.get('device_info', '')}\n"
            f"OS Version: {technical_details.get('os_version', '')}\n"
            f"Steps to Reproduce: {technical_details.get('steps_to_reproduce', '')}\n"
        )
    elif category == "Feature Request":
        ticket_title = f"[FEATURE REQUEST] {feature_details.get('feature_description', '')[:50]}"
        ticket_description = (
            f"Category: {category}\n"
            f"Priority: {priority}\n"
            f"Feature Description: {feature_details.get('feature_description', '')}\n"
            f"User Intent: {feature_details.get('user_intent', '')}\n"
            f"Estimated User Impact: {feature_details.get('user_impact', 'Medium')}\n"
        )
    else:
        ticket_title = f"[{category.upper()}] Feedback"
        ticket_description = f"Category: {category}\nPriority: {priority}\n"

    # Include metadata
    metadata = state.get("metadata", {})
    for key, value in metadata.items():
        ticket_description += f"{key}: {value}\n"

    processing_log.append(f"Ticket Created: {ticket_title}")

    # --- Update state ---
    state.update({
        "ticket_title": ticket_title,
        "ticket_description": ticket_description,
        "ticket_metadata": metadata,
        "processing_log": processing_log
    })

    return state




#async def quality_critic_agent(state: FeedbackState) -> FeedbackState:
#    return state

#Quality Critic Agent, which validates ticket completeness, accuracy, and flags for human review if needed.
async def quality_critic_agent(state: FeedbackState) -> FeedbackState:
    ticket_title = state.get("ticket_title", "")
    ticket_description = state.get("ticket_description", "")
    category = state.get("category", "")
    confidence = state.get("confidence", 1.0)
    processing_log: List[str] = state.get("processing_log", [])

    manual_review_flag = False
    qc_feedback = []

    # --- Basic checks ---
    if not ticket_title:
        manual_review_flag = True
        qc_feedback.append("Ticket title missing.")
    if not ticket_description:
        manual_review_flag = True
        qc_feedback.append("Ticket description missing.")
    if category in ["Bug", "Feature Request"] and confidence is not None and confidence < 0.7:
        manual_review_flag = True
        qc_feedback.append(f"Low confidence ({confidence}) for category {category}.")

    if manual_review_flag:
        processing_log.append(f"Quality Critic flagged manual review: {qc_feedback}")
    else:
        processing_log.append("Quality Critic passed: ticket looks complete.")

    # --- Update state ---
    state.update({
        "manual_review_flag": manual_review_flag,
        "qc_feedback": qc_feedback,
        "processing_log": processing_log
    })

    return state

async def end_node(state: FeedbackState) -> FeedbackState:
    return state


# === BUILD GRAPH ===
# PATTERN: Planning Pattern — router decides next node
# MODULE: Cognition
def get_next_node(state: FeedbackState) -> str:
    if state.get("manual_review_flag"):
        return "Human Review"

    category = state.get("category")

    if category == "Bug":
        return "Bug Analysis"
    elif category == "Feature Request":
        return "Feature Extraction"
    elif category in ["Praise", "Complaint"]:
        return "Ticket Creation"
    elif category == "Spam":
        return "End"
    else:
        return "End"


#builder = StateGraph(FinanceState)
builder = StateGraph(FeedbackState)

# PATTERN: Planning — adds modular nodes
# Add nodes
builder.add_node("CSV Reader", csv_reader_agent)
builder.add_node("Feedback Classifier", feedback_classifier_agent)
builder.add_node("Bug Analysis", bug_analysis_agent)
builder.add_node("Feature Extraction", feature_extraction_agent)
builder.add_node("Ticket Creation", ticket_creation_agent)
builder.add_node("Quality Critic", quality_critic_agent)
builder.add_node("End", end_node)

# PATTERN: Planning + ReAct orchestration (routing based on reasoning)
# Entry point
builder.set_entry_point("CSV Reader")

# Edges
builder.add_edge("CSV Reader", "Feedback Classifier")

builder.add_conditional_edges(
    "Feedback Classifier",
    get_next_node,
    {
        "Bug Analysis": "Bug Analysis",
        "Feature Extraction": "Feature Extraction",
        "Ticket Creation": "Ticket Creation",
        "End": "End"
    }
)

# Post-analysis flow
builder.add_edge("Bug Analysis", "Ticket Creation")
builder.add_edge("Feature Extraction", "Ticket Creation")
builder.add_edge("Ticket Creation", "Quality Critic")
builder.add_edge("Quality Critic", "End")

feedback_graph = builder.compile()

# === STREAMLIT UI ===
# MODULE: Interaction Layer (UI, not part of agent logic)
st.set_page_config(page_title="💸 FinAdvise", page_icon="💬", layout="centered")
st.title("💸 FinAdvise")
st.caption("Your personal finance assistant for stocks, expenses, budgets, and tailored advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "long_term_memory" not in st.session_state:
    st.session_state.long_term_memory = {}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # PATTERN: Planning + Action — triggers agent graph
            state = {
                "user_input": user_input,
                "intent": None,
                "data": None,
                "user_profile": st.session_state.get("user_profile", {}),
                "short_term_memory": {},
                "long_term_memory": st.session_state.long_term_memory,
                "hitl_flag": False
            }
            final_state = asyncio.run(finance_bot.ainvoke(state))
            bot_reply = final_state['data']['response']
            st.session_state.user_profile = final_state.get('user_profile', {})
            st.session_state.long_term_memory = final_state.get('long_term_memory', {})
            st.markdown(bot_reply)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
