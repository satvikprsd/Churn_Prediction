import os
import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from rag_store import retrieve_strategies

class AgentState(TypedDict):
    player_data: dict 
    churn_risk_score: float    # The probability score from the Random Forest model
    churn_prediction: int      # 1 (At Risk) or 0 (Safe)
    retrieved_strategies: List[str] 
    final_recommendation: str 
    error_flag: bool          

def retrieve_node(state: AgentState):
    """Queries ChromaDB based on the player's risk profile."""
    try:
        player_context = f"Player churn risk: {state['churn_risk_score']}. Metrics: {state['player_data']}"
        
        docs = retrieve_strategies(player_context, k=2)
        strategies = [doc.page_content for doc in docs]
        
        return {"retrieved_strategies": strategies, "error_flag": False}
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return {"retrieved_strategies": [], "error_flag": True}

def generate_node(state: AgentState):
    """Uses the LLM and RAG context to generate the structured output."""
    if state.get("error_flag"):
        return state
        
    try:
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)
        
        prompt = f"""
        You are an expert Game Engagement AI. Analyze the following player data and retrieved retention strategies.
        Player Data: {json.dumps(state['player_data'])}
        Churn Risk Score: {state['churn_risk_score']}
        Retrieved Strategies: {state['retrieved_strategies']}
        
        You MUST format your response exactly with these five headings:
        1. Summary: (Player Behavior Overview)
        2. Analysis: (Churn Risk Interpretation based on data)
        3. Plan: (Engagement & Retention Recommendations using the retrieved strategies)
        4. Refs: (Quote the specific retrieved strategies used)
        5. Disclaimer: (State that this is an AI recommendation and UX impacts should be tested)
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"final_recommendation": response.content, "error_flag": False}
    except Exception as e:
        print(f"Generation Error: {e}")
        return {"final_recommendation": "", "error_flag": True}

def fallback_node(state: AgentState):
    """Executes if the LLM fails or API limits are reached."""
    fallback_message = (
        "1. Summary: Data processing error.\n"
        "2. Analysis: Unable to process churn risk at this time.\n"
        "3. Plan: Default to standard retention protocol (e.g., send generalized login bonus email).\n"
        "4. Refs: System Fallback Triggered.\n"
        "5. Disclaimer: Automated fallback response due to API disruption."
    )
    return {"final_recommendation": fallback_message}

def route_after_generation(state: AgentState):
    """Determines if the graph should finish or trigger the fallback."""
    if state.get("error_flag"):
        return "fallback"
    return END

def build_agent_graph():
    """Compiles the workflow graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("fallback", fallback_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    
    workflow.add_conditional_edges(
        "generate",
        route_after_generation,
        {
            "fallback": "fallback",
            END: END
        }
    )
    workflow.add_edge("fallback", END)
    
    return workflow.compile()

def run_engagement_agent(player_data: dict, risk_score: float, prediction: int):
    """Entry point to execute the agent for a specific player."""
    graph = build_agent_graph()
    
    initial_state = AgentState(
        player_data=player_data,
        churn_risk_score=risk_score,
        churn_prediction=prediction,
        retrieved_strategies=[],
        final_recommendation="",
        error_flag=False
    )
    
    final_state = graph.invoke(initial_state)
    return final_state["final_recommendation"]