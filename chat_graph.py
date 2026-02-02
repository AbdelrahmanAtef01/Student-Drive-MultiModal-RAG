import os
import httpx
from typing import List, Literal, Annotated, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from vector_store import VectorDB

# Setup
load_dotenv()
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
APP_A_URL = "http://localhost:8000"

# Shared Vector DB
vector_db = VectorDB()

# --- MODEL SETUP ---
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL_ID"), 
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    max_output_tokens=1024
)

# --- PYDANTIC SCHEMAS ---
class IngestVideoInput(BaseModel):
    url: str = Field(
        description="The full YouTube URL to ingest. Example: https://www.youtube.com/watch?v=xxxxx"
    )
    admin_id: str = Field(
        default="admin", 
        description="The ID of the admin user requesting the ingest."
    )

class UpdateChunkInput(BaseModel):
    chunk_id: str = Field(
        description="""
        The EXACT unique ID of the chunk to update. 
        CRITICAL: You must extract this from the latest previous 'search_knowledge_base' result. 
        It typically looks like a UUID (e.g., '550e8400-e29b...') or a hash. 
        NEVER guess. If you don't see an ID in the context, search first.
        """
    )
    new_text: str = Field(
        description="The complete, corrected text that should replace the old chunk."
    )

class SearchInput(BaseModel):
    query: str = Field(
        description="The search query to find relevant information in the knowledge base."
    )
    file_ids: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of specific File IDs to restrict the search to."
    )

# --- TOOLS ---
@tool("ingest_youtube_video", args_schema=IngestVideoInput)
def ingest_youtube_video(url: str, admin_id: str = "admin"):
    """
    (Admin Only) Use this tool to download/ingest a YouTube video into the database.
    This tool is REQUIRED if the user asks to add, download, or ingest a video.
    """
    print(f"\n[TOOL EXEC] ingest_youtube_video called with: {url}")
    try:
        response = httpx.post(
            f"{APP_A_URL}/ingest-youtube",
            headers={"x-api-key": API_SECRET_KEY},
            json={"url": url, "admin_id": admin_id},
            timeout=10.0
        )
        if response.status_code == 200:
            data = response.json()
            return f"Success: Video queued. File ID: {data.get('file_id')}"
        return f"Error: {response.text}"
    except Exception as e:
        return f"Connection Error: {str(e)}"

@tool("update_knowledge_chunk", args_schema=UpdateChunkInput)
def update_knowledge_chunk(chunk_id: str, new_text: str):
    """
    (Admin Only) Use this tool to update the text content of a specific knowledge chunk.
    This tool is REQUIRED if the user asks to correct, change, or update specific text.
    """
    print(f"\n[TOOL EXEC] update_knowledge_chunk called for: {chunk_id}")
    try:
        response = httpx.post(
            f"{APP_A_URL}/update-chunk",
            headers={"x-api-key": API_SECRET_KEY},
            json={"chunk_id": chunk_id, "new_text": new_text},
            timeout=5.0
        )
        if response.status_code == 200:
            return "Success: Chunk updated."
        return f"Error: {response.text}"
    except Exception as e:
        return f"Connection Error: {str(e)}"

@tool("search_knowledge_base", args_schema=SearchInput)
def search_knowledge_base(query: str, file_ids: Optional[List[str]] = None):
    """
    ALWAYS USE THIS FIRST. Searches the vector database for relevant course content.
    Input: A specific query string (e.g., 'neural networks architecture').
    If 'file_ids' are provided, search ONLY within those files.
    """
    scope_msg = f" (Scope: {len(file_ids)} files)" if file_ids else " (Scope: Global)"
    print(f"\n[TOOL] search_knowledge_base: '{query}'{scope_msg}")
    try:
        # Pass filter logic to VectorDB
        results = vector_db.query(
            query, 
            n_results=5,
            where={"file_id": {"$in": file_ids}} if file_ids else None 
        )
        
        formatted_results = ""
        for r in results:
            meta = r.get('metadata', {})
            file_id = meta.get('file_id', 'unknown')
            chunk_id = r.get('id', 'unknown')
            
            # Source Formatting
            if file_id.startswith("yt_"):
                vid_id = file_id.replace("yt_", "")
                link = f"https://youtube.com/watch?v={vid_id}"
                if 'start' in meta: link += f"&t={int(float(meta['start']))}s"
                loc = f"Time {meta.get('start',0)}s"
            else:
                link = f"https://drive.google.com/file/d/{file_id}/view"
                loc = f"Page {meta.get('page', '?')}"

            formatted_results += f"--- [ID: {chunk_id}] [Link: {link}] [{loc}] ---\n{r.get('text', '')}\n\n"
        
        return formatted_results if formatted_results else "No relevant results found."

    except Exception as e:
        return f"Database Error: {str(e)}"

# Bind tools
tools = [ingest_youtube_video, update_knowledge_chunk, search_knowledge_base]
llm_with_tools = llm.bind_tools(tools)

# --- GRAPH DEFINITION ---
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    user_role: Literal["student", "admin"]

def agent_node(state: AgentState):
    role_instruction = "You are a helpful Teaching Assistant."
    
    # --- DYNAMIC INSTRUCTIONS BASED ON ROLE ---
    if state.user_role == "admin":
        role_instruction += """
        You are an ADMIN. You have special permissions to modify the database.
        
        ADMIN RULES:
        1. If asked to 'ingest' a video, MUST use 'ingest_youtube_video'.
        2. If asked to 'update' text, MUST use 'update_knowledge_chunk'.
        3. DO NOT hallucinate success.
        
        ### HANDLING CORRECTIONS (CRITICAL):
        If the user says a previous answer was WRONG (e.g., "No, actually X is Y"):
        1. Look at the 'search_knowledge_base' output in the chat history.
        2. Find the chunk that contained the wrong info. It looks like: `--- [ID: 550e8400-e29b...] ...`
        3. EXTRACT that exact `chunk_id` (e.g., "550e8400-e29b...").
        4. Call `update_knowledge_chunk` using that ID and the user's corrected text.
        
        DO NOT ask the user for the ID. Find it yourself in the history.
        """
    else:
        role_instruction += " You are a STUDENT assistant. You answer questions based ONLY on the database."

    # --- SHARED CRITICAL RULES ---
    system_prompt = f"""
    {role_instruction}
    
    GENERAL RULES:
    1. You do not know anything about the course content yourself. 
    2. For any content questions, you MUST use the 'search_knowledge_base' tool.
    3. Do not apologize. Do not say you can't. Just call the relevant tool.
    
    RESPONSE FORMAT:
    - If you used a tool, report the tool's actual output.
    - If searching: "Explanation here. [Source: <INSERT_LINK_FROM_TOOL>]"
    """
    
    conversation = [SystemMessage(content=system_prompt)] + state.messages
    response = llm_with_tools.invoke(conversation)

    # --- PRINTS ---
    print(f"\n--- [AGENT DECISION] ---")
    print(f"Role: {state.user_role}")
    print(f"Tool Calls: {response.tool_calls}")
    if hasattr(response, 'invalid_tool_calls') and response.invalid_tool_calls:
         print(f"INVALID TOOL CALLS: {response.invalid_tool_calls}")
    # ------------------------

    return {"messages": [response]}

# Custom Router
def router(state: AgentState):
    last_msg = state.messages[-1]
    
    # 1. Check for Valid Tool Calls
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        print(f"[ROUTER] Routing to 'tools' node.")
        return "tools"
    
    # 2. Check for Invalid Tool Calls
    if hasattr(last_msg, 'invalid_tool_calls') and last_msg.invalid_tool_calls:
        print(f"[ROUTER] Routing to END (Invalid Tool Call Detected).")
        return END

    print(f"[ROUTER] Routing to END (No tools used).")
    return END

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router,
)
workflow.add_edge("tools", "agent")

chat_app = workflow.compile()