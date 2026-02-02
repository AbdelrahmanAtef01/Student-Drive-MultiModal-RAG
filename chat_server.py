import uvicorn
import os
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from chat_graph import chat_app
from firebase_manager import FirebaseManager

# Setup
load_dotenv()
app = FastAPI(title="Chat Engine Core", version="3.0")

# --- CORS BLOCK---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)
# ---------------------

# Firebase Init
try:
    firebase = FirebaseManager()
    print("System Memory (Firebase): Connected")
except Exception as e:
    print(f"System Memory Error: {e}")

# --- API MODELS ---
class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    role: Literal["student", "admin"] = "student" 
    message: str
    filter_file_ids: Optional[List[str]] = None

# --- ROUTES ---
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    print(f"\n[INCOMING] User: {req.user_id} | Session: {req.session_id}")

    try:
        # 1. Role Validation
        real_role = firebase.get_user_role(req.user_id)
        
        # 2. History Retrieval
        history_dicts = firebase.get_chat_history(req.session_id, limit=10)
        lc_history = []
        for m in history_dicts:
            if m['role'] == 'user': lc_history.append(HumanMessage(content=m['content']))
            elif m['role'] == 'ai': lc_history.append(AIMessage(content=m['content']))
        
        # 3. Context Construction
        input_message = HumanMessage(content=req.message)
        
        # STRICT FILTER INJECTION
        if req.filter_file_ids:
            print(f"  Context Scope: {len(req.filter_file_ids)} files selected.")
            sys_note = (
                f"\n\nSYSTEM_OVERRIDE: The user has restricted this query to specific files. "
                f"You MUST pass this exact list to 'search_knowledge_base' as the 'file_ids' argument: "
                f"{req.filter_file_ids}"
            )
            input_message.content += sys_note

        # 4. Save User Input
        firebase.save_message(req.session_id, "user", req.message)

        # 5. Execute Graph
        initial_state = {
            "messages": lc_history + [input_message],
            "user_role": real_role
        }
        
        final_state = chat_app.invoke(initial_state)
        
        # 6. Extract Response & Tool Metadata
        all_tool_calls = []
        final_response_text = ""
        
        for msg in final_state['messages']:
            if isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        all_tool_calls.append(tc['name'])
                if msg.content:
                    final_response_text = msg.content

        unique_tool_calls = list(set(all_tool_calls))
        if unique_tool_calls: print(f"   Executed: {unique_tool_calls}")

        # 7. Save AI Output
        firebase.save_message(req.session_id, "ai", final_response_text)

        return {
            "response": final_response_text,
            "role_used": real_role,
            "tool_calls": unique_tool_calls
        }

    except Exception as e:
        print(f"Core Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("chat_server:app", host="0.0.0.0", port=8001, reload=True)