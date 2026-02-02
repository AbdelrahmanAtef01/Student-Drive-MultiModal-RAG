import os
import uvicorn
import sys
import json
import traceback
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from drive_processor import DriveProcessor
from chunking_engine import SemanticChunker
from vector_store import VectorDB
from workflow import app_workflow 

# Setup
load_dotenv()
app = FastAPI(title="App A: Ingestion Engine", version="5.0")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# --- GLOBAL INITIALIZATION ---
print("[Server] Pre-loading AI Engines... (This happens only once)")
vector_db = VectorDB()
chunker = SemanticChunker()

# Models
class DocumentRequest(BaseModel):
    file_id: str

class YouTubeRequest(BaseModel):
    url: str
    admin_id: str

class DeleteRequest(BaseModel):
    file_id: str

class UpdateChunkRequest(BaseModel):
    chunk_id: str
    new_text: str

# --- PIPELINE LOGIC ---
def run_pipeline(file_id: str, direct_url: str = None):
    print(f"\n[API] Starting Task for ID: {file_id}")
    try:
        local_path = None
        
        # 1. Determine Source
        if direct_url:
            # YouTube Mode
            local_path = direct_url
            print(f"[API] Using Direct URL: {local_path}")
        else:
            # Drive Mode
            processor = DriveProcessor() 
            local_path = processor.download_file(file_id)
        
        if not local_path: 
            print("[API] No valid path or URL found.")
            return

        # 2. Invoke Graph
        print(f"[API] Invoking Workflow for: {local_path}")
        inputs = {"file_path": local_path, "file_id": file_id}
        final_state = app_workflow.invoke(inputs)
        
        if final_state.get("status") == "failed":
            print(f"[API] Graph Failed: {final_state.get('error')}")
            return

        # 3. Save JSON 
        rag_data = final_state.get("rag_ready_data")
        
        # Determine output folder name
        if direct_url:
            safe_name = f"yt_{file_id.replace('yt_', '')}"
        else:
            safe_name = os.path.splitext(os.path.basename(local_path))[0]
            
        output_dir = os.path.join("data_output_final", safe_name)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        json_path = os.path.join(output_dir, "rag_ready_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, indent=2, ensure_ascii=False)
        print(f"   Data saved to: {json_path}")

        # 4. Chunk & Embed
        print(f"[API] Chunking & Embedding...")
        chunks = chunker.process_json(json_path, file_id)
        if chunks:
            vector_db.delete_file(file_id)
            vector_db.add_chunks(chunks)
            print("[API] Pipeline Complete.")
        else:
             print("[API] No chunks generated.")

    except Exception as e:
        print(f"[API] Critical Error: {e}")
        traceback.print_exc()

# --- ENDPOINTS ---

@app.post("/process-document")
async def process_document(req: DocumentRequest, bg_tasks: BackgroundTasks, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET_KEY: raise HTTPException(403, "Invalid Key")
    bg_tasks.add_task(run_pipeline, req.file_id)
    return {"status": "queued", "file_id": req.file_id}

@app.post("/ingest-youtube")
async def ingest_youtube(req: YouTubeRequest, bg_tasks: BackgroundTasks, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET_KEY: raise HTTPException(403, "Invalid Key")
    
    video_id = req.url.split("v=")[-1] if "v=" in req.url else "video"
    fake_file_id = f"yt_{video_id}"
    
    bg_tasks.add_task(run_pipeline, fake_file_id, req.url)
    return {"status": "queued_youtube", "file_id": fake_file_id}

@app.post("/delete-document")
async def delete_document(req: DeleteRequest, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET_KEY: raise HTTPException(403, "Invalid Key")
    vector_db.delete_file(req.file_id)
    return {"status": "deleted"}

@app.post("/update-chunk")
async def update_chunk(req: UpdateChunkRequest, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET_KEY: raise HTTPException(403, "Invalid Key")
    vector_db.update_chunk(req.chunk_id, req.new_text)
    return {"status": "updated"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)