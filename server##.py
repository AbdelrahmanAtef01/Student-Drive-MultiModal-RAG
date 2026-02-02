import os
import uvicorn
import sys
import traceback
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from drive_processor import DriveProcessor
from chunking_engine import SemanticChunker
from vector_store import VectorDB

# Setup
load_dotenv()
app = FastAPI(title="Drive RAG Pipeline API", version="2.0")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# Initialize Helpers
vector_db = VectorDB()
chunker = SemanticChunker()

# --- DATA MODELS ---
class DocumentRequest(BaseModel):
    file_id: str

class DeleteRequest(BaseModel):
    file_id: str

class UpdateChunkRequest(BaseModel):
    chunk_id: str
    new_text: str

# --- BACKGROUND TASK LOGIC ---
def run_full_pipeline(file_id: str):
    print(f"\n[API] Background Task Started for Drive ID: {file_id}")
    try:
        # A. DOWNLOAD
        processor = DriveProcessor()
        local_path = processor.download_file(file_id)
        
        if not local_path:
            print(f"[API] Failed to download file: {file_id}")
            return

        # B. PROCESS
        # This runs the heavy orchestrator and saves 'rag_ready_data.json'
        processor.orchestrator.process_file(local_path)
        
        # C. LOCATE JSON OUTPUT
        fname = os.path.splitext(os.path.basename(local_path))[0]
        json_path = os.path.join("data_output_final", fname, "rag_ready_data.json")
        
        if not os.path.exists(json_path):
             print(f"[API] JSON output not found at: {json_path}")
             return

        # D. CHUNK & EMBED
        print(f"[API] Chunking data for Vector DB...")
        chunks = chunker.process_json(json_path, file_id)
        
        if chunks:
            print(f"[API] Embedding {len(chunks)} chunks into Chroma...")
            vector_db.delete_file(file_id) 
            vector_db.add_chunks(chunks)
            print(f"[API] Pipeline Complete. RAG Ready!")
        else:
            print(f"[API] No chunks generated (Empty file?)")

    except Exception as e:
        print(f"[API] Pipeline Failed: {e}")
        traceback.print_exc()

# --- API ENDPOINTS ---
@app.post("/process-document")
async def process_document(
    request: DocumentRequest, 
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(None)
):
    """
    1. Downloads file from Drive.
    2. Runs Models.
    3. Chunks text.
    4. Embeds into ChromaDB.
    """
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    if not request.file_id:
        raise HTTPException(status_code=400, detail="file_id is required")

    # Add to background queue so API returns instantly
    background_tasks.add_task(run_full_pipeline, request.file_id)

    return {
        "status": "queued",
        "message": "Full RAG pipeline started in background.",
        "file_id": request.file_id
    }

@app.post("/delete-document")
async def delete_document(req: DeleteRequest, x_api_key: str = Header(None)):
    """Removes a file and all its chunks from the Vector DB"""
    if x_api_key != API_SECRET_KEY: raise HTTPException(status_code=403, detail="Invalid Key")
    
    vector_db.delete_file(req.file_id)
    return {"status": "deleted", "file_id": req.file_id}

@app.post("/update-chunk")
async def update_chunk(req: UpdateChunkRequest, x_api_key: str = Header(None)):
    """Updates the text of a specific chunk and recalculates its embedding"""
    if x_api_key != API_SECRET_KEY: raise HTTPException(status_code=403, detail="Invalid Key")
    
    vector_db.update_chunk(req.chunk_id, req.new_text)
    return {"status": "updated", "chunk_id": req.chunk_id}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)