from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import shutil
import uuid
from rag import ingest_document, chat_with_context, get_all_sources, delete_source

app = FastAPI(title="StudySnap AI Backend")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []
    selected_source_ids: Optional[List[str]] = None

class UrlRequest(BaseModel):
    url: str

@app.post("/api/sources/upload")
async def upload_sources(files: List[UploadFile] = File(...)):
    """Upload multiple files, process them, and store embeddings in ChromaDB."""
    uploaded_sources = []
    for file in files:
        try:
            file_id = str(uuid.uuid4())
            file_path = f"uploads/{file_id}_{file.filename}"
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # Process and ingest document into ChromaDB
            doc_meta = ingest_document(file_path, file.filename, file_id)
            uploaded_sources.append(doc_meta)
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            # Continue with other files
            pass
            
    if not uploaded_sources:
        raise HTTPException(status_code=400, detail="Failed to process any of the uploaded files.")
        
    return {"success": True, "sources": uploaded_sources}

@app.post("/api/sources/url")
async def add_url_source(request: UrlRequest):
    """Add a URL or YouTube link as a source."""
    try:
        url = request.url
        file_id = str(uuid.uuid4())
        # Pass the URL directly as the filename, file_path can be empty or None since the function handles it
        doc_meta = ingest_document("", url, file_id)
        return {"success": True, "source": doc_meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sources")
async def list_sources():
    """List all ingested sources from the database."""
    sources = get_all_sources()
    return {"sources": sources}

@app.delete("/api/sources/{file_id}")
async def remove_source(file_id: str):
    """Delete a source from ChromaDB."""
    try:
        delete_source(file_id)
        # We should also try to delete the local file if we stored it
        for filename in os.listdir("uploads"):
            if filename.startswith(file_id):
                os.remove(os.path.join("uploads", filename))
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with the AI using context retrieved from ChromaDB via LM Studio."""
    try:
        response_text, sources_used = await chat_with_context(request.message, request.history, request.selected_source_ids)
        return {"response": response_text, "sources": sources_used}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
