import os
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

load_dotenv()

class VectorDB:
    def __init__(self):
        self.chroma_path = os.getenv("CHROMA_PATH")

        # Initialize Client
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.client.get_or_create_collection(name="rag_collection")
        
        # Gemini Embedding
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.embed_model = os.getenv("EMBEDDING_MODEL_ID", "models/text-embedding-004")
        self.genai_client = genai.Client(api_key=self.api_key)

    def _generate_embeddings(self, texts):
        """
        Generates embeddings in batches of 90 to respect Google API limits.
        """
        BATCH_SIZE = 90
        all_embeddings = []
        
        print(f"   Embedding {len(texts)} texts in batches of {BATCH_SIZE}...")

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            try:
                # API Call
                result = self.genai_client.models.embed_content(
                    model=self.embed_model,
                    contents=batch,
                )
                
                # Extract vectors
                batch_embeddings = [e.values for e in result.embeddings]
                all_embeddings.extend(batch_embeddings)
                
                # Tiny sleep
                time.sleep(0.2)
                
            except Exception as e:
                print(f"   Batch Embedding Error (Indices {i}-{i+len(batch)}): {e}")
                return []

        return all_embeddings

    def add_chunks(self, chunks):
        """Upsert chunks into Chroma"""
        if not chunks: return
        
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # 1. Generate Embeddings
        embeddings = self._generate_embeddings(texts)
        
        # 2. Safety Check
        if len(embeddings) != len(ids):
            print(f"   Critical Error: Sent {len(ids)} texts but got {len(embeddings)} vectors.")
            print("   (Data was NOT saved to Vector DB)")
            return

        # 3. Upsert to Chroma
        DB_BATCH_SIZE = 500
        for i in range(0, len(ids), DB_BATCH_SIZE):
            end = i + DB_BATCH_SIZE
            self.collection.upsert(
                ids=ids[i:end],
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end]
            )
        
        print(f"   Successfully stored {len(chunks)} chunks in Vector DB.")

    def delete_file(self, file_id):
        """Delete all chunks associated with a specific file_id"""
        try:
            self.collection.delete(where={"file_id": file_id})
            print(f"   Cleaned old data for File ID: {file_id}")
        except Exception as e:
            print(f"   Delete Warning: {e}")

    def delete_chunk(self, chunk_id):
        """Delete a single specific chunk"""
        self.collection.delete(ids=[chunk_id])

    def update_chunk(self, chunk_id, new_text):
        """Reembed and update a single chunk"""
        # Recalculate embedding
        try:
            result = self.genai_client.models.embed_content(
                model=self.embed_model,
                contents=[new_text],
            )
            new_embedding = [e.values for e in result.embeddings][0]
            
            # Update in DB
            self.collection.update(
                ids=[chunk_id],
                documents=[new_text],
                embeddings=[new_embedding]
            )
            print(f"   Updated Chunk: {chunk_id}")
        except Exception as e:
            print(f"   Update Failed: {e}")

    def get_stats(self):
        return {"count": self.collection.count()}

    # --- UPDATED QUERY METHOD ---
    def query(self, query_text: str, n_results: int = 5, where: dict = None):
        """
        Semantic search using the same embedding model.
        Accepts optional 'where' dictionary for filtering.
        """
        try:
            # 1. Generate Embedding for the query string
            result = self.genai_client.models.embed_content(
                model=self.embed_model,
                contents=[query_text],
            )
            query_embedding = [e.values for e in result.embeddings][0]

            # 2. Query Chroma using the vector and the filter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )

            # 3. Format results into a clean list of dicts
            formatted_results = []
            
            # Check if any results were found
            if not results['ids'] or not results['ids'][0]:
                return []

            # Loop through the results
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "score": results['distances'][0][i] if 'distances' in results else 0.0
                })
                
            return formatted_results

        except Exception as e:
            print(f"   Search Query Error: {e}")
            return []