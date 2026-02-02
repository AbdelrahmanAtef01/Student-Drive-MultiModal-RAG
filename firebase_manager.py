import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os

# Initialization
if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS"))
    firebase_admin.initialize_app(cred)

db = firestore.client()

class FirebaseManager:
    
    def get_user_role(self, user_id: str):
        """Fetches user role for permission checks."""
        doc = db.collection('users').document(user_id).get()
        if doc.exists:
            return doc.to_dict().get('role', 'student')
        return 'student'

    def get_chat_history(self, session_id: str, limit: int = 10):
        """
        Retrieves the last N messages for context window.
        """
        messages_ref = db.collection('chat_sessions').document(session_id).collection('messages')
        
        # Get newest 10 (Descending)
        query = messages_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
        docs = query.stream()
        
        history = []
        for doc in docs:
            data = doc.to_dict()
            history.append({
                "role": data.get("role"),
                "content": data.get("content")
            })
            
        return history[::-1] 

    def save_message(self, session_id: str, role: str, content: str):
        """Saves a message to Firestore."""
        session_ref = db.collection('chat_sessions').document(session_id)
        
        # 1. Add to sub collection
        message_data = {
            "role": role,
            "content": content,
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        session_ref.collection('messages').add(message_data)
        
        # 2. Update parent session
        session_ref.set({
            "last_message": content[:50] + "..." if len(content) > 50 else content,
            "updated_at": firestore.SERVER_TIMESTAMP
        }, merge=True)

    def log_ingestion(self, file_id: str, metadata: dict):
        """Logs a new file to the Library."""
        db.collection('library').document(file_id).set({
            "uploaded_at": firestore.SERVER_TIMESTAMP,
            **metadata
        })