import os
import sys
import re
import io
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from pipeline_orchasterator import PipelineOrchestrator

# Setup
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

class DriveProcessor:
    def __init__(self):
        self.creds_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        self.scopes = ['https://www.googleapis.com/auth/drive.readonly']
        
        if not os.path.exists(self.creds_file):
            print(f"Error: Service account file '{self.creds_file}' not found.", file=sys.stderr)
            sys.exit(1)

        try:
            self.creds = service_account.Credentials.from_service_account_file(
                self.creds_file, scopes=self.scopes
            )
            self.service = build('drive', 'v3', credentials=self.creds)
            print("Connected to Google Drive API.", file=sys.stderr)
        except Exception as e:
            print(f"Auth Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Initialize the RAG Pipeline
        #self.orchestrator = PipelineOrchestrator()

    def extract_id_from_url(self, url):
        """Extracts File ID from various Google Drive URL formats."""
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
            r'/open\?id=([a-zA-Z0-9_-]+)',
            r'/document/d/([a-zA-Z0-9_-]+)',
            r'/presentation/d/([a-zA-Z0-9_-]+)'
        ]
        for p in patterns:
            match = re.search(p, url)
            if match:
                return match.group(1)
        return url

    def download_file(self, file_id, output_folder="data_input"):
        try:
            # 1. Get Metadata
            file_meta = self.service.files().get(fileId=file_id).execute()
            name = file_meta.get('name')
            mime_type = file_meta.get('mimeType')
            
            print(f"   Found File: {name} ({mime_type})")
            
            # 2. Determine Download Mode
            request = None
            ext = ""
            
            # A. Native Google Docs (Export)
            if "application/vnd.google-apps.document" in mime_type:
                print("   Exporting Google Doc to PDF...")
                request = self.service.files().export_media(fileId=file_id, mimeType='application/pdf')
                ext = ".pdf"
            elif "application/vnd.google-apps.presentation" in mime_type:
                print("   Exporting Google Slides to PDF...")
                request = self.service.files().export_media(fileId=file_id, mimeType='application/pdf')
                ext = ".pdf"
            elif "application/vnd.google-apps.spreadsheet" in mime_type:
                print("   Exporting Google Sheet to PDF...")
                request = self.service.files().export_media(fileId=file_id, mimeType='application/pdf')
                ext = ".pdf"
            elif "application/vnd.google-apps.folder" in mime_type:
                print("   Cannot download a folder directly. Please provide a file ID.")
                return None
            else:
                # B. Binary Files
                print("   Downloading Binary...")
                request = self.service.files().get_media(fileId=file_id)
                # Keep original extension 
                if "." not in name:
                    if "pdf" in mime_type: ext = ".pdf"
                    elif "mp4" in mime_type: ext = ".mp4"
                    elif "mpeg" in mime_type or "mp3" in mime_type: ext = ".mp3"
                    elif "image" in mime_type: ext = ".jpg"

            # 3. Execute Download
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # Ensure name has extension
            if not name.lower().endswith(ext) and ext:
                name += ext
                
            save_path = os.path.join(output_folder, name)
            
            fh = io.FileIO(save_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            print(f"   Downloaded to: {save_path}")
            return save_path

        except Exception as e:
            print(f"   Drive Error: {e}")
            return None

    def run(self, input_string):
        file_id = self.extract_id_from_url(input_string)
        print(f"Processing Drive ID: {file_id}")
        
        local_path = self.download_file(file_id)
        
        if local_path:
            print(f"Triggering RAG Pipeline for: {local_path}")
            self.orchestrator.process_file(local_path)

if __name__ == "__main__":
    processor = DriveProcessor()
    
    # Input Loop
    print("\n--- Google Drive RAG Processor ---")
    print("Paste a Google Drive Link or ID (or type 'EXIT'):")
    
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue
            
        processor.run(user_input)
        print("\nReady for next file.")