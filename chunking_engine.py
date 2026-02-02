import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SemanticChunker:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

    def process_json(self, json_path, file_id):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = []
        
        # Audio Handling
        if isinstance(data, list) and len(data) > 0 and "source" in data[0] and data[0]["source"] == "audio_transcription":
            return self._process_audio(data, file_id)

        # Document Handling
        for page in data:
            page_num = page.get('page', 1)
            for block in page.get('blocks', []):
                content = block.get('content', '')
                b_type = block.get('type', 'text')
                action = block.get('action', '') 
                
                if not content or len(content) < 5: continue

                # Determine Final Type for RAG
                final_type = "text"
                if b_type == 'table': final_type = "table"
                elif b_type == 'image' or 'vlm_described' in action: final_type = "image_description"
                elif b_type == 'handwritten': final_type = "handwritten_text"
                elif b_type == 'title': final_type = "title"

                # 1. Tables
                if final_type == 'table':
                    chunks.append({
                        "id": f"{file_id}_p{page_num}_{block['bbox'][1]}",
                        "text": content,
                        "metadata": {
                            "file_id": file_id,
                            "page": page_num,
                            "type": "table",
                            "source": "document"
                        }
                    })
                
                # 2. Text/Images/Handwritten (Recursive Split)
                else:
                    split_texts = self.text_splitter.split_text(content)
                    for i, text in enumerate(split_texts):
                        chunks.append({
                            "id": f"{file_id}_p{page_num}_{block['bbox'][1]}_{i}",
                            "text": text,
                            "metadata": {
                                "file_id": file_id,
                                "page": page_num,
                                "type": final_type,
                                "source": "document"
                            }
                        })
        return chunks

    def _process_audio(self, data, file_id):
        chunks = []
        segments = data[0].get("segments", [])
        current_chunk = ""
        start_time = 0.0
        
        for i, seg in enumerate(segments):
            text = seg.get('text', '')
            if not current_chunk: start_time = seg.get('start', 0.0)
            current_chunk += text + " "
            
            if len(current_chunk) > 1000:
                chunks.append({
                    "id": f"{file_id}_audio_{start_time}",
                    "text": current_chunk.strip(),
                    "metadata": {
                        "file_id": file_id,
                        "type": "audio_transcript",
                        "timestamp": start_time,
                        "source": "audio"
                    }
                })
                current_chunk = ""

        if current_chunk:
             chunks.append({
                "id": f"{file_id}_audio_{start_time}",
                "text": current_chunk.strip(),
                "metadata": {
                    "file_id": file_id,
                    "type": "audio_transcript",
                    "timestamp": start_time,
                    "source": "audio"
                }
            })
        return chunks