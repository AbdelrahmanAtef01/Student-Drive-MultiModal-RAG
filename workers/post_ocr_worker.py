import sys
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Setup Environment
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

class PostOCRWorker:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_id = os.getenv("GEMINI_MODEL_ID")
        
        if not self.api_key:
            print("   [Post-OCR] Error: GEMINI_API_KEY not found in .env", file=sys.stderr)
            sys.exit(1)

        print(f"   [Post-OCR] Initializing Gemini Client ({self.model_id})...", file=sys.stderr)
        self.client = genai.Client(api_key=self.api_key)

    def refine_text(self, blocks):
        if not blocks:
            return []

        # 1. Extract raw text to save tokens
        raw_lines = [b.get('text', '') for b in blocks if b.get('type') == 'text']
        full_text = "\n".join(raw_lines)

        if not full_text.strip():
            return blocks

        # 2. Construct Prompt
        # We ask Gemini to fix the text but KEEP the line structure so we can map it back 
        # (or just return the cleaned full text as a summary).
        prompt = f"""
        You are a post processing AI for an OCR system. 
        Your task is to correct the following text extracted from a document (Arabic/English).
        
        Rules:
        1. Fix spelling errors and OCR artifacts (e.g., '1l' -> 'll', 'rn' -> 'm').
        2. Fix Arabic/English script direction issues if present.
        3. Merge broken words that were split across lines.
        4. Return ONLY the cleaned text. Do not add markdown or explanations.
        
        Raw Text:
        {full_text}
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1
                )
            )
            
            cleaned_text = response.text.strip()
            
            # 3. Create a Refined Block
            refined_block = {
                "type": "refined_text",
                "bbox": [0, 0, 0, 0],
                "content": cleaned_text,
                "source": "gemini_2.5_flash"
            }
            
            blocks.append(refined_block)
            return blocks

        except Exception as e:
            print(f"   [Post-OCR] API Error: {e}", file=sys.stderr)
            return blocks

if __name__ == "__main__":
    worker = PostOCRWorker()
    print("   [Post-OCR] Ready.", file=sys.stderr)
    
    # Read JSON from stdin
    for line in sys.stdin:
        input_data = line.strip()
        if not input_data: continue
        if input_data == "EXIT": break
        
        try:
            blocks = json.loads(input_data)
            refined_blocks = worker.refine_text(blocks)
            print(json.dumps(refined_blocks, ensure_ascii=False))
            sys.stdout.flush()
        except json.JSONDecodeError:
            continue