import sys
import json
import os
import warnings
import traceback
import torch
import pandas as pd
from PIL import Image
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from rapidocr_onnxruntime import RapidOCR
from dotenv import load_dotenv
import tabulate

# 1. Setup Environment
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

class TableWorker:
    def __init__(self):
        self.model_name = os.getenv("TABLE_MODEL_ID")
        
        print(f"   [Table Worker] Loading {self.model_name}...", file=sys.stderr)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = TableTransformerForObjectDetection.from_pretrained(self.model_name).to(self.device)
        except Exception:
            print(f"   [Table Worker] Model Load Failed!", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
            
        self.ocr_engine = RapidOCR()
        print(f"   [Table Worker] Ready on {self.device}.", file=sys.stderr)

    def get_structure(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
        return results

    def process_table(self, image_path):
        if not os.path.exists(image_path): 
            print(f"   [Table Worker] Image not found: {image_path}", file=sys.stderr)
            return {"markdown": ""}

        try:
            image = Image.open(image_path).convert("RGB")
            
            # A. Get Structure
            structure = self.get_structure(image)
            rows = []
            cols = []
            
            for label, box in zip(structure["labels"], structure["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                label_name = self.model.config.id2label[label.item()]
                if label_name == 'table row': rows.append(box)
                elif label_name == 'table column': cols.append(box)
            
            rows.sort(key=lambda x: x[1])
            cols.sort(key=lambda x: x[0])

            if not rows or not cols:
                return {"markdown": "[TABLE_STRUCTURE_NOT_DETECTED]"}

            # B. Get Text
            ocr_result, _ = self.ocr_engine(image_path)
            if not ocr_result: return {"markdown": ""}

            # C. Fusion
            grid = [["" for _ in range(len(cols))] for _ in range(len(rows))]
            
            for item in ocr_result:
                coords, text, _ = item
                cx = (coords[0][0] + coords[2][0]) / 2
                cy = (coords[0][1] + coords[2][1]) / 2
                
                r_idx = -1
                for i, (rx1, ry1, rx2, ry2) in enumerate(rows):
                    if ry1 < cy < ry2:
                        r_idx = i; break
                
                c_idx = -1
                for j, (cx1, cy1, cx2, cy2) in enumerate(cols):
                    if cx1 < cx < cx2:
                        c_idx = j; break
                
                if r_idx != -1 and c_idx != -1:
                    grid[r_idx][c_idx] += text + " "

            # D. Convert to Markdown
            df = pd.DataFrame(grid)
  
            try:
                markdown = df.to_markdown(index=False, headers=[])
            except TypeError:
                markdown = df.to_markdown(index=False, header=False)
            
            return {"markdown": markdown}

        except Exception:
            print(f"\n   [Table Worker] ERROR PROCESSING {image_path}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {"markdown": "[TABLE_RECONSTRUCTION_FAILED]"}

if __name__ == "__main__":
    worker = TableWorker()
    
    for line in sys.stdin:
        path = line.strip()
        if not path: continue
        if path == "EXIT": break
        
        result = worker.process_table(path)
        print(json.dumps(result, ensure_ascii=False))
        sys.stdout.flush()