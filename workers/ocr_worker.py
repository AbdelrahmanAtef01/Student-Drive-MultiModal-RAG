import sys
import json
import os
import warnings
from rapidocr_onnxruntime import RapidOCR

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

class PersistentOCR:
    def __init__(self):
        print("   [OCR Worker] Loading RapidOCR (ONNX Production Engine)...", file=sys.stderr)
        
        # Initialize Engine
        try:
            self.engine = RapidOCR()
            print("   [OCR Worker] Ready.", file=sys.stderr)
        except Exception as e:
            print(f"   [OCR Worker] Init Failed: {e}", file=sys.stderr)
            sys.exit(1)

    def process_image(self, image_path):
        if not os.path.exists(image_path):
            return []

        try:
            # Run Inference
            result, _ = self.engine(image_path)
            
            blocks = []
            if result:
                for line in result:
                    # line format: [coords, text, confidence]
                    coords, text, conf = line
                    
                    # Convert coords to integer bbox [x1, y1, x2, y2]
                    x_vals = [int(p[0]) for p in coords]
                    y_vals = [int(p[1]) for p in coords]
                    
                    blocks.append({
                        "bbox": [min(x_vals), min(y_vals), max(x_vals), max(y_vals)],
                        "type": "text",
                        "text": text,
                        "confidence": float(conf)
                    })
            return blocks

        except Exception as e:
            print(f"   [OCR Worker] Processing Error: {e}", file=sys.stderr)
            return []

if __name__ == "__main__":
    worker = PersistentOCR()
    
    # Persistent Loop
    for line in sys.stdin:
        image_path = line.strip()
        if not image_path: continue
        
        if image_path == "EXIT":
            break
            
        data = worker.process_image(image_path)
        
        # Output clean JSON
        print(json.dumps(data, ensure_ascii=False))
        sys.stdout.flush()