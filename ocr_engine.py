import sys
import json
import os
import logging
from paddleocr import PaddleOCR

# Suppress Paddle logs
os.environ["GLOG_minloglevel"] = "3"

# Prevent Paddle and OpenCV prints
logging.getLogger("ppocr").setLevel(logging.ERROR)

def run_ocr(image_path):
    # Initialize Paddle
    ocr = PaddleOCR(use_angle_cls=True, lang='ar', show_log=False)
    
    try:
        result = ocr.ocr(image_path, cls=True)
    except Exception:
        print("[]")
        return

    blocks = []
    if result and result[0]:
        for line in result[0]:
            coords = line[0] 
            text = line[1][0]
            
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)

            blocks.append({
                "bbox": [x1, y1, x2, y2],
                "type": "text",
                "text": text
            })
            
    print(json.dumps(blocks))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[]")
        sys.exit(1)
        
    img_path = sys.argv[1]
    run_ocr(img_path)