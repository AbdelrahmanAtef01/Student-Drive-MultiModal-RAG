import sys
import torch
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import json
import warnings
from dotenv import load_dotenv

# 1. Force UTF-8 on stdout
sys.stdout.reconfigure(encoding='utf-8')

# 2. Suppress Warnings
warnings.filterwarnings("ignore")

# 3. Load Environment Variables
load_dotenv()

class FlorenceVLM:
    def __init__(self):
        # Config from .env
        self.model_id = os.getenv("VLM_MODEL_ID")
        env_device = os.getenv("VLM_DEVICE")
        self.device = env_device if torch.cuda.is_available() and env_device == "cuda" else "cpu"
        
        # FORCE FLOAT32 for maximum stability on Windows
        self.torch_dtype = torch.float32

        print(f"   [VLM Worker] Loading Foundation Model: {self.model_id}...", file=sys.stderr)
        print(f"   [VLM Worker] Device: {self.device} (Stable FP32 Mode)", file=sys.stderr)

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        
        # Load model in standard precision
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            attn_implementation="eager"
        ).to(self.device)

    def describe_image(self, image_path):
        try:
            if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                return "[Error: Image file missing or empty]"

            image = Image.open(image_path).convert("RGB")
            
            if image.width < 10 or image.height < 10:
                return "[Error: Image too small]"

            task_prompt = "<MORE_DETAILED_CAPTION>"
            
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # STABLE GENERATION
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1,
                use_cache=False 
            )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            # Manual Cleanup
            description = generated_text.replace("<s>", "").replace("</s>", "").replace(task_prompt, "").strip()
            return description

        except Exception as e:
            return f"VLM Error: {str(e)}"

if __name__ == "__main__":
    vlm = FlorenceVLM()
    print("   [VLM Worker] Ready for inference.", file=sys.stderr)
    
    for line in sys.stdin:
        image_path = line.strip()
        if not image_path: continue
        
        if image_path == "EXIT":
            break
            
        description = vlm.describe_image(image_path)
        
        result = json.dumps({"path": image_path, "description": description})
        print(result)
        sys.stdout.flush()