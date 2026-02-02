import sys
import json
import os
import warnings
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from tqdm import tqdm 

# 1. Setup Environment
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("   [Audio Worker] Error: faster-whisper not installed.", file=sys.stderr)
    sys.exit(1)

class AudioWorker:
    def __init__(self):
        # Load Config from .env
        self.model_size = os.getenv("WHISPER_MODEL_SIZE")
        self.device = os.getenv("WHISPER_DEVICE")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE")

        print(f"   [Audio Worker] Loading Whisper {self.model_size}...", file=sys.stderr)
        
        try:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print(f"   [Audio Worker] Ready on {self.device} ({self.compute_type}).", file=sys.stderr)
        except Exception as e:
            print(f"   [Audio Worker] Model Load Error: {e}", file=sys.stderr)
            sys.exit(1)

    def process_audio(self, audio_path):
        if not os.path.exists(audio_path): return {"error": "File not found"}

        try:
           # Start Transcription
            segments, info = self.model.transcribe(audio_path, language="ar", beam_size=5)
            
            print(f"   [Audio Worker] Detected Language: {info.language} | Duration: {round(info.duration, 2)}s", file=sys.stderr)
            
            transcript_blocks = []
            
            # 2. Iterate
            with tqdm(total=round(info.duration), unit="sec", file=sys.stderr, desc="   🎙️ Transcribing") as pbar:
                last_pos = 0
                for segment in segments:
                    transcript_blocks.append({
                        "start": round(segment.start, 2),
                        "end": round(segment.end, 2),
                        "text": segment.text.strip()
                    })
                    current_pos = round(segment.end)
                    update_amount = current_pos - last_pos
                    if update_amount > 0:
                        pbar.update(update_amount)
                        last_pos = current_pos
            
            return {
                "language": info.language,
                "duration": round(info.duration, 2),
                "blocks": transcript_blocks
            }

        except Exception as e:
            print(f"   [Audio Worker] Transcription Error: {e}", file=sys.stderr)
            return {"error": str(e)}

if __name__ == "__main__":
    worker = AudioWorker()
    
    for line in sys.stdin:
        path = line.strip()
        if not path: continue
        if path == "EXIT": break
        
        result = worker.process_audio(path)
        print(json.dumps(result, ensure_ascii=False))
        sys.stdout.flush()