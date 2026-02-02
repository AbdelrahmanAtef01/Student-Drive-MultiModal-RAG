import os
import json
import subprocess
import sys
from PIL import Image
import uuid
from layout_engine import HybridLayoutEngine

class PipelineOrchestrator:
    def __init__(self):
        self.layout_engine = HybridLayoutEngine()
        self.vlm_process = None
        self.ocr_process = None
        self.table_process = None
        self.audio_process = None
        
        # --- CONFIG: Supported File Types ---
        self.SUPPORTED_AUDIO = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma']
        self.SUPPORTED_VIDEO = ['.mp4', '.mkv', '.mov', '.avi', '.wmv', '.webm']

    def start_workers(self):
        print("\n Starting Workers...")
        
        # 1. VLM
        self.vlm_process = subprocess.Popen(
            [sys.executable, "workers/vlm_worker.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, encoding='utf-8', bufsize=1
        )

        # 2. OCR
        self.ocr_process = subprocess.Popen(
            [sys.executable, "workers/ocr_worker.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, encoding='utf-8', bufsize=1
        )

        # 3. Table worker
        self.table_process = subprocess.Popen(
            [sys.executable, "workers/table_worker.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, encoding='utf-8', bufsize=1
        )

        # 4. Audio Worker
        self.audio_process = subprocess.Popen(
            [sys.executable, "workers/audio_worker.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, encoding='utf-8', bufsize=1
        )

    def stop_workers(self):
        print("\n Stopping Workers...")
        for proc in [self.vlm_process, self.ocr_process, self.table_process, self.audio_process]:
            if proc:
                try:
                    proc.stdin.write("EXIT\n")
                    proc.stdin.flush()
                    proc.wait(timeout=2)
                except:
                    proc.kill()

    def query_worker(self, process, data, worker_name="Worker"):
        if not process: return None
        try:
            if isinstance(data, (dict, list)): msg = json.dumps(data, ensure_ascii=False)
            else: msg = str(data)
                
            process.stdin.write(f"{msg}\n")
            process.stdin.flush()
            response = process.stdout.readline()
            if response: return json.loads(response)
        except Exception as e:
            print(f"{worker_name} Error: {e}")
        return None

    # ============================
    #      AUDIO / VIDEO LOGIC
    # ============================
    def extract_audio_from_video(self, video_path):
        """Uses FFmpeg to rip audio from video to a temp WAV file"""
        audio_path = video_path.rsplit(".", 1)[0] + "_temp_audio.wav"
        print(f"   Extracting audio from video: {video_path}...")
        try:
            # fast extract, mono, 16k Hz
            subprocess.run([
                "ffmpeg", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", 
                "-y", audio_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return audio_path
        except Exception as e:
            print(f"   FFmpeg extraction failed: {e}")
            return None

    def process_media(self, file_path, output_root):
        file_ext = os.path.splitext(file_path)[1].lower()
        temp_audio_path = None
        target_path = file_path

        # 1. Handle Video -> Audio Conversion
        if file_ext in self.SUPPORTED_VIDEO:
            temp_audio_path = self.extract_audio_from_video(file_path)
            if not temp_audio_path: return []
            target_path = temp_audio_path
        
        # 2. Transcribe
        result = self.query_worker(self.audio_process, os.path.abspath(target_path), "Whisper")
        
        # 3. Cleanup Temp File
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        if not result or "error" in result:
            print(f"   Transcription Failed: {result.get('error') if result else 'Unknown'}")
            return []

        # 4. Format for RAG
        final_data = [{
            "source": "audio_transcription",
            "file_name": os.path.basename(file_path),
            "language": result.get("language"),
            "duration": result.get("duration"),
            "segments": result.get("blocks", [])
        }]
        
        return final_data

    # ============================
    #      DOCUMENT LOGIC
    # ============================
    def run_workers_on_blocks(self, blocks, original_image):
        for block in blocks:
            action = block.get('action')
            bbox = block['bbox']
            b_type = block.get('type')
            content = block.get('content', '')

            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]: continue
            temp_path = f"temp_worker_{uuid.uuid4()}.jpg"
            processed = False

            try:
                # A. TABLE LOGIC
                if b_type == 'table':
                    is_empty = (content is None) or (content.strip() == "")
                    is_binary = "[IMAGE_BINARY]" in content
                    if is_empty or is_binary:
                        crop = original_image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                        crop.save(temp_path)
                        result = self.query_worker(self.table_process, os.path.abspath(temp_path), "Table")
                        markdown = result.get("markdown", "") if result else ""
                        if markdown and "[TABLE_" not in markdown:
                            block['content'] = markdown
                            block['action'] = "local_table_reconstructed"
                        else:
                            ocr_results = self.query_worker(self.ocr_process, os.path.abspath(temp_path), "OCR")
                            if isinstance(ocr_results, list):
                                detected_text = "\n".join([item['text'] for item in ocr_results])
                                block['content'] = detected_text if detected_text.strip() else "[NO_TEXT_FOUND]"
                                block['action'] = "table_fallback_ocr"
                        processed = True

                # B. VLM LOGIC
                elif action == "crop_image" and b_type == 'image':
                    crop = original_image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                    crop.save(temp_path)
                    result = self.query_worker(self.vlm_process, os.path.abspath(temp_path), "VLM")
                    if isinstance(result, dict):
                        block['content'] = f"[IMAGE_DESCRIPTION]\n{result.get('description', '')}"
                        block['action'] = "vlm_described"
                    processed = True

                # C. OCR LOGIC
                elif action == "send_to_ocr":
                    crop = original_image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                    crop.save(temp_path)
                    ocr_results = self.query_worker(self.ocr_process, os.path.abspath(temp_path), "OCR")
                    if isinstance(ocr_results, list):
                        detected_text = "\n".join([item['text'] for item in ocr_results])
                        if detected_text.strip():
                            block['content'] = detected_text
                            block['action'] = "ocr_extracted"
                        else:
                            block['content'] = "[OCR_NO_TEXT_FOUND]"
                    processed = True
            except Exception as e:
                print(f"   Worker Failed on Block: {e}")
            finally:
                if processed and os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass
        return blocks

    def process_file(self, file_path, output_root="data_output_final"):
        print(f"\n Pipeline Started: {file_path}")
        self.start_workers()

        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            fname = os.path.splitext(os.path.basename(file_path))[0]
            save_path = os.path.join(output_root, fname, "rag_ready_data.json")
            
            # --- ROUTING LOGIC ---
            if file_ext in self.SUPPORTED_AUDIO or file_ext in self.SUPPORTED_VIDEO:
                # AUDIO/VIDEO PIPELINE
                if not os.path.exists(os.path.join(output_root, fname)):
                    os.makedirs(os.path.join(output_root, fname))
                
                final_data = self.process_media(file_path, output_root)
                
            else:
                # DOCUMENT PIPELINE (PDF, PPTX, Images)
                layout_results = self.layout_engine.process_file(file_path, output_root, visualize=True)
                final_data = []

                for page_data in layout_results:
                    page_num = page_data['page']
                    blocks = page_data['blocks']
                    print(f"   --- Post-Processing Page {page_num} ---")
                    
                    cleaned_blocks = []
                    for b in blocks:
                        if b['type'] in ['text', 'title'] and (not b.get('content') or len(b.get('content').strip()) < 2):
                            continue
                        cleaned_blocks.append(b)
                    
                    img_path = os.path.join(output_root, fname, f"annotated_page_{page_num}.jpg")
                    if os.path.exists(img_path):
                        original_img = Image.open(img_path)
                        blocks = self.run_workers_on_blocks(cleaned_blocks, original_img)
                    
                    final_data.append({"page": page_num, "blocks": blocks})

            # Save Final JSON
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Done! Data saved to: {save_path}")

        finally:
            self.stop_workers()

if __name__ == "__main__":
    orchestrator = PipelineOrchestrator()
    orchestrator.process_file("data_input/2026-01-30 19-45-54.mp4")