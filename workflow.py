import os
import uuid
import subprocess
from PIL import Image
from langgraph.graph import StateGraph, END
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from graph_state import GraphState
from worker_manager import manager
from layout_engine import HybridLayoutEngine

# Initialize Engine
layout_engine = HybridLayoutEngine()

# --- CONFIG ---
SUPPORTED_AUDIO = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma']
SUPPORTED_VIDEO = ['.mp4', '.mkv', '.mov', '.avi', '.wmv', '.webm']

# --- HELPERS ---
def extract_audio_from_video(video_path):
    audio_path = video_path.rsplit(".", 1)[0] + "_temp_audio.wav"
    try:
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", 
            "-y", audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return audio_path
    except Exception as e:
        print(f"   FFmpeg extraction failed: {e}")
        return None

def get_video_id(url):
    try:
        query = urlparse(url)
        if query.hostname == 'youtu.be': return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch': return parse_qs(query.query).get('v', [None])[0]
            if query.path[:7] == '/embed/': return query.path.split('/')[2]
            if query.path[:3] == '/v/': return query.path.split('/')[2]
    except: pass
    return None

# --- NODES ---
def node_router(state: GraphState):
    file_path = state['file_path']
    if "youtube.com" in file_path or "youtu.be" in file_path:
        return {"file_type": "youtube", "status": "routed"}
    ext = os.path.splitext(file_path)[1].lower()
    if ext in SUPPORTED_AUDIO or ext in SUPPORTED_VIDEO:
        return {"file_type": "media", "status": "routed"}
    else:
        return {"file_type": "document", "status": "routed"}

def node_process_youtube(state: GraphState):
    print(f"   Processing YouTube: {state['file_path']}")
    video_url = state['file_path']
    video_id = get_video_id(video_url)
    
    if not video_id:
        return {"error": "Invalid YouTube URL", "status": "failed"}
    
    try:
        yt_api = YouTubeTranscriptApi()
        transcript = yt_api.fetch(video_id=video_id, languages=["en"])

        # Format segments
        segments = []
        for t in transcript:
            segments.append({
                "start": round(t['start'], 2),
                "end": round(t['start'] + t['duration'], 2),
                "text": t['text']
            })
            
        final_data = [{
            "source": "youtube_transcript",
            "file_name": f"YouTube_{video_id}",
            "url": video_url,
            "video_id": video_id,
            "segments": segments
        }]
        
        return {"rag_ready_data": final_data, "status": "completed"}
        
    except Exception as e:
        print(f"   YouTube Error: {e}")
        return {"error": f"Transcript unavailable: {str(e)}", "status": "failed"}

def node_process_media(state: GraphState):
    file_path = state['file_path']
    ext = os.path.splitext(file_path)[1].lower()
    target_path = file_path
    temp_audio_path = None

    if ext in SUPPORTED_VIDEO:
        temp_audio_path = extract_audio_from_video(file_path)
        if not temp_audio_path: 
            return {"error": "Video extraction failed", "status": "failed"}
        target_path = temp_audio_path
    
    print(f"   Transcribing Audio ({target_path})...")
    result = manager.query("audio", os.path.abspath(target_path))
    
    if temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    if not result or "error" in result:
        return {"error": str(result), "status": "failed"}

    final_data = [{
        "source": "audio_transcription",
        "file_name": os.path.basename(file_path),
        "language": result.get("language"),
        "duration": result.get("duration"),
        "segments": result.get("blocks", [])
    }]
    return {"rag_ready_data": final_data, "status": "completed"}

def node_analyze_layout(state: GraphState):
    print(f"   Running Layout Engine on: {state['file_path']}")
    try:
        results = layout_engine.process_file(state['file_path'], output_root="data_output_final", visualize=True)
        return {"layout_results": results, "status": "analyzed"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

def node_enrich_content(state: GraphState):
    print("   Running Workers on Blocks...")
    layout_data = state['layout_results']
    file_path = state['file_path']
    fname = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join("data_output_final", fname)
    
    final_pages = []

    for page_data in layout_data:
        page_num = page_data['page']
        blocks = page_data['blocks']
        
        cleaned_blocks = []
        for b in blocks:
            if b['type'] in ['text', 'title'] and (not b.get('content') or len(b.get('content').strip()) < 2):
                continue
            cleaned_blocks.append(b)
        
        img_path = os.path.join(output_dir, f"annotated_page_{page_num}.jpg")
        original_img = None
        if os.path.exists(img_path):
            original_img = Image.open(img_path)
        else:
            print(f"   Warning: Image not found {img_path}")
            continue

        for block in cleaned_blocks:
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
                        crop = original_img.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                        crop.save(temp_path)
                        result = manager.query("table", os.path.abspath(temp_path))
                        markdown = result.get("markdown", "") if result else ""
                        
                        if markdown and "[TABLE_" not in markdown:
                            block['content'] = markdown
                            block['action'] = "local_table_reconstructed"
                        else:
                            ocr_results = manager.query("ocr", os.path.abspath(temp_path))
                            if isinstance(ocr_results, list):
                                detected_text = "\n".join([item['text'] for item in ocr_results])
                                block['content'] = detected_text if detected_text.strip() else "[NO_TEXT_FOUND]"
                                block['action'] = "table_fallback_ocr"
                        processed = True

                # B. VLM LOGIC
                elif action == "crop_image" and b_type == 'image':
                    crop = original_img.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                    crop.save(temp_path)
                    result = manager.query("vlm", os.path.abspath(temp_path))
                    if isinstance(result, dict):
                        block['content'] = f"[IMAGE_DESCRIPTION]\n{result.get('description', '')}"
                        block['action'] = "vlm_described"
                    processed = True

                # C. OCR LOGIC
                elif action == "send_to_ocr":
                    crop = original_img.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                    crop.save(temp_path)
                    ocr_results = manager.query("ocr", os.path.abspath(temp_path))
                    if isinstance(ocr_results, list):
                        detected_text = "\n".join([item['text'] for item in ocr_results])
                        if detected_text.strip():
                            block['content'] = detected_text
                            block['action'] = "ocr_extracted"
                        else:
                            block['content'] = "[OCR_NO_TEXT_FOUND]"
                    processed = True
            
            except Exception as e:
                print(f"   Worker Logic Failed on Block: {e}")
            finally:
                if processed and os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass
            
        final_pages.append({"page": page_num, "blocks": cleaned_blocks})

    return {"rag_ready_data": final_pages, "status": "completed"}

# --- CONDITIONAL EDGES ---
def route_next_step(state: GraphState):
    if state['status'] == "failed": return END
    if state['file_type'] == "youtube": return "process_youtube"
    if state['file_type'] == "media": return "process_media"
    return "analyze_layout"

# --- GRAPH BUILD ---
workflow = StateGraph(GraphState)

workflow.add_node("router", node_router)
workflow.add_node("process_media", node_process_media)
workflow.add_node("process_youtube", node_process_youtube)
workflow.add_node("analyze_layout", node_analyze_layout)
workflow.add_node("enrich_content", node_enrich_content)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_next_step,
    {
        "process_media": "process_media",
        "process_youtube": "process_youtube",
        "analyze_layout": "analyze_layout",
        END: END
    }
)

workflow.add_edge("process_media", END)
workflow.add_edge("process_youtube", END)
workflow.add_edge("analyze_layout", "enrich_content")
workflow.add_edge("enrich_content", END)

app_workflow = workflow.compile()