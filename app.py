import mediapipe as mp
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import yt_dlp
import tempfile
import os
import json
from groq import Groq
from rapidocr_onnxruntime import RapidOCR
from youtube_transcript_api import YouTubeTranscriptApi
import re
import urllib.request

# --- PAGE CONFIG ---
st.set_page_config(page_title="QuantTube Analyzer", page_icon="📈", layout="wide")

# --- INITIALIZATION ---
@st.cache_resource
def load_ocr():
    return RapidOCR()

@st.cache_resource
def load_mediapipe():
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

ocr_engine = load_ocr()
mp_face_detection = load_mediapipe()

# --- HELPER FUNCTIONS ---

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def fetch_video_data(url):
    video_id = extract_video_id(url)
    if not video_id:
        return None, None, None, "Invalid YouTube URL"

    # Use system temp directory to prevent Streamlit Cloud read-only errors
    temp_dir = tempfile.gettempdir()

    # 1. Get Transcript
    transcript_text = ""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([t['text'] for t in transcript_list])
    except Exception:
        transcript_text = "No transcript available."

    # 2. Download Thumbnail
    thumb_path = os.path.join(temp_dir, "temp_thumb.jpg")
    try:
        ydl_thumb_opts = {'quiet': True, 'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_thumb_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            thumb_url = info.get('thumbnail')
            if thumb_url:
                urllib.request.urlretrieve(thumb_url, thumb_path)
    except Exception:
        thumb_path = None

    # 3. Download Video (Lowest quality to save RAM/Bandwidth)
    video_path = os.path.join(temp_dir, "temp_video.mp4")
    try:
        ydl_vid_opts = {
            'quiet': True, 
            'format': 'worst', 
            'outtmpl': video_path,
        }
        with yt_dlp.YoutubeDL(ydl_vid_opts) as ydl:
            ydl.download([url])
    except Exception:
        video_path = None

    return thumb_path, video_path, transcript_text[:1500], None

def analyze_thumbnail(image_path):
    if not image_path or not os.path.exists(image_path):
        return {"error": "Could not load thumbnail"}

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Contrast & Sharpness (OpenCV LAB)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    contrast_score = np.std(l_channel) 
    sharpness_score = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

    # 2. Face & Pose Detection (MediaPipe)
    results = mp_face_detection.process(img_rgb)
    face_count = 0
    face_centered = False
    if results.detections:
        face_count = len(results.detections)
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            face_center_x = bboxC.xmin + (bboxC.width / 2)
            face_center_y = bboxC.ymin + (bboxC.height / 2)
            if 0.2 < face_center_x < 0.8 and 0.2 < face_center_y < 0.8:
                face_centered = True

    # 3. Text Ratio (RapidOCR)
    text_word_count = 0
    try:
        result, _ = ocr_engine(image_path)
        if result is not None:
            for line in result:
                text_word_count += len(line[1].split())
    except Exception:
        pass

    # Calculate Final Thumbnail Score (0-100)
    contrast_norm = min(contrast_score / 50.0, 1.0) * 30
    sharpness_norm = min(sharpness_score / 1000.0, 1.0) * 20
    face_score = 30 if face_count > 0 else 0
    center_score = 10 if face_centered else 0
    text_penalty = max(0, (text_word_count - 5) * 5) 
    
    final_score = max(0, min(100, (contrast_norm + sharpness_norm + face_score + center_score) - text_penalty))

    return {
        "score": round(final_score, 1),
        "contrast": round(contrast_score, 2),
        "sharpness": round(sharpness_score, 2),
        "faces": face_count,
        "face_centered": face_centered,
        "text_words": text_word_count
    }

def analyze_hook_video(video_path):
    if not video_path or not os.path.exists(video_path):
        return {"error": "Could not load video"}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Fallback
    
    # STRICT RAM SAFEGUARD: Only analyze the first 30 seconds
    max_frames = int(fps * 30)
    sample_rate = max(1, int(fps / 2)) 
    
    cuts = 0
    prev_frame = None
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                if mean_diff > 15.0: 
                    cuts += 1
            prev_frame = gray
        frame_count += 1
        
    cap.release()
    cpm = cuts * 2 
    return {"cuts_detected": cuts, "cpm": cpm}

def analyze_script_with_llm(transcript, cpm):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return {"error": "No Groq API Key found in Streamlit secrets."}

    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert YouTube strategist specializing in highly technical, B2B, and quantitative finance channels.
    Analyze this 30-second hook transcript and the visual pacing data.
    
    Visual Pacing: {cpm} Cuts Per Minute.
    Transcript: "{transcript}"
    
    Evaluate the hook on a scale of 1-10 for:
    1. Pattern Interrupt
    2. Value Proposition
    3. Jargon Density
    
    Output STRICTLY in JSON format with these keys:
    "pattern_interrupt_score" (int),
    "value_prop_score" (int),
    "jargon_score" (int),
    "overall_hook_score" (int 0-100),
    "critique" (string, 2 sentences max),
    "rewrite_suggestion" (string, a punchier 2-sentence rewrite).
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# --- STREAMLIT UI ---

st.title("📈 QuantTube Analyzer")
st.markdown("Proprietary Computer Vision & NLP pipeline for optimizing Algo-Trading & Tech YouTube content.")

with st.sidebar:
    st.header("⚙️ Settings")
    if "GROQ_API_KEY" not in st.secrets:
        st.warning("No Groq API Key found. LLM Hook Analysis will be disabled.")

url_input = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
    if not url_input:
        st.error("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Fetching video data, thumbnails, and transcripts..."):
            thumb_path, video_path, transcript, error = fetch_video_data(url_input)
            
        if error:
            st.error(error)
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🖼️ Thumbnail Analysis")
                if thumb_path and os.path.exists(thumb_path):
                    st.image(thumb_path, use_column_width=True)
                    
                with st.spinner("Running Computer Vision..."):
                    thumb_metrics = analyze_thumbnail(thumb_path)
                    
                if "error" in thumb_metrics:
                    st.error(thumb_metrics["error"])
                else:
                    score = thumb_metrics["score"]
                    st.metric("Thumbnail Score", f"{score}/100", delta="Optimize for CTR")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Contrast", thumb_metrics["contrast"])
                    m2.metric("Sharpness", round(thumb_metrics["sharpness"], 0))
                    m3.metric("Faces", thumb_metrics["faces"])
                    
                    if thumb_metrics["text_words"] > 5:
                        st.warning(f"⚠️ **Text Clutter:** {thumb_metrics['text_words']} words. Keep it under 5.")
                    else:
                        st.success(f"✅ **Text Ratio:** {thumb_metrics['text_words']} words. Perfect.")
                        
                    if thumb_metrics["faces"] > 0 and not thumb_metrics["face_centered"]:
                        st.warning("⚠️ **Composition:** Face detected, but not centered.")

            with col2:
                st.subheader("🎬 Hook Analysis (First 30s)")
                
                with st.spinner("Analyzing visual pacing..."):
                    vid_metrics = analyze_hook_video(video_path)
                    
                if "error" in vid_metrics:
                    st.error(vid_metrics["error"])
                else:
                    cpm = vid_metrics["cpm"]
                    st.metric("Visual Pacing", f"{cpm} Cuts/Min")
                    if cpm < 10:
                        st.warning("⚠️ **Pacing:** Too slow. Aim for 15-25 CPM.")
                    else:
                        st.success("✅ **Pacing:** Excellent visual retention.")
                
                if "GROQ_API_KEY" in st.secrets and transcript != "No transcript available.":
                    with st.spinner("Running LLM on transcript..."):
                        llm_data = analyze_script_with_llm(transcript, cpm)
                        
                    if "error" in llm_data:
                        st.error(llm_data["error"])
                    else:
                        st.markdown("---")
                        s1, s2, s3 = st.columns(3)
                        s1.metric("Pattern Interrupt", f"{llm_data.get('pattern_interrupt_score', 0)}/10")
                        s2.metric("Value Prop", f"{llm_data.get('value_prop_score', 0)}/10")
                        s3.metric("Jargon Control", f"{llm_data.get('jargon_score', 0)}/10")
                        
                        hook_score = llm_data.get("overall_hook_score", 0)
                        st.progress(hook_score / 100)
                        st.caption(f"Overall Hook Score: {hook_score}/100")
                        
                        st.info(f"**AI Critique:** {llm_data.get('critique', 'N/A')}")
                        st.success(f"**Suggested Rewrite:** {llm_data.get('rewrite_suggestion', 'N/A')}")
                else:
                    st.info("Add your free Groq API key to Streamlit secrets to unlock AI script analysis.")

            # Cleanup temp files safely
            for f in [thumb_path, video_path]:
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
