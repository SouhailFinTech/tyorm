import streamlit as st
import cv2
import numpy as np
import yt_dlp
import tempfile
import os
import json
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
import re
import urllib.request

st.set_page_config(page_title="QuantTube Analyzer", page_icon="📈", layout="wide")

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_face_cascade()

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def fetch_video_data(url):
    video_id = extract_video_id(url)
    if not video_id:
        return None, None, None, "Invalid YouTube URL"

    temp_dir = tempfile.gettempdir()

    transcript_text = ""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([t['text'] for t in transcript_list])
    except Exception:
        transcript_text = "No transcript available."

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
    if img is None:
        return {"error": "Failed to decode image"}
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    contrast_score = np.std(l_channel) 
    sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_count = len(faces)
    
    face_centered = False
    if face_count > 0:
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        img_h, img_w = gray.shape
        face_center_x = (x + w/2) / img_w
        face_center_y = (y + h/2) / img_h
        if 0.2 < face_center_x < 0.8 and 0.2 < face_center_y < 0.8:
            face_centered = True

    b, g, r = cv2.split(img)
    vibrancy = np.mean([np.std(b), np.std(g), np.std(r)])

    contrast_norm = min(contrast_score / 50.0, 1.0) * 30
    sharpness_norm = min(sharpness_score / 1000.0, 1.0) * 20
    vibrancy_norm = min(vibrancy / 60.0, 1.0) * 10
    face_score = 30 if face_count > 0 else 0
    center_score = 10 if face_centered else 0
    
    final_score = max(0, min(100, contrast_norm + sharpness_norm + vibrancy_norm + face_score + center_score))

    return {
        "score": round(final_score, 1),
        "contrast": round(contrast_score, 2),
        "sharpness": round(sharpness_score, 2),
        "vibrancy": round(vibrancy, 2),
        "faces": face_count,
        "face_centered": face_centered
    }

def analyze_hook_video(video_path):
    if not video_path or not os.path.exists(video_path):
        return {"error": "Could not load video"}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 
    
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
        return {"error": "No Groq API Key found."}

    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert YouTube strategist for technical/finance channels.
    Analyze this hook transcript and visual pacing.
    
    Visual Pacing: {cpm} Cuts Per Minute.
    Transcript: "{transcript}"
    
    Evaluate 1-10: Pattern Interrupt, Value Proposition, Jargon Density.
    Output STRICT JSON:
    "pattern_interrupt_score" (int),
    "value_prop_score" (int),
    "jargon_score" (int),
    "overall_hook_score" (int 0-100),
    "critique" (string),
    "rewrite_suggestion" (string).
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

st.title("📈 QuantTube Analyzer")
st.markdown("Proprietary CV & NLP pipeline for Algo-Trading YouTube optimization.")

with st.sidebar:
    st.header("⚙️ Settings")
    if "GROQ_API_KEY" not in st.secrets:
        st.warning("No Groq API Key found in Secrets.")

url_input = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
    if not url_input:
        st.error("Please enter a URL.")
    else:
        with st.spinner("Fetching data..."):
            thumb_path, video_path, transcript, error = fetch_video_data(url_input)
            
        if error:
            st.error(error)
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🖼️ Thumbnail Analysis")
                if thumb_path and os.path.exists(thumb_path):
                    st.image(thumb_path, use_column_width=True)
                    
                with st.spinner("Running CV..."):
                    thumb_metrics = analyze_thumbnail(thumb_path)
                    
                if "error" in thumb_metrics:
                    st.error(thumb_metrics["error"])
                else:
                    score = thumb_metrics["score"]
                    st.metric("Thumbnail Score", f"{score}/100", delta="Optimize for CTR")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Contrast", thumb_metrics["contrast"])
                    m2.metric("Sharpness", round(thumb_metrics["sharpness"], 0))
                    m3.metric("Vibrancy", thumb_metrics["vibrancy"])
                    
                    st.write(f"**Faces Detected:** {thumb_metrics['faces']}")
                    if thumb_metrics["faces"] > 0 and not thumb_metrics["face_centered"]:
                        st.warning("⚠️ Face detected, but not centered.")
                    elif thumb_metrics["faces"] > 0:
                        st.success("✅ Face composition is strong.")

            with col2:
                st.subheader("🎬 Hook Analysis (First 30s)")
                
                with st.spinner("Analyzing pacing..."):
                    vid_metrics = analyze_hook_video(video_path)
                    
                if "error" in vid_metrics:
                    st.error(vid_metrics["error"])
                else:
                    cpm = vid_metrics["cpm"]
                    st.metric("Visual Pacing", f"{cpm} Cuts/Min")
                    if cpm < 10:
                        st.warning("⚠️ Pacing too slow. Aim for 15-25 CPM.")
                    else:
                        st.success("✅ Excellent visual retention.")
                
                if "GROQ_API_KEY" in st.secrets and transcript != "No transcript available.":
                    with st.spinner("Running LLM..."):
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
                        st.progress(min(hook_score, 100) / 100)
                        st.caption(f"Overall Hook Score: {hook_score}/100")
                        
                        st.info(f"**AI Critique:** {llm_data.get('critique', 'N/A')}")
                        st.success(f"**Suggested Rewrite:** {llm_data.get('rewrite_suggestion', 'N/A')}")
                else:
                    st.info("Add Groq API key to Secrets to unlock AI script analysis.")

            for f in [thumb_path, video_path]:
                if f and os.path.exists(f):
                    try: os.remove(f)
                    except: pass
