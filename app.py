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

# --- PAGE CONFIG ---
st.set_page_config(page_title="QuantTube Analyzer", page_icon="📈", layout="wide")

# --- INITIALIZATION ---
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_face_cascade()

# --- HELPER FUNCTIONS ---
def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def fetch_thumbnail_and_transcript(url):
    video_id = extract_video_id(url)
    if not video_id:
        return None, None, "Invalid YouTube URL"

    temp_dir = tempfile.gettempdir()

    # 1. Get Transcript
    transcript_text = ""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([t['text'] for t in transcript_list])
    except Exception:
        transcript_text = "No transcript available."

    # 2. Download Thumbnail
    thumb_path = os.path.join(temp_dir, f"thumb_{video_id}.jpg")
    try:
        ydl_thumb_opts = {'quiet': True, 'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_thumb_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            thumb_url = info.get('thumbnail')
            if thumb_url:
                urllib.request.urlretrieve(thumb_url, thumb_path)
    except Exception:
        thumb_path = None

    return thumb_path, transcript_text[:1500], None

def analyze_thumbnail(image_path):
    if not image_path or not os.path.exists(image_path):
        return {"error": "Could not load thumbnail"}

    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Failed to decode image"}
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    contrast_score = float(np.std(l_channel))
    sharpness_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

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
    vibrancy = float(np.mean([np.std(b), np.std(g), np.std(r)]))

    contrast_norm = min(contrast_score / 50.0, 1.0) * 30
    sharpness_norm = min(sharpness_score / 2000.0, 1.0) * 20
    vibrancy_norm = min(vibrancy / 60.0, 1.0) * 10
    face_score = 30 if face_count > 0 else 0
    center_score = 10 if face_centered else 0
    
    final_score = int(round(max(0, min(100, contrast_norm + sharpness_norm + vibrancy_norm + face_score + center_score))))

    return {
        "score": final_score,
        "contrast": round(contrast_score, 1),
        "sharpness": round(sharpness_score, 0),
        "vibrancy": round(vibrancy, 1),
        "faces": face_count,
        "face_centered": face_centered
    }

def analyze_hook_video(video_path):
    if not video_path or not os.path.exists(video_path):
        return {"error": "Could not load video file"}

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video"}
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): 
            fps = 30
        
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
    except Exception as e:
        return {"error": f"Video analysis failed: {str(e)[:100]}"}

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

# --- UI ---
st.title("📈 QuantTube Analyzer")
st.markdown("Proprietary CV & NLP pipeline for Algo-Trading YouTube optimization.")

with st.sidebar:
    st.header("⚙️ Settings")
    if "GROQ_API_KEY" not in st.secrets:
        st.warning("No Groq API Key in Secrets.")

# INPUT SECTION
col_url, col_upload = st.columns(2)

with col_url:
    st.subheader("1. YouTube URL")
    st.caption("Used for Thumbnail & Auto-Fetch Transcript")
    url_input = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")

with col_upload:
    st.subheader("2. Video File")
    st.caption("Upload MP4 for Hook Analysis")
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], label_visibility="collapsed")

# MANUAL TRANSCRIPT FALLBACK
st.subheader("3. Transcript (Optional)")
st.caption("Paste your script here if auto-fetch fails")
manual_transcript = st.text_area("Or paste your hook script manually (first 30 seconds)", 
                                  height=100, 
                                  placeholder="In this video, I'm going to show you how to validate Bitcoin trading strategies in just 8 minutes...")

if st.button("🚀 Analyze", type="primary", use_container_width=True):
    if not url_input and not uploaded_file and not manual_transcript:
        st.error("Please provide a YouTube URL, upload a video, or paste a transcript.")
    else:
        with st.spinner("Processing..."):
            # Handle URL Data
            thumb_path = None
            transcript = ""
            if url_input:
                thumb_path, transcript, url_error = fetch_thumbnail_and_transcript(url_input)
                if url_error:
                    st.error(url_error)

            # Handle Uploaded Video
            video_path = None
            if uploaded_file is not None:
                temp_dir = tempfile.gettempdir()
                video_path = os.path.join(temp_dir, "uploaded_hook_video.mp4")
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

        # DISPLAY RESULTS
        col1, col2 = st.columns(2)
        
        # THUMBNAIL
        with col1:
            st.subheader("🖼️ Thumbnail Analysis")
            if thumb_path and os.path.exists(thumb_path):
                st.image(thumb_path, use_column_width=True)
                with st.spinner("Analyzing thumbnail..."):
                    thumb_metrics = analyze_thumbnail(thumb_path)
                    
                if "error" in thumb_metrics:
                    st.error(thumb_metrics["error"])
                else:
                    score = thumb_metrics["score"]
                    st.metric("Thumbnail Score", f"{score}/100", delta="Optimize for CTR")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Contrast", thumb_metrics["contrast"])
                    m2.metric("Sharpness", thumb_metrics["sharpness"])
                    m3.metric("Vibrancy", thumb_metrics["vibrancy"])
                    
                    st.write(f"**Faces Detected:** {thumb_metrics['faces']}")
                    if thumb_metrics["faces"] > 0 and not thumb_metrics["face_centered"]:
                        st.warning("⚠️ Face detected, but not centered.")
                    elif thumb_metrics["faces"] > 0:
                        st.success("✅ Face composition is strong.")
            else:
                st.info("Provide a YouTube URL to analyze the thumbnail.")

        # HOOK
        with col2:
            st.subheader("🎬 Hook Analysis (First 30s)")
            
            if video_path and os.path.exists(video_path):
                with st.spinner("Analyzing video pacing..."):
                    vid_metrics = analyze_hook_video(video_path)
                    
                if "error" in vid_metrics:
                    st.error(vid_metrics["error"])
                else:
                    cpm = vid_metrics["cpm"]
                    st.metric("Visual Pacing", f"{cpm} Cuts/Min")
                    
                    # Niche-specific feedback
                    if cpm < 5:
                        st.error("⚠️ **Very Slow:** Consider adding B-roll or zoom cuts every 5-6 seconds")
                    elif cpm < 10:
                        st.success("✅ **Good for Technical Content:** Perfect pace for algo-trading education")
                    elif cpm < 20:
                        st.success("✅ **Excellent Pacing:** High energy while maintaining clarity")
                    else:
                        st.warning("⚠️ **Very Fast:** Ensure viewers can follow the technical details")
                    
                    # Determine which transcript to use
                    final_transcript = manual_transcript if manual_transcript else transcript
                    
                    # LLM Analysis with specific error handling
                    if "GROQ_API_KEY" not in st.secrets:
                        st.warning("⚠️ **Missing API Key:** Add your Groq API key to Streamlit Secrets.")
                    elif not final_transcript or final_transcript == "No transcript available.":
                        st.warning("⚠️ **Missing Transcript:** Either paste the YouTube URL in Box 1 OR manually paste your script in Box 3.")
                    else:
                        with st.spinner("Running AI script analysis..."):
                            llm_data = analyze_script_with_llm(final_transcript, cpm)
                            
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
                st.info("Upload an MP4 file to analyze the video hook.")

        # Cleanup
        for f in [thumb_path, video_path]:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass
