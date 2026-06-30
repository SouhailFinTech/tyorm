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
st.set_page_config(page_title="QuantTube Analyzer Pro", page_icon="📈", layout="wide")

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

    transcript_text = ""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([t['text'] for t in transcript_list])
    except Exception:
        transcript_text = "No transcript available."

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

    # AUTO-CROP BLACK BARS
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    non_zero = cv2.findNonZero(thresh)
    if non_zero is not None:
        x, y, w, h = cv2.boundingRect(non_zero)
        img = img[y:y+h, x:x+w]
        gray = gray[y:y+h, x:x+w]

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
            if not ret: break
                
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
        return {"cuts_detected": cuts, "cpm": cuts * 2}
    except Exception as e:
        return {"error": f"Video analysis failed: {str(e)[:100]}"}

def detect_boring_signals(video_path):
    if not video_path or not os.path.exists(video_path):
        return {"error": "No video file"}

    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        max_frames = int(fps * 30)
        sample_rate = 3 
        
        stagnant_count = 0
        total_comparisons = 0
        motion_scores = []
        prev_frame = None
        frame_count = 0

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret: break

            if frame_count % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 180)) 

                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    motion = np.mean(diff)
                    motion_scores.append(motion)

                    if motion < 8.0: 
                        stagnant_count += 1
                    total_comparisons += 1

                prev_frame = gray
            frame_count += 1

        cap.release()

        if total_comparisons == 0:
            return {"boring_score": 50, "stagnation_rate": 0, "avg_motion": 0, "is_boring": False, "verdict": "Could not analyze."}

        stagnation_rate = (stagnant_count / total_comparisons) * 100
        avg_motion = np.mean(motion_scores)
        motion_penalty = max(0, 15 - avg_motion) * 3 
        boring_score = int(min(100, (stagnation_rate * 0.5) + motion_penalty))
        is_boring = boring_score > 50

        return {
            "boring_score": boring_score,
            "stagnation_rate": round(stagnation_rate, 1),
            "avg_motion": round(avg_motion, 2),
            "is_boring": is_boring,
            "verdict": "⚠️ BORING - Add visual variety" if is_boring else "✅ ENGAGING - Good visual dynamics"
        }
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)[:100]}"}

def analyze_title_with_llm(title, transcript, topic):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are a YouTube SEO expert for technical/finance content.
    Current Title: "{title}"
    Topic: {topic}
    Transcript Snippet: "{transcript[:300]}"
    
    Output STRICT JSON:
    "title_score" (int 0-100),
    "character_count" (int),
    "is_optimal_length" (bool),
    "alternative_titles" (array of 3 strings),
    "recommended_keywords" (array of 5 strings),
    "emotional_triggers" (string),
    "improvement_notes" (string)
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def generate_thumbnail_brief(title, transcript, topic):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are a YouTube thumbnail designer expert for technical/finance channels.
    Title: "{title}"
    Topic: {topic}
    Transcript Snippet: "{transcript[:300]}"
    
    Output STRICT JSON:
    "thumbnail_text" (string, max 5 words),
    "color_scheme" (object with background, text, accent hex codes),
    "layout" (string description),
    "visual_elements" (array of strings),
    "midjourney_prompt" (string, detailed),
    "style" (string),
    "dos" (array of 3 strings),
    "donts" (array of 3 strings),
    "thumbnail_score_prediction" (int 0-100)
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def analyze_script_with_llm(transcript, cpm):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert YouTube strategist for technical/finance channels.
    Visual Pacing: {cpm} Cuts Per Minute.
    Transcript: "{transcript}"
    
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
st.title("📈 QuantTube Analyzer Pro")
st.markdown("Proprietary CV & NLP pipeline for Algo-Trading YouTube optimization.")

with st.sidebar:
    st.header("⚙️ Settings")
    if "GROQ_API_KEY" not in st.secrets:
        st.warning("No Groq API Key in Secrets.")
    st.markdown("---")
    st.info("**Pro Features:**\n- Title Optimizer\n- Thumbnail Brief\n- A/B Thumbnail Comparator")

# INPUT SECTION
st.subheader("📥 Inputs")
col_url, col_upload = st.columns(2)

with col_url:
    url_input = st.text_input("1. YouTube URL (For Original Thumb & Transcript)", placeholder="https://www.youtube.com/watch?v=...")

with col_upload:
    uploaded_file = st.file_uploader("2. Video File (For Hook & Boring Analysis)", type=["mp4", "mov", "avi"])

col_title, col_topic = st.columns(2)
with col_title:
    title_input = st.text_input("3. Video Title", placeholder="e.g., How I Backtested Bitcoin Strategies")
with col_topic:
    topic_input = st.text_input("4. Main Topic/Keyword", placeholder="e.g., Bitcoin backtesting, Python algo")

st.subheader(" Thumbnail A/B Testing")
new_thumb_file = st.file_uploader("5. Upload your NEW/AI-Generated Thumbnail to compare", type=["jpg", "png", "jpeg"])

st.subheader("📝 Transcript (Optional)")
manual_transcript = st.text_area("Paste hook script if auto-fetch fails", height=80)

# BUTTONS
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    run_analysis = st.button("🚀 Full Analysis", type="primary", use_container_width=True)
with col_btn2:
    quick_thumb = st.button("🎨 Thumbnail Brief Only", use_container_width=True)

if run_analysis or quick_thumb:
    if not title_input and not quick_thumb:
        st.error("Please enter a video title.")
    elif not topic_input and not quick_thumb:
        st.error("Please enter the main topic.")
    elif not url_input and not uploaded_file and not manual_transcript and not new_thumb_file:
        st.error("Please provide at least one input.")
    else:
        with st.spinner("Processing..."):
            thumb_path = None
            transcript = ""
            if url_input:
                thumb_path, transcript, url_error = fetch_thumbnail_and_transcript(url_input)
                if url_error: st.error(url_error)

            video_path = None
            if uploaded_file:
                temp_dir = tempfile.gettempdir()
                video_path = os.path.join(temp_dir, "uploaded_hook_video.mp4")
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            new_thumb_path = None
            if new_thumb_file:
                temp_dir = tempfile.gettempdir()
                new_thumb_path = os.path.join(temp_dir, "new_thumb_comparison.jpg")
                with open(new_thumb_path, "wb") as f:
                    f.write(new_thumb_file.getbuffer())

        final_transcript = manual_transcript if manual_transcript else transcript
        
        # === THUMBNAIL A/B COMPARATOR ===
        st.markdown("---")
        st.subheader("🖼️ Thumbnail A/B Comparator")
        
        orig_metrics = None
        new_metrics = None
        
        if thumb_path and os.path.exists(thumb_path):
            orig_metrics = analyze_thumbnail(thumb_path)
        if new_thumb_path and os.path.exists(new_thumb_path):
            new_metrics = analyze_thumbnail(new_thumb_path)

        if orig_metrics and new_metrics:
            col_orig, col_new = st.columns(2)
            
            with col_orig:
                st.markdown("#### 🅰️ Original Thumbnail")
                st.image(thumb_path, use_column_width=True)
                st.metric("Score", f"{orig_metrics['score']}/100")
                st.write(f"Contrast: {orig_metrics['contrast']} | Sharpness: {orig_metrics['sharpness']}")
                st.write(f"Faces: {orig_metrics['faces']}")
                
            with col_new:
                st.markdown("#### ️ New/AI Thumbnail")
                st.image(new_thumb_path, use_column_width=True)
                
                # Calculate Deltas
                score_delta = new_metrics['score'] - orig_metrics['score']
                contrast_delta = round(new_metrics['contrast'] - orig_metrics['contrast'], 1)
                
                st.metric("Score", f"{new_metrics['score']}/100", delta=f"{score_delta} pts vs Original")
                st.write(f"Contrast: {new_metrics['contrast']} (Δ {contrast_delta}) | Sharpness: {new_metrics['sharpness']}")
                st.write(f"Faces: {new_metrics['faces']}")
                
            st.markdown("---")
            if score_delta > 5:
                st.success(f"🏆 **Winner: New Thumbnail!** It scores {score_delta} points higher. Use this one!")
            elif score_delta < -5:
                st.error(f"⚠️ **Winner: Original Thumbnail.** The new one dropped by {abs(score_delta)} points. Stick to the original.")
            else:
                st.info(f"️ **Tie Game.** Both thumbnails are statistically similar. Test both via YouTube A/B testing!")
                
        elif orig_metrics:
            st.markdown("#### 🅰️ Original Thumbnail Analysis")
            st.image(thumb_path, use_column_width=True)
            st.metric("Score", f"{orig_metrics['score']}/100")
            st.write(f"Contrast: {orig_metrics['contrast']} | Sharpness: {orig_metrics['sharpness']} | Vibrancy: {orig_metrics['vibrancy']}")
            st.write(f"Faces: {orig_metrics['faces']}")
        elif new_metrics:
            st.markdown("#### 🅱️ New Thumbnail Analysis")
            st.image(new_thumb_path, use_column_width=True)
            st.metric("Score", f"{new_metrics['score']}/100")
            st.write(f"Contrast: {new_metrics['contrast']} | Sharpness: {new_metrics['sharpness']} | Vibrancy: {new_metrics['vibrancy']}")
            st.write(f"Faces: {new_metrics['faces']}")
        else:
            st.info("Upload an Original (via URL) or New Thumbnail to see metrics.")

        if not quick_thumb:
            # === TITLE OPTIMIZATION ===
            if title_input:
                st.markdown("---")
                st.subheader("📝 Title Optimization")
                with st.spinner("Analyzing title..."):
                    title_analysis = analyze_title_with_llm(title_input, final_transcript, topic_input)
                
                if "error" not in title_analysis:
                    col_t1, col_t2, col_t3 = st.columns(3)
                    col_t1.metric("Title Score", f"{title_analysis.get('title_score', 0)}/100")
                    col_t2.metric("Characters", title_analysis.get('character_count', 0))
                    col_t3.metric("Length", "✅ Optimal" if title_analysis.get('is_optimal_length') else "️ Adjust")
                    
                    st.markdown("**Alternative Titles:**")
                    for i, alt in enumerate(title_analysis.get('alternative_titles', []), 1):
                        st.info(f"**{i}.** {alt}")

            # === THUMBNAIL BRIEF ===
            if title_input:
                st.markdown("---")
                st.subheader("🎨 AI Thumbnail Brief & Prompt")
                with st.spinner("Generating brief..."):
                    thumb_brief = generate_thumbnail_brief(title_input, final_transcript, topic_input)
                
                if "error" not in thumb_brief:
                    st.metric("Predicted CTR Score", f"{thumb_brief.get('thumbnail_score_prediction', 0)}/100")
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        st.markdown(f"**Text:** {thumb_brief.get('thumbnail_text')}")
                        st.markdown(f"**Colors:** {thumb_brief.get('color_scheme')}")
                        st.markdown(f"**Layout:** {thumb_brief.get('layout')}")
                    with col_b2:
                        st.markdown("**Midjourney Prompt:**")
                        st.code(thumb_brief.get('midjourney_prompt', ''), language="text")

            # === HOOK & BORING ANALYSIS ===
            if video_path and os.path.exists(video_path):
                st.markdown("---")
                st.subheader("🎬 Hook & Retention Analysis")
                
                with st.spinner("Analyzing pacing..."):
                    vid_metrics = analyze_hook_video(video_path)
                if "error" not in vid_metrics:
                    cpm = vid_metrics["cpm"]
                    st.metric("Visual Pacing", f"{cpm} Cuts/Min")
                    if cpm < 10: st.success("✅ Good for Technical Content")
                    elif cpm < 20: st.success("✅ Excellent Pacing")
                    else: st.warning("⚠️ Very Fast")

                with st.spinner("Detecting boring signals..."):
                    boring_metrics = detect_boring_signals(video_path)
                if "error" not in boring_metrics:
                    st.metric("Boring Score", f"{boring_metrics['boring_score']}/100", delta="Lower is better")
                    if boring_metrics['is_boring']:
                        st.error("🚨 BORING - Add B-roll, zoom cuts, or screen recordings!")
                    else:
                        st.success("✅ ENGAGING - Good visual dynamics.")

                # === SCRIPT ANALYSIS ===
                if "GROQ_API_KEY" in st.secrets and final_transcript and final_transcript != "No transcript available.":
                    with st.spinner("Running AI script analysis..."):
                        llm_data = analyze_script_with_llm(final_transcript, cpm)
                    if "error" not in llm_data:
                        s1, s2, s3 = st.columns(3)
                        s1.metric("Pattern Interrupt", f"{llm_data.get('pattern_interrupt_score', 0)}/10")
                        s2.metric("Value Prop", f"{llm_data.get('value_prop_score', 0)}/10")
                        s3.metric("Jargon Control", f"{llm_data.get('jargon_score', 0)}/10")
                        st.info(f"**Critique:** {llm_data.get('critique')}")
                        st.success(f"**Rewrite:** {llm_data.get('rewrite_suggestion')}")

        # Cleanup
        for f in [thumb_path, video_path, new_thumb_path]:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass
