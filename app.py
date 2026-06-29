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

    # --- FIX: AUTO-CROP BLACK BARS ---
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    non_zero = cv2.findNonZero(thresh)
    if non_zero is not None:
        x, y, w, h = cv2.boundingRect(non_zero)
        img = img[y:y+h, x:x+w]
        gray = gray[y:y+h, x:x+w]
    # ----------------------------------

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

def detect_boring_signals(video_path):
    """Analyzes visual stagnation and motion to detect boring segments."""
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
    """Analyze and optimize YouTube title for SEO and CTR"""
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return {"error": "No Groq API Key found."}

    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are a YouTube SEO expert specializing in technical/finance content.
    
    Current Title: "{title}"
    Video Topic: {topic}
    Transcript Snippet: "{transcript[:300]}"
    
    Analyze this title and provide:
    1. Title score (0-100) based on: curiosity, specificity, keyword optimization, length (ideal 50-60 chars)
    2. Character count
    3. 3 optimized alternative titles that are more clickable and SEO-friendly
    4. Top 5 keywords that should be in the title
    5. Emotional trigger analysis (curiosity/urgency/specificity)
    
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
    """Generate a complete thumbnail brief with text, colors, and AI image prompt"""
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return {"error": "No Groq API Key found."}

    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are a YouTube thumbnail designer expert for technical/finance channels.
    
    Video Title: "{title}"
    Video Topic: {topic}
    Transcript Snippet: "{transcript[:300]}"
    
    Create a complete thumbnail brief that will maximize CTR for an algo-trading/finance audience.
    
    Include:
    1. Thumbnail text (max 5 words, must be readable on mobile)
    2. Color scheme (specific hex codes for background, text, accents)
    3. Layout description (where to place elements)
    4. Visual elements to include (charts, logos, arrows, etc.)
    5. A detailed Midjourney/DALL-E prompt to generate the thumbnail
    6. Thumbnail style (professional/energetic/minimalist/etc.)
    7. Do's and Don'ts for this specific thumbnail
    
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
st.title("📈 QuantTube Analyzer Pro")
st.markdown("Proprietary CV & NLP pipeline for Algo-Trading YouTube optimization.")

with st.sidebar:
    st.header("⚙️ Settings")
    if "GROQ_API_KEY" not in st.secrets:
        st.warning("No Groq API Key in Secrets.")
    
    st.markdown("---")
    st.info("**Pro Features:**\n- Title Optimizer\n- Thumbnail Brief Generator\n- AI Script Analysis")

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

# TITLE INPUT
st.subheader("3. Video Title")
st.caption("Required for Title Optimization & Thumbnail Brief")
title_input = st.text_input("Enter your video title", placeholder="e.g., How I Backtested Bitcoin Strategies in 8 Minutes", label_visibility="collapsed")

# TOPIC/KEYWORD
st.subheader("4. Main Topic/Keyword")
st.caption("What is this video primarily about?")
topic_input = st.text_input("Main topic", placeholder="e.g., Bitcoin backtesting, algo trading, Python strategy", label_visibility="collapsed")

# MANUAL TRANSCRIPT FALLBACK
st.subheader("5. Transcript (Optional)")
st.caption("Paste your script here if auto-fetch fails")
manual_transcript = st.text_area("Or paste your hook script manually (first 30 seconds)", 
                                  height=100, 
                                  placeholder="In this video, I'm going to show you how to validate Bitcoin trading strategies in just 8 minutes...")

# ANALYZE BUTTON
analyze_col1, analyze_col2 = st.columns([3, 1])
with analyze_col1:
    run_analysis = st.button("🚀 Full Analysis", type="primary", use_container_width=True)
with analyze_col2:
    quick_thumb = st.button("🎨 Thumbnail Brief Only", use_container_width=True)

if run_analysis or quick_thumb:
    if not title_input and not quick_thumb:
        st.error("Please enter a video title for optimization.")
    elif not topic_input and not quick_thumb:
        st.error("Please enter the main topic/keyword.")
    elif not url_input and not uploaded_file and not manual_transcript:
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

        # Use manual transcript if provided, otherwise use fetched
        final_transcript = manual_transcript if manual_transcript else transcript
        
        # === TITLE OPTIMIZATION ===
        if title_input and not quick_thumb:
            st.markdown("---")
            st.subheader("📝 Title Optimization")
            
            with st.spinner("Analyzing title SEO and generating alternatives..."):
                title_analysis = analyze_title_with_llm(title_input, final_transcript, topic_input)
            
            if "error" in title_analysis:
                st.error(title_analysis["error"])
            else:
                col_t1, col_t2, col_t3 = st.columns(3)
                
                with col_t1:
                    st.metric("Title Score", f"{title_analysis.get('title_score', 0)}/100")
                
                with col_t2:
                    st.metric("Character Count", title_analysis.get('character_count', 0))
                
                with col_t3:
                    optimal = "✅ Optimal" if title_analysis.get('is_optimal_length', False) else "⚠️ Too long/short"
                    st.metric("Length", optimal)
                
                st.markdown(f"**Emotional Triggers:** {title_analysis.get('emotional_triggers', 'N/A')}")
                st.markdown(f"**Improvement Notes:** {title_analysis.get('improvement_notes', 'N/A')}")
                
                st.markdown("---")
                st.markdown("### 🎯 Recommended Alternative Titles:")
                
                alternatives = title_analysis.get('alternative_titles', [])
                for i, alt in enumerate(alternatives, 1):
                    st.info(f"**Option {i}:** {alt}")
                
                st.markdown("---")
                st.markdown("### 🔑 Recommended Keywords:")
                keywords = title_analysis.get('recommended_keywords', [])
                st.write(", ".join(keywords))

        # === THUMBNAIL BRIEF ===
        if title_input:
            st.markdown("---")
            st.subheader("🎨 AI-Generated Thumbnail Brief")
            
            with st.spinner("Generating thumbnail brief and Midjourney prompt..."):
                thumb_brief = generate_thumbnail_brief(title_input, final_transcript, topic_input)
            
            if "error" in thumb_brief:
                st.error(thumb_brief["error"])
            else:
                # Prediction score
                pred_score = thumb_brief.get('thumbnail_score_prediction', 0)
                st.metric("Predicted Thumbnail CTR Score", f"{pred_score}/100")
                
                col_b1, col_b2 = st.columns(2)
                
                with col_b1:
                    st.markdown("### 📋 Thumbnail Specifications:")
                    st.markdown(f"**Text on Thumbnail:** {thumb_brief.get('thumbnail_text', 'N/A')}")
                    st.markdown(f"**Style:** {thumb_brief.get('style', 'N/A')}")
                    
                    st.markdown("**Color Scheme:**")
                    colors = thumb_brief.get('color_scheme', {})
                    if colors:
                        st.code(f"Background: {colors.get('background', '#000000')}")
                        st.code(f"Text: {colors.get('text', '#FFFFFF')}")
                        st.code(f"Accent: {colors.get('accent', '#00FF00')}")
                    
                    st.markdown("**Layout:**")
                    st.write(thumb_brief.get('layout', 'N/A'))
                    
                    st.markdown("**Visual Elements:**")
                    for element in thumb_brief.get('visual_elements', []):
                        st.write(f"• {element}")
                
                with col_b2:
                    st.markdown("### ✅ Do's:")
                    for do_item in thumb_brief.get('dos', []):
                        st.success(f"✓ {do_item}")
                    
                    st.markdown("### ❌ Don'ts:")
                    for dont_item in thumb_brief.get('donts', []):
                        st.error(f"✗ {dont_item}")
                
                st.markdown("---")
                st.markdown("### 🤖 Midjourney/DALL-E Prompt:")
                st.code(thumb_brief.get('midjourney_prompt', ''), language="text")
                
                st.info("💡 **How to use:** Copy this prompt into Midjourney, DALL-E 3, or Leonardo AI to generate your thumbnail!")

        if not quick_thumb:
            # === THUMBNAIL ANALYSIS (if URL provided) ===
            if thumb_path and os.path.exists(thumb_path):
                st.markdown("---")
                st.subheader("🖼️ Current Thumbnail Analysis")
                
                st.image(thumb_path, use_column_width=True)
                
                with st.spinner("Analyzing current thumbnail..."):
                    thumb_metrics = analyze_thumbnail(thumb_path)
                    
                if "error" in thumb_metrics:
                    st.error(thumb_metrics["error"])
                else:
                    score = thumb_metrics["score"]
                    st.metric("Current Thumbnail Score", f"{score}/100", delta="Optimize for CTR")
                    
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
                        st.info("ℹ️ No face detected. (Normal for faceless channels)")

            # === HOOK ANALYSIS ===
            if video_path and os.path.exists(video_path):
                st.markdown("---")
                st.subheader("🎬 Hook Analysis (First 30s)")
                
                with st.spinner("Analyzing video pacing..."):
                    vid_metrics = analyze_hook_video(video_path)
                    
                if "error" in vid_metrics:
                    st.error(vid_metrics["error"])
                else:
                    cpm = vid_metrics["cpm"]
                    st.metric("Visual Pacing", f"{cpm} Cuts/Min")
                    
                    if cpm < 5:
                        st.error("⚠️ **Very Slow:** Consider adding B-roll or zoom cuts every 5-6 seconds")
                    elif cpm < 10:
                        st.success("✅ **Good for Technical Content:** Perfect pace for algo-trading education")
                    elif cpm < 20:
                        st.success("✅ **Excellent Pacing:** High energy while maintaining clarity")
                    else:
                        st.warning("⚠️ **Very Fast:** Ensure viewers can follow the technical details")

                with st.spinner("Detecting boring signals..."):
                    boring_metrics = detect_boring_signals(video_path)
                    
                if "error" in boring_metrics:
                    st.error(boring_metrics["error"])
                else:
                    st.markdown("---")
                    st.subheader("🎯 Boring Detector")
                    
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        st.metric("Boring Score", f"{boring_metrics['boring_score']}/100", delta="Lower is better")
                    with col_b2:
                        st.metric("Visual Stagnation", f"{boring_metrics['stagnation_rate']}%", delta="High = Too static")
                    with col_b3:
                        st.metric("Motion Level", f"{boring_metrics['avg_motion']}", delta="Higher = Dynamic")
                    
                    if boring_metrics['is_boring']:
                        st.error(f"🚨 **{boring_metrics['verdict']}**")
                        st.warning("**Fixes:**\n"
                                  "- Add B-roll footage every 10-15 seconds\n"
                                  "- Use zoom cuts (punch in/out)\n"
                                  "- Add text overlays/graphics\n"
                                  "- Show screen recordings/code demos")
                    else:
                        st.success(f"✅ **{boring_metrics['verdict']}**")
                        st.info("Your video maintains good visual interest throughout!")

                # === AI SCRIPT ANALYSIS ===
                if "GROQ_API_KEY" not in st.secrets:
                    st.warning("⚠️ **Missing API Key:** Add your Groq API key to Streamlit Secrets.")
                elif not final_transcript or final_transcript == "No transcript available.":
                    st.warning("⚠️ **Missing Transcript:** Either paste the YouTube URL in Box 1 OR manually paste your script in Box 5.")
                else:
                    with st.spinner("Running AI script analysis..."):
                        llm_data = analyze_script_with_llm(final_transcript, cpm)
                        
                    if "error" in llm_data:
                        st.error(llm_data["error"])
                    else:
                        st.markdown("---")
                        st.subheader("📊 Script Quality Metrics")
                        s1, s2, s3 = st.columns(3)
                        s1.metric("Pattern Interrupt", f"{llm_data.get('pattern_interrupt_score', 0)}/10")
                        s2.metric("Value Prop", f"{llm_data.get('value_prop_score', 0)}/10")
                        s3.metric("Jargon Control", f"{llm_data.get('jargon_score', 0)}/10")
                        
                        hook_score = llm_data.get("overall_hook_score", 0)
                        st.progress(min(hook_score, 100) / 100)
                        st.caption(f"Overall Hook Score: {hook_score}/100")
                        
                        st.info(f"**AI Critique:** {llm_data.get('critique', 'N/A')}")
                        st.success(f"**Suggested Rewrite:** {llm_data.get('rewrite_suggestion', 'N/A')}")

        # Cleanup
        for f in [thumb_path, video_path]:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass
