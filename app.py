import subprocess
subprocess.run(["apt-get", "update"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "libglib2.0-0"], capture_output=True)

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

# --- INITIALIZATION (FIXED CASCADE LOADING) ---
@st.cache_resource
def load_face_cascade():
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        if not cascade.empty(): return cascade
    except Exception: pass
    
    try:
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        temp_path = os.path.join(tempfile.gettempdir(), "haarcascade_frontalface_default.xml")
        if not os.path.exists(temp_path): urllib.request.urlretrieve(url, temp_path)
        cascade = cv2.CascadeClassifier(temp_path)
        if not cascade.empty(): return cascade
    except Exception: pass
    return None

face_cascade = load_face_cascade()

# --- HELPER FUNCTIONS ---
def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def fetch_thumbnail_and_transcript(url):
    video_id = extract_video_id(url)
    if not video_id: return None, None, "Invalid YouTube URL"
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
            if thumb_url: urllib.request.urlretrieve(thumb_url, thumb_path)
    except Exception:
        thumb_path = None
    return thumb_path, transcript_text[:2500], None

def analyze_thumbnail(image_path, niche_mode="Technical"):
    if not image_path or not os.path.exists(image_path):
        return {"error": "Could not load thumbnail"}
    img = cv2.imread(image_path)
    if img is None: return {"error": "Failed to decode image"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    b, g, r = cv2.split(img)
    vibrancy = float(np.mean([np.std(b), np.std(g), np.std(r)]))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])) * 100
    
    face_count = 0; face_centered = False
    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_count = len(faces)
        if face_count > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            img_h, img_w = gray.shape
            face_center_x = (x + w/2) / img_w
            face_center_y = (y + h/2) / img_h
            if 0.2 < face_center_x < 0.8 and 0.2 < face_center_y < 0.8: face_centered = True

    c_norm = min(contrast_score / 50.0, 1.0); s_norm = min(sharpness_score / 2000.0, 1.0)
    v_norm = min(vibrancy / 60.0, 1.0); e_norm = min(edge_density / 15.0, 1.0)
    face_pts = 1.0 if face_count > 0 else 0.0; center_pts = 1.0 if face_centered else 0.0
    final_score = 0
    if niche_mode == "Technical": final_score = (c_norm*20)+(s_norm*15)+(v_norm*10)+(e_norm*35)+(face_pts*15)+(center_pts*5)
    elif niche_mode == "Finance": final_score = (c_norm*20)+(s_norm*15)+(v_norm*15)+(e_norm*20)+(face_pts*25)+(center_pts*5)
    else: final_score = (c_norm*20)+(s_norm*10)+(v_norm*25)+(e_norm*10)+(face_pts*30)+(center_pts*5)
    final_score = int(round(max(0, min(100, final_score))))
    return {"score": final_score, "contrast": round(contrast_score, 1), "sharpness": round(sharpness_score, 0),
            "vibrancy": round(vibrancy, 1), "info_density": round(edge_density, 1), "faces": face_count, "face_centered": face_centered}

def analyze_hook_video(video_path):
    if not video_path or not os.path.exists(video_path): return {"error": "Could not load video file"}
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return {"error": "Failed to open video"}
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30
        max_frames = int(fps * 30); sample_rate = max(1, int(fps / 2))
        cuts = 0; prev_frame = None; frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if frame_count % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (21, 21), 0)
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray); mean_diff = np.mean(diff)
                    if mean_diff > 15.0: cuts += 1
                prev_frame = gray
            frame_count += 1
        cap.release()
        return {"cuts_detected": cuts, "cpm": cuts * 2}
    except Exception as e: return {"error": f"Video analysis failed: {str(e)[:100]}"}

def detect_boring_signals(video_path):
    if not video_path or not os.path.exists(video_path): return {"error": "No video file"}
    try:
        cap = cv2.VideoCapture(video_path); fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        max_frames = int(fps * 30); sample_rate = 3; stagnant_count = 0; total_comparisons = 0
        motion_scores = []; prev_frame = None; frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if frame_count % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); gray = cv2.resize(gray, (320, 180))
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray); motion = np.mean(diff); motion_scores.append(motion)
                    if motion < 8.0: stagnant_count += 1
                    total_comparisons += 1
                prev_frame = gray
            frame_count += 1
        cap.release()
        if total_comparisons == 0: return {"boring_score": 50, "stagnation_rate": 0, "avg_motion": 0, "is_boring": False, "verdict": "Could not analyze."}
        stagnation_rate = (stagnant_count / total_comparisons) * 100; avg_motion = np.mean(motion_scores)
        motion_penalty = max(0, 15 - avg_motion) * 3; boring_score = int(min(100, (stagnation_rate * 0.5) + motion_penalty))
        is_boring = boring_score > 50
        return {"boring_score": boring_score, "stagnation_rate": round(stagnation_rate, 1), "avg_motion": round(avg_motion, 2),
                "is_boring": is_boring, "verdict": "⚠️ BORING - Add visual variety" if is_boring else "✅ ENGAGING - Good visual dynamics"}
    except Exception as e: return {"error": f"Analysis failed: {str(e)[:100]}"}

# --- LLM FUNCTIONS ---
def generate_thumbnail_brief(title, transcript, topic):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    prompt = f"""You are a YouTube thumbnail designer expert for technical/finance channels. Title: "{title}". Topic: {topic}. Transcript Snippet: "{transcript[:300]}". Output STRICT JSON: "thumbnail_text" (string, max 5 words), "color_scheme" (object with background, text, accent hex codes), "layout" (string description), "visual_elements" (array of strings), "midjourney_prompt" (string, detailed), "style" (string), "dos" (array of 3 strings), "donts" (array of 3 strings), "thumbnail_score_prediction" (int 0-100)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.7, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def analyze_title_with_llm(title, transcript, topic, is_short=False):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    if is_short:
        prompt = f"""You are a YouTube Shorts SEO expert. Current Title: "{title}". Topic: {topic}. RULES: 1. Must be under 50 chars. 2. High curiosity, NO clickbait. 3. No "How to". Output STRICT JSON: "title_score" (int), "character_count" (int), "is_optimal_length" (bool), "alternative_titles" (array of 3 strings), "recommended_keywords" (array of 5 strings)"""
    else:
        prompt = f"""You are a YouTube SEO expert for technical/finance content. Current Title: "{title}". Topic: {topic}. Transcript Snippet: "{transcript[:300]}". Output STRICT JSON: "title_score" (int), "character_count" (int), "is_optimal_length" (bool), "alternative_titles" (array of 3 strings), "recommended_keywords" (array of 5 strings), "emotional_triggers" (string), "improvement_notes" (string)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.5, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def generate_shorts_description(title, topic):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    prompt = f"""You are a YouTube Shorts SEO expert. Title: "{title}". Topic: {topic}. Generate a Shorts description. RULES: 1. Max 2 sentences. 2. Pack with technical keywords. 3. Generate exactly 5 targeted hashtags. Output STRICT JSON: "short_description" (string), "hashtags" (array of 5 strings)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.3, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def compress_script_with_llm(full_script, is_short=False):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    if is_short:
        prompt = f"""You are a Ruthless Technical Editor for YouTube Shorts. TASK: Compress this script to fit STRICTLY under 60 seconds. RULES: 1. TARGET WORD COUNT: 130 to 150 words MAX. 2. PRESERVE ALL DATA. 3. CUT THE FLUFF. Original Script: "{full_script}". Output STRICT JSON: "original_word_count" (int), "compressed_word_count" (int), "estimated_seconds" (int), "compressed_script" (string)"""
    else:
        prompt = f"""You are a Ruthless Technical Editor. TASK: Rewrite to improve pacing. RULES: DO NOT SUMMARIZE. PRESERVE ALL DATA. CUT THE FLUFF. TARGET: Reduce word count by 40-60%. Original Script: "{full_script}". Output STRICT JSON: "original_word_count" (int), "compressed_word_count" (int), "compression_ratio" (string), "compressed_script" (string)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.2, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def analyze_script_with_llm(problem, mechanism, payoff, cpm, is_short=False):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    if is_short:
        prompt = f"""You are an elite YouTube Shorts Strategist. Visual Pacing: {cpm} CPM. Elements: Problem: {problem} | Mechanism: {mechanism} | Payoff: {payoff}. TASK: Write a 45s Shorts script. RULES: 1. HOOK IN FIRST 3 SECONDS (Max 15 words). 2. No intros. 3. Total word count MUST be under 120 words. Output STRICT JSON: "pattern_interrupt_score" (int), "value_prop_score" (int), "jargon_score" (int), "overall_hook_score" (int), "critique" (string), "script_rewrite" (string)"""
    else:
        prompt = f"""You are an elite YouTube Strategist. Visual Pacing: {cpm} CPM. Elements: Problem: {problem} | Mechanism: {mechanism} | Payoff: {payoff}. TASK: Write a punchy, 30-second Cold Open script. Start IMMEDIATELY with the Problem. NO FLUFF. Output STRICT JSON: "pattern_interrupt_score" (int), "value_prop_score" (int), "jargon_score" (int), "overall_hook_score" (int), "critique" (string), "script_rewrite" (string)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.3, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def generate_x_thread(topic, transcript):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    prompt = f"""You are a top 1% Quantitative Researcher on X. Topic: {topic}. Source: "{transcript[:1500]}". TASK: Write a 6-tweet thread. RULES: 1. TWEET 1 (Hook): Under 280 chars. Contrarian take or hard data. NO "In this thread...". 2. TWEETS 2-4 (Meat): Methodology, bullet points, technical terms. 3. TWEET 5 (Reality Check): Brutal truth or final metric. 4. TWEET 6 (CTA & Trap): Follow CTA + specific question to force replies. Output STRICT JSON: "tweet_1" (string), "tweet_2" (string), "tweet_3" (string), "tweet_4" (string), "tweet_5" (string), "tweet_6" (string), "engagement_question" (string)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.6, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def generate_threads_post(topic, transcript):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    prompt = f"""You are a professional Quantitative Trader on Threads. Topic: {topic}. Source: "{transcript[:1000]}". TASK: Write a single, high-impact post (max 400 chars). RULES: 1. Clean, conversational, authoritative. 2. Strong hook. 3. Use line breaks. 4. Suggest an "Image Idea" to attach. 5. NO HASHTAGS. Output STRICT JSON: "post_text" (string), "image_idea" (string)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.5, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def analyze_text_hook(text, platform):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    prompt = f"""You are a viral social media strategist for technical/finance creators. Platform: {platform}. User's First Post: "{text}". Evaluate from 0-100 based on: 1. Curiosity Gap. 2. Authority. 3. Formatting. Output STRICT JSON: "hook_score" (int), "strengths" (array of 2 strings), "weaknesses" (array of 2 strings), "rewrite_suggestion" (string)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.4, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

# --- UI ---
st.title(" QuantTube Analyzer Pro")
st.markdown("Proprietary CV & NLP pipeline for Algo-Trading YouTube optimization.")

with st.sidebar:
    st.header("⚙️ Settings")
    if "GROQ_API_KEY" not in st.secrets: st.warning("No Groq API Key in Secrets.")
    st.markdown("---")
    st.info("**Pro Features:**\n- Long-form & Shorts Mode\n- X & Threads Generator\n- Niche-Aware Scoring\n- Hook Builder\n- Script Compressor\n- A/B Comparator")

format_mode = st.radio("🎬 Content Format:", ["Long-form Video (8+ mins)", "YouTube Short (< 60s)", "X (Twitter) Thread", "Threads Post"], horizontal=True)
is_short = (format_mode == "YouTube Short (< 60s)")
is_x = (format_mode == "X (Twitter) Thread")
is_threads = (format_mode == "Threads Post")
is_text_platform = is_x or is_threads

if is_short: st.info("📱 **Shorts Mode Active:** AI will enforce <150 words, <50 char titles, and >30 CPM pacing.")
elif is_text_platform: st.info("📱 **Text Platform Active:** AI will optimize for dwell time, bookmarks, and replies.")

st.subheader("📥 Inputs")
col_url, col_upload = st.columns(2)
with col_url: url_input = st.text_input("1. YouTube URL (For Original Thumb & Transcript)", placeholder="https://www.youtube.com/watch?v=...")
with col_upload: uploaded_file = st.file_uploader("2. Video File (For Hook & Boring Analysis)", type=["mp4", "mov", "avi"])

col_title, col_topic = st.columns(2)
with col_title: title_input = st.text_input("3. Video Title / Post Topic", placeholder="e.g., Why EMA crossovers fail on BTC")
with col_topic: topic_input = st.text_input("4. Main Topic/Keyword", placeholder="e.g., Bitcoin backtesting, Python algo")

st.subheader("🎯 Content Niche Mode")
niche_mode = st.selectbox("Select your channel type:", ["Technical (Algo/Coding/Tutorials)", "Finance (Stocks/Crypto/Business)", "Entertainment (Vlogs/Lifestyle)"])

st.subheader("🖼️ Thumbnail A/B Testing")
new_thumb_file = st.file_uploader("5. Upload your NEW/AI-Generated Thumbnail to compare", type=["jpg", "png", "jpeg"])

st.subheader("✍️ Your Description (For Analysis)")
user_description = st.text_area("Paste YOUR existing description here...", height=100)

st.subheader("🎣 Hook Builder (Provide the Ingredients)")
col_p, col_m, col_pay = st.columns(3)
with col_p: problem_input = st.text_area("The Problem (Pain points, bad stats)", height=100)
with col_m: mechanism_input = st.text_area("The Mechanism (Your specific solution)", height=100)
with col_pay: payoff_input = st.text_area("The Payoff (The result/deliverable)", height=100)

st.subheader("✂️ Full Script Compressor")
full_script_input = st.text_area("Paste your full script here...", height=200)

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1: run_analysis = st.button("🚀 Full Analysis", type="primary", use_container_width=True)
with col_btn2: seo_only = st.button(" SEO Only", use_container_width=True)

if run_analysis or seo_only:
    if not title_input: st.error("Please enter a title/topic.")
    elif not topic_input: st.error("Please enter the main topic.")
    elif not url_input and not uploaded_file and not new_thumb_file and not seo_only and not problem_input and not full_script_input and not is_text_platform:
        st.error("Please provide at least one input.")
    else:
        mode_name = niche_mode.split(" ")[0]
        with st.spinner("Processing..."):
            thumb_path = None; transcript = ""
            if url_input:
                thumb_path, transcript, url_error = fetch_thumbnail_and_transcript(url_input)
                if url_error: st.error(url_error)
            video_path = None
            if uploaded_file:
                temp_dir = tempfile.gettempdir(); video_path = os.path.join(temp_dir, "uploaded_hook_video.mp4")
                with open(video_path, "wb") as f: f.write(uploaded_file.getbuffer())
            new_thumb_path = None
            if new_thumb_file:
                temp_dir = tempfile.gettempdir(); new_thumb_path = os.path.join(temp_dir, "new_thumb_comparison.jpg")
                with open(new_thumb_path, "wb") as f: f.write(new_thumb_file.getbuffer())

        final_transcript = transcript
        
        # === SCRIPT COMPRESSOR ===
        if full_script_input:
            st.markdown("---"); st.subheader("✂️ Script Pacing Compressor")
            with st.spinner("Ruthlessly editing your script..."): compression_data = compress_script_with_llm(full_script_input, is_short)
            if "error" in compression_data: st.error(compression_data["error"])
            else:
                col_w1, col_w2, col_w3 = st.columns(3)
                col_w1.metric("Original Words", compression_data.get('original_word_count', 0)); col_w2.metric("Compressed Words", compression_data.get('compressed_word_count', 0))
                if is_short:
                    col_w3.metric("Est. Time", f"~{compression_data.get('estimated_seconds', 0)}s")
                    if compression_data.get('compressed_word_count', 0) > 150: st.error("⚠️ Still too long! Must be under 150 words for Shorts.")
                    else: st.success("✅ Perfect length for a 60s Short!")
                else: col_w3.metric("Time Saved", f"~{compression_data.get('compression_ratio', '0%')}")
                st.markdown("### 📜 Compressed Script"); st.text_area("Compressed Version", value=compression_data.get('compressed_script', ''), height=400)

        # === SEO / DESCRIPTION ===
        if is_short and (run_analysis or seo_only):
            st.markdown("---"); st.subheader("📱 Shorts SEO & Description")
            with st.spinner("Generating Shorts metadata..."): shorts_seo = generate_shorts_description(title_input, topic_input)
            if "error" not in shorts_seo:
                st.markdown("### 📄 Shorts Description (Copy-Paste)"); st.text_area("Description", value=shorts_seo.get('short_description', ''), height=100)
                st.markdown("### #️ Hashtags"); st.code(" ".join(shorts_seo.get('hashtags', [])), language="text")
        elif user_description and (run_analysis or seo_only) and not is_short and not is_text_platform:
            st.markdown("---"); st.subheader("📊 Your Description Analysis"); st.info("Description analysis is optimized for Long-form. Use Shorts SEO for vertical content.")

        # === X & THREADS GENERATORS ===
        if is_text_platform and (run_analysis or seo_only):
            st.markdown("---")
            if is_x:
                st.subheader("🐦 X (Twitter) Thread Generator")
                with st.spinner("Drafting a viral quant thread..."): thread_data = generate_x_thread(topic_input, final_transcript if final_transcript else "Topic: " + title_input)
                if "error" in thread_data: st.error(thread_data["error"])
                else:
                    st.markdown("###  Your 6-Tweet Thread (Copy & Paste)")
                    for i in range(1, 7): st.text_area(f"Tweet {i}", value=thread_data.get(f"tweet_{i}", ''), height=100, key=f"tweet_{i}_ui")
                    st.markdown("### 🪤 The Engagement Trap (Post as a reply)"); st.success(thread_data.get('engagement_question', 'N/A'))
            elif is_threads:
                st.subheader(" Threads Post Generator")
                with st.spinner("Drafting an aesthetic Threads post..."): threads_data = generate_threads_post(topic_input, final_transcript if final_transcript else "Topic: " + title_input)
                if "error" in threads_data: st.error(threads_data["error"])
                else:
                    st.markdown("### 📝 Your Threads Post"); st.text_area("Post Text", value=threads_data.get('post_text', ''), height=200)
                    st.markdown("### 🖼️ Visual Asset Idea"); st.info(threads_data.get('image_idea', 'N/A'))
            
            st.markdown("---"); st.subheader("🎯 Text Hook Analyzer"); st.caption("Paste your first tweet or Threads post here to see if it's strong enough to stop the scroll.")
            user_text_hook = st.text_area("Paste your draft hook here...", height=100, key="text_hook_input")
            if st.button("📊 Analyze Text Hook", use_container_width=True):
                if user_text_hook:
                    with st.spinner("Analyzing text hook..."): hook_analysis = analyze_text_hook(user_text_hook, format_mode)
                    if "error" not in hook_analysis:
                        col_h1, col_h2 = st.columns(2)
                        with col_h1: st.metric("Hook Score", f"{hook_analysis.get('hook_score', 0)}/100")
                        with col_h2: st.metric("Platform", format_mode)
                        col_h3, col_h4 = st.columns(2)
                        with col_h3:
                            st.markdown("**✅ Strengths:**")
                            for s in hook_analysis.get('strengths', []): st.success(f"• {s}")
                        with col_h4:
                            st.markdown("**⚠️ Weaknesses:**")
                            for w in hook_analysis.get('weaknesses', []): st.error(f"• {w}")
                        st.markdown("**🔥 AI Rewrite Suggestion:**"); st.info(hook_analysis.get('rewrite_suggestion', 'N/A'))
                else: st.warning("Please paste a text hook to analyze.")

        # === THUMBNAIL A/B COMPARATOR ===
        st.markdown("---"); st.subheader(f"🖼️ Thumbnail A/B Comparator ({mode_name} Mode)")
        orig_metrics = None; new_metrics = None
        if thumb_path and os.path.exists(thumb_path): orig_metrics = analyze_thumbnail(thumb_path, mode_name)
        if new_thumb_path and os.path.exists(new_thumb_path): new_metrics = analyze_thumbnail(new_thumb_path, mode_name)
        if orig_metrics and new_metrics:
            col_orig, col_new = st.columns(2)
            with col_orig: st.markdown("#### ️ Original Thumbnail"); st.image(thumb_path, use_column_width=True); st.metric("Score", f"{orig_metrics['score']}/100")
            with col_new: st.markdown("#### 🅱️ New/AI Thumbnail"); st.image(new_thumb_path, use_column_width=True); score_delta = new_metrics['score'] - orig_metrics['score']; st.metric("Score", f"{new_metrics['score']}/100", delta=f"{score_delta} pts vs Original")
            if score_delta > 5: st.success(f" **Winner: New Thumbnail!** +{score_delta} pts.")
            elif score_delta < -5: st.error(f"⚠️ **Winner: Original Thumbnail.** -{abs(score_delta)} pts.")
            else: st.info(f"⚖️ **Tie Game.**")
        elif orig_metrics: st.image(thumb_path, use_column_width=True); st.metric("Score", f"{orig_metrics['score']}/100")

        # === TITLE OPTIMIZATION ===
        if title_input and not is_text_platform:
            st.markdown("---"); st.subheader("📝 Title Optimization")
            with st.spinner("Analyzing title..."): title_analysis = analyze_title_with_llm(title_input, final_transcript, topic_input, is_short)
            if "error" not in title_analysis:
                col_t1, col_t2, col_t3 = st.columns(3)
                col_t1.metric("Title Score", f"{title_analysis.get('title_score', 0)}/100"); col_t2.metric("Characters", title_analysis.get('character_count', 0))
                col_t3.metric("Length", "✅ Optimal" if title_analysis.get('is_optimal_length') else "⚠️ Adjust")
                st.markdown("**Alternative Titles:**")
                for i, alt in enumerate(title_analysis.get('alternative_titles', []), 1): st.info(f"**{i}.** {alt}")

        # === HOOK & RETENTION ANALYSIS ===
        if video_path and os.path.exists(video_path) and not is_text_platform:
            st.markdown("---"); st.subheader("🎬 Hook & Retention Analysis")
            with st.spinner("Analyzing pacing..."): vid_metrics = analyze_hook_video(video_path)
            if "error" not in vid_metrics:
                cpm = vid_metrics["cpm"]; st.metric("Visual Pacing", f"{cpm} Cuts/Min")
                if is_short:
                    if cpm < 20: st.error("⚠️ BORING FOR SHORTS! Need 30+ CPM.")
                    elif cpm < 40: st.warning("⚠️ Good, but aim for 40+ CPM for Shorts.")
                    else: st.success("✅ VIRAL PACING! Excellent for Shorts.")
                else:
                    if cpm < 10: st.success("✅ Good for Technical Content")
                    elif cpm < 20: st.success("✅ Excellent Pacing")
                    else: st.warning("⚠️ Very Fast")
            
            with st.spinner("Detecting boring signals..."): boring_metrics = detect_boring_signals(video_path)
            if "error" not in boring_metrics:
                st.metric("Boring Score", f"{boring_metrics['boring_score']}/100", delta="Lower is better")
                if boring_metrics['is_boring']: st.error("🚨 BORING - Add visual variety!")
                else: st.success("✅ ENGAGING - Good visual dynamics.")

            # === RESTORED THUMBNAIL BRIEF SECTION ===
            if title_input:
                st.markdown("---"); st.subheader("🎨 AI Thumbnail Brief & Prompt")
                with st.spinner("Generating brief..."): thumb_brief = generate_thumbnail_brief(title_input, final_transcript, topic_input)
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

            if problem_input or mechanism_input or payoff_input:
                st.markdown("---"); st.subheader("🎣 AI Hook Builder")
                with st.spinner("Weaving your ingredients..."): llm_data = analyze_script_with_llm(problem_input, mechanism_input, payoff_input, cpm, is_short)
                if "error" not in llm_data:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Pattern Interrupt", f"{llm_data.get('pattern_interrupt_score', 0)}/10")
                    s2.metric("Value Prop", f"{llm_data.get('value_prop_score', 0)}/10")
                    s3.metric("Jargon Control", f"{llm_data.get('jargon_score', 0)}/10")
                    st.success(llm_data.get('script_rewrite', 'N/A'))

        for f in [thumb_path, video_path, new_thumb_path]:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass
