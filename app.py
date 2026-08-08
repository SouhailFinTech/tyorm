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

# === NEW: also return raw timed transcript segments (needed for hook-window isolation) ===
def fetch_timed_transcript(url):
    video_id = extract_video_id(url)
    if not video_id: return None
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception:
        return None

def get_hook_window_text(timed_transcript, window_seconds=15):
    """Isolate only the words spoken in the first N seconds — the true click->watch hook window."""
    if not timed_transcript:
        return ""
    words = []
    for seg in timed_transcript:
        if seg.get('start', 0) <= window_seconds:
            words.append(seg.get('text', ''))
        else:
            break
    return " ".join(words).strip()

def analyze_thumbnail(image_path, niche_mode="Technical", is_faceless=False):
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
    if face_cascade is not None and not is_faceless:
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

    if is_faceless:
        # No face channel: the eye has nothing to anchor on except contrast, bold
        # text/graphics (edge density), and color pop. Redistribute the ~20-30pts
        # that would've gone to face detection into contrast + edge density + vibrancy,
        # since those three carry the entire attention-grabbing job on a faceless thumb.
        final_score = (c_norm*30)+(s_norm*15)+(v_norm*20)+(e_norm*35)
    elif niche_mode == "Technical": final_score = (c_norm*20)+(s_norm*15)+(v_norm*10)+(e_norm*35)+(face_pts*15)+(center_pts*5)
    elif niche_mode == "Finance": final_score = (c_norm*20)+(s_norm*15)+(v_norm*15)+(e_norm*20)+(face_pts*25)+(center_pts*5)
    else: final_score = (c_norm*20)+(s_norm*10)+(v_norm*25)+(e_norm*10)+(face_pts*30)+(center_pts*5)
    final_score = int(round(max(0, min(100, final_score))))
    return {"score": final_score, "contrast": round(contrast_score, 1), "sharpness": round(sharpness_score, 0),
            "vibrancy": round(vibrancy, 1), "info_density": round(edge_density, 1), "faces": face_count, "face_centered": face_centered,
            "is_faceless": is_faceless}

# === NEW: Mobile Legibility Score ===
# A thumbnail can score high full-size and still fail on the phone feed, where most
# impressions actually happen. This simulates real browse sizes and measures whether
# there's still enough contrast/edge structure to read at a glance.
def analyze_mobile_legibility(image_path):
    if not image_path or not os.path.exists(image_path):
        return {"error": "Could not load thumbnail"}
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Failed to decode image"}

    # Real-world render sizes: desktop grid (~336x188) and mobile feed (~120x67)
    sizes = {"desktop_grid": (336, 188), "mobile_feed": (120, 67)}
    results = {}
    for label, (w, h) in sizes.items():
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Edge density after downscale = proxy for "is there still readable structure"
        edges_small = cv2.Canny(gray_small, 50, 150)
        edge_density_small = float(np.count_nonzero(edges_small) / (w * h)) * 100
        # Local contrast after downscale = proxy for "does text/subject still pop"
        contrast_small = float(np.std(gray_small))
        results[label] = {
            "edge_density": round(edge_density_small, 2),
            "contrast": round(contrast_small, 1),
        }

    mobile = results["mobile_feed"]
    # Empirically: legible small thumbnails keep edge_density > ~4 and contrast > ~35
    # after aggressive downscale. Below that, text/faces blur into mush.
    edge_ok = mobile["edge_density"] >= 4.0
    contrast_ok = mobile["contrast"] >= 35.0
    legibility_score = int(round(
        min(mobile["edge_density"] / 8.0, 1.0) * 50 +
        min(mobile["contrast"] / 60.0, 1.0) * 50
    ))
    if legibility_score >= 70:
        verdict = "✅ Reads clearly at mobile-feed size."
    elif legibility_score >= 45:
        verdict = "⚠️ Borderline — simplify: fewer elements, bigger contrast between subject and background."
    else:
        verdict = "❌ Likely illegible in the mobile feed — this is probably costing you CTR on 60-70% of impressions."

    return {
        "legibility_score": legibility_score,
        "desktop_grid": results["desktop_grid"],
        "mobile_feed": mobile,
        "verdict": verdict,
    }

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
def generate_thumbnail_brief(title, transcript, topic, is_faceless=False):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    face_instruction = (
        "This is a FACELESS channel — no presenter face, ever. Design around bold typography, "
        "data visualizations (charts, candlesticks, code snippets), strong color-blocking, and a "
        "consistent recognizable template/branding system instead of a face. Do not suggest a face or presenter."
        if is_faceless else
        "A presenter face can be used if it strengthens the thumbnail."
    )
    prompt = f"""You are a YouTube thumbnail designer expert for technical/finance channels. Title: "{title}". Topic: {topic}. Transcript Snippet: "{transcript[:300]}". {face_instruction} Output STRICT JSON: "thumbnail_text" (string, max 5 words), "color_scheme" (object with background, text, accent hex codes), "layout" (string description), "visual_elements" (array of strings), "midjourney_prompt" (string, detailed), "style" (string), "dos" (array of 3 strings), "donts" (array of 3 strings), "thumbnail_score_prediction" (int 0-100)"""
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

# === NEW: Hook Window Deep-Analysis (click -> watch) ===
# Scores ONLY what's actually spoken/shown in the first ~15s, not the whole script.
# This is the moment that decides whether a click becomes a view or an instant bounce.
def analyze_hook_window_with_llm(hook_text, title, topic, niche_mode="Technical"):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    if not hook_text:
        return {"error": "No hook-window text available — paste your first-15-seconds script manually below."}
    client = Groq(api_key=api_key)
    prompt = f"""You are a YouTube retention specialist for {niche_mode} finance/quant creators.
Video Title: "{title}"
Topic: {topic}
EXACT words spoken in the first ~15 seconds: "{hook_text}"

Score this hook window on what determines whether a viewer who just clicked keeps watching past 15s:
1. Curiosity gap (did it open a question the viewer needs answered?)
2. Promise clarity (is it obvious what specific payoff they'll get?)
3. Pattern interrupt (does it avoid a generic/slow intro — "hey guys welcome back" style openers are a major retention killer)
4. Relevance match to the title (does the hook deliver on what the title/thumbnail promised, avoiding bait-and-switch?)

Output STRICT JSON: "curiosity_gap_score" (int 0-100), "promise_clarity_score" (int 0-100), "pattern_interrupt_score" (int 0-100), "title_match_score" (int 0-100), "overall_hook_window_score" (int 0-100), "biggest_risk" (string, the single most likely reason a viewer would bounce in these 15s), "rewritten_hook" (string, a stronger version of these first 15 seconds)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.4, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

# === NEW: Subscribe Funnel Advisor (watch -> subscribe) ===
# Your original tool had zero logic for the actual conversion goal you stated: turning
# viewers into subscribers. This detects CTA presence/timing and generates CTAs that
# fit a quant/finance audience (who bounce off hypey generic "smash that subscribe" asks).
def analyze_subscribe_funnel(transcript, title, topic, niche_mode="Technical"):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    if not transcript:
        return {"error": "No transcript available to analyze."}
    client = Groq(api_key=api_key)
    prompt = f"""You are a YouTube channel growth strategist specializing in technical/finance/quant creators, where audiences are skeptical of hypey asks and respond better to earned, specific CTAs.

Video Title: "{title}"
Topic: {topic}
Full transcript (may be truncated): "{transcript[:3000]}"

TASK: Analyze this transcript for subscribe-conversion mechanics only.
1. Detect whether there is any verbal subscribe/follow ask in the transcript, and roughly where (early/mid/late/none).
2. Judge the "value-before-ask" ratio: does the creator deliver real, specific value (a number, a method, a concrete insight) BEFORE any ask, or does the ask come too early/generic?
3. Flag if the ask is generic/hypey ("smash that subscribe button") vs specific and earned (tied to a concrete reason to come back).
4. Generate a SOFT-ASK line (placed right after a value payoff mid-video) and a HARD-ASK line (for the outro), both written for a technical/quant/finance audience — no hype language, tie the ask to a specific, credible reason to subscribe (e.g. a concrete recurring series, a specific edge/insight they'll miss otherwise).

Output STRICT JSON: "cta_detected" (bool), "cta_timing" (string: "early"/"mid"/"late"/"none"), "cta_style" (string: "generic_hype"/"specific_earned"/"none"), "value_before_ask_score" (int 0-100), "diagnosis" (string, 1-2 sentences on what's likely hurting sub-conversion here), "soft_ask_line" (string), "hard_ask_line" (string)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.4, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

# === NEW: Impressions/Discovery Advisor (how the video gets found at all) ===
# Click/Watch/Subscribe optimize a video that's ALREADY being shown to someone.
# This step is upstream of all of that: it's what gets YouTube to show it in the
# first place. Two real traffic sources matter here — Search (SEO) and
# Suggested/Browse (topic-clustering + session-time signals). External and
# notification traffic exist too but aren't something a 3-month channel controls yet.
def generate_impressions_strategy(title, topic, transcript, niche_mode="Technical", is_faceless=False):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    faceless_note = (
        "The channel is faceless, so discovery strategy should lean harder on topic "
        "consistency, a recognizable branding system, and search intent — since there's "
        "no personality-driven audience pull to rely on early on."
        if is_faceless else ""
    )
    prompt = f"""You are a YouTube growth strategist specializing in small/new technical-finance/quant channels (this one is ~3 months old and still building initial traction).
Title: "{title}"
Topic/Keyword: {topic}
Niche: {niche_mode}
Transcript snippet: "{transcript[:800] if transcript else 'N/A'}"
{faceless_note}

TASK: Give a concrete impressions/discovery plan. YouTube surfaces videos mainly through (a) Search, when the title/description/tags match what people actually type, and (b) Suggested/Browse, which clusters videos by topic similarity and rewards videos that keep people watching more of the SAME topic cluster (session time), not just this one video. New/small channels get almost nothing from (c) Notifications/subscribers yet, and (d) external traffic is channel-dependent.

Output STRICT JSON:
"search_keywords" (array of 8 realistic search phrases a target viewer would actually type into YouTube for this topic — not generic SEO fluff, actual query phrasing),
"tags_to_use" (array of 10 YouTube tags, ordered broad-to-specific),
"suggested_video_cluster_strategy" (string, 2-3 sentences on what adjacent/related videos or a series structure would build a topic cluster YouTube can reliably suggest this channel's videos within),
"upload_cadence_advice" (string, 1-2 sentences realistic for a solo creator),
"session_time_tactic" (string, one concrete tactic to increase watch-next behavior, e.g. end screens, playlists, or pinned comment linking related videos),
"biggest_impressions_blocker" (string, the single most likely reason a 3-month-old channel in this niche is getting few impressions)
"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.5, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

# === NEW: Long-form Description SEO Analyzer (was a dead placeholder before) ===
# Descriptions matter for Impressions (Stage 0): the first ~150 chars show in search
# results, and YouTube's search/suggested systems parse the full text for keyword
# relevance. This actually analyzes what you pasted instead of just telling you to
# use the Shorts tool.
def analyze_longform_description(description, title, topic, target_keywords=None):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    if not description:
        return {"error": "No description provided."}
    client = Groq(api_key=api_key)
    kw_note = f"Target search keywords from the Impressions plan: {target_keywords}" if target_keywords else ""
    prompt = f"""You are a YouTube SEO specialist for technical/finance/quant channels.
Title: "{title}"
Topic/Keyword: {topic}
{kw_note}
User's current description: "{description}"

TASK: Evaluate this description for search/discovery performance (NOT for persuasion/CR — that's a separate job).
1. Check the first ~150 characters specifically, since that's what shows in search results and suggested feed before "show more" is clicked — is the core keyword and hook present there, or buried later?
2. Check keyword coverage against the topic/target keywords — is it naturally present or missing entirely?
3. Check for a clear video summary a search algorithm can parse (not just hashtags/links dumped at the top).

Output STRICT JSON: "first_150_chars_ok" (bool), "first_150_chars_issue" (string, what's wrong with the opening if anything), "keyword_coverage_score" (int 0-100), "missing_keywords" (array of strings, keywords from the topic that are absent but should be there), "rewritten_description" (string, an improved full description, 3-5 sentences plus a short keyword line, staying accurate to the original content)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.4, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

# --- UI ---
st.title("📈 QuantTube Analyzer Pro")
st.markdown("Proprietary CV & NLP pipeline for Algo-Trading YouTube optimization — **Click → Watch → Subscribe funnel.**")

with st.sidebar:
    st.header("️ Settings")
    if "GROQ_API_KEY" not in st.secrets: st.warning("No Groq API Key in Secrets.")
    st.markdown("---")
    st.info("**Pro Features:**\n- Long-form & Shorts Mode\n- Mobile Legibility Score (NEW)\n- Hook-Window Deep Analysis (NEW)\n- Subscribe Funnel Advisor (NEW)\n- X & Threads Generator\n- Niche-Aware Scoring\n- Hook Builder\n- Script Compressor\n- A/B Comparator")

format_mode = st.radio("🎬 Content Format:", ["Long-form Video (8+ mins)", "YouTube Short (< 60s)", "X (Twitter) Thread", "Threads Post"], horizontal=True)
is_short = (format_mode == "YouTube Short (< 60s)")
is_x = (format_mode == "X (Twitter) Thread")
is_threads = (format_mode == "Threads Post")
is_text_platform = is_x or is_threads

if is_short: st.info(" **Shorts Mode Active:** AI will enforce <150 words, <50 char titles, and >30 CPM pacing.")
elif is_text_platform: st.info("📱 **Text Platform Active:** AI will optimize for dwell time, bookmarks, and replies.")

st.subheader("📥 Inputs")
col_url, col_upload = st.columns(2)
with col_url: url_input = st.text_input("1. YouTube URL (For Original Thumb & Transcript)", placeholder="https://www.youtube.com/watch?v=...")
with col_upload: uploaded_file = st.file_uploader("2. Video File (For Hook & Boring Analysis)", type=["mp4", "mov", "avi"])

col_title, col_topic = st.columns(2)
with col_title: title_input = st.text_input("3. Video Title / Post Topic", placeholder="e.g., Why EMA crossovers fail on BTC")
with col_topic: topic_input = st.text_input("4. Main Topic/Keyword", placeholder="e.g., Bitcoin backtesting, Python algo")

st.subheader("🎯 Content Niche Mode")
col_niche, col_faceless = st.columns([3, 1])
with col_niche:
    niche_mode = st.selectbox("Select your channel type:", ["Technical (Algo/Coding/Tutorials)", "Finance (Stocks/Crypto/Business)", "Entertainment (Vlogs/Lifestyle)"])
with col_faceless:
    is_faceless = st.checkbox("Faceless channel", value=True, help="Removes face-detection scoring from the thumbnail grader and redirects that weight to contrast/text/graphics, since there's no presenter face to score.")

st.subheader("🖼️ Thumbnail A/B Testing")
new_thumb_file = st.file_uploader("5. Upload your NEW/AI-Generated Thumbnail to compare", type=["jpg", "png", "jpeg"])

st.subheader("✍️ Your Description (For Analysis)")
user_description = st.text_area("Paste YOUR existing description here...", height=100)

# === NEW INPUT: manual hook-window override (in case transcript timing is unavailable) ===
st.subheader("🎣 First 15 Seconds (Hook Window)")
manual_hook_input = st.text_area(
    "Optional: paste EXACTLY what you say/show in the first 15 seconds. If left blank and a YouTube URL is provided, this is auto-extracted from the transcript timestamps.",
    height=80
)

st.subheader("🎣 Hook Builder (Provide the Ingredients)")
col_p, col_m, col_pay = st.columns(3)
with col_p: problem_input = st.text_area("The Problem (Pain points, bad stats)", height=100)
with col_m: mechanism_input = st.text_area("The Mechanism (Your specific solution)", height=100)
with col_pay: payoff_input = st.text_area("The Payoff (The result/deliverable)", height=100)

st.subheader("✂️ Full Script Compressor")
full_script_input = st.text_area("Paste your full script here...", height=200)

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1: run_analysis = st.button("🚀 Full Analysis", type="primary", use_container_width=True)
with col_btn2: seo_only = st.button("📝 SEO Only", use_container_width=True)

if run_analysis or seo_only:
    if not title_input: st.error("Please enter a title/topic.")
    elif not topic_input: st.error("Please enter the main topic.")
    elif not url_input and not uploaded_file and not new_thumb_file and not seo_only and not problem_input and not full_script_input and not is_text_platform and not manual_hook_input:
        st.error("Please provide at least one input.")
    else:
        mode_name = niche_mode.split(" ")[0]
        with st.spinner("Processing..."):
            thumb_path = None; transcript = ""; timed_transcript = None
            if url_input:
                thumb_path, transcript, url_error = fetch_thumbnail_and_transcript(url_input)
                if url_error: st.error(url_error)
                timed_transcript = fetch_timed_transcript(url_input)
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
                st.markdown("###  Shorts Description (Copy-Paste)"); st.text_area("Description", value=shorts_seo.get('short_description', ''), height=100)
                st.markdown("### #️⃣ Hashtags"); st.code(" ".join(shorts_seo.get('hashtags', [])), language="text")
        elif user_description and (run_analysis or seo_only) and not is_short and not is_text_platform:
            st.markdown("---"); st.subheader("📊 Your Description Analysis")
            with st.spinner("Analyzing description SEO..."):
                desc_result = analyze_longform_description(user_description, title_input, topic_input)
            if "error" in desc_result:
                st.warning(desc_result["error"])
            else:
                d1, d2 = st.columns(2)
                d1.metric("First 150 Chars OK", "Yes" if desc_result.get("first_150_chars_ok") else "No")
                d2.metric("Keyword Coverage", f"{desc_result.get('keyword_coverage_score', 0)}/100")
                if not desc_result.get("first_150_chars_ok"):
                    st.warning(f"**Opening issue:** {desc_result.get('first_150_chars_issue', 'N/A')}")
                missing = desc_result.get("missing_keywords", [])
                if missing:
                    st.markdown("**Missing keywords:** " + ", ".join(missing))
                st.markdown("**Rewritten description:**")
                st.text_area("Improved Description", value=desc_result.get("rewritten_description", ""), height=150)

        # === X & THREADS GENERATORS ===
        if is_text_platform and (run_analysis or seo_only):
            st.markdown("---")
            if is_x:
                st.subheader("🐦 X (Twitter) Thread Generator")
                with st.spinner("Drafting a viral quant thread..."): thread_data = generate_x_thread(topic_input, final_transcript if final_transcript else "Topic: " + title_input)
                if "error" in thread_data: st.error(thread_data["error"])
                else:
                    st.markdown("### 🧵 Your 6-Tweet Thread (Copy & Paste)")
                    for i in range(1, 7): st.text_area(f"Tweet {i}", value=thread_data.get(f"tweet_{i}", ''), height=100, key=f"tweet_{i}_ui")
                    st.markdown("### 🪤 The Engagement Trap (Post as a reply)"); st.success(thread_data.get('engagement_question', 'N/A'))
            elif is_threads:
                st.subheader("🧵 Threads Post Generator")
                with st.spinner("Drafting an aesthetic Threads post..."): threads_data = generate_threads_post(topic_input, final_transcript if final_transcript else "Topic: " + title_input)
                if "error" in threads_data: st.error(threads_data["error"])
                else:
                    st.markdown("### 📝 Your Threads Post"); st.text_area("Post Text", value=threads_data.get('post_text', ''), height=200)
                    st.markdown("### ️ Visual Asset Idea"); st.info(threads_data.get('image_idea', 'N/A'))

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
                        st.markdown("** AI Rewrite Suggestion:**"); st.info(hook_analysis.get('rewrite_suggestion', 'N/A'))
                else: st.warning("Please paste a text hook to analyze.")

        # Everything below this point is the core Click -> Watch -> Subscribe funnel,
        # so skip it entirely for text-platform runs.
        if not is_text_platform:

            # === STAGE 0: IMPRESSIONS — how the video gets shown at all (NEW) ===
            st.markdown("---"); st.header("0️⃣ IMPRESSIONS — Getting Seen (NEW)")
            st.caption("Click/Watch/Subscribe all optimize a video that's already being shown to someone. This is upstream of that: what actually gets YouTube to surface it.")
            if title_input and topic_input:
                with st.spinner("Building discovery plan..."):
                    imp_result = generate_impressions_strategy(title_input, topic_input, final_transcript, mode_name, is_faceless)
                if "error" in imp_result:
                    st.warning(imp_result["error"])
                else:
                    st.error(f"**Likely blocker right now:** {imp_result.get('biggest_impressions_blocker', 'N/A')}")
                    col_i1, col_i2 = st.columns(2)
                    with col_i1:
                        st.markdown("**🔎 Real search phrases to target:**")
                        for kw in imp_result.get("search_keywords", []):
                            st.markdown(f"- {kw}")
                    with col_i2:
                        st.markdown("**🏷️ Tags (broad → specific):**")
                        st.code(", ".join(imp_result.get("tags_to_use", [])), language="text")
                    st.markdown("**🧩 Topic-cluster / series strategy (Suggested & Browse traffic):**")
                    st.info(imp_result.get("suggested_video_cluster_strategy", "N/A"))
                    col_i3, col_i4 = st.columns(2)
                    with col_i3:
                        st.markdown("**📅 Upload cadence:**")
                        st.write(imp_result.get("upload_cadence_advice", "N/A"))
                    with col_i4:
                        st.markdown("**⏱️ Session-time tactic:**")
                        st.write(imp_result.get("session_time_tactic", "N/A"))
            else:
                st.info("Enter a title and topic/keyword above to generate the discovery plan.")

            # === STAGE 1: CLICK — THUMBNAIL A/B COMPARATOR + MOBILE LEGIBILITY ===
            st.markdown("---"); st.header("1️⃣ CLICK — Thumbnail & Title")
            st.subheader(f"🖼️ Thumbnail A/B Comparator ({mode_name} Mode)")
            orig_metrics = None; new_metrics = None
            if thumb_path and os.path.exists(thumb_path): orig_metrics = analyze_thumbnail(thumb_path, mode_name, is_faceless)
            if new_thumb_path and os.path.exists(new_thumb_path): new_metrics = analyze_thumbnail(new_thumb_path, mode_name, is_faceless)
            if orig_metrics and new_metrics:
                col_orig, col_new = st.columns(2)
                with col_orig: st.markdown("#### 🅰️ Original Thumbnail"); st.image(thumb_path, use_container_width=True); st.metric("Score", f"{orig_metrics['score']}/100")
                with col_new: st.markdown("#### 🅱️ New/AI Thumbnail"); st.image(new_thumb_path, use_container_width=True); score_delta = new_metrics['score'] - orig_metrics['score']; st.metric("Score", f"{new_metrics['score']}/100", delta=f"{score_delta} pts vs Original")
                if score_delta > 5: st.success(f"🏆 **Winner: New Thumbnail!** +{score_delta} pts.")
                elif score_delta < -5: st.error(f"️ **Winner: Original Thumbnail.** -{abs(score_delta)} pts.")
                else: st.info(f"⚖️ **Tie Game.**")
            elif orig_metrics: st.image(thumb_path, use_container_width=True); st.metric("Score", f"{orig_metrics['score']}/100")

            # NEW: Mobile Legibility Score, run on whichever thumbnail(s) are present
            legibility_targets = []
            if thumb_path and os.path.exists(thumb_path): legibility_targets.append(("Original", thumb_path))
            if new_thumb_path and os.path.exists(new_thumb_path): legibility_targets.append(("New/AI", new_thumb_path))
            if legibility_targets:
                st.markdown("#### 📱 Mobile Legibility Score (NEW)")
                st.caption("Most impressions happen in the mobile feed at ~120x67px. A thumbnail that looks great full-size can still fail here — this tests it directly.")
                leg_cols = st.columns(len(legibility_targets))
                for col, (label, path) in zip(leg_cols, legibility_targets):
                    leg = analyze_mobile_legibility(path)
                    with col:
                        st.markdown(f"**{label}**")
                        if "error" in leg:
                            st.error(leg["error"])
                        else:
                            st.metric("Legibility Score", f"{leg['legibility_score']}/100")
                            st.caption(leg["verdict"])

            # === TITLE OPTIMIZATION ===
            if title_input:
                st.markdown("#### 📝 Title Optimization")
                with st.spinner("Analyzing title..."): title_analysis = analyze_title_with_llm(title_input, final_transcript, topic_input, is_short)
                if "error" not in title_analysis:
                    col_t1, col_t2, col_t3 = st.columns(3)
                    col_t1.metric("Title Score", f"{title_analysis.get('title_score', 0)}/100"); col_t2.metric("Characters", title_analysis.get('character_count', 0))
                    col_t3.metric("Length", "✅ Optimal" if title_analysis.get('is_optimal_length') else "⚠️ Adjust")
                    st.markdown("**Alternative Titles:**")
                    for i, alt in enumerate(title_analysis.get('alternative_titles', []), 1): st.info(f"**{i}.** {alt}")

            # === STAGE 2: WATCH — HOOK WINDOW + PACING + BORING SIGNALS ===
            st.markdown("---"); st.header("2️⃣ WATCH — Hook & Retention")

            # NEW: Hook Window Deep-Analysis (first ~15s only)
            hook_window_text = manual_hook_input.strip() if manual_hook_input.strip() else get_hook_window_text(timed_transcript)
            if title_input:
                st.subheader("🎣 First-15-Seconds Hook Analysis (NEW)")
                st.caption("Scored on ONLY the words spoken in the click->watch window — not the full script.")
                if hook_window_text:
                    with st.spinner("Scoring the hook window..."):
                        hook_window_result = analyze_hook_window_with_llm(hook_window_text, title_input, topic_input, mode_name)
                    if "error" in hook_window_result:
                        st.warning(hook_window_result["error"])
                    else:
                        hc1, hc2, hc3, hc4 = st.columns(4)
                        hc1.metric("Curiosity Gap", f"{hook_window_result.get('curiosity_gap_score', 0)}/100")
                        hc2.metric("Promise Clarity", f"{hook_window_result.get('promise_clarity_score', 0)}/100")
                        hc3.metric("Pattern Interrupt", f"{hook_window_result.get('pattern_interrupt_score', 0)}/100")
                        hc4.metric("Title Match", f"{hook_window_result.get('title_match_score', 0)}/100")
                        st.metric("Overall Hook-Window Score", f"{hook_window_result.get('overall_hook_window_score', 0)}/100")
                        st.error(f"**Biggest bounce risk:** {hook_window_result.get('biggest_risk', 'N/A')}")
                        st.success(f"**Rewritten hook:** {hook_window_result.get('rewritten_hook', 'N/A')}")
                else:
                    st.info("No hook-window text found. Provide a YouTube URL with captions, or paste your first-15-seconds script above.")

            # === HOOK & PACING ANALYSIS (video upload) ===
            cpm = None
            if video_path and os.path.exists(video_path):
                st.subheader("🎬 Visual Pacing & Boring-Signal Analysis")
                with st.spinner("Analyzing pacing..."): vid_metrics = analyze_hook_video(video_path)
                if "error" not in vid_metrics:
                    cpm = vid_metrics["cpm"]; st.metric("Visual Pacing", f"{cpm} Cuts/Min")
                    if is_short:
                        if cpm < 20: st.error("⚠️ BORING FOR SHORTS! Need 30+ CPM.")
                        elif cpm < 40: st.warning("️ Good, but aim for 40+ CPM for Shorts.")
                        else: st.success("✅ VIRAL PACING! Excellent for Shorts.")
                    else:
                        if cpm < 10: st.success("✅ Good for Technical Content")
                        elif cpm < 20: st.success("✅ Excellent Pacing")
                        else: st.warning("️ Very Fast")

                with st.spinner("Detecting boring signals..."): boring_metrics = detect_boring_signals(video_path)
                if "error" not in boring_metrics:
                    st.metric("Boring Score", f"{boring_metrics['boring_score']}/100", delta="Lower is better")
                    if boring_metrics['is_boring']:
                        st.error(" BORING - Add visual variety!")
                        st.warning("⚠️ **This connects directly to Impressions (Stage 0).** Keywords and tags only get a video its FIRST few impressions. What happens after — retention/watch time on those first views — is what YouTube's algorithm uses to decide whether to keep showing it. A high boring score means early viewers are likely bouncing fast, which caps impressions regardless of how good your keywords are. Fixing pacing here is an impressions fix, not just a watch-time fix.")
                    else: st.success("✅ ENGAGING - Good visual dynamics.")

                if title_input:
                    st.markdown("#### 🎨 AI Thumbnail Brief & Prompt")
                    with st.spinner("Generating brief..."): thumb_brief = generate_thumbnail_brief(title_input, final_transcript, topic_input, is_faceless)
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
                    st.markdown("#### 🧠 AI Hook Builder")
                    with st.spinner("Weaving your ingredients..."): llm_data = analyze_script_with_llm(problem_input, mechanism_input, payoff_input, cpm or 10, is_short)
                    if "error" not in llm_data:
                        s1, s2, s3 = st.columns(3)
                        s1.metric("Pattern Interrupt", f"{llm_data.get('pattern_interrupt_score', 0)}/10")
                        s2.metric("Value Prop", f"{llm_data.get('value_prop_score', 0)}/10")
                        s3.metric("Jargon Control", f"{llm_data.get('jargon_score', 0)}/10")
                        st.success(llm_data.get('script_rewrite', 'N/A'))

            # === STAGE 3: SUBSCRIBE — NEW FUNNEL ADVISOR ===
            st.markdown("---"); st.header("3️⃣ SUBSCRIBE — Conversion Advisor (NEW)")
            st.caption("Your only real conversion goal on YouTube. This checks if/when/how you ask, and generates CTAs that fit a skeptical quant/finance audience.")
            if final_transcript and final_transcript != "No transcript available.":
                with st.spinner("Analyzing subscribe-conversion mechanics..."):
                    sub_result = analyze_subscribe_funnel(final_transcript, title_input, topic_input, mode_name)
                if "error" in sub_result:
                    st.warning(sub_result["error"])
                else:
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("CTA Detected", "Yes" if sub_result.get("cta_detected") else "No")
                    sc2.metric("CTA Timing", sub_result.get("cta_timing", "none").title())
                    sc3.metric("Value-Before-Ask", f"{sub_result.get('value_before_ask_score', 0)}/100")
                    st.markdown(f"**CTA Style:** {sub_result.get('cta_style', 'none').replace('_', ' ').title()}")
                    st.warning(f"**Diagnosis:** {sub_result.get('diagnosis', 'N/A')}")
                    st.markdown("**Suggested soft-ask (mid-video, right after a value payoff):**")
                    st.info(sub_result.get("soft_ask_line", "N/A"))
                    st.markdown("**Suggested hard-ask (outro):**")
                    st.success(sub_result.get("hard_ask_line", "N/A"))
            else:
                st.info("Provide a YouTube URL with captions to run the subscribe-funnel analysis (needs the full transcript).")

        for f in [thumb_path, video_path, new_thumb_path]:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass
