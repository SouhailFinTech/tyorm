import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import yt_dlp
import tempfile
import os
import json
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
import re
import urllib.request
import zipfile

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

# ---------------------------------------------------------------------------
# CHANNEL-SPECIFIC RULES — derived from this channel's own real Studio data,
# not generic copywriting advice. These get injected as HARD constraints into
# every title/thumbnail prompt below, so the AI can't quietly ignore them.
# Update this if/when new patterns get confirmed from real performance data.
# ---------------------------------------------------------------------------
CHANNEL_TITLE_RULES = """This channel has DATA-BACKED rules from its own real performance history. Apply these as HARD constraints, not stylistic suggestions — a title that violates rule 1 or 2 is a failed output, not a valid alternative:
1. FORMULA (required): every title must follow [Personal action] + [specific number] + [concrete outcome]. Shape: "I [did X] and got [specific number/%] in [timeframe]." A title with no specific number tied to a personal outcome is not acceptable, even if it's catchy.
2. NEVER use a question mark. On this channel, titles with "?" measurably underperform 3x (0.77% vs 2.23% real CTR). If a draft is phrased as a question, rewrite it as a statement.
3. Thumbnails must visualize the SPECIFIC RESULT/NUMBER (e.g. a before/after comparison like "22% vs 79%"), not a generic process screenshot (plain chart or code with no result shown)."""

# Shorts follow a DIFFERENT winning pattern than long-form on this channel — Search
# is the dominant discovery lever for Shorts (not Browse, which is what drives
# long-form), so the title's job is to mirror a real, commonly-searched, tutorial/
# action-intent phrase — not tell a personal-outcome story like long-form titles do.
SHORTS_TITLE_RULES = """This channel has DATA-BACKED rules from its own real Shorts performance — apply as HARD constraints:
1. FORMAT (required): title must be either a TUTORIAL promise ("How I [did X] in [short timeframe]") or a LISTICLE ("[N] reasons/ways [claim]") — both won on this channel. Abstract finance concepts, clichés ("the only free lunch"), or dry analytical framing ("X reality check") measurably flopped (near-zero views) even when technically accurate.
2. The core phrase must mirror something people ACTUALLY search for on YouTube (e.g. "algo trading bot", "trading bot python", "auto trading bot") — not a niche/jargon term with low real search volume, even if it's topically correct. A title can rank 80%+ of its (tiny) traffic from Search and still flop if the underlying query itself has almost no volume — the phrase has to be one real people commonly type, not just a technically-relevant one.
3. NEVER use a question mark (same rule as long-form, still holds for Shorts)."""

# Facebook Reel captions follow a DIFFERENT winning pattern again — derived from this
# page's own 36-post export, not assumed. The page had been cross-posting full
# YouTube descriptions as captions, which measurably hurt reach.
FACEBOOK_CAPTION_RULES = """This page has DATA-BACKED rules from its own real Facebook Reels performance — apply as HARD constraints:
1. LENGTH: caption must be SHORT — under 150 characters, one punchy stat-led line. On this page, the two best-performing posts were both under 91 characters; posts over 200 characters (full YouTube-style descriptions) measurably underperformed.
2. NEVER include a link in the caption body. Posts with a link averaged 29% fewer impressions than posts without one — Facebook suppresses reach on off-platform links. If a link is needed, it goes in the first comment, not the caption (say so explicitly in your output).
3. NEVER include a multi-step off-platform funnel (e.g. "subscribe, screenshot, email me for X") in the caption. This measurably cost reach on this page (110 vs 153 avg impressions) — it reads as YouTube-native, not Facebook-native.
4. Hashtags: use around 5, not 15-20. More hashtags showed no reach benefit on this page's real data.
5. STRUCTURE (required): a bare stat alone isn't enough — pair it with a short stakes-setting frame so the reader knows why the number matters, not just what it is. Shape: "[Frame] [Stat]." e.g. "RSI Reality Check: 41% → 58% Win Rate" or "Trend Following Reality Check: 40% WR Needed" (this page's actual top post). A stat with no frame (e.g. just "41% to 58% win rate boost") is a weaker, less complete output than one with a frame attached — always attach one.
6. When a before/after transformation stat is available (e.g. "41% to 58%"), always prefer it over a standalone/negative stat (e.g. "9/10 lost money") — a transformation is what actually won on this page; a lone negative number reads as discouraging, not compelling."""


# ---------------------------------------------------------------------------
# PHASE 1: Calibration & History — closes the loop between predicted scores
# and what actually happened on YouTube. Without this, every score in this
# app is an untested guess. With it, you can see whether thumbnail_score
# actually predicts CTR, and recalibrate the heuristics once there's enough
# real data.
#
# NOTE ON PERSISTENCE: this stores history in a local CSV on the app's own
# filesystem. On free hosting (e.g. Streamlit Community Cloud), that storage
# is NOT permanent — it resets on redeploy/reboot. Download the history CSV
# periodically (button provided below) so you don't lose it. A real database
# (Phase 4) is the permanent fix; this is the free version that works today.
# ---------------------------------------------------------------------------
HISTORY_FILE = "channel_history.csv"
HISTORY_COLUMNS = [
    "timestamp", "video_title", "format",
    "predicted_thumbnail_score", "predicted_title_score",
    "predicted_hook_score", "predicted_boring_score",
    "actual_ctr_pct", "actual_avd_pct", "actual_impressions",
    "actual_views", "actual_subs_gained",
]

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            for col in HISTORY_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[HISTORY_COLUMNS]
        except Exception:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.DataFrame(columns=HISTORY_COLUMNS)

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

def log_prediction_row(video_title, format_label, thumb_score=None, title_score=None, hook_score=None, boring_score=None):
    """Called right after an analysis run to record what the tool PREDICTED,
    before any real outcome is known. Actual outcomes get filled in later via
    CSV import once the video has real Studio data."""
    df = load_history()
    new_row = {col: None for col in HISTORY_COLUMNS}
    new_row.update({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "video_title": video_title,
        "format": format_label,
        "predicted_thumbnail_score": thumb_score,
        "predicted_title_score": title_score,
        "predicted_hook_score": hook_score,
        "predicted_boring_score": boring_score,
    })
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_history(df)
    return df

def parse_youtube_studio_csv(uploaded_file):
    """Flexible parser for YouTube Studio's export. Accepts EITHER a single per-video
    CSV, OR the full .zip bundle Studio actually exports — which contains three files
    (a per-video table, a daily chart-data file, and a channel totals file). Only the
    table file has per-video title + CTR + impressions; this picks that one out
    automatically instead of requiring the user to unzip and find it themselves.
    Also matches French-locale column headers (e.g. "Titre de la vidéo", "Taux de
    clics par impression (%)"), since Studio exports in the account's UI language."""
    filename = getattr(uploaded_file, "name", "") or ""

    def find_col(df, possible_substrings):
        for col in df.columns:
            col_lower = str(col).lower()
            for sub in possible_substrings:
                if sub in col_lower:
                    return col
        return None

    candidate_dfs = []
    if filename.lower().endswith(".zip"):
        try:
            zf = zipfile.ZipFile(uploaded_file)
        except Exception as e:
            return None, f"Could not read zip file: {e}"
        for name in zf.namelist():
            if name.lower().endswith(".csv"):
                try:
                    with zf.open(name) as f:
                        df = pd.read_csv(f)
                    candidate_dfs.append((name, df))
                except Exception:
                    continue
        if not candidate_dfs:
            return None, "No CSV files found inside that zip."
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            return None, f"Could not read CSV: {e}"
        candidate_dfs.append((filename or "uploaded.csv", df))

    # Among the candidate CSVs (1 if a plain CSV was uploaded, 3 if it was the zip),
    # pick the one that actually has a per-video title column AND a CTR or
    # impressions column — the chart-data and totals files lack one or both of these,
    # so this reliably isolates the right file without asking the user to know which
    # one it is.
    title_col = ctr_col = impressions_col = df_to_use = None
    for name, cand_df in candidate_dfs:
        # Note: "Content"/"Contenu" is the video-ID column in real Studio exports, NOT
        # the title — deliberately excluded here so it can't shadow the real title
        # column ("Video title" / "Titre de la vidéo").
        t = find_col(cand_df, ["video title", "titre de la vidéo", "titre de la video", "titre"])
        c = find_col(cand_df, ["click-through rate", "click through rate", "ctr", "taux de clics"])
        i = find_col(cand_df, ["impressions"])
        if t and (c or i):
            title_col, ctr_col, impressions_col, df_to_use = t, c, i, cand_df
            break

    if df_to_use is None:
        return None, "Couldn't find a per-video table with a title + CTR/impressions column. If you uploaded the Studio zip, make sure it's the unmodified export; if a single CSV, make sure it's the 'Table data' file, not the chart-data or totals file."

    avd_col = find_col(df_to_use, ["average percentage viewed", "average % viewed", "avg % viewed", "average view percentage", "pourcentage moyen visionné", "pourcentage moyen vu"])
    views_col = find_col(df_to_use, ["views", "vues"])
    subs_col = find_col(df_to_use, ["subscribers", "abonnés", "abonnes"])

    parsed_rows = []
    for _, row in df_to_use.iterrows():
        title_val = row.get(title_col)
        if pd.isna(title_val) or not str(title_val).strip():
            continue  # skips the "Total" aggregate row, which has an empty title
        parsed_rows.append({
            "video_title": str(title_val).strip(),
            "actual_ctr_pct": row.get(ctr_col) if ctr_col else None,
            "actual_avd_pct": row.get(avd_col) if avd_col else None,
            "actual_impressions": row.get(impressions_col) if impressions_col else None,
            "actual_views": row.get(views_col) if views_col else None,
            "actual_subs_gained": row.get(subs_col) if subs_col else None,
        })
    return parsed_rows, None

def merge_actuals_into_history(parsed_rows):
    """Matches imported CSV rows to existing logged predictions by video title
    (case-insensitive substring match, since Studio sometimes truncates titles).
    Unmatched rows get appended as new history entries with no prediction data."""
    df = load_history()
    matched_count = 0
    new_count = 0
    for prow in parsed_rows:
        title = str(prow.get("video_title", "")).strip()
        if not title:
            continue
        match_idx = None
        for idx, hrow in df.iterrows():
            existing_title = str(hrow.get("video_title", "")).strip()
            if existing_title and existing_title.lower() != "nan" and (
                existing_title.lower() in title.lower() or title.lower() in existing_title.lower()
            ):
                match_idx = idx
                break
        if match_idx is not None:
            for k in ["actual_ctr_pct", "actual_avd_pct", "actual_impressions", "actual_views", "actual_subs_gained"]:
                if prow.get(k) is not None and not pd.isna(prow.get(k)):
                    df.at[match_idx, k] = prow[k]
            matched_count += 1
        else:
            new_row = {col: None for col in HISTORY_COLUMNS}
            new_row.update(prow)
            new_row["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
            new_row["format"] = "imported"
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            new_count += 1
    save_history(df)
    return df, matched_count, new_count

def compute_calibration(df):
    """Correlates predicted scores against actual outcomes wherever both exist
    for the same video. This is the actual answer to 'do these scores mean
    anything' — not a vibe, a number."""
    results = {}
    pairs = [
        ("predicted_thumbnail_score", "actual_ctr_pct", "Thumbnail Score vs CTR"),
        ("predicted_title_score", "actual_ctr_pct", "Title Score vs CTR"),
        ("predicted_hook_score", "actual_avd_pct", "Hook Score vs Avg % Viewed"),
        ("predicted_boring_score", "actual_avd_pct", "Boring Score vs Avg % Viewed (expect negative)"),
    ]
    for pred_col, actual_col, label in pairs:
        if pred_col not in df.columns or actual_col not in df.columns:
            results[label] = {"correlation": None, "n": 0}
            continue
        sub = df[[pred_col, actual_col]].copy()
        sub[pred_col] = pd.to_numeric(sub[pred_col], errors="coerce")
        sub[actual_col] = pd.to_numeric(sub[actual_col], errors="coerce")
        sub = sub.dropna()
        if len(sub) >= 3:
            corr = np.corrcoef(sub[pred_col].astype(float), sub[actual_col].astype(float))[0, 1]
            results[label] = {"correlation": round(float(corr), 2), "n": len(sub)}
        else:
            results[label] = {"correlation": None, "n": len(sub)}
    return results

# ---------------------------------------------------------------------------
# PHASE 2 (channel-audit half): works directly on real imported Studio data —
# no predictions, no backfilling, no waiting. Ranks your actual videos and
# checks whether simple title patterns correlate with real CTR on YOUR
# channel specifically, using the data you already have.
# ---------------------------------------------------------------------------
def extract_title_features(title):
    """Deterministic, no LLM — cheap enough to run on every row."""
    title = title or ""
    return {
        "length": len(title),
        "word_count": len(title.split()),
        "has_number": bool(re.search(r"\d", title)),
        "has_question": "?" in title,
        "hashtag_count": title.count("#"),
        "has_colon": ":" in title,
    }

def compute_channel_audit(df):
    """Ranks real videos by CTR and checks whether simple title features
    correlate with real CTR/views on this specific channel. Only uses rows
    with real actual_ctr_pct data — works even with zero logged predictions."""
    working = df.copy()
    working["actual_ctr_pct"] = pd.to_numeric(working.get("actual_ctr_pct"), errors="coerce")
    working["actual_views"] = pd.to_numeric(working.get("actual_views"), errors="coerce")
    working = working.dropna(subset=["actual_ctr_pct"])
    working = working[working["video_title"].astype(str).str.strip() != ""]
    working = working[working["video_title"].astype(str).str.lower() != "total"]

    if working.empty:
        return None

    ranked = working.sort_values("actual_ctr_pct", ascending=False).reset_index(drop=True)

    feat_rows = []
    for _, row in working.iterrows():
        feats = extract_title_features(str(row["video_title"]))
        feats["actual_ctr_pct"] = row["actual_ctr_pct"]
        feat_rows.append(feats)
    feat_df = pd.DataFrame(feat_rows)

    feature_insights = {}
    n = len(feat_df)
    if n >= 5:
        for feat_col in ["length", "word_count", "hashtag_count"]:
            sub = feat_df[[feat_col, "actual_ctr_pct"]].dropna()
            if len(sub) >= 5 and sub[feat_col].std() > 0:
                corr = np.corrcoef(sub[feat_col].astype(float), sub["actual_ctr_pct"].astype(float))[0, 1]
                feature_insights[feat_col] = round(float(corr), 2)
        for bool_col in ["has_number", "has_question", "has_colon"]:
            with_feat = feat_df[feat_df[bool_col] == True]["actual_ctr_pct"]
            without_feat = feat_df[feat_df[bool_col] == False]["actual_ctr_pct"]
            if len(with_feat) >= 2 and len(without_feat) >= 2:
                feature_insights[bool_col] = {
                    "avg_ctr_with": round(float(with_feat.mean()), 2),
                    "avg_ctr_without": round(float(without_feat.mean()), 2),
                    "n_with": len(with_feat),
                    "n_without": len(without_feat),
                }

    return {
        "ranked": ranked,
        "feature_insights": feature_insights,
        "n_videos": n,
        "avg_ctr": round(float(working["actual_ctr_pct"].mean()), 2),
        "median_ctr": round(float(working["actual_ctr_pct"].median()), 2),
    }

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

# === NEW: Spoken-Keyword Search Checker ===
# YouTube indexes SPOKEN words via captions/transcript for search ranking, not just
# title/description/tags. A video can have a perfect title and still under-rank for
# search if the target keyword is never actually said. This is deterministic —
# real string matching against the real transcript, not an LLM guess.
def check_keywords_spoken(timed_transcript, keywords, early_window_seconds=60):
    if not timed_transcript or not keywords:
        return None
    full_text = " ".join(seg.get("text", "") for seg in timed_transcript).lower()
    early_text = " ".join(
        seg.get("text", "") for seg in timed_transcript if seg.get("start", 0) <= early_window_seconds
    ).lower()

    results = []
    for kw in keywords:
        kw_clean = str(kw).strip().lower()
        if not kw_clean:
            continue
        said_at_all = kw_clean in full_text
        said_early = kw_clean in early_text
        first_timestamp = None
        if said_at_all:
            running = ""
            for seg in timed_transcript:
                running += " " + seg.get("text", "").lower()
                if kw_clean in running:
                    first_timestamp = seg.get("start", 0)
                    break
        results.append({
            "keyword": kw,
            "said_at_all": said_at_all,
            "said_early": said_early,
            "first_timestamp": first_timestamp,
        })
    return results

# === NEW: Shorts vertical-format check ===
# A Short uploaded in landscape gets letterboxed/cropped badly by the Shorts player,
# which tanks completion rate immediately — checking this takes one line and catches
# a mistake that's otherwise invisible until after upload.
def check_vertical_aspect(video_path):
    if not video_path or not os.path.exists(video_path):
        return {"error": "No video file to check"}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w == 0 or h == 0:
        return {"error": "Could not read dimensions"}
    is_vertical = h > w
    aspect = round(w / h, 3)
    return {"width": w, "height": h, "is_vertical": is_vertical, "aspect_ratio": aspect}

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
    prompt = f"""{CHANNEL_TITLE_RULES}

You are a YouTube thumbnail designer expert for technical/finance channels. Title: "{title}". Topic: {topic}. Transcript Snippet: "{transcript[:300]}". {face_instruction} Follow rule 3 above strictly — the thumbnail concept must visualize the specific number/result from the title. Output STRICT JSON: "thumbnail_text" (string, max 5 words, should include the specific number if the title has one), "color_scheme" (object with background, text, accent hex codes), "layout" (string description), "visual_elements" (array of strings), "midjourney_prompt" (string, detailed), "style" (string), "dos" (array of 3 strings), "donts" (array of 3 strings), "thumbnail_score_prediction" (int 0-100)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.7, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def analyze_title_with_llm(title, transcript, topic, is_short=False):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    if is_short:
        prompt = f"""{SHORTS_TITLE_RULES}

You are a YouTube Shorts SEO expert. Current Title: "{title}". Topic: {topic}. Additional constraint: must be under 50 chars. Follow the Shorts rules above strictly when scoring and generating alternatives. Output STRICT JSON: "title_score" (int, penalize heavily if the format/search-phrase/question-mark rules above are violated), "character_count" (int), "is_optimal_length" (bool), "alternative_titles" (array of 3 strings, ALL must follow the Shorts rules above), "recommended_keywords" (array of 5 strings)"""
    else:
        prompt = f"""{CHANNEL_TITLE_RULES}

You are a YouTube SEO expert for technical/finance content. Current Title: "{title}". Topic: {topic}. Transcript Snippet: "{transcript[:300]}". Follow the channel rules above strictly when scoring and generating alternatives. Output STRICT JSON: "title_score" (int, penalize heavily if the formula/question-mark rules above are violated), "character_count" (int), "is_optimal_length" (bool), "alternative_titles" (array of 3 strings, ALL must follow the formula above), "recommended_keywords" (array of 5 strings), "emotional_triggers" (string), "improvement_notes" (string)"""
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
    prompt = f"""You are a top 1% Quantitative Researcher on X. Topic: {topic}. Source: "{transcript[:1500]}". TASK: Write a 6-tweet thread. RULES: 1. TWEET 1 (Hook): Under 280 chars. Contrarian take or hard data. NO "In this thread...". No external links in tweet 1 — X's algorithm suppresses reach on tweets with outbound links. 2. TWEETS 2-4 (Meat): Methodology, bullet points, technical terms. 3. TWEET 5 (Reality Check): Brutal truth or final metric. 4. TWEET 6 (CTA & Trap): Follow CTA + specific question to force replies. Output STRICT JSON: "tweet_1" (string), "tweet_2" (string), "tweet_3" (string), "tweet_4" (string), "tweet_5" (string), "tweet_6" (string), "engagement_question" (string)"""
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

# === NEW: Deterministic character-count validator (real Python len(), not the LLM's
# self-reported "under 280 chars" claim, which is unverified and sometimes wrong) ===
def validate_tweet_lengths(tweets, limit=280):
    results = []
    for i, t in enumerate(tweets, 1):
        length = len(t) if t else 0
        results.append({
            "index": i,
            "length": length,
            "over_limit": length > limit,
            "has_link": bool(re.search(r"https?://", t or "")),
        })
    return results

def validate_threads_post_length(text, limit=500):
    length = len(text) if text else 0
    return {"length": length, "over_limit": length > limit}

# === NEW: Facebook Reel Caption — deterministic validation + generation, separate
# tag from X/Threads since this page's real data shows a different winning pattern
# (short, link-free, funnel-free, stat-led) than either of those platforms. ===
def check_hashtags_for_subscribe_push(hashtags):
    """Catches the same off-platform-push pattern the funnel-CTA check catches in
    caption body text, but in the hashtag list — a #SubscribeNow hashtag is the same
    energy that measurably cost this page reach, just relocated."""
    if not hashtags:
        return []
    flagged = []
    for h in hashtags:
        h_clean = str(h).lstrip("#").lower()
        if re.search(r"subscribe|followme|follow.?now|joinnow", h_clean):
            flagged.append(h)
    return flagged

def validate_facebook_caption(caption, length_limit=150):
    caption = caption or ""
    return {
        "length": len(caption),
        "over_limit": len(caption) > length_limit,
        "has_link": bool(re.search(r"https?://", caption)),
        "has_funnel_cta": bool(re.search(r"screenshot|email me|gumroad|subscribe.{0,20}(email|screenshot|dm)", caption, re.IGNORECASE)),
        "hashtag_count": caption.count("#"),
    }

def generate_facebook_caption(topic, transcript):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    prompt = f"""{FACEBOOK_CAPTION_RULES}

You are writing a Facebook Reel caption for a quant/algo-trading page. Topic: {topic}. Source content: "{transcript[:1200] if transcript else 'Topic: ' + topic}".

TASK: Write ONE caption strictly following the rules above. STAT SELECTION: the source may contain several numbers — scan all of them and prefer a POSITIVE BEFORE/AFTER TRANSFORMATION (e.g. "41% to 58% win rate", a specific improvement) over a standalone negative/failure stat (e.g. "9/10 lost money") when both are present. A transformation stat is what actually won on this page before; a lone negative stat without the fix attached reads as discouraging, not compelling. If a link would normally be relevant, do NOT put it in the caption — instead note it should go in the first comment. Hashtags must be topical/content-related only (e.g. #AlgoTrading, #QuantFinance) — NEVER a subscribe/follow-push hashtag like #SubscribeNow or #FollowMe. That's the same off-platform-push pattern that cost this page real reach, just moved into a hashtag instead of the caption body.

Output STRICT JSON: "caption" (string, under 150 chars), "hashtags" (array of 5 strings, topical only, no subscribe/follow-push hashtags), "link_placement_note" (string, e.g. "Put your video/product link in the first comment, not here")"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.5, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

# === NEW: Thread Quality / Retention Analyzer (works on generated OR pasted-in threads) ===
# X's algorithm rewards dwell time and reply engagement on a thread, not just the hook.
# A thread that opens strong but loses the reader by tweet 3 gets throttled the same as
# a weak hook would. This scores the WHOLE thread, and flags exactly where it would drop off.
def analyze_thread_quality(tweets, platform="X (Twitter) Thread"):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    if not tweets or not any(tweets):
        return {"error": "No thread content to analyze."}
    client = Groq(api_key=api_key)
    numbered = "\n".join(f"Tweet {i+1}: {t}" for i, t in enumerate(tweets) if t)
    prompt = f"""You are a growth strategist for technical/finance/quant creators on {platform}, where the algorithm rewards dwell time (reading the whole thread) and reply engagement, not just impressions on tweet 1.

Full thread:
{numbered}

TASK:
1. Identify the single tweet in this thread MOST likely to cause a reader to stop scrolling/reading (weakest transition, jargon spike, or lost momentum) — not necessarily the first one.
2. Score overall thread cohesion — does each tweet earn the next one, or does it feel like a list of disconnected facts?
3. Score the final engagement question/CTA on whether it would actually generate replies (specific, answerable, mildly provocative) vs generic ("thoughts?").
4. Flag if the thread reads as authoritative/credible for a skeptical quant audience, or too generic/AI-sounding.

Output STRICT JSON: "weakest_tweet_index" (int), "weakest_tweet_reason" (string), "cohesion_score" (int 0-100), "reply_bait_score" (int 0-100), "authority_score" (int 0-100), "overall_thread_score" (int 0-100), "single_biggest_fix" (string)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.4, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}


# === NEW: Hook Window Deep-Analysis (click -> watch) ===
# Scores ONLY what's actually spoken/shown in the first ~15s, not the whole script.
# This is the moment that decides whether a click becomes a view or an instant bounce.
def analyze_hook_window_with_llm(hook_text, title, topic, niche_mode="Technical", is_short=False):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    if not hook_text:
        window_label = "3 seconds" if is_short else "15 seconds"
        return {"error": f"No hook-window text available — paste your first-{window_label} script manually below."}
    client = Groq(api_key=api_key)
    window_seconds = 3 if is_short else 15
    format_note = (
        "This is a SHORT — there is no click decision, only a swipe-away decision. The viewer decides "
        "to keep watching in roughly the first 3 seconds, and the FIRST FRAME must already show the "
        "payoff or a strong visual, not build up to it. A slow open, even a 2-second one, is fatal here."
        if is_short else
        "This is a LONG-FORM video — the viewer already clicked, so the hook's job is to confirm the "
        "click was worth it and open a curiosity gap for the next few minutes."
    )
    prompt = f"""You are a YouTube retention specialist for {niche_mode} finance/quant creators.
Video Title: "{title}"
Topic: {topic}
{format_note}
EXACT words spoken/shown in the first ~{window_seconds} seconds: "{hook_text}"

Score this hook window on what determines whether a viewer keeps watching:
1. Curiosity gap (did it open a question the viewer needs answered?)
2. Promise clarity (is it obvious what specific payoff they'll get?)
3. Pattern interrupt (does it avoid a generic/slow intro — "hey guys welcome back" style openers are a major retention killer)
4. Relevance match to the title (does the hook deliver on what the title/thumbnail promised, avoiding bait-and-switch?)

Output STRICT JSON: "curiosity_gap_score" (int 0-100), "promise_clarity_score" (int 0-100), "pattern_interrupt_score" (int 0-100), "title_match_score" (int 0-100), "overall_hook_window_score" (int 0-100), "biggest_risk" (string, the single most likely reason a viewer would bounce in this window), "rewritten_hook" (string, a stronger version of this hook window)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.4, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

# === NEW: Subscribe Funnel Advisor (watch -> subscribe) ===
# Your original tool had zero logic for the actual conversion goal you stated: turning
# viewers into subscribers. This detects CTA presence/timing and generates CTAs that
# fit a quant/finance audience (who bounce off hypey generic "smash that subscribe" asks).
def analyze_subscribe_funnel(transcript, title, topic, niche_mode="Technical", is_short=False):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    if not transcript:
        return {"error": "No transcript available to analyze."}
    client = Groq(api_key=api_key)
    if is_short:
        format_note = (
            "This is a SHORT — there is essentially no room for a spoken mid-video ask without killing "
            "pacing (Shorts run under 60s and every second of narration competes with completion rate). "
            "A subscribe ask here should be a brief ON-SCREEN TEXT overlay in the last 1-2 seconds "
            "(not spoken), timed to land right after the payoff/punchline, plus optionally one word "
            "in the caption/hashtag line. Do NOT recommend a spoken CTA line that would eat into pacing."
        )
    else:
        format_note = (
            "This is LONG-FORM — there's room for both a brief spoken soft-ask after a value payoff "
            "mid-video, and a slightly longer hard-ask in the outro."
        )
    prompt = f"""You are a YouTube channel growth strategist specializing in technical/finance/quant creators, where audiences are skeptical of hypey asks and respond better to earned, specific CTAs.

Video Title: "{title}"
Topic: {topic}
{format_note}
Full transcript (may be truncated): "{transcript[:3000]}"

TASK: Analyze this transcript for subscribe-conversion mechanics only, matched to the format above.
1. Detect whether there is any verbal or on-screen subscribe/follow ask in the transcript, and roughly where (early/mid/late/none).
2. Judge the "value-before-ask" ratio: does the creator deliver real, specific value (a number, a method, a concrete insight) BEFORE any ask, or does the ask come too early/generic?
3. Flag if the ask is generic/hypey ("smash that subscribe button") vs specific and earned (tied to a concrete reason to come back).
4. Generate a SOFT-ASK line and a HARD-ASK line appropriate to the format above (for Shorts: short on-screen text, not spoken; for long-form: spoken lines), both written for a technical/quant/finance audience — no hype language, tied to a specific, credible reason to subscribe.

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
def generate_impressions_strategy(title, topic, transcript, niche_mode="Technical", is_faceless=False, is_short=False):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    client = Groq(api_key=api_key)
    faceless_note = (
        "The channel is faceless, so discovery strategy should lean harder on topic "
        "consistency, a recognizable branding system, and search intent — since there's "
        "no personality-driven audience pull to rely on early on."
        if is_faceless else ""
    )
    if is_short:
        format_context = (
            "This is a YOUTUBE SHORT (<60s), which is discovered almost entirely differently from "
            "long-form: the Shorts feed is a swipe-driven autoplay surface, not a search/browse-grid "
            "surface. Impressions here are driven overwhelmingly by (a) completion rate / rewatches "
            "(does the viewer watch to the end, or better, loop it), (b) shares and sends, not clicks, "
            "and (c) hashtags in the description, which carry more weight here than full tag lists do "
            "for long-form. Search still matters but is secondary. Title/thumbnail CTR is nearly "
            "irrelevant since there's no click decision in the Shorts feed — only a swipe-away decision "
            "in the first ~1-2 seconds."
        )
        extra_fields = (
            '"loopability_tactic" (string, one specific way to make the last second connect back to '
            'the first second so the video rewatches instead of ending flatly), '
            '"hashtags_to_use" (array of 5 hashtags, since these matter more than long-tags for Shorts discovery), '
        )
    else:
        format_context = "This is a LONG-FORM video, discovered mainly through Search and Suggested/Browse (topic-clustering + session time)."
        extra_fields = ""

    prompt = f"""You are a YouTube growth strategist specializing in small/new technical-finance/quant channels (this one is ~3 months old and still building initial traction).
Title: "{title}"
Topic/Keyword: {topic}
Niche: {niche_mode}
Transcript snippet: "{transcript[:800] if transcript else 'N/A'}"
{faceless_note}
{format_context}

TASK: Give a concrete impressions/discovery plan matched to this format.

Output STRICT JSON:
"search_keywords" (array of 8 realistic search phrases a target viewer would actually type into YouTube for this topic — not generic SEO fluff, actual query phrasing),
"tags_to_use" (array of 10 YouTube tags, ordered broad-to-specific),
{extra_fields}"suggested_video_cluster_strategy" (string, 2-3 sentences on what adjacent/related videos or a series structure would build a topic cluster this platform can reliably suggest this channel's videos within),
"upload_cadence_advice" (string, 1-2 sentences realistic for a solo creator),
"session_time_tactic" (string, one concrete tactic to increase watch-next/rewatch behavior appropriate to this format),
"biggest_impressions_blocker" (string, the single most likely reason a 3-month-old channel in this niche is getting few impressions on this format)
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

# === NEW: Pre-Upload Title & Thumbnail Generator — works from the raw SCRIPT of a
# video that hasn't been published (or even filmed) yet, since the AI can't "watch"
# your video before it exists. This is also where Browse-optimization (the channel
# formula) and Search-optimization (real query phrases) get reconciled into one
# title instead of treating them as separate, conflicting jobs. ===
def generate_title_thumb_from_script(script_text, topic, niche_mode="Technical", is_faceless=False, search_keywords=None, is_short=False):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key: return {"error": "No Groq API Key found."}
    if not script_text or not script_text.strip():
        return {"error": "No script provided — paste your video's script/outline above."}
    client = Groq(api_key=api_key)
    face_instruction = (
        "This is a FACELESS channel — design around bold typography, data visualizations, "
        "and color-blocking, never a presenter face."
        if is_faceless else "A presenter face can be used if it helps."
    )
    rules_block = SHORTS_TITLE_RULES if is_short else CHANNEL_TITLE_RULES
    kw_note = (
        f"Real search phrases your audience actually types (from channel discovery data): {search_keywords}. "
        "Where possible without breaking the format rules, try to make ONE candidate title naturally contain "
        "one of these phrases."
        if search_keywords else
        "No pre-researched search phrases were provided for this run."
    )
    prompt = f"""{rules_block}

You are a YouTube title/thumbnail strategist for a {niche_mode} channel. This video has NOT been uploaded yet — here is its full script/outline, which is your only source of truth about what's actually in it:
"{script_text[:3000]}"
Topic: {topic}
{kw_note}
{face_instruction}

TASK: Generate 3 title candidates, ALL strictly following the rules above — pull the specific number/outcome/phrase from the actual script content, don't invent one. STAT SELECTION: if the script contains several numbers, prefer a positive before/after transformation (e.g. "41% to 58%") over a standalone negative/failure stat when both are present — a transformation is what's actually won on this channel before; a lone negative number reads as discouraging, not compelling. For each title, note whether it naturally contains one of the given search phrases. Then write a thumbnail brief for the strongest candidate.

Output STRICT JSON: "titles" (array of 3 objects, each with "title" (string), "contains_search_phrase" (bool), "matched_phrase" (string or null)), "thumbnail_text" (string, max 5 words), "layout" (string), "midjourney_prompt" (string, detailed), "why_this_works" (string, 1-2 sentences tying it to this channel's own real performance pattern)"""
    try:
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.6, response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

# --- UI ---
st.title("📈 QuantTube Analyzer Pro")
st.markdown("Proprietary CV & NLP pipeline for Algo-Trading YouTube optimization — **Click → Watch → Subscribe funnel.**")

# === PHASE 2 (channel-audit half): shows immediately if real Studio data has been
# imported, using zero predictions — works on real CTR data alone. ===
_audit_history = load_history()
_audit_result = compute_channel_audit(_audit_history) if not _audit_history.empty else None
if _audit_result:
    with st.expander(f"📈 Channel Audit — {_audit_result['n_videos']} real videos analyzed (NEW, Phase 2)", expanded=False):
        st.caption("Runs directly on your imported Studio data — no predictions needed. Ranks your actual videos and checks whether simple title patterns correlate with real CTR on this specific channel.")
        ac1, ac2 = st.columns(2)
        ac1.metric("Channel avg CTR", f"{_audit_result['avg_ctr']}%")
        ac2.metric("Channel median CTR", f"{_audit_result['median_ctr']}%")

        ranked = _audit_result["ranked"]
        st.markdown("**🏆 Top 5 by real CTR:**")
        top5 = ranked.head(5)[["video_title", "actual_ctr_pct", "actual_views", "actual_impressions"]]
        st.dataframe(top5, use_container_width=True, hide_index=True)
        st.markdown("**🔻 Bottom 5 by real CTR (audit targets):**")
        bottom5 = ranked.tail(5)[["video_title", "actual_ctr_pct", "actual_views", "actual_impressions"]]
        st.dataframe(bottom5, use_container_width=True, hide_index=True)

        insights = _audit_result["feature_insights"]
        if insights:
            st.markdown("**📊 Title patterns vs real CTR on your channel:**")
            for key, val in insights.items():
                if isinstance(val, dict):
                    diff = val["avg_ctr_with"] - val["avg_ctr_without"]
                    direction = "higher" if diff > 0 else "lower"
                    label_map = {"has_number": "Has a number", "has_question": "Has a '?'", "has_colon": "Has a ':'"}
                    st.write(f"- **{label_map.get(key, key)}:** {val['avg_ctr_with']}% avg CTR (n={val['n_with']}) vs {val['avg_ctr_without']}% without (n={val['n_without']}) — {abs(round(diff,2))}pp {direction}")
                else:
                    label_map = {"length": "Title length (chars)", "word_count": "Word count", "hashtag_count": "Hashtag count"}
                    st.write(f"- **{label_map.get(key, key)} vs CTR:** r = {val}")
            st.caption("These are correlations on your own past videos, not causal proof — but with 50+ real data points, they're a far better guide than guessing.")
        else:
            st.info("Not enough videos with real CTR data yet for pattern detection (need 5+).")

with st.sidebar:
    st.header("️ Settings")
    if "GROQ_API_KEY" not in st.secrets: st.warning("No Groq API Key in Secrets.")
    st.markdown("---")
    st.info("**Pro Features:**\n- Long-form & Shorts Mode\n- Mobile Legibility Score (NEW)\n- Hook-Window Deep Analysis (NEW)\n- Subscribe Funnel Advisor (NEW)\n- X & Threads Generator\n- Niche-Aware Scoring\n- Hook Builder\n- Script Compressor\n- A/B Comparator\n- Calibration & History (NEW)")

    st.markdown("---")
    st.subheader("📊 Calibration & History (NEW)")
    st.caption("Upload your real YouTube Studio CSV export to check whether this tool's scores actually predict real performance.")
    studio_csv = st.file_uploader("Upload Studio export (.zip or .csv)", type=["csv", "zip"], key="studio_csv_upload")
    if studio_csv is not None:
        if st.button("Import CSV", key="import_csv_btn", use_container_width=True):
            parsed, err = parse_youtube_studio_csv(studio_csv)
            if err:
                st.error(err)
            elif not parsed:
                st.warning("No usable rows found in that CSV.")
            else:
                _, matched, new = merge_actuals_into_history(parsed)
                st.success(f"Imported: {matched} matched to logged predictions, {new} added as new rows.")

    history_df = load_history()
    if not history_df.empty:
        st.markdown(f"**{len(history_df)} video(s) logged**")
        with st.expander("View history table"):
            st.dataframe(history_df, use_container_width=True, height=200)
        calib = compute_calibration(history_df)
        with st.expander("View calibration (predicted vs actual)"):
            for label, res in calib.items():
                if res["correlation"] is not None:
                    st.write(f"**{label}:** r = {res['correlation']} (n={res['n']})")
                else:
                    st.write(f"**{label}:** not enough matched data yet (n={res['n']}, need 3+)")
            st.caption("r close to +1 or -1 means the score is a strong predictor. r near 0 means it isn't — that's a signal to recalibrate the heuristic, not a bug.")
        csv_bytes = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download history CSV", csv_bytes, file_name="channel_history.csv", mime="text/csv", use_container_width=True)
    else:
        st.caption("No history yet — run an analysis and log it below, or import a Studio CSV.")
    st.caption("⚠️ Stored on the app's own server storage — not permanent on free hosting. Download periodically.")

format_mode = st.radio("🎬 Content Format:", ["Long-form Video (8+ mins)", "YouTube Short (< 60s)", "X (Twitter) Thread", "Threads Post", "Facebook Reel Caption"], horizontal=True)
is_short = (format_mode == "YouTube Short (< 60s)")
is_x = (format_mode == "X (Twitter) Thread")
is_threads = (format_mode == "Threads Post")
is_facebook = (format_mode == "Facebook Reel Caption")
is_text_platform = is_x or is_threads or is_facebook

if is_short: st.info(" **Shorts Mode Active:** AI will enforce <150 words, <50 char titles, and >30 CPM pacing.")
elif is_text_platform: st.info("📱 **Text Platform Active:** AI will optimize for dwell time, bookmarks, and replies.")

# === Inputs — text platforms (X/Threads) get a stripped-down form. Video file upload,
# thumbnail A/B testing, niche/faceless scoring, hook-window/hook-builder, and the
# script compressor are all video-specific and don't apply to a text post, so they're
# not shown here anymore instead of sitting unused above a Threads/X run. ===
if is_text_platform:
    st.subheader("📥 Inputs")
    col_topic1, col_topic2 = st.columns(2)
    with col_topic1:
        url_input = st.text_input(
            "Source URL (optional)",
            placeholder="https://youtube.com/watch?v=... — pulls the transcript as source material, or leave blank"
        )
    with col_topic2:
        topic_input = st.text_input("Main Topic/Keyword", placeholder="e.g., Bitcoin backtesting, Python algo")
    title_input = st.text_input("Post Topic", placeholder="e.g., Why EMA crossovers fail on BTC")

    # NEW: proper source-content box for text platforms. Previously, generation with
    # no URL fell back to just "Topic: " + title_input as the "transcript" — barely
    # enough for the AI to pull a real number/result from, which is why generic hype
    # copy (e.g. "5x Your Trading Edge") showed up instead of a real stat. This gives
    # it actual source material to work from, same role the Script Compressor plays
    # for long-form.
    st.subheader("📝 Script / Source Content (optional but recommended)")
    st.caption("Paste the script, outline, or key result of the post/video this content is for. Without this, generation has to guess at numbers instead of pulling a real one — that's what produced generic hype copy in earlier tests.")
    text_platform_script_input = st.text_area(
        "Paste your script, outline, or key result here...",
        height=150,
        placeholder="e.g. 'I backtested RSI mean-reversion on 4 years of BTC/USD data... Result: 58% win rate, profit factor 1.7...'"
    )

    # Video/thumbnail-only inputs stay defined but empty/None so the shared processing
    # code below doesn't break — they're simply never shown or used in this mode.
    uploaded_file = None
    new_thumb_file = None
    user_description = ""
    manual_hook_input = ""
    problem_input = mechanism_input = payoff_input = ""
    full_script_input = ""
    niche_mode = "Technical (Algo/Coding/Tutorials)"  # unused for X/Threads scoring, kept for shared code path
    is_faceless = False
else:
    text_platform_script_input = ""  # not used outside text-platform mode, defined so downstream code path is safe
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
    _hook_window_label_seconds = 3 if is_short else 15
    st.subheader(f"🎣 First {_hook_window_label_seconds} Seconds (Hook Window)")
    manual_hook_input = st.text_area(
        f"Optional: paste EXACTLY what you say/show in the first {_hook_window_label_seconds} seconds. If left blank and a YouTube URL is provided, this is auto-extracted from the transcript timestamps.",
        height=80
    )

    st.subheader("🎣 Hook Builder (Provide the Ingredients)")
    col_p, col_m, col_pay = st.columns(3)
    with col_p: problem_input = st.text_area("The Problem (Pain points, bad stats)", height=100)
    with col_m: mechanism_input = st.text_area("The Mechanism (Your specific solution)", height=100)
    with col_pay: payoff_input = st.text_area("The Payoff (The result/deliverable)", height=100)

    st.subheader("✂️ Full Script Compressor")
    full_script_input = st.text_area("Paste your full script here...", height=200)

    # === NEW: Pre-Upload Title & Thumbnail Generator ===
    st.subheader("🎯 Title & Thumbnail Generator — From Script (NEW)")
    st.caption("For a video that isn't uploaded yet. Since the AI can't watch it, this uses the script you paste above as its only source of truth, and strictly applies this channel's own data-backed title formula instead of generic suggestions.")
    if st.button("🎯 Generate Data-Backed Title & Thumbnail", use_container_width=True):
        if not full_script_input.strip():
            st.warning("Paste your script in the Full Script Compressor box above first — that's what this reads from.")
        else:
            with st.spinner("Generating from your script..."):
                pre_upload_result = generate_title_thumb_from_script(full_script_input, topic_input, niche_mode.split(" ")[0], is_faceless, is_short=is_short)
            if "error" in pre_upload_result:
                st.warning(pre_upload_result["error"])
            else:
                st.markdown("**📝 Title candidates (formula-compliant):**")
                for i, t in enumerate(pre_upload_result.get("titles", []), 1):
                    tag = "🔎 contains a real search phrase" if t.get("contains_search_phrase") else "📡 Browse/CTR-optimized only"
                    st.info(f"**{i}.** {t.get('title', 'N/A')}  \n_{tag}" + (f" ({t.get('matched_phrase')})_" if t.get("matched_phrase") else "_"))
                st.markdown("**🎨 Thumbnail brief for the strongest candidate:**")
                st.markdown(f"**Text overlay:** {pre_upload_result.get('thumbnail_text', 'N/A')}")
                st.markdown(f"**Layout:** {pre_upload_result.get('layout', 'N/A')}")
                st.code(pre_upload_result.get("midjourney_prompt", ""), language="text")
                st.success(f"**Why this fits your channel:** {pre_upload_result.get('why_this_works', 'N/A')}")

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
                # Only attempt the YouTube-specific fetch if this actually looks like
                # a YouTube URL. Previously this ran unconditionally, so pasting any
                # other link (e.g. a Facebook post URL, valid for the Facebook Caption
                # generator's optional source field) threw a scary "Invalid YouTube
                # URL" error even though nothing was actually broken.
                if "youtube.com" in url_input or "youtu.be" in url_input:
                    thumb_path, transcript, url_error = fetch_thumbnail_and_transcript(url_input)
                    if url_error: st.error(url_error)
                    timed_transcript = fetch_timed_transcript(url_input)
                elif not is_text_platform:
                    st.error("That doesn't look like a YouTube URL.")
                # else: text-platform mode with a non-YouTube URL (e.g. a Facebook
                # link) — silently skipped, since the Script/Source box is the real
                # source of content there and the URL field is optional.
            video_path = None
            if uploaded_file:
                temp_dir = tempfile.gettempdir(); video_path = os.path.join(temp_dir, "uploaded_hook_video.mp4")
                with open(video_path, "wb") as f: f.write(uploaded_file.getbuffer())
            new_thumb_path = None
            if new_thumb_file:
                temp_dir = tempfile.gettempdir(); new_thumb_path = os.path.join(temp_dir, "new_thumb_comparison.jpg")
                with open(new_thumb_path, "wb") as f: f.write(new_thumb_file.getbuffer())

        final_transcript = transcript

        # Preferred source content for X/Threads/Facebook generation, in priority
        # order: pasted script > fetched URL transcript > bare topic string (last
        # resort — this is what produced generic hype copy in earlier tests, since
        # there's no real content for the AI to pull a number/result from).
        text_platform_source = (
            text_platform_script_input.strip() if text_platform_script_input.strip()
            else (final_transcript if final_transcript else "Topic: " + title_input)
        )

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
                with st.spinner("Drafting a viral quant thread..."): thread_data = generate_x_thread(topic_input, text_platform_source)
                if "error" in thread_data: st.error(thread_data["error"])
                else:
                    generated_tweets = [thread_data.get(f"tweet_{i}", "") for i in range(1, 7)]

                    # NEW: real character-count + link-penalty validation (deterministic,
                    # not the LLM's self-reported "under 280 chars" claim)
                    length_checks = validate_tweet_lengths(generated_tweets)
                    st.markdown("### 🧵 Your 6-Tweet Thread (Copy & Paste)")
                    for i, tweet in enumerate(generated_tweets, 1):
                        check = length_checks[i - 1]
                        label = f"Tweet {i}  —  {check['length']}/280 chars"
                        if check["over_limit"]:
                            label += "  ⚠️ OVER LIMIT"
                        if check["has_link"]:
                            label += "  🔗 has link (reach penalty risk)"
                        st.text_area(label, value=tweet, height=100, key=f"tweet_{i}_ui")
                    if any(c["over_limit"] for c in length_checks):
                        st.error("⚠️ One or more tweets exceed 280 characters and will be rejected or truncated on posting — trim before publishing.")
                    if any(c["has_link"] for c in length_checks[:1]):
                        st.warning("🔗 Tweet 1 contains a link — X's algorithm measurably suppresses reach on tweets with outbound links. Move the link to a reply instead.")
                    st.markdown("### 🪤 The Engagement Trap (Post as a reply)"); st.success(thread_data.get('engagement_question', 'N/A'))

                    # NEW: whole-thread quality/retention scoring, not just the hook
                    st.markdown("#### 📊 Thread Quality Analysis (NEW)")
                    with st.spinner("Scoring thread cohesion and drop-off risk..."):
                        quality = analyze_thread_quality(generated_tweets, "X (Twitter) Thread")
                    if "error" in quality:
                        st.warning(quality["error"])
                    else:
                        q1, q2, q3, q4 = st.columns(4)
                        q1.metric("Cohesion", f"{quality.get('cohesion_score', 0)}/100")
                        q2.metric("Reply-Bait Quality", f"{quality.get('reply_bait_score', 0)}/100")
                        q3.metric("Authority", f"{quality.get('authority_score', 0)}/100")
                        q4.metric("Overall", f"{quality.get('overall_thread_score', 0)}/100")
                        st.error(f"**Weakest tweet: #{quality.get('weakest_tweet_index', '?')}** — {quality.get('weakest_tweet_reason', 'N/A')}")
                        st.info(f"**Biggest fix:** {quality.get('single_biggest_fix', 'N/A')}")

            elif is_threads:
                st.subheader("🧵 Threads Post Generator")
                with st.spinner("Drafting an aesthetic Threads post..."): threads_data = generate_threads_post(topic_input, text_platform_source)
                if "error" in threads_data: st.error(threads_data["error"])
                else:
                    post_text = threads_data.get('post_text', '')
                    length_check = validate_threads_post_length(post_text)
                    label = f"Post Text — {length_check['length']}/500 chars"
                    if length_check["over_limit"]: label += "  ⚠️ OVER LIMIT"
                    st.markdown("### 📝 Your Threads Post"); st.text_area(label, value=post_text, height=200)
                    if length_check["over_limit"]:
                        st.error("⚠️ This post exceeds Threads' 500-character limit and will be rejected on posting.")
                    st.markdown("### ️ Visual Asset Idea"); st.info(threads_data.get('image_idea', 'N/A'))

            elif is_facebook:
                st.subheader("📘 Facebook Reel Caption Generator (NEW)")
                st.caption("Separate ruleset from X/Threads — built from this page's own 36-post export, not generic advice.")
                with st.spinner("Drafting a caption from your page's own winning pattern..."):
                    fb_data = generate_facebook_caption(topic_input, text_platform_source)
                if "error" in fb_data: st.error(fb_data["error"])
                else:
                    caption_text = fb_data.get("caption", "")
                    check = validate_facebook_caption(caption_text)
                    label = f"Caption — {check['length']}/150 chars"
                    if check["over_limit"]: label += "  ⚠️ OVER LIMIT"
                    if check["has_link"]: label += "  🔗 has link"
                    if check["has_funnel_cta"]: label += "  ⚠️ funnel CTA detected"
                    st.markdown("### 📝 Your Caption"); st.text_area(label, value=caption_text, height=80)
                    if check["over_limit"]:
                        st.error("⚠️ Over 150 chars — this page's top posts were all under 91 chars. Trim it.")
                    if check["has_link"]:
                        st.error("🔗 Contains a link — this measurably cost 29% reach on this page's real data. Move it to the first comment.")
                    if check["has_funnel_cta"]:
                        st.warning("⚠️ Looks like a subscribe/screenshot/email funnel — this pattern cost real reach on this page. Consider dropping it from the caption.")
                    fb_hashtags = fb_data.get("hashtags", [])
                    bad_hashtags = check_hashtags_for_subscribe_push(fb_hashtags)
                    st.markdown("**Hashtags (aim for ~5, not 15-20):**")
                    st.code(" ".join(f"#{h.lstrip('#')}" for h in fb_hashtags), language="text")
                    if bad_hashtags:
                        st.warning(f"⚠️ {', '.join(bad_hashtags)} — this is the same off-platform-push pattern that cost reach in the caption body, just in hashtag form. Swap for a topical hashtag instead.")
                    st.info(f"**Link placement:** {fb_data.get('link_placement_note', 'Put any link in the first comment, not the caption.')}")

                st.markdown("---")
                st.subheader("🎯 Optimize Your Own Draft (paste existing caption)")
                pasted_fb_caption = st.text_area("Paste your existing Facebook caption...", height=100, key="pasted_fb_caption")
                if st.button("📊 Analyze My Caption", use_container_width=True, key="analyze_fb_caption_btn"):
                    if pasted_fb_caption.strip():
                        pcheck = validate_facebook_caption(pasted_fb_caption)
                        pasted_hashtags = re.findall(r"#\w+", pasted_fb_caption)
                        bad_pasted_hashtags = check_hashtags_for_subscribe_push(pasted_hashtags)
                        flags = []
                        if pcheck["over_limit"]: flags.append("⚠️ over 150 chars")
                        if pcheck["has_link"]: flags.append("🔗 has link (reach penalty)")
                        if pcheck["has_funnel_cta"]: flags.append("⚠️ funnel CTA (reach penalty)")
                        if bad_pasted_hashtags: flags.append(f"⚠️ subscribe-push hashtag(s): {', '.join(bad_pasted_hashtags)} (same reach penalty, hashtag form)")
                        st.metric("Length", f"{pcheck['length']} chars")
                        st.metric("Hashtags", pcheck["hashtag_count"])
                        if flags:
                            for f in flags: st.warning(f)
                        else:
                            st.success("✅ Matches this page's winning pattern — short, no link, no funnel CTA, no subscribe-push hashtags.")
                    else:
                        st.warning("Paste a caption first.")

            if not is_facebook:
                st.markdown("---")
                st.subheader("🎯 Optimize Your Own Draft (paste existing content)")
                st.caption("Already have a thread or post written? Paste it here to get the same scoring the generator above gets — this works on your own drafts, not just AI-generated ones.")
            if is_x:
                pasted_thread = st.text_area(
                    "Paste your thread, one tweet per line...",
                    height=150, key="pasted_thread_input"
                )
                if st.button("📊 Analyze My Thread", use_container_width=True):
                    if pasted_thread.strip():
                        pasted_tweets = [line.strip() for line in pasted_thread.split("\n") if line.strip()]
                        checks = validate_tweet_lengths(pasted_tweets)
                        for i, (tweet, check) in enumerate(zip(pasted_tweets, checks), 1):
                            flag = "  ⚠️ OVER 280" if check["over_limit"] else ""
                            flag += "  🔗 link" if check["has_link"] else ""
                            st.text_area(f"Tweet {i} — {check['length']} chars{flag}", value=tweet, height=60, key=f"pasted_tweet_{i}")
                        with st.spinner("Scoring your thread..."):
                            quality = analyze_thread_quality(pasted_tweets, "X (Twitter) Thread")
                        if "error" not in quality:
                            q1, q2, q3, q4 = st.columns(4)
                            q1.metric("Cohesion", f"{quality.get('cohesion_score', 0)}/100")
                            q2.metric("Reply-Bait Quality", f"{quality.get('reply_bait_score', 0)}/100")
                            q3.metric("Authority", f"{quality.get('authority_score', 0)}/100")
                            q4.metric("Overall", f"{quality.get('overall_thread_score', 0)}/100")
                            st.error(f"**Weakest tweet: #{quality.get('weakest_tweet_index', '?')}** — {quality.get('weakest_tweet_reason', 'N/A')}")
                            st.info(f"**Biggest fix:** {quality.get('single_biggest_fix', 'N/A')}")
                    else:
                        st.warning("Paste a thread first.")
            elif is_threads:
                pasted_post = st.text_area("Paste your Threads post...", height=150, key="pasted_threads_input")
                if st.button("📊 Analyze My Post", use_container_width=True):
                    if pasted_post.strip():
                        check = validate_threads_post_length(pasted_post)
                        flag = "  ⚠️ OVER 500" if check["over_limit"] else ""
                        st.metric("Length", f"{check['length']} chars{flag}")
                    else:
                        st.warning("Paste a post first.")

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
            # Score holders for Phase 1 prediction-logging — filled in as each stage
            # computes its score, then logged together at the end of the run.
            log_thumb_score = None
            log_title_score = None
            log_hook_score = None
            log_boring_score = None

            # === STAGE 0: IMPRESSIONS — how the video gets shown at all (NEW) ===
            st.markdown("---"); st.header("0️⃣ IMPRESSIONS — Getting Seen (NEW)")
            st.caption("Click/Watch/Subscribe all optimize a video that's already being shown to someone. This is upstream of that: what actually gets YouTube to surface it.")
            if title_input and topic_input:
                with st.spinner("Building discovery plan..."):
                    imp_result = generate_impressions_strategy(title_input, topic_input, final_transcript, mode_name, is_faceless, is_short)
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
                        if is_short:
                            st.markdown("**#️⃣ Hashtags (drive more here than tags for Shorts):**")
                            st.code(" ".join(f"#{h.lstrip('#')}" for h in imp_result.get("hashtags_to_use", [])), language="text")
                        else:
                            st.markdown("**🏷️ Tags (broad → specific):**")
                            st.code(", ".join(imp_result.get("tags_to_use", [])), language="text")
                    if is_short and imp_result.get("loopability_tactic"):
                        st.markdown("**🔁 Loopability tactic (make it rewatch, not just finish):**")
                        st.success(imp_result.get("loopability_tactic"))
                    st.markdown("**🧩 Topic-cluster / series strategy:**")
                    st.info(imp_result.get("suggested_video_cluster_strategy", "N/A"))
                    col_i3, col_i4 = st.columns(2)
                    with col_i3:
                        st.markdown("**📅 Upload cadence:**")
                        st.write(imp_result.get("upload_cadence_advice", "N/A"))
                    with col_i4:
                        st.markdown("**⏱️ Session-time / rewatch tactic:**")
                        st.write(imp_result.get("session_time_tactic", "N/A"))

                    # NEW: Spoken-Keyword Search Checker — deterministic, real transcript check
                    if timed_transcript and imp_result.get("search_keywords"):
                        st.markdown("#### 🎙️ Are your target keywords actually SPOKEN? (NEW)")
                        st.caption("YouTube indexes your spoken words via captions for search ranking — a keyword in your title alone isn't enough. This checks your real transcript, not a guess.")
                        spoken_check = check_keywords_spoken(timed_transcript, imp_result.get("search_keywords", []))
                        if spoken_check:
                            for res in spoken_check:
                                if res["said_early"]:
                                    st.success(f"✅ \"{res['keyword']}\" — said at {res['first_timestamp']:.0f}s (within the first minute — good for indexing)")
                                elif res["said_at_all"]:
                                    st.warning(f"⚠️ \"{res['keyword']}\" — said at {res['first_timestamp']:.0f}s, but not until later. Saying it earlier strengthens search relevance.")
                                else:
                                    st.error(f"❌ \"{res['keyword']}\" — never said in the video. If this is a keyword you want to rank for, work it into the script naturally.")
            else:
                st.info("Enter a title and topic/keyword above to generate the discovery plan.")

            # === STAGE 1: CLICK — THUMBNAIL & TITLE (long-form) / SWIPE CHECK (Shorts) ===
            st.markdown("---")
            if is_short:
                st.header("1️⃣ SWIPE — Cover Frame & Vertical Format")
                st.caption("Shorts have no click decision — there's no thumbnail-driven CTR the way long-form has. This stage instead checks the two things that actually matter before a viewer even sees your hook: the video is genuinely vertical, and the auto-picked cover frame isn't broken.")
                if video_path and os.path.exists(video_path):
                    aspect_result = check_vertical_aspect(video_path)
                    if "error" in aspect_result:
                        st.warning(aspect_result["error"])
                    else:
                        if aspect_result["is_vertical"]:
                            st.success(f"✅ Vertical format confirmed ({aspect_result['width']}×{aspect_result['height']}, ratio {aspect_result['aspect_ratio']}).")
                        else:
                            st.error(f"❌ This video is {aspect_result['width']}×{aspect_result['height']} — landscape/square, not vertical. The Shorts player will letterbox or crop it, which hurts completion rate before your hook even gets a chance. Re-export at 1080×1920 (9:16).")
                else:
                    st.info("Upload the video file above to check vertical format.")
                if thumb_path and os.path.exists(thumb_path):
                    st.markdown("You can still set a custom cover image, but it's a minor lever here — put your effort into the first-3-seconds hook below instead.")
                    st.image(thumb_path, use_container_width=True, caption="Current cover/thumbnail reference")
            else:
                st.header("1️⃣ CLICK — Thumbnail & Title")
                st.subheader(f"🖼️ Thumbnail A/B Comparator ({mode_name} Mode)")
                orig_metrics = None; new_metrics = None
                if thumb_path and os.path.exists(thumb_path): orig_metrics = analyze_thumbnail(thumb_path, mode_name, is_faceless)
                if new_thumb_path and os.path.exists(new_thumb_path): new_metrics = analyze_thumbnail(new_thumb_path, mode_name, is_faceless)
                if orig_metrics and "score" in orig_metrics: log_thumb_score = orig_metrics["score"]
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

            # === TITLE OPTIMIZATION (both formats) ===
            if title_input:
                st.markdown("#### 📝 Title Optimization")
                with st.spinner("Analyzing title..."): title_analysis = analyze_title_with_llm(title_input, final_transcript, topic_input, is_short)
                if "error" not in title_analysis:
                    log_title_score = title_analysis.get('title_score')
                    col_t1, col_t2, col_t3 = st.columns(3)
                    col_t1.metric("Title Score", f"{title_analysis.get('title_score', 0)}/100"); col_t2.metric("Characters", title_analysis.get('character_count', 0))
                    col_t3.metric("Length", "✅ Optimal" if title_analysis.get('is_optimal_length') else "⚠️ Adjust")
                    st.markdown("**Alternative Titles:**")
                    for i, alt in enumerate(title_analysis.get('alternative_titles', []), 1): st.info(f"**{i}.** {alt}")

            # === STAGE 2: WATCH — HOOK WINDOW + PACING + BORING SIGNALS ===
            st.markdown("---"); st.header("2️⃣ WATCH — Hook & Retention")

            # NEW: Hook Window Deep-Analysis (first ~3s for Shorts, ~15s for long-form)
            hook_window_seconds = 3 if is_short else 15
            hook_window_text = manual_hook_input.strip() if manual_hook_input.strip() else get_hook_window_text(timed_transcript, window_seconds=hook_window_seconds)
            if title_input:
                st.subheader(f"🎣 First-{hook_window_seconds}-Seconds Hook Analysis (NEW)")
                if is_short:
                    st.caption("For Shorts this window IS the swipe-away decision — there's no click to fall back on, so this score matters more here than the thumbnail ever would.")
                else:
                    st.caption("Scored on ONLY the words spoken in the click->watch window — not the full script.")
                if hook_window_text:
                    with st.spinner("Scoring the hook window..."):
                        hook_window_result = analyze_hook_window_with_llm(hook_window_text, title_input, topic_input, mode_name, is_short)
                    if "error" in hook_window_result:
                        st.warning(hook_window_result["error"])
                    else:
                        log_hook_score = hook_window_result.get('overall_hook_window_score')
                        hc1, hc2, hc3, hc4 = st.columns(4)
                        hc1.metric("Curiosity Gap", f"{hook_window_result.get('curiosity_gap_score', 0)}/100")
                        hc2.metric("Promise Clarity", f"{hook_window_result.get('promise_clarity_score', 0)}/100")
                        hc3.metric("Pattern Interrupt", f"{hook_window_result.get('pattern_interrupt_score', 0)}/100")
                        hc4.metric("Title Match", f"{hook_window_result.get('title_match_score', 0)}/100")
                        st.metric("Overall Hook-Window Score", f"{hook_window_result.get('overall_hook_window_score', 0)}/100")
                        st.error(f"**Biggest bounce risk:** {hook_window_result.get('biggest_risk', 'N/A')}")
                        st.success(f"**Rewritten hook:** {hook_window_result.get('rewritten_hook', 'N/A')}")
                else:
                    st.info(f"No hook-window text found. Provide a YouTube URL with captions, or paste your first-{hook_window_seconds}-seconds script above.")

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
                    log_boring_score = boring_metrics.get('boring_score')
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
                    sub_result = analyze_subscribe_funnel(final_transcript, title_input, topic_input, mode_name, is_short)
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

            # === PHASE 1: Log this run's predictions to history ===
            st.markdown("---")
            st.subheader("💾 Log This Prediction (NEW)")
            st.caption("Saves today's predicted scores against this video's title. Once you have real Studio numbers for this video (days/weeks later), import the Studio CSV above and it'll match by title automatically — that comparison is what tells you if these scores are actually worth trusting.")
            log_preview_cols = st.columns(4)
            log_preview_cols[0].metric("Thumbnail", log_thumb_score if log_thumb_score is not None else "—")
            log_preview_cols[1].metric("Title", log_title_score if log_title_score is not None else "—")
            log_preview_cols[2].metric("Hook", log_hook_score if log_hook_score is not None else "—")
            log_preview_cols[3].metric("Boring", log_boring_score if log_boring_score is not None else "—")
            if st.button("💾 Save this run to history", use_container_width=True):
                if not title_input:
                    st.warning("Enter a video title above before logging — it's the key used to match real Studio data to this prediction later.")
                else:
                    log_prediction_row(title_input, format_mode, log_thumb_score, log_title_score, log_hook_score, log_boring_score)
                    st.success(f"Logged predictions for \"{title_input}\". Check the sidebar Calibration & History panel.")

        for f in [thumb_path, video_path, new_thumb_path]:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass
