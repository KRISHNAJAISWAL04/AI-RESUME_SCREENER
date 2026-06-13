import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

# Page config for the absolute premium look
st.set_page_config(
    page_title="AI RESUME SCREENER",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS for SaaS-grade Experience
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Root Variables */
:root {
    --obsidian: #05070A;
    --electric-violet: #8B5CF6;
    --deep-pink: #EC4899;
    --glass: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.08);
}

/* Global Reset */
html, body, [data-testid="stSidebar"], .stMarkdown, p, div, span {
    font-family: 'Outfit', sans-serif !important;
}

/* Restore Material Icons font for icon elements so glyphs render correctly */
[data-testid="stIconMaterial"], .material-icons {
    font-family: 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-variant: normal !important;
    text-transform: none !important;
    line-height: 1 !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
}

.stApp {
    background: radial-gradient(circle at 0% 0%, rgba(139, 92, 246, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 100% 100%, rgba(236, 72, 153, 0.05) 0%, transparent 50%),
                #05070A !important;
    color: #E2E8F0;
}

/* Hero Section Animation */
@keyframes heroFadeUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-container {
    text-align: center;
    padding: 3rem 0;
    animation: heroFadeUp 1s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

.hero-title {
    background: linear-gradient(135deg, #FFFFFF 30%, #8B5CF6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 4rem;
    margin-bottom: 1rem;
    letter-spacing: -2px;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #94A3B8;
    max-width: 800px;
    margin: 0 auto;
    font-weight: 300;
    line-height: 1.6;
}

/* TARGETED WIDGET STYLING - Fixing the Overlap */
/* Target the container of the text area */
div[data-testid="stTextArea"] {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 2rem;
}

div[data-testid="stTextArea"] textarea {
    background-color: transparent !important;
    color: #FFFFFF !important;
    border: none !important;
    font-size: 1.1rem !important;
}

/* Target the File Uploader */
div[data-testid="stFileUploader"] {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 2rem;
}

div[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}

/* Ensure uploader layout and buttons do not expand or overlap */
div[data-testid="stFileUploader"] {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    flex-wrap: wrap !important;
}

div[data-testid="stFileUploader"] .stButton > button,
div[data-testid="stFileUploader"] button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: auto !important;
    min-width: 120px !important;
    white-space: nowrap !important;
    overflow: visible !important;
    padding: 0.6rem 0.9rem !important;
    font-size: 0.95rem !important;
}

/* Prevent text duplication/overflow inside uploader */
div[data-testid="stFileUploader"] [data-testid="stMarkdown"] {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Hide any pseudo-elements or duplicated overlay text inside uploader buttons */
div[data-testid="stFileUploader"] button::before,
div[data-testid="stFileUploader"] button::after {
    display: none !important;
    content: none !important;
}

/* Reset any transforms/positioning that may layer duplicate text */
div[data-testid="stFileUploader"] button, 
div[data-testid="stFileUploader"] button > span {
    position: relative !important;
    transform: none !important;
    top: auto !important;
    left: auto !important;
    line-height: 1 !important;
}

/* Specifically hide Streamlit's nested generic text node that duplicates 'upload' */
div[data-testid="stFileUploader"] button > div > div:first-child {
    display: none !important;
}

/* Hide the icon text fallback inside the uploader (icon font may be overridden by global font) */
div[data-testid="stFileUploader"] [data-testid="stIconMaterial"] {
    display: none !important;
}

/* Custom Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #8B5CF6 0%, #C084FC 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 1rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 10px 20px rgba(139, 92, 246, 0.2) !important;
    display: inline-block !important;
    width: auto !important;
    min-width: 160px !important;
}

div.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 30px rgba(139, 92, 246, 0.4) !important;
    opacity: 0.9;
}

/* Secondary Button (Clear All) */
div.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #94A3B8 !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    width: auto !important;
    padding: 0.4rem 1.2rem !important;
    font-size: 0.9rem !important;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: var(--electric-violet) !important;
    color: white !important;
}

/* File Chips */
.file-chip {
    display: inline-flex;
    align-items: center;
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 100px;
    padding: 0.4rem 1rem;
    margin: 0.4rem;
    color: #C4B5FD;
    font-size: 0.85rem;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease;
}

.file-chip:hover {
    background: rgba(139, 92, 246, 0.2);
    border-color: var(--electric-violet);
    transform: scale(1.05);
}

/* Candidate Cards */
.candidate-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(0,0,0,0.2));
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 24px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.candidate-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: var(--electric-violet);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
}

.excerpt-text {
    margin: 1rem 0;
    color: #94A3B8;
    font-size: 0.95rem;
    line-height: 1.6;
    border-left: 4px solid var(--electric-violet);
    padding-left: 1rem;
}

.badge {
    padding: 0.3rem 0.8rem;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-strong { background: rgba(34, 197, 94, 0.1); color: #4ADE80; border: 1px solid rgba(34, 197, 94, 0.2); }
.badge-good { background: rgba(234, 179, 8, 0.1); color: #FACC15; border: 1px solid rgba(234, 179, 8, 0.2); }
.badge-low { background: rgba(239, 68, 68, 0.1); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.2); }

.progress-bar-container {
    width: 100%;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 100px;
    height: 8px;
    margin-top: 1.5rem;
    overflow: hidden;
}

.progress-bar-fill {
    background: linear-gradient(90deg, #8B5CF6 0%, #EC4899 100%);
    height: 100%;
    border-radius: 100px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #05070A !important;
    border-right: 1px solid var(--glass-border);
}

</style>
""", unsafe_allow_html=True)

# Model loading with caching
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

def read_pdf(file):
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content: text += content + "\\n"
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
    return text

# Default Job Description
default_jd = """IBM AI Engineer
Required Skills:
- Python, Machine Learning, Deep Learning
- Generative AI, LangChain, RAG, LLMs
- Prompt Engineering, Vector Databases (FAISS)
- Docker, Kubernetes, AWS Cloud Services
- Excellent Communication & Collaboration Skills"""

# --- UI START ---

# 1. Hero Section
st.markdown('''
<div class="hero-container">
    <div class="hero-title">AI RESUME SCREENER</div>
    <div class="hero-subtitle">Revolutionizing recruitment with high-precision semantic embeddings and real-time candidate ranking.</div>
</div>
''', unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.markdown("### ⚙️ Analysis Settings")
top_k = st.sidebar.slider("Display Top Candidates", 1, 20, 5)
st.sidebar.markdown("---")
st.sidebar.info("Engine: `all-MiniLM-L6-v2`\\nStatus: Ready")

# Guided Flow Logic
# We use a vertical flow to prevent the "Overlap" issue completely.

# SECTION 1: ROLE DEFINITION
st.markdown("#### 🎯 Step 1: Define Target Role")
job_description = st.text_area(
    "Paste the Job Description below",
    default_jd,
    height=200,
    label_visibility="collapsed",
    placeholder="Describe the ideal candidate..."
)

# SECTION 2: TALENT ACQUISITION
st.markdown("#### 📂 Step 2: Upload Candidate Portfolios")
col_up, col_btn = st.columns([5, 1])
with col_up:
    uploaded_files = st.file_uploader(
        "Upload PDF Resumes",
        type=["pdf"],
        accept_multiple_files=True,
        key="resume_uploader",
        label_visibility="collapsed"
    )
with col_btn:
    if st.button("Clear All", key="clear_btn"):
        st.session_state["saved_files"] = []
        st.rerun()

# Persistence
if uploaded_files:
    st.session_state["saved_files"] = uploaded_files

# File Chip Preview
if "saved_files" in st.session_state and st.session_state["saved_files"]:
    chips_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin: 1rem 0 2rem 0;">'
    for file in st.session_state["saved_files"]:
        chips_html += f'<div class="file-chip">📄 {file.name[:20]}...</div>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)

st.markdown("---")

# SECTION 3: ANALYSIS TRIGGER
# Center the button
_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    if st.button("🚀 Run Intelligence Screening"):
        if "saved_files" not in st.session_state or not st.session_state["saved_files"]:
            st.warning("Please upload at least one resume first.")
            st.stop()

        with st.spinner("🚀 Orchestrating semantic vectors and computing matches..."):
            # PERFORMANCE OPTIMIZATION: BATCH ENCODING

            # 1. Encode JD (once)
            jd_embedding = model.encode([job_description])

            # 2. Extract all texts first
            all_texts = []
            file_names = []
            for file in st.session_state["saved_files"]:
                file.seek(0)
                text = read_pdf(file)
                if text:
                    all_texts.append(text)
                    file_names.append(file.name)

            if not all_texts:
                st.error("Could not extract text from the uploaded PDFs.")
                st.stop()

            # 3. Batch encode all resumes (MUCH FASTER)
            resume_embeddings = model.encode(all_texts)

            # 4. Calculate similarities
            similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]

            results = []
            for i in range(len(file_names)):
                score = round(similarities[i] * 100, 2)
                results.append({
                    "name": file_names[i],
                    "score": score,
                    "text_snippet": all_texts[i][:140].strip().replace('\\n', ' ') + "..."
                })

            results.sort(key=lambda x: x["score"], reverse=True)

        st.success("Analysis Complete!")

        # Intelligence Dashboard
        st.markdown("#### 🏆 Intelligence Dashboard")
        res_col1, res_col2 = st.columns([3, 2], gap="large")

        with res_col1:
            report = "AI Resume Match Report\\n" + "="*40 + "\\n\\n"
            for rank, candidate in enumerate(results[:top_k], start=1):
                score = candidate["score"]
                if score >= 80: badge_class, status = "badge-strong", "Strong Match"
                elif score >= 60: badge_class, status = "badge-good", "Good Match"
                else: badge_class, status = "badge-low", "Low Match"

                card_html = f'''
                <div class="candidate-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #FFFFFF; font-size: 1.2rem; font-weight:700;">#{rank} — {candidate['name']}</h4>
                        <span class="badge {badge_class}">{status}</span>
                    </div>
                    <div class="excerpt-text">
                        <strong style="color: #8B5CF6; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px;">Analysis Preview:</strong><br>
                        {candidate['text_snippet']}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                        <span style="font-weight: 700; color: #8B5CF6; font-size: 1.2rem;">{score}% Match</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: {max(0, min(score, 100))}%;"></div>
                    </div>
                </div>
                '''
                st.markdown(card_html, unsafe_allow_html=True)
                report += f"Rank #{rank}\\nCandidate: {candidate['name']}\\nScore: {score}%\\nStatus: {status}\\n{'-'*40}\\n"

            st.download_button("📥 Download Match Report", report, "match_report.txt", "text/plain")

        with res_col2:
            # Wrap chart in a visually consistent way
            st.markdown("##### Match Distribution")
            chart_df = pd.DataFrame({
                "Candidate": [c["name"]for c in results[:top_k]],
                "Match %": [c["score"] for c in results[:top_k]]
            }).set_index("Candidate")
            st.bar_chart(chart_df, height=300)
