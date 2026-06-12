import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

# Page config for premium look
st.set_page_config(
    page_title="AI Resume Screening Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Dark Theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Global Reset & Base Styling */
html, body, [data-testid="stSidebar"], .stMarkdown, p, div, span {
    font-family: 'Outfit', sans-serif !important;
}

/* Background Ambient Glow */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(138, 35, 135, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(242, 113, 33, 0.06) 0%, transparent 40%),
                #0a0b10 !important;
}

/* App Header Styling */
.header-container {
    text-align: center;
    margin-bottom: 3rem;
    padding-top: 1.5rem;
}

.main-header {
    background: linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}

.subheader-text {
    font-size: 1.25rem;
    color: #94a3b8;
    max-width: 750px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 300;
}

/* Glassmorphism Input Containers */
.input-panel {
    background: rgba(30, 41, 59, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}

/* Custom Candidate Card Styling */
.candidate-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(12px);
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.candidate-card:hover {
    transform: translateY(-5px);
    border-color: rgba(236, 72, 153, 0.4);
    box-shadow: 0 12px 40px rgba(236, 72, 153, 0.15);
    background: rgba(255, 255, 255, 0.04);
}

/* Match Badges with neon-glow effects */
.badge {
    display: inline-block;
    padding: 0.4rem 0.9rem;
    border-radius: 30px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.badge-strong {
    background: rgba(34, 197, 94, 0.1);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.25);
    box-shadow: 0 0 15px rgba(34, 197, 94, 0.15);
}

.badge-good {
    background: rgba(234, 179, 8, 0.1);
    color: #facc15;
    border: 1px solid rgba(234, 179, 8, 0.25);
    box-shadow: 0 0 15px rgba(234, 179, 8, 0.15);
}

.badge-low {
    background: rgba(239, 68, 68, 0.1);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.25);
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);
}

/* Custom Progress Bar inside Candidate Card */
.progress-bar-container {
    width: 100%;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 30px;
    height: 10px;
    margin-top: 1.25rem;
    overflow: hidden;
}

.progress-bar-fill {
    background: linear-gradient(90deg, #a855f7 0%, #ec4899 50%, #f97316 100%);
    height: 100%;
    border-radius: 30px;
}

/* Sidebar Custom Theme */
[data-testid="stSidebar"] {
    background-color: #08090d;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Reset file uploader style conflict */
div[data-testid="stFileUploader"] button {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Premium Screening Button style */
div.stButton > button {
    background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem 2.5rem !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.5px;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 8px 25px rgba(236, 72, 153, 0.25) !important;
    width: 100%;
    margin-top: 1.5rem;
}

.stButton>button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 30px rgba(236, 72, 153, 0.45) !important;
    opacity: 0.95;
}

</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

model = load_model()

# PDF Reader
def read_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

# Default IBM JD
default_jd = """IBM AI Engineer

Required Skills:
- Python, Machine Learning, Deep Learning
- Generative AI, LangChain, RAG, LLMs
- Prompt Engineering, Vector Databases (FAISS)
- Docker, Kubernetes, AWS Cloud Services
- Excellent Communication & Collaboration Skills"""

# Header Section
st.markdown("""
<div class="header-container">
    <div class="main-header">🤖 AI Resume Screening Agent</div>
    <div class="subheader-text">Leveraging state-of-the-art semantic embedding algorithms to match and rank applicant files instantly.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### ⚙️ Control Settings")
top_k = st.sidebar.slider(
    "Show Top Candidates",
    min_value=1,
    max_value=20,
    value=5
)

st.sidebar.info("Model active: `all-MiniLM-L6-v2` (SentenceTransformer)")

# Main Layout: 2 Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.subheader("📄 Target Role Profile")
    job_description = st.text_area(
        "Target Role Requirements",
        default_jd,
        height=260,
        placeholder="Paste job details here...",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.subheader("📂 Upload Candidate Portfolios")
    uploaded_files = st.file_uploader(
        "Upload PDF Resumes",
        type=["pdf"],
        accept_multiple_files=True,
        key="resume_uploader",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Temporary session state memory for persistence
if uploaded_files:
    st.session_state["saved_files"] = uploaded_files

# Preview Section
if "saved_files" in st.session_state and st.session_state["saved_files"]:
    st.markdown("### 📎 Uploaded Files")
    preview_cols = st.columns(min(len(st.session_state["saved_files"]), 6))
    for i, file in enumerate(st.session_state["saved_files"]):
        with preview_cols[i % 6]:
            st.code(f"📄 {file.name[:20]}...", language="text")

st.divider()

# Action Button
if st.button("🚀 Start Semantic Screening"):
    if "saved_files" not in st.session_state or not st.session_state["saved_files"]:
        st.warning("Please upload at least one PDF resume to analyze.")
        st.stop()

    with st.spinner("Analyzing semantics and calculating score matches..."):
        jd_embedding = model.encode([job_description])
        results = []
        
        for file in st.session_state["saved_files"]:
            file.seek(0)
            resume_text = read_pdf(file)
            resume_embedding = model.encode([resume_text])
            
            similarity = cosine_similarity(
                jd_embedding,
                resume_embedding
            )[0][0]
            
            score = round(similarity * 100, 2)
            results.append({
                "name": file.name,
                "score": score,
                "text_snippet": resume_text[:120].strip().replace('\n', ' ') + "..."
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)

    st.success("Screening Complete!")
    st.subheader("🏆 Candidate Leaderboard")
    
    report = "AI Resume Match Report\n" + "="*40 + "\n\n"
    
    for rank, candidate in enumerate(results[:top_k], start=1):
        score = candidate["score"]
        
        # Determine status & badge
        if score >= 80:
            badge_class = "badge-strong"
            status = "Strong Match"
        elif score >= 60:
            badge_class = "badge-good"
            status = "Good Match"
        else:
            badge_class = "badge-low"
            status = "Low Match"
            
        # Premium card representation with progress bar aligned inside the card container
        card_html = f"""
        <div class="candidate-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <h4 style="margin: 0; color: #ffffff; font-size: 1.35rem; font-weight:600;">#{rank} — {candidate['name']}</h4>
                <span class="badge {badge_class}">{status}</span>
            </div>
            <p style="margin: 0.5rem 0 0.75rem 0; color: #94a3b8; font-size: 0.95rem; line-height: 1.5;">
                <strong>Preview:</strong> {candidate['text_snippet']}
            </p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <span style="font-weight: 700; color: #ec4899; font-size: 1.15rem;">{score}% Match</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: {max(0, min(score, 100))}%;"></div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
        report += (
            f"Rank #{rank}\n"
            f"Candidate: {candidate['name']}\n"
            f"Match Score: {score}%\n"
            f"Status: {status}\n"
            + "-" * 40 + "\n"
        )
        
    st.markdown("### 💾 Export Results")
    st.download_button(
        label="📥 Download Match Report",
        data=report,
        file_name="resume_match_report.txt",
        mime="text/plain"
    )
