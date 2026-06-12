import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

# Page config for high-end look
st.set_page_config(
    page_title="AI Resume Screening Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Main font style */
html, body, [data-testid="stSidebar"], .stMarkdown, p, div, span {
    font-family: 'Outfit', sans-serif !important;
}

/* Header gradient style */
.main-header {
    background: linear-gradient(135deg, #8A2387 0%, #E94057 50%, #F27121 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem;
    margin-bottom: 0.5rem;
}

.subheader-text {
    font-size: 1.15rem;
    color: #a0aec0;
    margin-bottom: 2rem;
}

/* Modern Card Styling */
.candidate-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}

.candidate-card:hover {
    transform: translateY(-4px);
    border-color: rgba(233, 64, 87, 0.4);
    box-shadow: 0 8px 30px rgba(233, 64, 87, 0.15);
    background: rgba(255, 255, 255, 0.05);
}

/* Badge styling */
.badge {
    display: inline-block;
    padding: 0.35rem 0.85rem;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-strong {
    background: rgba(46, 213, 115, 0.15);
    color: #2ed573;
    border: 1px solid rgba(46, 213, 115, 0.3);
}

.badge-good {
    background: rgba(255, 165, 2, 0.15);
    color: #ffa502;
    border: 1px solid rgba(255, 165, 2, 0.3);
}

.badge-low {
    background: rgba(255, 71, 87, 0.15);
    color: #ff4757;
    border: 1px solid rgba(255, 71, 87, 0.3);
}

/* Progress bar color tweaking */
div[data-testid="stProgress"] > div > div > div > div {
    background-image: linear-gradient(to right, #8A2387, #E94057, #F27121);
}

/* Sidebar premium adjustments */
[data-testid="stSidebar"] {
    background-color: #0f111a;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Buttons style */
.stButton>button {
    background: linear-gradient(135deg, #8A2387 0%, #E94057 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(233, 64, 87, 0.3) !important;
    width: 100%;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(233, 64, 87, 0.5) !important;
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
st.markdown('<div class="main-header">🤖 AI Resume Screening Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">Deploying semantic intelligence to rank and match applicants with jobs instantly.</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### ⚙️ Settings")
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
    st.subheader("📄 Job Description")
    job_description = st.text_area(
        "Target Role Requirements",
        default_jd,
        height=260,
        placeholder="Paste job details here...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF Resumes",
        type=["pdf"],
        accept_multiple_files=True,
        key="resume_uploader",
        label_visibility="collapsed"
    )

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
            
        # Premium card markdown html representation
        card_html = f"""
        <div class="candidate-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h4 style="margin: 0; color: #ffffff; font-size: 1.25rem;">#{rank} — {candidate['name']}</h4>
                <span class="badge {badge_class}">{status}</span>
            </div>
            <p style="margin: 0.5rem 0; color: #a0aec0; font-size: 0.9rem;">
                <strong>Preview:</strong> {candidate['text_snippet']}
            </p>
            <div style="display: flex; align-items: center; gap: 1rem; margin-top: 0.8rem;">
                <span style="font-weight: 700; color: #E94057; font-size: 1.1rem;">{score}% Match</span>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        st.progress(int(max(0, min(score, 100))))
        
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
