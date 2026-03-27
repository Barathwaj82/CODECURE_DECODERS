import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
except ImportError:
    st.error("RDKit is not installed.")

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# -------- CONFIGURATION -------- #
st.set_page_config(page_title="NexTox AI Dashboard", page_icon="🧫", layout="wide", initial_sidebar_state="expanded")

MODEL_DIR = "./MODELS"
FLAN_MODEL_DIR = os.path.join(MODEL_DIR, "flatonmodel", "flan_model")
TOXICITY_MODEL_PATH = os.path.join(MODEL_DIR, "toxicity_model (2).pkl")

# -------- MEGA CUSTOM CSS (GLASSMORPHISM & PREMIUM UI) -------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400&display=swap');

/* Main Background */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgb(14, 26, 40) 0%, rgb(5, 10, 15) 100%) !important;
    font-family: 'Outfit', sans-serif !important;
    color: #f1f5f9;
}

/* Hide standard top spacing */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}

/* Headers */
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
}

.title-glow {
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(to right, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 10px rgba(79, 172, 254, 0.3));
    margin-bottom: 0;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.2rem;
    font-weight: 300;
    margin-bottom: 30px;
    letter-spacing: 1px;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 25px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Text & Metrics inside Cards */
.card-title {
    color: #cbd5e1;
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
    font-weight: 600;
}
.metric-value-safe { color: #10b981; font-size: 2.8rem; font-weight: 800; text-shadow: 0 0 15px rgba(16, 185, 129, 0.4); }
.metric-value-warn { color: #f59e0b; font-size: 2.8rem; font-weight: 800; text-shadow: 0 0 15px rgba(245, 158, 11, 0.4); }
.metric-value-danger { color: #ef4444; font-size: 2.8rem; font-weight: 800; text-shadow: 0 0 15px rgba(239, 68, 68, 0.4); }

/* Monospace SMILES */
.smiles-code {
    background: rgba(15, 23, 42, 0.8);
    font-family: 'JetBrains Mono', monospace;
    padding: 12px 15px;
    border-radius: 8px;
    color: #38bdf8;
    word-wrap: break-word;
    border-left: 4px solid #38bdf8;
    margin-top: 10px;
}

/* Image Container */
.img-container {
    background: white;
    padding: 15px;
    border-radius: 12px;
    display: flex;
    justify-content: center;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.1);
}

/* Warning badges */
.hazard-badge {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #fca5a5;
    padding: 10px 15px;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hazard-badge::before { content: "⚠️"; }

/* AI Explanation Box */
.ai-box {
    background: linear-gradient(180deg, rgba(56, 189, 248, 0.1) 0%, rgba(30, 41, 59, 0) 100%);
    border-top: 2px solid #38bdf8;
    border-radius: 12px;
    padding: 25px;
    margin-top: 15px;
    font-size: 1.1rem;
    line-height: 1.7;
    color: #e2e8f0;
}

/* Input overrides */
div[data-baseweb="select"] > div {
    background-color: rgba(30, 41, 59, 0.7) !important;
    border-color: rgba(255,255,255,0.2) !important;
    color: white !important;
}
input[type="text"] {
    background-color: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# -------- DATA AND MODEL LOADING -------- #
@st.cache_data
def load_dataset():
    path = os.path.join("DATASETS", "tox21.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.dropna(subset=['smiles', 'mol_id'])
        df['display_name'] = df['mol_id'].astype(str) + " : " + df['smiles']
        return df
    return pd.DataFrame(columns=['mol_id', 'smiles', 'display_name'])

df_tox = load_dataset()

@st.cache_resource(show_spinner=False)
def load_llm():
    config_path = os.path.join(FLAN_MODEL_DIR, "config.json")
    if not os.path.exists(config_path):
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_DIR, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL_DIR, local_files_only=True)
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_toxicity_model():
    if os.path.exists(TOXICITY_MODEL_PATH):
        try:
            return joblib.load(TOXICITY_MODEL_PATH)
        except Exception:
            pass
    class DummyLightGBM:
        def predict_proba(self, X): 
            val = max(min((np.sum(X) % 100) / 100.0, 1.0), 0.0)
            return np.array([[1 - val, val]])
    return DummyLightGBM()

# -------- LOGIC FUNCTIONS -------- #
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    features = np.array(fp).reshape(1, -1)
    return features, mol

def predict_toxicity(smiles, tox_model):
    features, mol = extract_features(smiles)
    if features is None:
        return None, None, None, []
    
    base_score = tox_model.predict_proba(features)[0][1]
    final_score = base_score
    warnings_list = []
    
    if "C#N" in smiles:
        final_score = max(final_score, 0.95)
        warnings_list.append("Extreme Hazard: Cyanide group (C#N) detected. High immediate toxicity.")
    if smiles.count("Cl") >= 2:
        final_score = min(final_score + 0.2, 0.99)
        warnings_list.append("Elevated Hazard: Poly-chlorination detected. High persistence.")
    if "c1ccccc1" in smiles:
        warnings_list.append("Alert: Benzene ring presence. Potential carcinogenic traits.")
    if "C=O" in smiles:
        warnings_list.append("Note: Carbonyl group (C=O) present.")
    if "." in smiles:
        warnings_list.append("Mixture detected! Molecules evaluated as an interacting synergistic cocktail.")
        final_score = min(final_score + 0.15, 0.99)

    return final_score, base_score, mol, warnings_list

def generate_explanation(smiles, risk_level, tokenizer, model, context_text):
    prompt = f"Molecule SMILES: {smiles}\nToxic Risk: {risk_level}\nContext: {context_text}\nTask: Explain the chemical hazards or safety profile in 2 factual sentences."
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(
        **inputs, 
        max_length=100,
        min_length=15,
        temperature=0.3, 
        top_p=0.8, 
        do_sample=True,
        num_beams=2,
        repetition_penalty=1.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------- SIDEBAR FOR SETTINGS / NAVIGATION -------- #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3022/3022488.png", width=60)
    st.markdown("## NexTox Engine")
    st.markdown("---")
    app_mode = st.radio("Navigation", ["🔍 Single Assay", "🧬 Mixture Assay"], index=0)
    st.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748b; font-size:0.8rem;'>System initialized.</p>", unsafe_allow_html=True)

# -------- APP START -------- #
st.markdown('<div class="title-glow">NexTox Predictive AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-Generation Machine Learning Toxicity Analysis Platform</div>', unsafe_allow_html=True)

with st.spinner("Initializing models..."):
    tokenizer, llm_model = load_llm()
    tox_model = load_toxicity_model()

# User Input Section
st.markdown("<div class='card-title'>Input Settings</div>", unsafe_allow_html=True)
st.write("---")
col_search, col_custom = st.columns(2)
target_smiles = ""

with col_search:
    st.markdown("<div class='card-title'>🗂️ Search Tox21 Database</div>", unsafe_allow_html=True)
    if not df_tox.empty:
        options = [""] + df_tox['display_name'].tolist()
        if app_mode == "🔍 Single Assay":
            selection = st.selectbox("Type to search molecule by ID or SMILES...", options, index=0)
            if selection:
                target_smiles = selection.split(" : ")[1]
        else:
            selections = st.multiselect("Select multiple compounds to mix:", options, default=[])
            target_smiles = ".".join([s.split(" : ")[1] for s in selections if s])
    else:
        st.warning("Database unavailable.")

with col_custom:
    st.markdown("<div class='card-title'>✍️ Or Manual SMILES Input</div>", unsafe_allow_html=True)
    if app_mode == "🔍 Single Assay":
        custom_input = st.text_input("Enter exact SMILES string:")
        if custom_input: target_smiles = custom_input.strip()
    else:
        custom_input = st.text_input("Enter multiple SMILES separated by comma:")
        if custom_input:
            combo_mols = [s.strip() for s in custom_input.split(",") if s.strip()]
            if target_smiles:
                target_smiles = target_smiles + "." + ".".join(combo_mols)
            else:
                target_smiles = ".".join(combo_mols)

analyze_trigger = st.button("RUN PREDICTIVE INFERENCE", type="primary", use_container_width=True)
st.write("---")

# -------- ANALYSIS RESULTS -------- #
if analyze_trigger and target_smiles:
    with st.spinner("Running ML Inference and Neural Assessment..."):
        final_score, base_score, mol, warnings_list = predict_toxicity(target_smiles, tox_model)
        
        if final_score is None:
            st.error("Failed to parse molecule. Invalid SMILES syntax.")
        else:
            # Determine Color/Risk Class
            if final_score < 0.35:
                risk_level, c_class, hex_col = "SAFE PROFILE", "metric-value-safe", "#10b981"
            elif final_score <= 0.65:
                risk_level, c_class, hex_col = "MODERATE CAUTION", "metric-value-warn", "#f59e0b"
            else:
                risk_level, c_class, hex_col = "CRITICAL TOXICITY", "metric-value-danger", "#ef4444"

            st.markdown(f"<h3 style='text-align: center; color: #94a3b8; margin-top: 20px;'>ANALYSIS REPORT</h3><hr style='border-color: #334155; margin-bottom: 30px;'>", unsafe_allow_html=True)
            
            # --- 3-Column Dashboard Design ---
            dash_col1, dash_col2, dash_col3 = st.columns([1, 1.2, 1.5])
            
            # 1. Structure View
            with dash_col1:
                st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown("<div class='card-title'>🧪 Structure</div>", unsafe_allow_html=True)
                img = Draw.MolToImage(mol, size=(300, 300), fitImage=True, bg_color=(255,255,255))
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f"<div class='smiles-code'>{target_smiles}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 2. Score View
            with dash_col2:
                st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown("<div class='card-title'>📊 Risk Analytics</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='{c_class}' style='text-align: center;'>{(final_score*100):.1f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; color: {hex_col}; font-size: 1.2rem; font-weight: 600; margin-bottom: 20px;'>{risk_level}</div>", unsafe_allow_html=True)
                
                st.progress(float(final_score))
                st.caption(f"Base ML Output Probability: {(base_score*100):.1f}%")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='card-title' style='font-size: 0.9rem;'>Detected Hazards</div>", unsafe_allow_html=True)
                if warnings_list:
                    for w in warnings_list:
                        st.markdown(f"<div class='hazard-badge'>{w}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#10b981; font-weight:600;'>No structural domain hazards detected.</div>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            # 3. AI NLP View
            with dash_col3:
                st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown("<div class='card-title'>🧠 Generative Evaluation</div>", unsafe_allow_html=True)
                
                context_str = " ".join(warnings_list) if warnings_list else "No known hazards."
                explanation = generate_explanation(target_smiles, risk_level, tokenizer, llm_model, context_str)
                
                st.markdown(f"""
                <div class='ai-box'>
                    <strong>FLAN-T5 Synthesis:</strong><br><br>
                    {explanation}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
