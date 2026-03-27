# 🧫 NexTox — AI-Powered Chemical Toxicity Prediction Platform

> **Hackathon Project** | Predictive AI · Cheminformatics · Generative NLP · Full-Stack  
> Built with LightGBM · RDKit · FLAN-T5 · Streamlit · Django REST · React

---

## 🔬 What is NexTox?

**NexTox** is a production-ready, offline AI platform for real-time chemical toxicity prediction and scientific explanation. Given a molecule's SMILES string, NexTox:

1. **Parses and visualizes** the molecular structure using RDKit
2. **Predicts toxicity risk** (0–100%) using a LightGBM model trained on the Tox21 dataset
3. **Applies domain-specific safety overrides** for known hazardous structural patterns (cyanide, poly-chlorination, benzene rings)
4. **Generates a scientific AI explanation** using a locally-running FLAN-T5 language model — **fully offline, no API keys required**
5. **Supports mixture analysis** — predict synergistic toxicity across multiple compounds simultaneously

---

## 🏗️ Project Architecture

```
MEDICAL_PREDECTION/
├── app.py                  # Streamlit standalone application (main entry point)
├── requirements.txt        # Python dependencies
├── INFO.md                 # Strategy, methodology, and technical deep-dive
├── README.md               # This file
│
├── BACKEND/                # Django REST API backend
│   ├── api/
│   │   ├── views.py        # Prediction logic, LLM inference, REST endpoints
│   │   └── urls.py         # API route definitions
│   └── backend/
│       └── settings.py     # Django configuration
│
├── FRONTEND/               # React + Vite frontend dashboard
│   ├── src/
│   │   └── App.jsx         # Main React app with radar charts & glassmorphism UI
│   ├── package.json
│   └── vite.config.js
│
├── MODELS/
│   ├── toxicity_model (2).pkl          # Trained LightGBM toxicity model
│   └── flatonmodel/
│       └── flan_model/                 # Locally-cached FLAN-T5 model weights
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer.json
│           └── tokenizer_config.json
│
└── DATASETS/
    ├── tox21.csv                       # Raw Tox21 dataset
    └── cleaned_tox21.csv               # Preprocessed dataset
```

---

## ⚙️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML Model | **LightGBM** | Toxicity probability prediction |
| Feature Engineering | **RDKit** | Morgan fingerprints, physicochemical descriptors |
| Generative AI | **FLAN-T5** (Google) | Scientific explanation generation |
| Standalone UI | **Streamlit** | Offline-capable single-file app |
| Backend API | **Django REST Framework** | Production-grade REST API |
| Frontend Dashboard | **React + Vite** | Interactive analytics dashboard |
| Data | **Tox21 Dataset** | 12,000+ compounds with toxicity labels |

---

## 🚀 Quick Start (Streamlit App)

### Prerequisites
- Python 3.9 or higher
- ~1.5 GB disk space (FLAN-T5 model is pre-loaded; no download needed)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/NexTox.git
cd NexTox

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install all dependencies
pip install -r requirements.txt
```

> **RDKit Note:** If `pip install rdkit` fails, try `pip install rdkit-pypi` or use conda: `conda install -c conda-forge rdkit`

### Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🌐 Running the Full-Stack Version (Django + React)

### Backend API

```bash
cd BACKEND
pip install -r ../requirements.txt
python manage.py migrate
python manage.py runserver
# API available at http://localhost:8000/api/
```

### Frontend Dashboard

```bash
cd FRONTEND
npm install
npm run dev
# Dashboard at http://localhost:5173
```

---

## 🧪 How to Use

1. **Launch** `streamlit run app.py`
2. **Choose mode**: Single Assay (one molecule) or Mixture Assay (multiple compounds)
3. **Enter a SMILES string** or search the Tox21 database directly in the UI
4. **Click "RUN PREDICTIVE INFERENCE"**
5. **View results**: molecular structure, risk score, hazard badges, and AI-generated scientific explanation

### Example SMILES to Try

| Molecule | SMILES | Expected Risk |
|---|---|---|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | Low |
| Benzene | `c1ccccc1` | Moderate |
| Cyanide compound | `C#N` | Critical |
| DDT (Pesticide) | `ClC(Cl)(Cl)c1ccc(cc1)C(c1ccc(Cl)cc1)c1ccc(Cl)cc1` | Critical |
| Ethanol | `CCO` | Low |

---

## 📊 Model Performance

- **Dataset**: Tox21 (12,060 compounds, 12 toxicity endpoints)
- **Algorithm**: LightGBM on 2048-bit Morgan Fingerprints (radius=2)
- **Domain Rules**: Chemical structure-based hard-constraint overrides applied post-inference

---

## 🔑 Key Features

- ✅ **Fully Offline** — No internet required after setup; FLAN-T5 runs locally
- ✅ **Dual Interface** — Streamlit standalone app + Django REST + React dashboard
- ✅ **Mixture Analysis** — Synergistic toxicity prediction for multi-compound inputs
- ✅ **Generative Explanations** — AI-written toxicology reports per molecule
- ✅ **Physicochemical Profiling** — MW, LogP, TPSA, H-bond donors/acceptors, ring count
- ✅ **Radar Charts** — Visual pathway vulnerability scores across 5 toxicity mechanisms
- ✅ **Glassmorphism UI** — Premium dark-mode design with animated cards

---

## 📄 License

This project is submitted for educational and hackathon purposes.  
Dataset credits: [Tox21 Data Challenge](https://tripod.nih.gov/tox21/challenge/)

---

*Built with ❤️ for the hackathon — March 2026*
