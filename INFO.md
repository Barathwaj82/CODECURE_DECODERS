# 📋 NexTox — Project Info, Strategy & Technical Methodology

> This document outlines the **strategy, design decisions, technical implementation**, and **future roadmap** for the NexTox AI Toxicity Prediction Platform.

---

## 🎯 Problem Statement

Chemical toxicity assessment is a critical bottleneck in drug discovery, environmental safety, and industrial chemical regulation. Traditional wet-lab toxicity testing is:
- **Expensive** — costs $1M+ per compound for full evaluation
- **Slow** — months to years per safety assessment
- **Limited** — cannot scale to millions of candidate molecules

**Our Solution**: NexTox uses machine learning on molecular fingerprints combined with a locally-deployed language model to provide:
- Instant toxicity risk scoring (**< 2 seconds per molecule**)
- AI-written scientific explanations without any external API dependency
- Structural hazard detection via domain rule-based overrides
- Mixture synergy analysis for multi-compound evaluations

---

## 🧠 Strategy & Approach

### Phase 1 — Data & Feature Engineering
- Sourced the **Tox21 dataset** (12,060 compounds, 12 validated endpoint labels)
- Used **RDKit** to convert SMILES strings into 2048-bit **Morgan Circular Fingerprints** (radius=2)
  - Morgan fingerprints capture local atomic environments up to 2 bond-hops in all directions
  - Binary bit-vectors encode structural subgraph presence, ideal for tree-based ML
- Extracted real **physicochemical descriptors** for radar visualization:
  - Molecular Weight, LogP (lipophilicity), TPSA, H-bond donors/acceptors, Ring Count

### Phase 2 — ML Model Selection & Training
- Selected **LightGBM** as the core prediction engine for multiple reasons:
  - Handles sparse binary fingerprint inputs natively
  - Extremely fast inference (< 50ms per sample)
  - Outperforms Random Forest and basic XGBoost on sparse high-dimensional data
  - Low memory footprint compared to deep neural networks
- Trained with **binary cross-entropy loss** for toxic/non-toxic classification
- Output: a **continuous probability score** (0.0 – 1.0) → mapped to risk tiers

### Phase 3 — Domain Safety Override System
- Implemented a **rule-based post-processing layer** that overrides ML output for known toxic structural patterns:
  - `C#N` → Cyanide group → Forces score ≥ 0.95 (extreme hazard)
  - `Cl × ≥ 2` → Poly-chlorination → Adds +0.20 score (persistent organochlorines)
  - `c1ccccc1` → Benzene ring → Flags potential carcinogenicity
  - `.` separator in SMILES → Mixture → Adds +0.15 synergy penalty
- This hybrid approach ensures **no known hazardous compound is underscored** by the ML model alone

### Phase 4 — Generative AI Explanation Engine
- Integrated **Google's FLAN-T5** (Sequence-to-Sequence language model) for explanation generation
- Model is **fully local** — cached in `MODELS/flatonmodel/flan_model/`, no API calls
- Custom prompt engineering ensures factual, domain-specific 2–3 sentence outputs:
  ```
  Molecule: {SMILES}
  Hazard Level: {risk_level}
  Context: {detected_hazards}
  Task: Write a scientific explanation of the toxicological profile.
  ```
- Generation parameters tuned to reduce hallucinations:
  - `temperature=0.2` (low randomness)
  - `repetition_penalty=1.05` (no repeated text)
  - `num_beams=2` (mild beam search for quality)

### Phase 5 — Full-Stack Architecture
- **Streamlit App** (`app.py`): Self-contained, single-command deployment for rapid demo
- **Django REST API** (`BACKEND/`): Production-grade API for scalable integration
- **React + Vite Dashboard** (`FRONTEND/`): Interactive analytics UI with:
  - Radar/spider charts for 5-pathway toxicity visualization
  - Glassmorphism card design with animated transitions
  - Real-time API query with async state management

---

## 🔬 How It Works — End-to-End Flow

```
User Input (SMILES String)
        │
        ▼
  RDKit Molecule Parser
  - Validates SMILES syntax
  - Generates 2D mol object
        │
        ▼
  Feature Extraction
  - 2048-bit Morgan Fingerprint
  - Physicochemical descriptors (MW, LogP, TPSA...)
        │
        ▼
  LightGBM Prediction
  - Returns base_score (0.0 – 1.0)
        │
        ▼
  Domain Override Rules
  - Cyanide, Chlorination, Benzene, Mixture checks
  - Adjusts final_score
        │
        ▼
  Risk Classification
  - < 35% → SAFE PROFILE
  - 35–65% → MODERATE CAUTION
  - > 65% → CRITICAL TOXICITY
        │
        ├────────────────────────────┐
        ▼                            ▼
  Molecule Visualization       FLAN-T5 Explanation
  (RDKit DrawToImage)          (Local Generative AI)
        │                            │
        └──────────────┬─────────────┘
                       ▼
              Results Dashboard
        (Risk score, Hazard badges,
         Radar chart, AI report)
```

---

## 📈 Toxicity Pathway Scoring

Five biological pathways are scored (0–100) using heuristic formulas derived from physicochemical properties and the ML output score:

| Pathway | Description | Key Drivers |
|---|---|---|
| **Nuclear Receptor** | Hormonal/genetic pathway disruption | LogP, base ML score |
| **Stress Response** | Oxidative/inflammatory pathway activation | TPSA, base ML score |
| **Genotoxicity** | DNA damage / mutagenicity potential | Ring count, benzene presence |
| **Cytotoxicity** | Cell death mechanism | Cyanide override, base ML score |
| **Endocrine Disruption** | Hormonal interference | H-bond acceptors, base ML score |

---

## 🌍 Impact & Applications

| Domain | Application |
|---|---|
| **Drug Discovery** | Early-stage ADMET toxicity triage of drug candidates |
| **Environmental Safety** | Screening industrial chemicals for ecological toxicity |
| **Chemical Regulation** | Supporting REACH compliance assessments |
| **Education** | Teaching toxicology through interactive AI explanations |
| **Research** | Rapid exploration of Tox21 chemical space |

---

## 🗺️ Roadmap — What We'll Build Next

### v2.0 — Model Enhancement
- [ ] Train on **all 12 Tox21 endpoints** simultaneously (multi-task learning)
- [ ] Upgrade to **Graph Neural Network** (GNN/MPNN) architecture for atom-level reasoning
- [ ] Add **Lipinski Rule of Five** + ADMET property prediction layer

### v2.1 — Data & Coverage
- [ ] Expand dataset with **ChEMBL** and **PubChem BioAssay** data
- [ ] Add **CAS number → SMILES** lookup via PubChem API
- [ ] Support **InChI and InChIKey** inputs alongside SMILES

### v2.2 — Explainability
- [ ] **Atom-level heatmaps** using SHAP values on Morgan bits
- [ ] **Grad-CAM equivalent visualization** for substructure attribution
- [ ] Export **PDF toxicology reports** from the dashboard

### v2.3 — Deployment
- [ ] **Docker containerization** for one-command deployment
- [ ] **CI/CD pipeline** with GitHub Actions
- [ ] **Streamlit Cloud** public deployment

---

## 👥 Team & Roles

| Role | Responsibility |
|---|---|
| ML Engineer | LightGBM model, RDKit feature pipeline, domain rules |
| AI/NLP Engineer | FLAN-T5 integration, prompt engineering, offline caching |
| Backend Engineer | Django REST API, database, authentication |
| Frontend Engineer | React dashboard, Vite build, radar charts, UI/UX |

---

## 🤝 Acknowledgements

- **Tox21 Dataset** — NIH National Center for Advancing Translational Sciences
- **RDKit** — Open-source cheminformatics library
- **FLAN-T5** — Google Research, released under Apache 2.0
- **LightGBM** — Microsoft Research

---

*Last Updated: March 2026 | NexTox Hackathon Submission*
