<div align="center">

# 🧬 CellRepair Health Educator — MedGemma Impact Challenge

**Patient-Friendly Cellular Health Education Powered by MedGemma 1.5 4B**

[![Quality Score](https://img.shields.io/badge/Quality_Score-4.68%2F5.0_(93.6%25)-brightgreen?style=for-the-badge)](https://powerforyou74.github.io/cellrepair-medgemma/dashboard/benchmark-dashboard.html)
[![AgentBeats](https://img.shields.io/badge/AgentBeats-96%25_Win_Rate-blueviolet?style=for-the-badge)](https://powerforyou74.github.io/cellrepair-medgemma/dashboard/benchmark-dashboard.html)
[![Edge AI](https://img.shields.io/badge/Edge_AI-8GB_VRAM-ff6b6b?style=for-the-badge)](#-edge-deployment)

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/cellrepairai/cellrepair-health-educator-medgemma)
[![MedGemma](https://img.shields.io/badge/Model-MedGemma%201.5%204B-4285F4?logo=google&logoColor=white)](https://huggingface.co/google/medgemma-1.5-4b-it)
[![Demo Video](https://img.shields.io/badge/Demo-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/4XUU19DFdJo)
[![Live Dashboard](https://img.shields.io/badge/📊_Live-Benchmark_Dashboard-8b5cf6?style=flat)](https://powerforyou74.github.io/cellrepair-medgemma/dashboard/benchmark-dashboard.html)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-222?logo=github&logoColor=white)](https://powerforyou74.github.io/cellrepair-medgemma/)
[![Release](https://img.shields.io/github/v/release/PowerForYou74/cellrepair-medgemma?color=blue)](https://github.com/PowerForYou74/cellrepair-medgemma/releases)

</div>

---

## 🎬 Demo Video

<div align="center">

[![CellRepair Health Educator Demo](https://img.youtube.com/vi/4XUU19DFdJo/maxresdefault.jpg)](https://youtu.be/4XUU19DFdJo)

▶️ [Watch the 3-minute demo on YouTube](https://youtu.be/4XUU19DFdJo)

</div>

---

## 📊 Live Benchmark Dashboard

> **Explore all results interactively →** [**CellRepair AI — Benchmark Dashboard**](https://powerforyou74.github.io/cellrepair-medgemma/dashboard/benchmark-dashboard.html)
>
> Chart.js visualizations · KPI cards · Hardware compatibility matrix · Filterable & sortable

---

## 🏆 Key Results at a Glance

| Metric | Value | Details |
|:---|:---:|:---|
| **MedGemma Quality Score** | **4.68 / 5.0 (93.6%)** | LLM-as-Judge across 6 clinical dimensions |
| **Patient Accessibility** | **5.00 / 5.0** | Perfect layperson readability |
| **Analogy Quality** | **5.00 / 5.0** | Complex biology → everyday metaphors |
| **AgentBeats Win Rate** | **96% (195/202)** | Head-to-head vs. GPT-4, Claude 3.5, Gemini Pro |
| **Edge Deployment** | **8 GB VRAM** | Runs on RTX 3060, Jetson Orin, Apple M1 |
| **Avg. Response Time** | **32.2 s** | On Tesla T4 GPU |
| **Peak GPU Memory** | **13.28 / 15.64 GB (84.9%)** | bfloat16 precision |

---

## 🔬 Overview

CellRepair Health Educator transforms complex cellular biology into clear, actionable patient explanations using Google's MedGemma 1.5 4B model. Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) on Kaggle.

**Tracks:** Main Track (Medical Education & Patient Empowerment) + Edge AI Prize

---

## 🥊 AgentBeats Benchmark — 96% Win Rate

CellRepair was evaluated head-to-head against leading LLMs across 202 patient education scenarios:

| Matchup | Wins | Losses | Ties | Win Rate |
|:---|:---:|:---:|:---:|:---:|
| CellRepair vs. GPT-4 | 63 / 67 | 3 | 1 | **94%** |
| CellRepair vs. Claude 3.5 Sonnet | 65 / 67 | 1 | 1 | **97%** |
| CellRepair vs. Gemini Pro | 67 / 68 | 1 | 0 | **99%** |
| **Total** | **195 / 202** | **5** | **2** | **96%** |

> *Evaluated on: medical accuracy, patient accessibility, actionable advice, analogy quality, and safety disclaimers.*

---

## ✨ What Makes This Submission Special

1. **Prompt Ablation Study** — 3-strategy comparison proving structured prompts improve quality by 40%+ over generic approaches
2. **LLM-as-Judge Evaluation** — MedGemma evaluates its own responses across 6 clinical dimensions with per-criterion justifications
3. **AgentBeats Benchmark** — Head-to-head comparison against GPT-4, Claude 3.5, Gemini Pro (96% win rate)
4. **Multi-Turn Conversation** — Full conversational context for follow-up patient questions
5. **Multimodal Vision** — Cell biology image analysis for visual patient education
6. **Edge Deployment Ready** — 4B params, 8 GB VRAM, fully local inference (HIPAA/GDPR compatible)

---

## 📁 Project Structure

```
cellrepair-medgemma/
├── README.md                          # This file
├── WRITEUP.md                         # Competition writeup
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── notebooks/
│   └── cellrepair_health_educator_v3.ipynb  # Main Kaggle notebook (v3.0)
├── src/
│   ├── demo_cellrepair_health.py      # Standalone demo script
│   ├── scoring_framework.py           # Quality evaluation framework
│   ├── edge_deployment_analysis.py    # Edge hardware analysis
│   ├── model_comparison.py            # MedGemma vs LLM comparison
│   └── image_analysis_cell.py         # Multimodal image analysis
├── dashboard/
│   └── benchmark-dashboard.html       # Interactive results dashboard
├── assets/
│   ├── scoring_radar.png              # Quality radar chart
│   ├── edge_deployment_chart.png      # Edge deployment comparison
│   ├── model_comparison_chart.png     # Model comparison chart
│   ├── medical_advantages.png         # MedGemma advantages
│   ├── inference_timeline.png         # Inference timeline
│   └── thumbnail.png                  # Project thumbnail
└── docs/
    ├── VIDEO_SCRIPT.md                # 3-minute demo video script
    └── QUICK_START.md                 # Quick start guide
```

---

## 🚀 Quick Start

### Run on Kaggle (Recommended)

1. Open the [Kaggle Notebook](https://www.kaggle.com/code/cellrepairai/cellrepair-health-educator-medgemma)
2. Enable GPU T4 x2 in Settings
3. Add your HF_TOKEN via Add-ons → Secrets
4. Click "Run All"

### Run Locally

```bash
git clone https://github.com/powerforyou74/cellrepair-medgemma.git
cd cellrepair-medgemma
pip install -r requirements.txt
huggingface-cli login   # Requires access to MedGemma
python src/demo_cellrepair_health.py
```

---

## 🏗️ Architecture

```
Patient Question → Structured Education Prompt → MedGemma 1.5 4B → Patient-Friendly Response
                                                                          ↓
                                                              LLM-as-Judge Self-Evaluation
```

The system uses a structured prompt (CellRepair v2) with emoji-segmented sections:

- 🔬 **What's happening in your cells** — Biology with analogies
- 💡 **Why this matters for you** — Personal health relevance
- ✅ **What you can do** — 3 actionable lifestyle tips
- ⚕️ **Disclaimer** — Consult your healthcare provider

---

## 🧪 5 Patient Education Scenarios

| # | Topic | Category | Time | Words |
|:---:|:---|:---|:---:|:---:|
| 1 | Autophagy — Cellular Self-Cleaning | Education | 32.0s | 330 |
| 2 | Free Radicals & Oxidative Stress | Clinical Support | 29.6s | 304 |
| 3 | Lifestyle & Cellular Health | Prevention | 36.4s | 394 |
| 4 | Chronic Inflammation | Education | 35.7s | 372 |
| 5 | Telomeres & Aging | Education | 27.3s | 296 |

---

## ⚡ Edge Deployment

CellRepair runs on consumer-grade hardware — no cloud required:

| Device | VRAM | Status |
|:---|:---:|:---:|
| NVIDIA RTX 3060 | 12 GB | ✅ Full speed |
| NVIDIA Jetson Orin | 8 GB | ✅ Optimized |
| Apple M1 (16 GB) | Shared | ✅ Compatible |
| Raspberry Pi 5 | 8 GB RAM | ⚠️ CPU-only, slow |

> **Privacy by design:** All inference runs locally. No patient data leaves the device. HIPAA & GDPR compatible.

---

## 🔧 Technical Details

- **Model:** google/medgemma-1.5-4b-it (4B parameters, multimodal)
- **Precision:** bfloat16 for memory efficiency
- **Decoding:** Greedy (do_sample=False) for reproducibility
- **APIs:** AutoProcessor + AutoModelForImageTextToText
- **GPU:** Tesla T4 (15.64 GB VRAM), peak usage 84.9%

---

## 👤 Author

**Oliver Winkel** — Founder & Developer, [CellRepair AI](https://cellrepair.ai)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Oliver_Winkel-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/cellrepair-systems)
[![GitHub](https://img.shields.io/badge/GitHub-PowerForYou74-181717?logo=github&logoColor=white)](https://github.com/PowerForYou74)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*Disclaimer: CellRepair Health Educator is a research prototype for educational purposes. Not a medical device. Always consult a qualified healthcare professional.*

**[📊 Benchmark Dashboard](https://powerforyou74.github.io/cellrepair-medgemma/dashboard/benchmark-dashboard.html)** · **[📓 Kaggle Notebook](https://www.kaggle.com/code/cellrepairai/cellrepair-health-educator-medgemma)** · **[🎬 YouTube Demo](https://youtu.be/4XUU19DFdJo)**

</div>
