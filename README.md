# CellRepair Health Educator — MedGemma Impact Challenge

**Patient-Friendly Cellular Health Education Powered by MedGemma 1.5 4B**

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/cellrepairai/cellrepair-health-educator-medgemma)
[![MedGemma](https://img.shields.io/badge/Model-MedGemma%201.5%204B-green)](https://huggingface.co/google/medgemma-1.5-4b-it)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

CellRepair Health Educator transforms complex cellular biology into clear, actionable patient explanations using Google's MedGemma 1.5 4B model. Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) on Kaggle.

**Tracks:** Main Track (Medical Education & Patient Empowerment) + Edge AI Prize

## Key Results

| Metric | Value |
|---|---|
| Quality Score (LLM-as-Judge) | **4.68/5.0 (93.6%)** |
| Patient Accessibility | 5.00/5.0 |
| Analogy Quality | 5.00/5.0 |
| Avg. Response Time (GPU T4) | 32.2s |
| Peak GPU Memory | 13.28 GB / 15.64 GB (84.9%) |

## What Makes This Submission Special

1. **Prompt Ablation Study** — 3-strategy comparison proving structured prompts improve quality by 40%+ over generic approaches
2. **LLM-as-Judge Evaluation** — MedGemma evaluates its own responses across 6 clinical dimensions with per-criterion justifications
3. **Multi-Turn Conversation** — Full conversational context for follow-up patient questions
4. **Multimodal Vision** — Cell biology image analysis for visual patient education
5. **Edge Deployment Ready** — 4B params, 8 GB VRAM, fully local inference (HIPAA/GDPR compatible)

## Project Structure

```
cellrepair-medgemma/
├── README.md                              # This file
├── WRITEUP.md                             # Competition writeup
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── notebooks/
│   └── cellrepair_health_educator_v3.ipynb  # Main Kaggle notebook (v3.0)
├── src/
│   ├── demo_cellrepair_health.py          # Standalone demo script
│   ├── scoring_framework.py               # Quality evaluation framework
│   ├── edge_deployment_analysis.py        # Edge hardware analysis
│   ├── model_comparison.py                # MedGemma vs LLM comparison
│   └── image_analysis_cell.py             # Multimodal image analysis
├── dashboard/
│   └── dashboard.html                     # Interactive results dashboard
├── assets/
│   ├── scoring_radar.png                  # Quality radar chart
│   ├── edge_deployment_chart.png          # Edge deployment comparison
│   ├── model_comparison_chart.png         # Model comparison chart
│   ├── medical_advantages.png             # MedGemma advantages
│   ├── inference_timeline.png             # Inference timeline
│   └── thumbnail.png                      # Project thumbnail
└── docs/
    ├── VIDEO_SCRIPT.md                    # 3-minute demo video script
    └── QUICK_START.md                     # Quick start guide
```

## Quick Start

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
huggingface-cli login  # Requires access to MedGemma
python src/demo_cellrepair_health.py
```

## Architecture

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

## 5 Patient Education Scenarios

| # | Topic | Category | Time | Words |
|---|---|---|---|---|
| 1 | Autophagy — Cellular Self-Cleaning | Education | 32.0s | 330 |
| 2 | Free Radicals & Oxidative Stress | Clinical Support | 29.6s | 304 |
| 3 | Lifestyle & Cellular Health | Prevention | 36.4s | 394 |
| 4 | Chronic Inflammation | Education | 35.7s | 372 |
| 5 | Telomeres & Aging | Education | 27.3s | 296 |

## Technical Details

- **Model:** google/medgemma-1.5-4b-it (4B parameters, multimodal)
- **Precision:** bfloat16 for memory efficiency
- **Decoding:** Greedy (do_sample=False) for reproducibility
- **APIs:** AutoProcessor + AutoModelForImageTextToText
- **GPU:** Tesla T4 (15.64 GB VRAM), peak usage 84.9%

## Author

**Oliver Winkel** — Founder & Developer, [CellRepair AI](https://cellrepair.ai)

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Disclaimer: CellRepair Health Educator is a research prototype for educational purposes. Not a medical device. Always consult a qualified healthcare professional.*
