#!/usr/bin/env python3
"""
MedGemma Impact Challenge – CellRepair AI
==========================================
Patient Education System for Cellular Health

Uses MedGemma 1.5 (HAI-DEF) to translate complex biomedical concepts
about cellular repair, regeneration, and health into patient-friendly
language that anyone can understand.

Requirements:
  - pip install -r requirements.txt
  - huggingface-cli login  (MedGemma is a gated model)
  - GPU recommended (runs on CPU but very slow)

Usage:
  python3 demo_cellrepair_health.py              # Full run with model
  python3 demo_cellrepair_health.py --dry-run     # Test without loading model
  python3 demo_cellrepair_health.py --text-only   # Text-only mode (no images)

Author: Oliver Winkel / CellRepair AI (cellrepair.ai)
License: Apache 2.0
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-1.5-4b-it")
DRY_RUN = "--dry-run" in sys.argv or os.environ.get("MEDGEMMA_DRY_RUN") == "1"
TEXT_ONLY = "--text-only" in sys.argv
OUTPUT_DIR = Path(__file__).parent / "outputs"

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are CellRepair Health Educator, a patient-friendly medical AI assistant "
    "powered by MedGemma. Your role is to explain complex biomedical concepts about "
    "cellular health, repair mechanisms, and regeneration in clear, accessible language. "
    "Always: (1) Use simple analogies and everyday language, (2) Explain WHY something "
    "matters for the patient's health, (3) Provide actionable lifestyle recommendations "
    "when appropriate, (4) Include a brief scientific context without jargon, "
    "(5) Add a disclaimer that this is educational content, not medical advice. "
    "Target audience: health-conscious adults without medical training."
)

# ─── Demo Scenarios ───────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "cellular_repair",
        "title": "Understanding Cellular Repair",
        "prompt": (
            "A patient asks: 'I keep hearing about cellular repair and regeneration. "
            "What does this actually mean for my body? How do my cells fix themselves, "
            "and what happens when this process doesn't work well?'\n\n"
            "Please explain cellular repair mechanisms in simple, patient-friendly language. "
            "Include what autophagy is, how DNA repair works, and why these processes matter "
            "for aging and disease prevention."
        ),
        "category": "education",
    },
    {
        "id": "oxidative_stress",
        "title": "Oxidative Stress Explained",
        "prompt": (
            "A 45-year-old patient says: 'My doctor mentioned oxidative stress in my blood "
            "work. I don't understand what that means. Is it dangerous? What can I do about it?'\n\n"
            "Explain oxidative stress, free radicals, and antioxidant defense in terms a "
            "non-expert can understand. Include practical lifestyle advice for reducing "
            "oxidative damage."
        ),
        "category": "clinical_support",
    },
    {
        "id": "lifestyle_cellular",
        "title": "Lifestyle and Cell Health",
        "prompt": (
            "A health-conscious person asks: 'What are the most evidence-based things I can "
            "do in my daily life to support my body's natural cell repair? I've heard about "
            "intermittent fasting, sleep, and exercise – how do these actually affect my cells?'\n\n"
            "Provide an evidence-based summary of lifestyle factors that support cellular "
            "health. Be specific about mechanisms (e.g., how fasting triggers autophagy) "
            "but keep the language accessible."
        ),
        "category": "prevention",
    },
    {
        "id": "inflammation_simple",
        "title": "Chronic Inflammation Demystified",
        "prompt": (
            "A patient with a family history of autoimmune disease asks: 'Everyone talks "
            "about inflammation being bad, but I thought inflammation helps fight infections. "
            "What's the difference between good and bad inflammation? And how does it affect "
            "my cells long-term?'\n\n"
            "Explain the difference between acute and chronic inflammation at the cellular "
            "level. Use simple analogies. Discuss how chronic inflammation damages cells "
            "and what patients can watch for."
        ),
        "category": "education",
    },
    {
        "id": "aging_cells",
        "title": "Why Do Our Cells Age?",
        "prompt": (
            "A 60-year-old asks: 'Why do we age? I know my cells are constantly being replaced, "
            "so why don't I just stay young forever? What's actually happening inside my cells "
            "as I get older?'\n\n"
            "Explain cellular aging in patient-friendly terms. Cover telomere shortening, "
            "cellular senescence, and the accumulation of damage. Discuss what modern science "
            "says about slowing these processes."
        ),
        "category": "education",
    },
]

# ─── Utility Functions ────────────────────────────────────────────────────────


def check_environment():
    """Check GPU availability and system info."""
    info = {
        "python": sys.version.split()[0],
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "mode": "dry-run" if DRY_RUN else "full",
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 1
            )
        else:
            info["gpu"] = "CPU only"
    except ImportError:
        info["torch"] = "not installed"
        info["cuda"] = False
        info["gpu"] = "N/A"
    return info


def load_model():
    """Load MedGemma 1.5 model and processor."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"  Loading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)

    print(f"  Loading model from {MODEL_ID}...")
    print("  (First run downloads ~8 GB – this may take a few minutes)")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    return model, processor


def generate_response(model, processor, scenario):
    """Generate a response for a given scenario using MedGemma."""
    import torch

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": scenario["prompt"]}],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Decode only the generated tokens (skip the input)
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(
        outputs[0][input_len:], skip_special_tokens=True
    )
    return response.strip()


def print_header():
    """Print a formatted header."""
    print()
    print("=" * 70)
    print("  MedGemma Impact Challenge – CellRepair AI")
    print("  Patient Education for Cellular Health")
    print("=" * 70)


def print_scenario(idx, scenario, response=None):
    """Print a formatted scenario with optional response."""
    print()
    print(f"─── Scenario {idx}: {scenario['title']} ──────────────────")
    print(f"  Category: {scenario['category']}")
    print(f"  Prompt (excerpt): {scenario['prompt'][:100]}...")
    if response:
        print()
        print("  ╔══ MedGemma Response ══════════════════════════════════════╗")
        for line in response.split("\n"):
            # Wrap long lines
            while len(line) > 64:
                print(f"  ║ {line[:64]}")
                line = line[64:]
            print(f"  ║ {line}")
        print("  ╚══════════════════════════════════════════════════════════╝")


# ─── Main ─────────────────────────────────────────────────────────────────────


def run_demo():
    """Run the full CellRepair Health Education demo."""
    print_header()

    # Environment check
    env = check_environment()
    print(f"\n  Python:  {env['python']}")
    print(f"  PyTorch: {env.get('torch', 'N/A')}")
    print(f"  GPU:     {env.get('gpu', 'N/A')}")
    print(f"  Model:   {env['model']}")
    print(f"  Mode:    {env['mode']}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if DRY_RUN:
        print("\n  [DRY RUN] Skipping model loading – showing scenarios only.\n")
        for i, scenario in enumerate(SCENARIOS, 1):
            print_scenario(i, scenario)
        print()
        print("─" * 70)
        print("  Dry-run complete. To run with MedGemma:")
        print("    1. huggingface-cli login")
        print(f"    2. Accept license: https://huggingface.co/{MODEL_ID}")
        print("    3. python3 demo_cellrepair_health.py")
        print("─" * 70)
        return

    # Load model
    print("\n  Loading MedGemma 1.5...")
    t0 = time.time()
    try:
        model, processor = load_model()
    except Exception as e:
        print(f"\n  ❌ Model loading failed: {e}")
        print()
        print("  Troubleshooting:")
        print("    1. Run: huggingface-cli login")
        print(f"    2. Accept license at: https://huggingface.co/{MODEL_ID}")
        print("    3. Ensure sufficient GPU memory (~8 GB for 4B model)")
        print("    4. For CPU-only: expect very slow inference (minutes per query)")
        sys.exit(1)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s\n")

    # Run scenarios
    results = []
    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n  Processing scenario {i}/{len(SCENARIOS)}: {scenario['title']}...")
        t0 = time.time()
        response = generate_response(model, processor, scenario)
        gen_time = time.time() - t0

        print_scenario(i, scenario, response)
        print(f"  (Generated in {gen_time:.1f}s)")

        results.append(
            {
                "scenario_id": scenario["id"],
                "title": scenario["title"],
                "category": scenario["category"],
                "prompt": scenario["prompt"],
                "response": response,
                "generation_time_sec": round(gen_time, 2),
            }
        )

    # Save results
    output_file = OUTPUT_DIR / f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "model": MODEL_ID,
                    "timestamp": datetime.now().isoformat(),
                    "environment": env,
                    "system_prompt": SYSTEM_PROMPT,
                    "num_scenarios": len(SCENARIOS),
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n  Results saved to: {output_file}")

    # Summary
    print()
    print("=" * 70)
    print("  DEMO COMPLETE")
    print(f"  Scenarios: {len(results)}")
    total_time = sum(r["generation_time_sec"] for r in results)
    print(f"  Total generation time: {total_time:.1f}s")
    avg_time = total_time / len(results)
    print(f"  Average per scenario: {avg_time:.1f}s")
    print()
    print("  Next steps:")
    print("    1. Review outputs in outputs/ directory")
    print("    2. Finalize WRITEUP.md with actual results")
    print("    3. Record 3-min demo video")
    print("    4. Submit on Kaggle")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
