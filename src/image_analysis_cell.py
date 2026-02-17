"""
MedGemma 1.5 Multimodal Image Analysis Cell
For CellRepair Health Educator - Kaggle MedGemma Impact Challenge

This cell demonstrates how to:
1. Load or generate a cell biology/microscopy image
2. Send it to MedGemma 1.5 with a patient question using multimodal format
3. Get patient-friendly explanations of medical images
4. Measure generation performance

Image source options:
- Download from Wikimedia Commons (free, CC-licensed medical images)
- Generate a synthetic cell diagram using matplotlib
"""

import time
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ============================================================================
# 1. IMAGE ACQUISITION: Download or Generate Cell Microscopy Image
# ============================================================================

def get_sample_cell_image():
    """
    Acquire a sample cell microscopy image from a free source.

    Option A: Download from Wikimedia Commons (requires internet)
    Option B: Generate synthetic cell diagram locally (always works)

    Returns:
        PIL.Image: RGB image of a cell or cell structure
    """

    try:
        # Option A: Try to download a real microscopy image from Wikimedia Commons
        # This is a fluorescent microscopy image of human cells (CC-licensed)
        # Using a specific image ID from Wikimedia Commons
        print("Attempting to download cell microscopy image from Wikimedia Commons...")

        # Example: Human cell nucleus with fluorescent staining
        # File ID pointing to open-licensed cell biology image
        wikimedia_url = (
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            "Human_cells_(Hoechst_nuclear_staining).jpg?width=800"
        )

        response = requests.get(wikimedia_url, timeout=10)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            print(f"Successfully loaded image from Wikimedia Commons: {image.size}")
            return image
        else:
            print(f"Download failed (status {response.status_code}), generating synthetic image...")

    except Exception as e:
        print(f"Download error ({type(e).__name__}), generating synthetic cell image...")

    # Option B: Generate a synthetic cell diagram
    return generate_synthetic_cell_image()


def generate_synthetic_cell_image():
    """
    Generate a synthetic cell diagram for demonstration.

    Creates a stylized cell image with:
    - Cell nucleus (round structure)
    - Cytoplasm (surrounding area)
    - Mitochondria (oval structures)
    - Cell membrane (outer boundary)

    Returns:
        PIL.Image: RGB image of synthetic cell
    """

    print("Generating synthetic cell diagram...")

    # Create image canvas
    width, height = 512, 512
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image, 'RGBA')

    # Background - light blue cytoplasm
    draw.rectangle([0, 0, width, height], fill=(220, 240, 255))

    # Cell membrane - dark blue outer circle
    membrane_margin = 20
    draw.ellipse(
        [membrane_margin, membrane_margin,
         width - membrane_margin, height - membrane_margin],
        outline=(20, 100, 150), width=3
    )

    # Nucleus - large purple circular structure in center
    nucleus_x, nucleus_y = width // 2, height // 2
    nucleus_radius = 80
    draw.ellipse(
        [nucleus_x - nucleus_radius, nucleus_y - nucleus_radius,
         nucleus_x + nucleus_radius, nucleus_y + nucleus_radius],
        fill=(180, 100, 200, 200), outline=(100, 50, 150), width=2
    )

    # Add nucleolus (darker center in nucleus)
    nucleolus_radius = 25
    draw.ellipse(
        [nucleus_x - nucleolus_radius, nucleus_y - nucleolus_radius,
         nucleus_x + nucleolus_radius, nucleus_y + nucleolus_radius],
        fill=(120, 40, 150), outline=(80, 20, 120), width=1
    )

    # Mitochondria - multiple elongated structures around nucleus
    mitochondria_positions = [
        (200, 150), (350, 180), (280, 320), (350, 380), (150, 300)
    ]

    for mx, my in mitochondria_positions:
        # Main mitochondrion body
        draw.ellipse(
            [mx - 30, my - 15, mx + 30, my + 15],
            fill=(255, 200, 100, 200), outline=(200, 150, 50), width=1
        )
        # Inner cristae (detail)
        draw.line([mx - 20, my, mx + 20, my], fill=(200, 150, 50), width=1)

    # Add some small vesicles/organelles scattered in cytoplasm
    vesicle_positions = [
        (100, 100), (420, 120), (110, 420), (420, 420), (250, 100), (300, 450)
    ]

    for vx, vy in vesicle_positions:
        draw.ellipse(
            [vx - 8, vy - 8, vx + 8, vy + 8],
            fill=(100, 200, 255), outline=(50, 150, 200), width=1
        )

    print("Synthetic cell image generated successfully")

    return image


# ============================================================================
# 2. MULTIMODAL MODEL SETUP
# ============================================================================

def initialize_medgemma_multimodal():
    """
    Initialize MedGemma 1.5 4B multimodal model and processor.

    Uses:
    - AutoProcessor: Handles both image and text preprocessing
    - AutoModelForImageTextToText: MedGemma 1.5 multimodal variant
    - Device: GPU (cuda) if available, otherwise CPU
    - Precision: bfloat16 for efficiency

    Returns:
        tuple: (processor, model, device)
    """

    print("Initializing MedGemma 1.5 multimodal model...")

    model_id = "google/medgemma-1.5-4b-it"

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Processor loaded from {model_id}")

    # Load model with optimizations
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Model loaded on {device} with bfloat16 precision")

    return processor, model, device


# ============================================================================
# 3. MULTIMODAL INFERENCE: Image + Text to Text
# ============================================================================

def analyze_cell_image_multimodal(image, processor, model, device, patient_question=None):
    """
    Send image + text question to MedGemma 1.5 and get patient-friendly response.

    This demonstrates the multimodal capability where:
    - Content is a list containing both image and text items
    - Images are processed as PIL Image objects
    - Text questions are formatted naturally
    - apply_chat_template() handles the multimodal formatting

    Args:
        image (PIL.Image): Cell microscopy image
        processor: AutoProcessor instance
        model: AutoModelForImageTextToText instance
        device (str): "cuda" or "cpu"
        patient_question (str): Question about the image in patient-friendly language

    Returns:
        dict: Contains explanation, generation_time_ms, input_tokens, output_tokens
    """

    if patient_question is None:
        patient_question = (
            "What do you see in this cell image? "
            "Please explain the different structures you observe in simple, patient-friendly terms."
        )

    print(f"\nPatient Question: {patient_question}")
    print("-" * 70)

    # ========================================================================
    # MULTIMODAL CHAT TEMPLATE FORMAT
    # ========================================================================
    # Content is a list with mixed types:
    # - {"type": "image"}: The PIL Image is passed separately to processor
    # - {"type": "text", "text": "..."}: The text question
    #
    # The processor.apply_chat_template() method converts this format
    # into the correct token sequence for MedGemma 1.5's vision encoder.
    # ========================================================================

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": patient_question
                }
            ]
        }
    ]

    # Apply chat template for multimodal input
    # Note: The processor expects the image to be passed separately in the
    # images parameter, while content list holds the structure
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    # Prepare inputs: both image and text together
    # The processor handles the vision encoding of the image automatically
    inputs = processor(
        images=image,  # PIL Image is passed here
        text=text,     # Template-formatted text with image placeholder
        return_tensors="pt",
        padding=True
    )

    # Move inputs to device (GPU or CPU)
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    # ========================================================================
    # GENERATION WITH TIMING
    # ========================================================================

    start_time = time.time()

    # Generate response with controlled sampling
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    generation_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Decode the response
    # Skip the input tokens to get only the generated text
    generated_text = processor.decode(
        output[0][inputs['input_ids'].shape[-1]:],
        skip_special_tokens=True
    )

    # Get token counts
    input_token_count = inputs['input_ids'].shape[-1]
    output_token_count = output.shape[-1] - input_token_count

    print(f"Medical Image Analysis (Patient-Friendly Explanation):")
    print(f"\n{generated_text}\n")

    return {
        "explanation": generated_text,
        "generation_time_ms": round(generation_time, 2),
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "tokens_per_second": round(
            output_token_count / (generation_time / 1000), 2
        ) if generation_time > 0 else 0
    }


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    """
    Complete pipeline for multimodal cell image analysis with MedGemma 1.5.

    Demonstrates:
    1. Image acquisition (download or synthetic)
    2. Model initialization with device mapping
    3. Multimodal inference with timing
    4. Performance metrics
    """

    print("=" * 70)
    print("MedGemma 1.5 Multimodal Cell Image Analysis")
    print("Kaggle MedGemma Impact Challenge - CellRepair Health Educator")
    print("=" * 70)

    # Step 1: Get sample image
    print("\n[STEP 1] Acquiring sample cell image...")
    image = get_sample_cell_image()
    print(f"Image shape: {image.size}, mode: {image.mode}")

    # Step 2: Initialize model
    print("\n[STEP 2] Initializing MedGemma 1.5 multimodal model...")
    processor, model, device = initialize_medgemma_multimodal()

    # Step 3: Run multimodal analysis
    print("\n[STEP 3] Running multimodal image analysis...")

    # Example patient questions
    patient_questions = [
        "What structures do you see in this cell image? Explain it as if speaking to a patient.",
        "Are there any visible organelles? What are their functions in simple terms?",
    ]

    results = []
    for i, question in enumerate(patient_questions, 1):
        print(f"\n--- Analysis {i}/{len(patient_questions)} ---")

        result = analyze_cell_image_multimodal(
            image,
            processor,
            model,
            device,
            patient_question=question
        )
        results.append(result)

    # Step 4: Performance summary
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        print(f"\nAnalysis {i}:")
        print(f"  Generation Time: {result['generation_time_ms']}ms")
        print(f"  Input Tokens: {result['input_tokens']}")
        print(f"  Output Tokens: {result['output_tokens']}")
        print(f"  Tokens/Second: {result['tokens_per_second']}")

    # Summary statistics
    avg_gen_time = sum(r['generation_time_ms'] for r in results) / len(results)
    total_output = sum(r['output_tokens'] for r in results)

    print(f"\nSummary:")
    print(f"  Average Generation Time: {avg_gen_time:.2f}ms")
    print(f"  Total Output Tokens: {total_output}")
    print(f"  Model: google/medgemma-1.5-4b-it")
    print(f"  Precision: bfloat16")
    print(f"  Device: {device}")

    return results


if __name__ == "__main__":
    results = main()
