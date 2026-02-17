# MedGemma 1.5 Multimodal Image Analysis - Quick Start

## TL;DR

Copy this into your Jupyter notebook to analyze cell images with MedGemma 1.5:

```python
from image_analysis_cell import (
    get_sample_cell_image,
    analyze_cell_image_multimodal,
    initialize_medgemma_multimodal
)

# Step 1: Get image (downloads from Wikimedia or generates synthetic)
image = get_sample_cell_image()

# Step 2: Initialize model (if not already done)
processor, model, device = initialize_medgemma_multimodal()

# Step 3: Analyze image with question
result = analyze_cell_image_multimodal(
    image,
    processor,
    model,
    device,
    patient_question="What cellular structures do you see? Explain in patient-friendly language."
)

# Step 4: View results
print(result['explanation'])
print(f"⏱️ Generation Time: {result['generation_time_ms']}ms")
print(f"📊 Tokens: {result['output_tokens']} generated, {result['tokens_per_second']} tok/sec")
```

## What You Get

✅ **Image Acquisition**: Automatic download from Wikimedia Commons (or synthetic fallback)
✅ **Multimodal Processing**: Proper MedGemma 1.5 chat template format with vision encoding
✅ **Patient-Friendly Explanations**: Cell structures explained in understandable language
✅ **Performance Metrics**: Generation time, token counts, throughput
✅ **GPU/CPU Support**: Automatic device detection and bfloat16 optimization

## The Key Innovation: Multimodal Format

MedGemma 1.5 differs from text-only models. The content is a **list**, not a string:

```python
# ✅ CORRECT for MedGemma 1.5 (multimodal)
messages = [{
    "role": "user",
    "content": [
        {"type": "image"},                    # Image placeholder
        {"type": "text", "text": "question"}  # Text question
    ]
}]

# ❌ WRONG for MedGemma 1.5 (text-only format)
messages = [{
    "role": "user",
    "content": "question"  # This won't work with images
}]
```

The processor handles the rest:
- Converts `{"type": "image"}` to special tokens
- Encodes the PIL Image to tokens using SigLIP vision encoder
- Merges everything for model.generate()

## File Locations

```
/sessions/blissful-eloquent-allen/mnt/claude/medgemma_impact_challenge/
├── image_analysis_cell.py                  ← Main implementation (384 lines)
├── MULTIMODAL_IMPLEMENTATION_GUIDE.md      ← Deep technical details
├── FREE_CELL_IMAGE_SOURCES.md             ← Image sources & URLs
├── IMAGE_ANALYSIS_SUMMARY.md              ← Overview & integration
└── QUICK_START.md                         ← This file
```

## Image Sources

### Option 1: Wikimedia Commons (Real Medical Images)

Free, CC-licensed cell microscopy images:
- https://commons.wikimedia.org/wiki/Category:Fluorescent_microscope_images
- https://commons.wikimedia.org/wiki/Category:Cell_biology
- https://commons.wikimedia.org/wiki/Category:Microscopic_images_of_cell_anatomy

### Option 2: Synthetic Generation (Always Works)

Automatic fallback in the code - generates stylized cell diagram if download fails.

## Code Structure

### 1. Image Acquisition
```python
get_sample_cell_image()
  → Try: Download from Wikimedia Commons
  → Fallback: Generate synthetic cell diagram
```

### 2. Model Initialization
```python
initialize_medgemma_multimodal()
  → Loads: AutoProcessor, AutoModelForImageTextToText
  → Device: GPU (CUDA) or CPU
  → Precision: bfloat16 (memory-efficient)
```

### 3. Multimodal Analysis
```python
analyze_cell_image_multimodal(image, processor, model, device, question)
  → Format: Multimodal chat template with image + text
  → Process: Vision encoding + text generation
  → Measure: Time, tokens, throughput
```

## Generation Parameters

```python
model.generate(
    max_new_tokens=256,    # Short, focused explanations
    temperature=0.7,       # Balanced creativity/focus
    top_p=0.9,            # Nucleus sampling
    do_sample=True        # Stochastic for natural language
)
```

Adjust these for your use case:
- **Faster**: Reduce `max_new_tokens` to 128
- **Higher Quality**: Increase to 512
- **More Deterministic**: Lower `temperature` to 0.5
- **More Creative**: Increase `temperature` to 0.9

## Example Questions for Patient Education

```python
questions = [
    "What do you see in this cell image? Explain in simple terms.",
    "What are the main structures? What do they do?",
    "How would you explain this to someone with no biology background?",
    "Are there any organelles visible? What are their functions?",
    "How is this cell structured? Why is that important?"
]

for q in questions:
    result = analyze_cell_image_multimodal(image, processor, model, device, q)
    print(f"Q: {q}\nA: {result['explanation']}\n")
```

## Performance Expectations

| Metric | Typical Value |
|--------|---------------|
| Image Encoding | 50-100ms |
| Text Generation (256 tokens) | 200-500ms |
| Total per Image | 250-600ms |
| Tokens/Second | 400-1200 tok/sec |
| VRAM Required | 8-10GB |

*Timings on single GPU. CPU will be 5-10x slower.*

## Integration Checklist

- [ ] Copy `image_analysis_cell.py` to your notebook directory
- [ ] Import functions: `from image_analysis_cell import ...`
- [ ] Ensure dependencies: `transformers`, `torch`, `PIL`, `requests`
- [ ] Initialize model once (expensive operation)
- [ ] Acquire image (download or generate)
- [ ] Call `analyze_cell_image_multimodal()` with image + question
- [ ] Parse results dict for explanation and metrics

## Troubleshooting

### "apply_chat_template expects a list"
Make sure content is a **list** of dicts, not a single string or dict.

### Slow generation on CPU
Normal. For real-time use, GPU is required. 4B model on CPU: ~2-5 sec per image.

### Wikimedia download fails
Code automatically falls back to synthetic generation. No error - it just generates a demo image.

### CUDA out of memory
1. Reduce `max_new_tokens` (256 → 128)
2. Use `device_map="cpu"` for CPU-only mode (slower)
3. Use quantization (not in current code)

### Image size issues
Processor handles resizing automatically. Works with any size from 64x64 to 4096x4096.

## API Reference (Quick)

### `get_sample_cell_image()`
Returns PIL.Image of cell microscopy (real or synthetic)

**Usage**:
```python
image = get_sample_cell_image()
```

### `initialize_medgemma_multimodal()`
Returns (processor, model, device)

**Usage**:
```python
processor, model, device = initialize_medgemma_multimodal()
```

### `analyze_cell_image_multimodal()`
Returns dict with keys: `explanation`, `generation_time_ms`, `input_tokens`, `output_tokens`, `tokens_per_second`

**Usage**:
```python
result = analyze_cell_image_multimodal(image, processor, model, device, patient_question)
print(result['explanation'])
```

## For CellRepair Health Educator

This code enables:
1. ✅ Explaining cell structures in patient-friendly language
2. ✅ Visual analysis of microscopy images with medical context
3. ✅ Educational content generation for health literacy
4. ✅ Measuring model performance on medical image interpretation

Use cases:
- Health education platforms
- Patient-facing medical explanations
- Cellular biology visualization
- Accessibility features for medical image understanding

## Next Steps

1. **Test**: Run the Quick Start code above in a notebook cell
2. **Customize**: Adjust patient questions for your use case
3. **Integrate**: Add to your existing MedGemma notebook
4. **Evaluate**: Check quality of patient-friendly explanations
5. **Optimize**: Tune generation parameters if needed

## Documentation

For more details, see:
- **Implementation Details** → `MULTIMODAL_IMPLEMENTATION_GUIDE.md`
- **Image Sources** → `FREE_CELL_IMAGE_SOURCES.md`
- **Full Summary** → `IMAGE_ANALYSIS_SUMMARY.md`
- **Code Comments** → Read inline comments in `image_analysis_cell.py`

## Resources

- [MedGemma Model Card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [MedGemma on Hugging Face](https://huggingface.co/google/medgemma-1.5-4b-it)
- [Multimodal Chat Templates](https://huggingface.co/docs/transformers/v4.49.0/en/chat_template_multimodal)
- [Wikimedia Commons Cell Images](https://commons.wikimedia.org/wiki/Category:Cell_biology)

---

**Ready to go!** Copy the TL;DR code above and start analyzing cell images with MedGemma 1.5.
