# Text-to-Text Transfer Transformer (T5) Quantized Model for Text Translation

This repository hosts a quantized version of the T5 model, fine-tuned for text translation tasks. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

## Model Details
- **Model Architecture:** T5  
- **Task:** Text Translation  
- **Dataset:** Hugging Face's `opus100`  
- **Quantization:** Float16
- **Supporting Languages:** English to French
- **Fine-tuning Framework:** Hugging Face Transformers  

## Usage
### Installation
```sh
pip install transformers torch
```

### Loading the Model
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/t5-text-translator"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

def translate_text(model, text, src_lang, tgt_lang):
    input_text = f"translate {src_lang} to {tgt_lang}: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Generate translation
    output_ids = model.generate(input_ids, max_length=50)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test Example
test_sentences = {"en-fr": "Hello, what is your name?"}

for lang_pair, sentence in test_sentences.items():
    src, tgt = lang_pair.split("-")
    print(f"{src} → {tgt}: {translate_text(model, sentence, src, tgt)}")
```

## 📊 ROUGE Evaluation Results
After fine-tuning the T5-Small model for text translation, we obtained the following ROUGE scores:

| **Metric**  | **Score** | **Meaning**  |
|------------|---------|--------------------------------------------------------------|
| **ROUGE-1**  | 0.4673 (~46%) | Measures overlap of unigrams (single words) between the reference and generated text. |
| **ROUGE-2**  | 0.2486 (~24%) | Measures overlap of bigrams (two-word phrases), indicating coherence and fluency. |
| **ROUGE-L**  | 0.4595 (~45%) | Measures longest matching word sequences, testing sentence structure preservation. |
| **ROUGE-Lsum**  | 0.4620 (~46%) | Similar to ROUGE-L but optimized for summarization tasks. |

## Fine-Tuning Details
### Dataset
The Hugging Face's `opus100` dataset was used, containing different types of translations of languages.

### Training
- **Number of epochs:** 3  
- **Batch size:** 8  
- **Evaluation strategy:** epoch  

### Quantization
Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

## Repository Structure
```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## Limitations
- The model may not generalize well to domains outside the fine-tuning dataset.
- Currently, it only supports English to French translations.
- Quantization may result in minor accuracy degradation compared to full-precision models.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
