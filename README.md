# HAI Indexer - LLaMA Factory & llama.cpp Integration

This repository combines **LLaMA-Factory** (for training fine-tuned language models) and **llama.cpp** (for efficient inference and model conversion) to provide a complete pipeline for training, merging, and deploying specialized AI models.

## 🎯 Repository Overview

This repo contains **two main components** that work together:

### 1. **LLaMA-Factory** (`src/llamafactory/`)
**Purpose**: Training framework for fine-tuning large language models

- **What it does**: Fine-tunes base models (like Mistral-7B) using LoRA/SFT on custom datasets
- **Output**: Trained LoRA adapters saved in `saves/` directory
- **Key features**:
  - LoRA fine-tuning (parameter-efficient)
  - Full SFT training
  - Multiple training methods (DPO, PPO, etc.)
  - Web UI for training management
  - API server for inference

### 2. **llama.cpp** (`llama.cpp/`)
**Purpose**: Efficient inference engine and model conversion tool

- **What it does**: Converts HuggingFace models to GGUF format and runs fast CPU/GPU inference
- **Output**: GGUF model files ready for Ollama or standalone inference
- **Key features**:
  - Model quantization (Q4, Q8, FP16, etc.)
  - Fast CPU inference
  - GPU acceleration support
  - Integration with Ollama
  - Memory-efficient inference

## 🔄 Complete Pipeline

```
Training Data (JSON) 
    ↓
[LLaMA-Factory] LoRA Training
    ↓
LoRA Adapters (saves/)
    ↓
[LLaMA-Factory] Merge LoRA → Full Model
    ↓
Merged HF Model (exports/)
    ↓
[llama.cpp] Convert to GGUF
    ↓
GGUF Model (exports/gguf/)
    ↓
[Ollama/llama.cpp] Deploy & Inference
```

## 📁 Repository Structure

```
LLaMA-Factory/
├── src/llamafactory/          # LLaMA-Factory training framework
│   ├── train/                 # Training loops
│   ├── model/                 # Model loading & adapters
│   ├── data/                  # Dataset processing
│   └── chat/                  # Inference & chat
│
├── llama.cpp/                 # llama.cpp inference engine (git submodule)
│   ├── llama-cli              # Command-line inference
│   ├── llama-quantize         # Model quantization
│   └── ...                    # Core inference code
│
├── data/                      # Training datasets (JSON format)
│   ├── hai_indexer_rag_training.json
│   ├── business_integration_training.json
│   └── ...
│
├── saves/                     # Trained LoRA adapters (gitignored)
│   └── Mistral-7B-Instruct-v0.2/lora/
│       └── trained_with_new_data/
│
├── exports/                   # Merged models & GGUF exports (gitignored)
│   ├── hai_indexer_mistral_7b_new_data/  # Merged HF model
│   └── gguf/                              # GGUF models
│
├── scripts/                    # Utility scripts
│   └── merge_trained_with_new_data.py
│
├── docker/                     # Docker setup for training
│   └── docker-cuda/
│
└── examples/                   # Training & inference configs
    ├── train_lora/
    └── inference/
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ (3.10+ recommended)
- CUDA-capable GPU (for training) or CPU (for inference)
- Docker (optional, for containerized training)

### Installation

```bash
# Clone the repository (including submodules)
git clone --recursive <your-repo-url>
cd LLaMA-Factory

# If you already cloned without --recursive, initialize submodules:
git submodule update --init --recursive

# Install LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation

# llama.cpp is included as a git submodule
# Build llama.cpp tools (optional, for advanced usage)
cd llama.cpp
make
```

### Training a Model (LLaMA-Factory)

```bash
# Train LoRA adapter on your dataset
llamafactory-cli train examples/train_lora/mistral7b_hai_indexer_sft.yaml

# Check trained adapters
ls saves/Mistral-7B-Instruct-v0.2/lora/
```

### Merging LoRA to Full Model

```bash
# Merge LoRA adapter into base model
python scripts/merge_trained_with_new_data.py

# Output: exports/hai_indexer_mistral_7b_new_data/
```

### Converting to GGUF (llama.cpp)

```bash
# Convert merged HF model to GGUF format
./convert_to_gguf.sh

# Output: exports/gguf/hai-indexer-mistral-7b-fp16.gguf
```

### Using with Ollama

```bash
cd exports/gguf
ollama create hai-indexer -f Modelfile
ollama run hai-indexer
```

## 📊 Current Models

### Latest Trained Model

- **Base Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Adapter Location**: `saves/Mistral-7B-Instruct-v0.2/lora/trained_with_new_data/`
- **Training Steps**: 620 steps (10 epochs)
- **Datasets Used**:
  - `hai_indexer_rag_training` (RAG context understanding)
  - `business_integration_training` (Business domain knowledge)
  - `company_kb_training` (Company knowledge base)
  - `entity_classification_training` (Entity recognition)
  - `hard_negative_hallucination` (Anti-hallucination)
  - `safety_guardrails` (Safety training)
  - And more...

### Merged Models

- **Full HF Model**: `exports/hai_indexer_mistral_7b_new_data/`
  - Format: HuggingFace SafeTensors
  - Use with: Transformers library, LLaMA-Factory inference

- **GGUF Model**: `exports/gguf/hai-indexer-mistral-7b-fp16.gguf`
  - Format: GGUF (llama.cpp format)
  - Use with: Ollama, llama.cpp, llama-cpp-python

## 🛠️ Usage Examples

### Query Trained Model (LoRA Adapter)

```bash
# Using LLaMA-Factory CLI
llamafactory-cli chat \
  --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter_name_or_path saves/Mistral-7B-Instruct-v0.2/lora/trained_with_new_data \
  --template mistral \
  --finetuning_type lora \
  --default_system "You are HAI Indexer, an AI assistant..."
```

### Query Merged Model (Full HF)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("exports/hai_indexer_mistral_7b_new_data")
tokenizer = AutoTokenizer.from_pretrained("exports/hai_indexer_mistral_7b_new_data")
# ... use model for inference
```

### Query GGUF Model (Ollama)

```bash
ollama run hai-indexer "Hello! Who are you?"
```

## 📝 Key Files

- **Training Config**: `examples/train_lora/mistral7b_hai_indexer_sft.yaml`
- **Merge Script**: `scripts/merge_trained_with_new_data.py`
- **GGUF Conversion**: `convert_to_gguf.sh`
- **Ollama Guide**: `OLLAMA_GUIDE.md`
- **Training Data**: `data/*.json`

## 🔒 What's NOT in Git

The following directories are gitignored (models are too large for git):

- `saves/` - Trained LoRA adapters
- `exports/` - Merged models and GGUF files
- `offload/` - Temporary conversion files
- `*.gguf`, `*.safetensors` - Model weight files

**Note**: Training configs, scripts, and data formats ARE in git. Only the actual model weights are excluded.

## 🐳 Docker Usage

For consistent training environment:

```bash
cd docker/docker-cuda
docker compose up -d
docker exec -it llamafactory bash

# Inside container, your saves/ and data/ are mounted
llamafactory-cli train examples/train_lora/mistral7b_hai_indexer_sft.yaml
```

## 📚 Documentation

- **LLaMA-Factory Docs**: See original [README](https://github.com/hiyouga/LLaMA-Factory) for full LLaMA-Factory features
- **llama.cpp Docs**: See [llama.cpp README](llama.cpp/README.md) for inference details
- **Ollama Guide**: `OLLAMA_GUIDE.md` - Complete guide for using GGUF models with Ollama
- **Training Guide**: `HAI_INDEXER_TRAINING.md` - Hai Indexer specific training documentation

## 🤝 Contributing

This is a specialized fork for Hai Indexer. For general LLaMA-Factory contributions, see the [original repository](https://github.com/hiyouga/LLaMA-Factory).

## 📄 License

- **LLaMA-Factory**: Apache 2.0 (see LICENSE)
- **llama.cpp**: MIT (see llama.cpp/LICENSE)
- **Models**: Inherit license from base model (Mistral-7B-Instruct-v0.2: Apache 2.0)

## 🙏 Acknowledgments

- **LLaMA-Factory**: [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Training framework
- **llama.cpp**: [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) - Inference engine
- **Base Model**: [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

---

**Summary**: This repo provides a complete pipeline from training (LLaMA-Factory) → merging → conversion (llama.cpp) → deployment (Ollama/llama.cpp) for specialized AI models.
