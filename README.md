# 🤖 Tiny GPT from Scratch (PyTorch)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A clean, modular, and fully documented implementation of an autoregressive, character-level **Generative Pre-trained Transformer (GPT)** built completely from scratch in **PyTorch**. 

This project demystifies the core mechanics powering modern Large Language Models (like GPT-4 and ChatGPT) — constructing every building block (tokenization, scaled dot-product attention, multi-head attention, feed-forward networks, residual streams, pre-layer normalization, and sampling strategies) by hand without high-level abstractions like Hugging Face.

---

## 📌 Table of Contents

- [Overview & Key Features](#-overview--key-features)
- [Architecture & Mechanics](#-architecture--mechanics)
- [Expected Benchmarks](#-expected-benchmarks)
- [Quick Start](#-quick-start)
  - [Option 1: Google Colab (Recommended)](#option-1-google-colab-recommended)
  - [Option 2: Local Installation](#option-2-local-installation)
- [Implementation Pipeline](#-implementation-pipeline)
- [Model Configurations](#-model-configurations)
- [Inference & Generation Strategies](#-inference--generation-strategies)
- [Attention Visualization](#-attention-visualization)
- [Training on Custom Datasets (Multilingual / Hindi / Marathi)](#-training-on-custom-datasets-multilingual--hindi--marathi)
- [Tiny GPT vs. Modern LLMs (GPT-3 / ChatGPT)](#-tiny-gpt-vs-modern-llms-gpt-3--chatgpt)
- [Ablations & Extensions](#-ablations--extensions)
- [References & Acknowledgments](#-references--acknowledgments)

---

## 🚀 Overview & Key Features

- **Built from First Principles:** Step-by-step evolution from a simple Bigram statistical model to a multi-million parameter Decoder-only Transformer.
- **Pure PyTorch Implementation:** No external deep learning abstractions; clear understanding of tensor shapes, matrix multiplications, and backpropagation flows.
- **Causal Self-Attention:** Manual implementation of Query, Key, and Value projections with lower-triangular causal masking to enforce unidirectional autoregression.
- **Pre-LayerNorm Transformer Blocks:** Modern architecture with pre-normalization and residual stream addition for stable deep network optimization.
- **Advanced Sampling Mechanics:** Text generation supporting temperature scaling, Top-K filtering, and prompt conditioning.
- **Attention Map Visualizer:** Built-in inspection tools to visualize how individual attention heads attend to preceding tokens in real time.
- **Multilingual Ready:** Full UTF-8 support enabling training on Devanagari scripts (Hindi, Marathi) and complex character sets out of the box.

---

## 🧠 Architecture & Mechanics

Tiny GPT implements a **Decoder-Only Transformer** architecture based on the principles established in *Attention Is All You Need* (Vaswani et al., 2017) and OpenAI's GPT series.

```
Input Characters
      │
      ▼
┌──────────────────────────────────────────┐
│ Token Embedding + Positional Embedding   │  -> (B, T, n_embd)
└──────────────────┬───────────────────────┘
                   │
         ┌─────────┴─────────┐
         │  Residual Stream  │ ◄──────────┐
         └─────────┬─────────┘            │
                   ▼                      │
       ┌───────────────────────┐          │
       │     LayerNorm (ln1)   │          │
       └───────────┬───────────┘          │
                   ▼                      │
       ┌───────────────────────┐          │
       │ Multi-Head Attention  │──────────┘
       │  (Causal Q, K, V)     │
       └───────────┬───────────┘
                   │
         ┌─────────┴─────────┐
         │  Residual Stream  │ ◄──────────┐
         └─────────┬─────────┘            │
                   ▼                      │
       ┌───────────────────────┐          │
       │     LayerNorm (ln2)   │          │
       └───────────┬───────────┘          │
                   ▼                      │
       ┌───────────────────────┐          │
       │  Feed-Forward Network │──────────┘
       │   (Linear -> GELU     │
       │    -> Linear)         │
       └───────────┬───────────┘
                   │ (Repeated for n_layer blocks)
                   ▼
       ┌───────────────────────┐
       │    Final LayerNorm    │
       └───────────┬───────────┘
                   ▼
       ┌───────────────────────┐
       │   LM Head (Linear)    │
       └───────────┬───────────┘
                   ▼
          Logits (B, T, vocab)
```

### Mathematical Formulation

1. **Scaled Dot-Product Attention:**
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
   where $M_{i,j} = 0$ for $j \le i$ and $-\infty$ for $j > i$ (causal mask).

2. **Transformer Block Forward Pass:**
   $$x' = x + \text{MultiHeadAttention}(\text{LayerNorm}(x))$$
   $$x_{\text{out}} = x' + \text{FeedForward}(\text{LayerNorm}(x'))$$

---

## 📊 Expected Benchmarks

Trained on the **Tiny Shakespeare** dataset (~1 MB, ~1,115,394 characters, vocabulary size = 65):

| Model Stage | Hardware | Training Time | Parameters | Validation Loss | Perplexity ($e^{\text{loss}}$) | Qualitative Output |
|---|---|---|---|---|---|---|
| **Random Guessing** | - | - | 0 | ~4.17 | ~65.0 | Complete random noise |
| **Bigram Model** | CPU / GPU | < 1 min | ~4.2 K | ~2.50 | ~12.2 | Frequent bigrams (`th`, `he`), non-words |
| **Default Tiny GPT** | Colab T4 GPU | ~10 mins | ~3.2 M | **~1.60** | **~4.95** | Plausible Shakespearean words, dialogues, syntax |
| **Large GPT (`BIG=True`)** | Colab T4 GPU | ~35-45 mins | ~10.7 M | **~1.50** | **~4.48** | Coherent character interactions, correct metre & poetic structure |

---

## ⚡ Quick Start

### Option 1: Google Colab (Recommended)

1. Navigate to [Google Colab](https://colab.research.google.com).
2. Go to **File → Upload notebook** and select [`tiny_gpt.ipynb`](file:///c:/Users/ADMIN/Downloads/Work/Projects/ML%20projects/09-tiny-gpt-from-scratch/tiny_gpt.ipynb).
3. Switch runtime to GPU: **Runtime → Change runtime type → T4 GPU → Save**.
4. Run all cells sequentially (**Shift + Enter** or **Runtime → Run all**).

### Option 2: Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sumant26/tiny-gpt-from-scratch-ml.git
   cd tiny-gpt-from-scratch-ml
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # or cpu
   pip install matplotlib jupyter ipykernel
   ```

4. **Launch Jupyter Lab / Notebook:**
   ```bash
   jupyter notebook tiny_gpt.ipynb
   ```

---

## 🛠️ Implementation Pipeline

The notebook walks through the step-by-step evolution of a generative language model:

1. **Environment Setup & Reproducibility:** Device selection (`cuda` vs. `cpu`), setting random seeds.
2. **Data Ingestion:** Automated download and inspection of the Tiny Shakespeare corpus (or custom text).
3. **Character-Level Tokenization:** Constructing forward mapping (`stoi`) and inverse mapping (`itos`), batch encoding/decoding, train/val split (90% / 10%).
4. **Context & Target Batching:** Generating `(B, T)` temporal slices where every index $t$ predicts token $t+1$.
5. **Baseline Bigram Model:** Lookup-table cross-entropy model demonstrating the upper-bound loss limit without attention mechanisms.
6. **Self-Attention Mechanics:**
   - Visualizing uniform historical averaging via lower-triangular matrices.
   - Dynamic weight calculation via $Q, K, V$ linear transformations and dot-product similarity.
   - Applying $-\infty$ causal masks and softmax normalization.
7. **Complete GPT Transformer:**
   - Single Attention Head (`Head`)
   - Multi-Head Attention (`MultiHeadAttention`)
   - 4x Expanded Feed-Forward Network (`FeedForward`) with GELU non-linearities
   - Residual Pre-LayerNorm Block (`Block`)
   - Positional + Token Embedding aggregator (`GPT`)
8. **Training Loop:** AdamW optimizer, cosine annealing learning rate scheduler, periodic evaluation on validation batches, loss curve plotting.
9. **Generation Engine:** Autoregressive sampling loop with temperature scaling and top-k filtering.
10. **Attention Heatmap Visualization:** Inspecting attention head patterns across sample sequences.
11. **Model Checkpointing:** Serializing weights and hyperparameters to `tiny_gpt.pt`.

---

## ⚙️ Model Configurations

| Hyperparameter | CPU Friendly | Default GPU (T4) | Large GPT (`BIG = True`) |
|---|---|---|---|
| `n_embd` (Embedding Dimension) | `128` | `256` | `384` |
| `n_head` (Attention Heads) | `4` | `4` | `6` |
| `head_size` (`n_embd // n_head`) | `32` | `64` | `64` |
| `n_layer` (Transformer Blocks) | `3` | `4` | `6` |
| `block_size` (Context Window) | `64` | `128` | `256` |
| `dropout` | `0.1` | `0.2` | `0.2` |
| `batch_size` | `32` | `64` | `64` |
| `max_iters` | `1500` | `3000` | `5000` |
| `lr` (Initial Learning Rate) | `1e-3` | `1e-3` | `3e-4` |
| **Total Parameters** | **~0.6 M** | **~3.2 M** | **~10.7 M** |

---

## 🎲 Inference & Generation Strategies

Once trained, generate text autoregressively starting from a prompt or initial token (e.g. newline `\n`):

```python
# Conditioning with a custom prompt
prompt = "ROMEO:\n"
encoded_prompt = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

# Sample with temperature and top-k
generated_tokens = model.generate(
    encoded_prompt, 
    max_new_tokens=400, 
    temperature=0.8, 
    top_k=20
)
print(decode(generated_tokens[0].tolist()))
```

### Sampling Parameters

- **Temperature ($T$):** Controls randomness in output distributions:
  - $T \in [0.2, 0.5]$ (Low): Deterministic, conservative, high repetition.
  - $T \in [0.7, 0.9]$ (Balanced): Natural balance between coherence and lexical variety.
  - $T \ge 1.2$ (High): Creative, diverse, higher probability of grammatical mistakes.
- **Top-K Filtering:** Restricts the candidate token pool to only the top $k$ highest-probability tokens prior to multinomial sampling, preventing improbable tail hallucinations.

---

## 🔍 Attention Visualization

The project includes built-in attention extraction allowing you to view the exact weights assigned by individual attention heads to preceding context:

```
Head 1 (Local context / spelling)        Head 2 (Grammar / Speaker Dialogue)
[F][i][r][s][t]                         [F][i][r][s][t]
 █ ░ ░ ░ ░                               █ ░ ░ ░ ░
 █ █ ░ ░ ░                               ░ █ ░ ░ ░
 ░ █ █ ░ ░                               ░ ░ █ ░ ░
 ░ ░ █ █ ░                               █ ░ ░ █ ░
 ░ ░ ░ █ █                               ░ █ ░ ░ █
```
*(Brighter intensities represent higher attention coefficients; the black upper triangle confirms zero future leakage via the causal mask).*

---

## 🇮🇳 Training on Custom Datasets (Multilingual / Hindi / Marathi)

Because the model operates directly on Unicode character points, it can train on any language without architecture modifications:

1. Prepare a UTF-8 text file (1 MB+ recommended, e.g. `marathi.txt` or `hindi.txt`).
2. Update Section 2 in [`tiny_gpt.ipynb`](file:///c:/Users/ADMIN/Downloads/Work/Projects/ML%20projects/09-tiny-gpt-from-scratch/tiny_gpt.ipynb):
   ```python
   TEXT_FILE = "marathi.txt"
   ```
3. Run the notebook. The tokenizer will automatically extract the complete Devanagari character set (vowels, consonants, matras, and conjunct markers / virama `्`).

---

## 🎓 Tiny GPT vs. Modern LLMs (GPT-3 / ChatGPT)

| Metric / Aspect | Tiny GPT (This Repository) | OpenAI GPT-3 | ChatGPT / GPT-4 |
|---|---|---|---|
| **Parameters** | ~1M – 10.7M | 175 Billion | Estimated 1+ Trillion (MoE) |
| **Tokenizer** | Character-level (~65 tokens) | Byte-Pair Encoding / BPE (~50k tokens) | Byte-Pair Encoding / tiktoken (~100k tokens) |
| **Context Length** | 128 – 256 characters | 2,048 tokens (~1,500 words) | 8,192 – 128,000+ tokens |
| **Training Corpus** | ~1 MB (Tiny Shakespeare) | ~570 GB (Common Crawl, Books, Web) | Multi-Terabyte Multimodal Web Data |
| **Objective** | Autoregressive Next-Token Prediction | Autoregressive Next-Token Prediction | Pre-training + SFT + RLHF / DPO |
| **Compute Requirement** | 1x Free T4 GPU (~10 min) | Thousands of A100/H100 clusters | Massive GPU Supercomputers |

---

## 🧪 Ablations & Extensions

Looking to explore deeper? Try these experiments:

1. **Residual Connections Ablation:** Remove `x = x + ...` skips from `Block.forward()`. Observe how gradient flow degrades and validation loss stalls.
2. **LayerNorm Placement:** Compare Pre-LN ($x + \text{attn}(\text{LN}(x))$) vs. original Post-LN ($\text{LN}(x + \text{attn}(x))$).
3. **Sub-word Tokenization:** Integrate OpenAI's `tiktoken` or Hugging Face's `tokenizers` (Byte-Pair Encoding) to compress token sequences and reduce sequence lengths.
4. **Rotary Position Embeddings (RoPE):** Replace absolute learned positional embeddings with RoPE (as used in LLaMA and Mistral).

---

## 📚 References & Acknowledgments

- **Andrej Karpathy:** ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) and the [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series.
- **Vaswani et al. (2017):** ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Original Transformer architecture paper).
- **Radford et al. (2019):** ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2 paper).
- **nanoGPT Repository:** [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) for scaled optimization.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use, modify, and distribute for educational and personal projects.
