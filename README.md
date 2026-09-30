# 🤖 Tiny GPT from Scratch

Build and train a character-level GPT (a mini ChatGPT-style Transformer) in PyTorch — every piece written by hand — on Shakespeare or your own Hindi/Marathi text.

**Skills:** language modelling · tokenisation · self-attention (Q/K/V, causal mask) · multi-head attention · Transformer blocks · residual connections · LayerNorm · training at scale · temperature sampling · attention visualisation

**Time:** a month or more · **Level:** advanced

## Run it
1. Open [Google Colab](https://colab.research.google.com) → **File → Upload notebook** → `tiny_gpt.ipynb`.
2. **Runtime → Change runtime type → T4 GPU** (strongly recommended).
3. Run top to bottom (~10 minutes of training; set `BIG = True` for a better, slower model).

## Watch alongside
Andrej Karpathy — [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) and the full [Zero to Hero](https://karpathy.ai/zero-to-hero.html) series.

## Expected results (Tiny Shakespeare, validation loss)
| Model | Val loss |
|---|---|
| Random guessing | ~4.17 |
| Bigram | ~2.5 |
| Default GPT (T4, ~10 min) | ~1.6 |
| `BIG = True` (T4, ~40 min) | ~1.5 |

## Use your own language
Upload a large UTF-8 `.txt` file (1 MB+) and set `TEXT_FILE` in section 2.
# tiny-gpt-from-scratch-ml
