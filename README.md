# GPT from scratch

A small character-level GPT (decoder-only transformer) in PyTorch, trained on Tiny Shakespeare. It follows Andrej Karpathy's "Let's build GPT" lecture. Despite the filename, `bigram.py` is the full transformer, not the bigram model from the start of the lecture.

## Requirements

- Python 3 (run with 3.13.7)
- PyTorch (run with 2.14.1, CPU build)
- A CUDA GPU is strongly recommended. The script uses it automatically if available.

## Install

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```
python bigram.py
```

It prints the parameter count, train/val loss every 500 steps, then 500 characters of generated text. Nothing is saved to disk.

```
10.715201 M parameters
step 0: train loss 4.2889, val loss 4.2889
...
```

The full default run is 5000 steps with batch size 64 and context 256. I only ran a shortened version (batch 16, context 64, 100 steps), and that alone took about 8 minutes on CPU, so don't try the defaults without a GPU.

## Config

Hyperparameters are plain variables at the top of `bigram.py`:

| Name | Default | Meaning |
| --- | --- | --- |
| `batch_size` | 64 | sequences per step |
| `block_size` | 256 | context length |
| `max_iters` | 5000 | training steps |
| `eval_interval` | 500 | steps between loss reports |
| `eval_iters` | 200 | batches averaged per loss report |
| `learning_rate` | 3e-4 | AdamW learning rate |
| `n_embd` / `n_head` / `n_layer` | 384 / 6 / 6 | model size |
| `dropout` | 0.2 | dropout rate |

## Data

`input.txt` is the Tiny Shakespeare corpus (~1.1 MB), read from the working directory, so run the script from this folder. Characters are the vocabulary; the first 90% of the text is train, the rest is validation.
