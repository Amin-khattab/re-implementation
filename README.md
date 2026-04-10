# re-implementation

A repo where I re-implement transformer models — some from scratch to really understand how they work, and others using pre-trained models to see how the big ones are actually used in practice. Splitting it this way helps me keep the "learning the internals" stuff separate from the "using real models" stuff.

## What's in here

### `Scratch_Models/`
Transformers I built from the ground up, no pre-trained weights.

| File | What it is |
|------|------------|
| `simple-Transformer.py` | A minimal transformer implementation — the bare bones |
| `simpler_transformer(explanation).py` | Same idea but with comments and explanations, basically my notes to myself on how each piece works |

### `pre_trained/`
Working with existing pre-trained models instead of training from scratch.

| File | What it is |
|------|------------|
| `ANA(Kurdish-language-Transformer).py` | A transformer setup for Kurdish — something I care about personally, trying to get decent results on a low-resource language |
| `GPT2-Transformer(simple).py` | Basic GPT-2 usage, just getting it running |
| `Gpt2_transformer(Pro).py` | A more advanced GPT-2 setup — fine-tuning, better config, closer to how you'd actually use it |
| `Ollama-Transformer.py` | Hooking into Ollama to run local models |

## Why I built this

Reading papers and watching explanations only gets you so far — at some point you have to actually write the thing yourself. The scratch models are me trying to understand attention, embeddings, and all the inner workings by building them. The pre-trained side is the opposite end: learning how to actually use production-level models instead of toy ones.

The Kurdish transformer is a side interest of mine — there aren't many good models for Kurdish out there and I wanted to mess around with building something useful in that direction.

## Running the scripts

You'll need Python with the usual ML stack:
