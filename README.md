# Homework 4: GPT2-based TTS on LJ Speech

**Authors:**\
[Roman Pavlosiuk](https://github.com/gllekkoff)\
[Iryna Denysova](https://github.com/Shnapa)

---

## Idea

The core idea is to treat text-to-speech as a language modeling problem. Instead of generating a spectrogram or waveform directly, we convert audio into a discrete token sequence using a neural codec, and then train GPT2 to predict those tokens given text. At inference time, generating audio is just autoregressive sampling — the same thing GPT does with text, just with a different vocabulary.

---

## Data

We used the [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) dataset — ~24 hours of single-speaker English speech with transcripts.

Loading works like this:
- Parse `metadata.csv` to get (id, text, normalized text) triples
- Build file paths from ids
- 90/10 train/val split with a fixed random seed

All audio is resampled to 16kHz before going into the codec. To avoid re-encoding every epoch, all audio files are tokenized once and cached to disk as `.pt` files. The cache filename includes a SHA1 hash of the split's sample IDs, so it auto-invalidates if the split changes.

Samples where `n_text + n_audio + 2 > 768` are dropped to keep sequences a manageable length. After filtering: ~11 700 train / ~1 300 val.

---

## Architecture

**Codec:** [FocalCodec 25Hz](https://huggingface.co/lucadellalib/focalcodec_25hz) encodes raw audio into a flat sequence of integers from a codebook of 8192 entries. 1 second of audio = 25 tokens. The codec is frozen — we never train it, it's just a feature extractor.

**Model (`GPT2TTS`):**
- Backbone: pretrained `GPT2Model` (117M params)
- `audio_emb`: an `nn.Embedding(8194, 768)` that maps audio token ids (plus 2 special tokens) into GPT2's hidden space
- `audio_head`: a `nn.Linear(768, 8194)` that projects hidden states back to audio logits
- The embedding and head weights are tied — same matrix, just transposed — which reduces parameters and keeps the embedding space consistent

Two special tokens:
- `BOS` (id = 8192) — separates text from audio
- `EOS` (id = 8193) — tells the model to stop

Each training sequence looks like:

```
[text token] ... [text token] [BOS] [audio token] ... [audio token] [EOS] [PAD] ...
```

---

## How `_build_inputs` works

This function is where the batch gets packed into a format GPT2 can consume. The tricky part is that text and audio have different lengths per sample, and they use different embedding spaces.

For each sample in the batch:
1. Text tokens go through GPT2's own word embedding (`base.wte`)
2. `BOS` is placed right after the last text token using `audio_emb`
3. Audio tokens follow, also embedded with `audio_emb`
4. `EOS` closes the sequence

The attention mask is set to 1 for all real positions and 0 for padding. The labels tensor is filled with -100 everywhere except the audio + EOS positions, which get the actual token ids. This means the loss is only computed on the audio part — the model is not penalized for whatever it does at text positions, since those are given as input context.

```
position:  0 .. tl-1 | tl  | tl+1 .. tl+al | tl+1+al | tl+2+al ..
content:   text        BOS   audio             EOS        PAD
label:     -100        -100  audio_ids         EOS_id     -100
```

---

## How `forward` works

After `_build_inputs` builds the `(B, L, H)` input tensor, the attention mask, and the labels, the forward pass is straightforward:

1. Pass inputs through GPT2: `h = base(inputs_embeds=inputs, attention_mask=mask)`
2. Project hidden states to audio logits: `logits = audio_head(h)`
3. Compute cross-entropy loss with a causal shift — the hidden state at position `t` predicts the token at position `t+1`, so we compare `logits[:, :-1, :]` against `labels[:, 1:]`
4. Positions with label `-100` are ignored by the loss

Logits are cast to float32 before the softmax to avoid instability under bf16.

---

## How `generate_audio` works

At inference time, the model generates audio tokens one by one given a text prompt.

1. Encode the input text with GPT2's tokenizer
2. Embed the text and append the `BOS` embedding to signal "now generate audio"
3. Run this prefix through GPT2 once to fill the KV cache — this is the efficient part, the whole text is processed in a single forward pass and the results are cached
4. Take the hidden state after `BOS` and project it to logits over the audio vocabulary
5. Apply temperature scaling: `logits / temperature` — higher temperature = more random, lower = more peaked
6. Apply top-k filtering: keep only the top-k logits, set the rest to `-inf`
7. Sample a token from the resulting distribution
8. If the sampled token is `EOS`, stop
9. Otherwise, embed the token and run a single-step forward pass using the cached KV values — only one token is processed per step from here on
10. Repeat until `EOS` or `max_new_tokens`
11. Decode the collected token ids back to audio with FocalCodec

The KV cache is what makes this fast — without it, every step would re-process the entire sequence from scratch.

---

## Training setup

- Framework: PyTorch Lightning
- Optimizer: AdamW, weight decay on all params except biases and LayerNorm weights
- Scheduler: OneCycleLR with cosine annealing, 10% warmup
- Precision: `bf16-mixed`
- Gradient accumulation: 2 steps → effective batch size 16
- Bottom 2 GPT2 blocks frozen to stabilize early training and save memory
- Gradient checkpointing enabled to reduce VRAM usage

### Hyperparameters

| Parameter | Value |
|---|---|
| Base model | gpt2 |
| Codec | focalcodec_25hz (8192 codebook) |
| Max sequence length | 768 tokens |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Batch size | 8 |
| Gradient accumulation | 2 |
| Epochs | 10 |
| Gradient clip | 1.0 |
| Warmup fraction | 10% |
| Frozen GPT2 blocks | bottom 2 |
| Precision | bf16-mixed |

---

## Training observations

- Initial loss was ~12.3, which is higher than random (random over 8194 tokens ≈ 9.0). That's expected — the audio head starts with random weights, so the model has no idea what audio tokens look like at the start
- Val loss dropped steadily across all 10 epochs, ending at **7.864**
- No instability or divergence observed
- The biggest time cost was the initial cache build — encoding ~13k audio files takes a while. After that, each epoch is fast
- Gradient checkpointing + bf16 kept memory usage well within the 8GB VRAM of the RTX 4060

Checkpoint saved to `artifacts/checkpoints/gpt2tts_last_state_dict.pt`. It contains the model state dict, config, codec name, tokenizer name, and sample rate.
