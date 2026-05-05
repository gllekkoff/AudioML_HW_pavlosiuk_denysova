# Homework 4: GPT2-based TTS on LJ Speech

**Authors:**\
[Roman Pavlosiuk](https://github.com/gllekkoff)\
[Iryna Denysova](https://github.com/Shnapa)

---

## Idea

The core idea is to treat text-to-speech as a language modeling problem. Instead of generating a spectrogram or waveform directly, we convert audio into a discrete token sequence using a neural codec, and then train GPT2 to predict those tokens given text. At inference time, generating audio is just autoregressive sampling - the same thing GPT does with text, just with a different vocabulary.

---

## Data

We used the LJ Speech dataset.

Loading works like this:
- Parse `metadata.csv` to get (id, text, normalized text) triples
- Build file paths from ids
- 90/10 train/val split with a fixed random seed

All audio is resampled to 16kHz before going into the codec. To avoid re-encoding every epoch, all audio files are tokenized once and cached to disk as `.pt` files. The cache filename includes a SHA1 hash of the split's sample IDs, so it auto-invalidates if the split changes.

Samples where `n_text + n_audio + 2 > 768` are removed so the sequences stay short enough to train. After filtering, we have around 11 700 training samples and around 1 300 validation samples.

---

## Architecture

**Codec:** [FocalCodec 25Hz](https://huggingface.co/lucadellalib/focalcodec_25hz) encodes raw audio into a flat sequence of integers from a codebook of 8192 entries. 1 second of audio = 25 tokens. The codec is frozen - we never train it, it's just a feature extractor.

**Model (`GPT2TTS`):**
- Backbone: pretrained `GPT2Model` (117M params)
- `audio_emb`: an `nn.Embedding(8194, 768)` that maps audio token ids (plus 2 special tokens) into GPT2's hidden space
- `audio_head`: a `nn.Linear(768, 8194)` that projects hidden states back to audio logits
- The embedding and head weights are tied - same matrix, just transposed - which reduces parameters and keeps the embedding space consistent

Two special tokens:
- `BOS` (id = 8192) - separates text from audio
- `EOS` (id = 8193) - tells the model to stop

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

The attention mask is set to 1 for all real positions and 0 for padding. The labels tensor is filled with -100 everywhere except the audio + EOS positions, which get the actual token ids. This means the loss is only computed on the audio part - the model is not penalized for whatever it does at text positions, since those are given as input context.

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
3. Compute cross-entropy loss with a causal shift - the hidden state at position `t` predicts the token at position `t+1`, so we compare `logits[:, :-1, :]` against `labels[:, 1:]`
4. Positions with label `-100` are ignored by the loss

Logits are cast to float32 before the softmax to avoid instability under bf16.

---

## How `generate_audio` works

At inference time, the model generates audio tokens one by one given a text prompt.

1. Encode the input text with GPT2's tokenizer
2. Embed the text and append the `BOS` embedding to signal "now generate audio"
3. Run this prefix through GPT2 once to fill the KV cache - this is the efficient part, the whole text is processed in a single forward pass and the results are cached
4. Take the hidden state after `BOS` and project it to logits over the audio vocabulary
5. Apply temperature scaling: `logits / temperature` - higher temperature = more random, lower = more peaked
6. Apply top-k filtering: keep only the top-k logits, set the rest to `-inf`
7. Sample a token from the resulting distribution
8. If the sampled token is `EOS`, stop
9. Otherwise, embed the token and run a single-step forward pass using the cached KV values - only one token is processed per step from here on
10. Repeat until `EOS` or `max_new_tokens`
11. Decode the collected token ids back to audio with FocalCodec

The KV cache is what makes this fast - without it, every step would re-process the entire sequence from scratch.

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

- Initial loss was around 12.3, which is higher than random guessing over 8194 tokens. This is expected because the audio head starts with random weights, so at the beginning the model does not know how to predict audio tokens.
- Val loss dropped steadily across all 10 epochs, ending at **7.864**
- No instability or divergence observed
- The slowest part was building the first token cache, because encoding around 13k audio files takes time. After the cache is created, each epoch is much faster.
- Gradient checkpointing + bf16 kept memory usage well within the 8GB VRAM of the RTX 4060

Checkpoint saved to `artifacts/checkpoints/gpt2tts_last_state_dict.pt`. It contains the model state dict, config, codec name, tokenizer name, and sample rate.

---

## Inference and evaluation

The inference and metrics part is implemented in `hw4_inference_metrics.ipynb`. This notebook does not train the model. It loads the trained checkpoint, generates speech for validation texts, and compares different decoding strategies.

Use the Python 3.10 evaluation environment for reproducible metric runs:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install -r requirements-eval.txt
python -m ipykernel install --user --name audioml-eval-py310 --display-name "AudioML Eval (Python 3.10)"
```

Select the `AudioML Eval (Python 3.10)` kernel in Jupyter/VS Code before running the notebook.

### What the notebook does

First, the notebook loads the trained `GPT2TTS` checkpoint, the GPT2 tokenizer, and FocalCodec. Then it rebuilds the LJ Speech validation split with the same random seed as in training and selects 50 validation texts.

For each text, we generate audio with four methods:

| Method | temperature | top_k | top_p | sampling |
|---|---:|---:|---:|---|
| `greedy` | 1.0 | 0 | 1.0 | no |
| `temperature` | 0.8 | 0 | 1.0 | yes |
| `top_k` | 1.0 | 50 | 1.0 | yes |
| `top_p` | 1.0 | 0 | 0.9 | yes |

This gives 200 generated samples in total: 50 texts for each of the four methods.

### Metrics used

We use three metrics:

- **CER** measures how understandable the generated speech is. Whisper transcribes the generated audio, and `jiwer.cer` compares this transcript with the target text. Lower CER is better.
- **UTMOSv2** estimates speech quality and naturalness without a reference audio file. Higher UTMOSv2 is better.
- **SECS** measures speaker similarity. It compares speaker embeddings of the generated audio and the original LJ Speech reference audio. Higher SECS means the voice is closer to the target speaker.

We use `UTMOSv2` instead of the older `utmos` package because the old version depends on `fairseq`, which was unstable in our environment.

### Results

The completed inference run generated 200 samples. The generated sequences were around 150-168 codec tokens on average. Since the codec produces 25 tokens per second, this is roughly 6 seconds of audio per sample.

The CER results show that the model produces speech-like audio, but the generated words are still often unclear. Whisper usually recognizes only parts of the target sentence, and sometimes it recognizes unrelated words. So the model learned to generate audio tokens, but intelligibility is still the main problem.

The best method in this run was `top_k` sampling. It has the lowest mean CER and the lowest median CER. Temperature sampling is second. Greedy decoding and `top_p` are weaker.

| Method | CER mean | CER median | Avg. generated tokens |
|---|---:|---:|---:|
| `top_k` | 0.8004 | 0.7780 | 152.10 |
| `temperature` | 0.8594 | 0.8263 | 158.66 |
| `greedy` | 0.9066 | 0.9115 | 167.66 |
| `top_p` | 0.9066 | 0.8366 | 160.20 |

Greedy decoding is too rigid: it often collapses into repetitive or unclear outputs. Temperature sampling adds randomness, which helps a bit. `top_k` works best because it keeps generation inside the most likely audio tokens but still allows some variation. `top_p` is less stable here: its median CER is better than greedy, but its mean CER is still high because some samples are much worse.

The current saved results contain the full CER evaluation for all 200 generated samples. UTMOSv2 and SECS are implemented in the updated notebook and should be rerun with the `AudioML Eval (Python 3.10)` kernel to complete the final naturalness and speaker-similarity comparison.

Overall, the best current decoding choice is `top_k`. It gives the most understandable outputs among the tested methods, but the high CER values show that the model still needs more training or stronger conditioning to produce accurate speech.
