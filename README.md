# HW1: Phoneme Recognition + Audio Filters

**Authors:** Roman Pavlosiuk, Iryna Denysova

---

## Task 1 - Phoneme Recognition on TIMIT

### 1. Data Overview

TIMIT is a dataset of 630 American English speakers, each reading 10 sentences recorded at 16 kHz. Each recording comes with a `.PHN` file that says exactly which phoneme was spoken at each moment.

- Train: 3696 utterances
- Test: 1344 utterances

The data is very clean - recorded in a studio with professional microphones. Labels are manually verified, not auto-generated.

**Problems we had to deal with:**

TIMIT has 61 phonemes, but a lot of them sound almost the same. The standard solution is to merge similar ones down to **39 phonemes** (for example `ao -> aa`, `ix -> ih`). All our results are measured on these 39.

We also removed SA sentences from the dataset. All 630 speakers read the exact same 2 SA sentences, so if we kept them the model could just memorize those specific recordings instead of actually learning phonemes.

The task itself is also just hard — phoneme boundaries in real speech are not sharp. One sound blends smoothly into the next, so even the labels are approximate.

---

### 2. Approach

We did not train a model from scratch. Instead, we took large pre-trained speech models, froze them, and used their internal features to train a small classifier.

The idea is that wav2vec2 and HuBERT were pre-trained on 960 hours of speech and already know a lot about how audio works. We just need to teach the classifier which features correspond to which phoneme.

**How the pipeline works:**
1. Pass each audio file through the SSL model once and save the output - one 768-dimensional vector per 20ms frame
2. Train a small MLP on those saved vectors using phoneme labels
3. At test time: run the MLP frame by frame, collapse consecutive identical predictions into a sequence, compare to the reference with edit distance

**The two SSL models we used:**

Both have the same structure inside: a CNN that converts raw audio into frames, then 12 transformer layers that add context. The difference is how they were trained:

- **wav2vec2-base** - trained to pick the correct speech unit from a set of fake ones (contrastive learning)
- **HuBERT-base** - trained to predict cluster labels (k-means clusters of MFCC features) at masked positions

**The MLP classifier:**
```
Linear(768 -> 512) -> LayerNorm -> ReLU -> Dropout(0.1) -> Linear(512 -> 39)
```
Trained for 15 epochs with Adam and cosine annealing.

**How we measure quality — PER (Phoneme Error Rate):**
```
PER = edit_distance(predicted, reference) / len(reference)
```
Lower is better. PER of 15% means roughly 85% of phonemes are correct.

---

### 3. Metrics

**Last layer (layer 12) results:**c

| Model | PER |
|---|---|
| wav2vec2-base | 38.31% |
| HuBERT-base | **15.09%** |

**Layer sweep — wav2vec2-base:**

| Layer | PER |
|---|---|
| 0 (CNN only) | 46.40% |
| 3 | 36.69% |
| **6** | **21.67%** |
| 9 | 22.37% |
| 12 (last) | 37.56% |

**Layer sweep — HuBERT-base:**

| Layer | PER |
|---|---|
| 0 (CNN only) | 48.19% |
| 3 | 42.18% |
| 6 | 22.92% |
| 9 | 17.20% |
| **12 (last)** | **15.11%** |

**Summary:**

| Configuration | PER |
|---|---|
| wav2vec2-base, last layer | 38.31% |
| wav2vec2-base, best layer (6) | 21.67% |
| HuBERT-base, last layer | 15.09% |
| HuBERT-base, best layer (12) | 15.11% |

---

### 4. Hypotheses and Results

**HuBERT is way better than wav2vec2 (15% vs 38%).**
We expected this going in. HuBERT was trained to predict cluster labels - which is basically a simplified version of phoneme classification. wav2vec2 used contrastive learning, which is less directly related to what phonemes are. So HuBERT's features naturally work better for our task.

**For wav2vec2, the best layer is layer 6, not layer 12.**
This was a bit surprising. We thought the last layer would have the most useful features. But it turned out to be one of the worst (37.56%). The top layers of wav2vec2 get shaped by the contrastive loss in a way that makes them less useful for phoneme classification. The middle layers still carry clean acoustic information before the model abstracts it away.

**For HuBERT, deeper is always better.**
PER goes down steadily from CNN output (48%) all the way to layer 12 (15%). HuBERT's pre-training pushes every layer toward phoneme-like representations, so there is no point where quality drops off.

**CNN alone is bad for both models**
Without the transformer layers, the features only capture what's happening in a tiny local window. That's not enough to reliably tell phonemes apart - you need context from surrounding frames too.

---