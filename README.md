# Audio ML Homeworks
This is main branch for Audio ML course Homeworks, it will be just clean with readme, for the homeworks check branches with related names: HW1, HW2, etc.

**Authors:**\
[Roman Pavlosiuk](https://github.com/gllekkoff)\
[Iryna Denysova](https://github.com/Shnapa)

## Homework 1: Phoneme Recognition + Audio Filters

# Automatic Speech Translation (AST) with Whisper

## Overview

This project implements a system for Automatic Speech Translation (AST).

The goal is to convert:
- English speech (audio) → Ukrainian text (translation)

This task is complex because it combines:
- Speech recognition (audio → text)
- Machine translation (text → another language)

---

## Dataset

We use the Google FLEURS dataset, which provides:
- audio samples
- original transcriptions
- translated text

![Dataset Info](./images/datainfo.png)

### What this shows
Each sample contains:
- `audio` — input speech  
- `transcription` — original English text  
- `translation` — Ukrainian translation  

### Important
The dataset is very small:
- ~130 training samples  
- ~60 validation samples  
- ~60 test samples  

This strongly limits model performance.

---

## Dataset Split

![Dataset Split](./images/trainsplit.png)

The split is correct, but total data is too small for effective training.

---

## Dataloader

![Dataloader](./images/dataload.png)

- Pipeline works correctly  
- But number of batches is small, so learning is limited  

---

## Audio Representation

![Audio Example](./images/audioex.png)

### What this shows
- Waveform (signal over time)  
- Frequency spectrum  

Audio is converted into features before model processing.  
This part works correctly and does not cause issues.

---

## Model

We use the pretrained Whisper model:
- ~37.8M parameters  
- multilingual support  

The model is large, but the dataset is small, so adaptation is weak.

---

## Training Process

![Training Logs](./images/train.png)

### Critical observation
Training: 0/?
Validation: 0/?

This indicates:
- training loop is limited or not fully effective  
- model did not properly iterate over the dataset  

The pipeline runs, but real learning is minimal.

---

## Baseline Predictions

![Baseline Predictions](./baseline.png)

Predictions before training (zero-shot).
- outputs are often incorrect or meaningless  
- model is not adapted to this dataset  

---

## Evaluation (COMET)

![COMET Results](./ast_comet.png)

- small improvement after training  
- model learned slightly, but not enough  

---

![Evaluation Table](./result.png)

- evaluation is correct (60 samples)  
- improvement is confirmed  
- results are still weak  

---

## Sample Predictions

![Predictions](./text.png)

- `src` — English input  
- `ref` — correct Ukrainian translation  
- `hyp` — model prediction  

So
- reference translations are correct  
- predictions are often:
  - incorrect  
  - unrelated  
  - grammatically broken  

The model failed to learn proper translation.

---

## Why Results Are Weak

### 1. Small dataset
- only ~130 training samples  
- not enough for deep learning  

### 2. Few training epochs
- only 3 epochs  
- insufficient training time  

### 3. Large model vs small data
- 37M parameters  
- cannot generalize from small dataset  

### 4. Training inefficiency
- logs show weak iteration (`0/?`)  

### 5. Task complexity
- AST combines speech recognition and translation  
- harder than single-task problems  

---

## Conclusion

The project successfully builds a full AST pipeline:
- data loading  
- preprocessing  
- model training  
- evaluation  

However:
- training is limited  
- improvement is small  
- predictions are mostly incorrect  

---

## Key Takeaway
The pipeline works correctly, but due to the very small dataset and limited training process, the model could not significantly improve translation quality.
