# Whisper Tibetan Model Evaluation Summary

## 0. Experiment Overview

This experiment evaluates and compares two fine-tuned **Whisper-small** models for **Tibetan Automatic Speech Recognition (ASR)**.  
The goal is to assess the effect of **using native Tibetan script tokens** (via a custom tokenizer) versus **Wylie transliteration** (using the default Whisper tokenizer).

### Model A — *Wylie-based Whisper model*
- **Model name:** `whisper-small-tibetan-wylie-checkpoint-4000`
- **Base model:** `openai/whisper-small`
- **Tokenizer:** Default Whisper tokenizer
- **Training data:** 5.6 hours of Tibetan audio paired with Wylie transliterations  
- **Training steps:** 4000  
- **Batch size:** 16  
- **Gradient accumulation:** 1  
- **Input:** Audio + Wylie transcript  
- **Evaluation:** After transcription, predicted Wylie text is converted back to Tibetan script and compared against the **Tibetan ground-truth** in the benchmark dataset.

### Model B — *Tibetan-script Whisper model with added tokens*
- **Model name:** `whisper-small-latin-added-tibetan-checkpoint-4000`
- **Base model:** `openai/whisper-small`
- **Tokenizer:** Custom tokenizer extending the default Whisper tokenizer with **added Tibetan script tokens**
- **Training data:** Same 5.6 hours of Tibetan audio
- **Training steps:** 4000  
- **Batch size:** 16  
- **Gradient accumulation:** 1  
- **Input:** Audio + Tibetan script transcript (tokenized using added Tibetan tokens)
- **Evaluation:** Direct comparison between the model’s Tibetan output and **Tibetan ground-truth**.

**Objective:**  
To determine whether directly training Whisper with a **native Tibetan tokenizer** improves transcription accuracy, token efficiency, and inference performance compared to a **Wylie transliteration-based approach**.

All experiments were performed on a **NVIDIA RTX 4090 (24GB VRAM)** GPU using 30-second audio samples (the Whisper maximum input window).

---

## 1. Word and Sentence Error Rates (WER / SER)

| model                                              | micro_wer | macro_wer | micro_ser | macro_ser | substitutions | insertions | deletions |
|----------------------------------------------------|-----------|-----------|-----------|-----------|----------------|-------------|------------|
| whisper-small-latin-added-tibetan-checkpoint-4000  | 0.607723  | 0.587186  | 0.565648  | 0.565680  | 7289           | 543         | 1478       |
| whisper-small-tibetan-wylie-checkpoint-4000_to_tibetan | 0.675397  | 0.712424  | 0.616562  | 0.656280  | 6741           | 1561        | 1846       |

**Summary:**  
- The **Latin-added Tibetan model** outperforms the **Wylie→Tibetan model** in both WER and SER across micro and macro averages.  
- The difference is especially clear in insertion and substitution error counts.

---

## 2. Character Error Rate (CER)

| model                                              | cer_mean  |
|----------------------------------------------------|-----------|
| whisper-small-latin-added-tibetan-checkpoint-4000  | 0.298808  |
| whisper-small-tibetan-wylie-checkpoint-4000_to_tibetan | 0.384848  |

**Summary:**  
- The **Tibetan-script model** achieves a **lower CER (~0.30)** compared to the **Wylie model (~0.38)**, showing higher character-level accuracy.  
- This suggests better modeling of native script structure and spelling consistency.

---

## 3. Tokenization Length Comparison  
(for ~30-second audio transcripts)

| Tokenizer Type                            | Transcript Language | Token Length | Notes |
|-------------------------------------------|----------------------|--------------|-------|
| Whisper Default Tokenizer                 | Wylie               | 271          | Within model limit (≤ 1024) |
| Whisper + Added Tibetan Tokens Tokenizer  | Tibetan             | 311          | Within model limit (≤ 1024) |
| Whisper Default Tokenizer                 | Tibetan             | 1551         | ❌ Exceeds model max input limit (1024) |

**Summary:**  
- The **default Whisper tokenizer** is inefficient on Tibetan script, producing over 1500 tokens for 30-second text — exceeding the model limit.  
- The **added-tokens tokenizer** keeps Tibetan input compact and fully processable.  
- This makes direct Tibetan script training feasible without transliteration.

---

## 4. Inference Time Benchmark  
(on ~30-second audios, Whisper max input, NVIDIA RTX 4090 24GB)

| Model                                              | GPU (VRAM)      | Avg Inference Time (sec) | Notes |
|----------------------------------------------------|------------------|---------------------------|-------|
| whisper-small-latin-added-tibetan-checkpoint-4000  | RTX 4090 (24GB) | 1.3                       | Stable average; small variation per run |
| whisper-small-tibetan-wylie-checkpoint-4000_to_tibetan | RTX 4090 (24GB) | 1.3                       | No significant difference between models |

**Summary:**  
- Both models achieve **~1.3 seconds inference time** for 30-second audio clips.  
- There is **no measurable difference** in inference latency between Wylie and Tibetan-tokenized models.  
- GPU utilization remains consistent across runs.

---

## 5. Tokenizer Vocabulary Size

| Tokenizer Type                            | Vocabulary Size | Notes |
|-------------------------------------------|------------------|-------|
| Whisper Default Tokenizer                 | 51,865           | Original Whisper vocabulary |
| Whisper + Added Tibetan Tokens Tokenizer  | 53,014           | Includes 1,149 additional Tibetan script tokens |

**Summary:**  
- The **custom Tibetan tokenizer** expands the base Whisper vocabulary by ~2.2%.  
- These extra tokens provide **direct Tibetan script coverage**, reducing token fragmentation and improving alignment with real-world Tibetan text.

---

## 6. Overall Observations

| Category              | Best Performing Model | Key Advantage |
|------------------------|----------------------|----------------|
| WER / SER             | Latin-added Tibetan   | Lower word and sentence error rates |
| CER                   | Latin-added Tibetan   | More accurate at character level |
| Tokenization Efficiency | Latin-added Tibetan   | Efficient native-script encoding |
| Vocabulary Coverage   | Latin-added Tibetan   | Broader support for Tibetan characters |
| Inference Speed       | Equal (both)          | No significant runtime difference |

---

## ✅ Conclusion

The **`whisper-small-latin-added-tibetan-checkpoint-4000`** model consistently outperforms the **`whisper-small-tibetan-wylie-checkpoint-4000_to_tibetan`** model across accuracy metrics (WER, SER, CER) while maintaining identical inference performance.

By leveraging a **custom tokenizer with added Tibetan tokens**, this model:
- Enables **direct training on Tibetan script** (no transliteration needed),
- Avoids **token overflow issues** (keeps sequences ≤1024 tokens),
- Achieves **higher accuracy** on native Tibetan benchmarks,
- And maintains **comparable inference speed**.

Overall, direct Tibetan tokenization is a **robust and scalable improvement** for Tibetan ASR tasks using Whisper architectures.

---

*Prepared for: OpenPecha / MonlamAI — Tibetan Speech-to-Text Evaluation Report*  
*Hardware: NVIDIA RTX 4090 (24 GB VRAM) · Training Data: 5.6 hours · Audio Length: ~30 seconds*
