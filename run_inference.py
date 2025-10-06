import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# -------------------------------
# 1. Load model + processor from HF
# -------------------------------
repo_name = "ganga4364/Garchen_rinpoche_whisper_generic_on_wylie_checkpoint-4000"  # your HF repo

processor = WhisperProcessor.from_pretrained(repo_name, language="Tibetan", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(repo_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -------------------------------
# 2. Load audio file
# -------------------------------
audio_path = "/workspace/data/wav_16k/STT_GR_0001_0002_17400_to_21800.wav"
waveform, sr = torchaudio.load(audio_path)

# Resample if needed
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    sr = 16000

# -------------------------------
# 3. Preprocess
# -------------------------------
inputs = processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt").to(device)

# -------------------------------
# 4. Run inference
# -------------------------------
with torch.no_grad():
    pred_ids = model.generate(
        inputs["input_features"],
        num_beams=4,
        max_length=225
    )

# Decode prediction
text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
print("Transcription:", text)
