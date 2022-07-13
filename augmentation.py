from audiomentations import (
    Compose,
    # AddBackgroundNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    # AddShortNoises,
    AirAbsorption,
    TimeStretch,
    PitchShift
)
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
import wave

# 水増し設定
augment = Compose([
    AddGaussianNoise(p=0.4),
    AddGaussianSNR(p=0.7),
    AirAbsorption(p=0.6),
    TimeStretch(p=0.4),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.7),
])

# base_dir = "kansaiben"
base_dir = "ese"
data_dir = "data"
augmented_dir = "augmented"

data_dir = os.path.join(base_dir, data_dir)
augmented_dir = os.path.join(base_dir, augmented_dir)
if os.path.exists(augmented_dir):
    shutil.rmtree(augmented_dir)
os.mkdir(augmented_dir)

n = 50
for i, fname in enumerate(os.listdir(data_dir)):
    print(fname)
    with wave.open(os.path.join(data_dir, fname)) as w:
        channel, sample_width, sample_rate, frames, _, _ = w.getparams()
        print(channel, sample_width, sample_rate, frames)
        data = np.frombuffer(w.readframes(frames), dtype="int16").astype(np.float32)
    print(i, data, len(data))

    for j in range(n):
        augmented = augment(samples=data, sample_rate=sample_rate)
        print(" ", j, augmented, len(augmented))
        with wave.open(os.path.join(augmented_dir, f"augmented{i * n + j:0>4}.wav"), "w") as w:
            w.setnchannels(channel)
            w.setsampwidth(sample_width)
            w.setframerate(sample_rate)
            w.setnframes(frames)
            w.writeframes(augmented.astype("int16").tobytes())
