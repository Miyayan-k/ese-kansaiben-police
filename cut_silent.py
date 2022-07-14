import os
import shutil
import numpy as np
import wave


# augmentationの直前に実施。

# テストしたい場合
base_dir = "test_data"
data_dir = "test_data/cut_silent"

# 関西弁実施時
# base_dir = "kansaiben/data"
# data_dir = "kansaiben/cut_silent"

# エセ関西弁実施時
# base_dir = "ese/data"
# data_dir = "ese/cut_silent"

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.mkdir(data_dir)

files = os.listdir(base_dir)
wav_files = [f for f in files if os.path.isfile(os.path.join(base_dir, f))]

for fname in wav_files:
    if fname.startswith('.'):
        continue
    with wave.open(os.path.join(base_dir, fname)) as w:
        channel, sample_width, sample_rate, frames, comptype, compname = w.getparams()
        print(channel, sample_width, sample_rate, frames, comptype, compname)
        data = np.frombuffer(w.readframes(frames), dtype="int16").astype(np.float32)
        t = np.arange(0, len(data))/sample_rate
        # 平均的に考慮して、静寂を2000以下に設定。あまりに静かなデータはすべてカットされる可能性に留意
        thres = 2000
        amp = np.abs(data)
        b = amp > thres
        # 声の認知開始ポイント
        start_voice = b.tolist().index(True)
        # 認知後の配列を取得
        voice = data[start_voice:]
    with wave.open(os.path.join(data_dir, fname), "wb") as w:
        w.setnchannels(channel)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(voice.astype("int16").tobytes())
