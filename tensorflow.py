import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
import audiomentations

# シード値固定
# seed = 0
# tf.random.set_seed(seed)
# np.random.seed(seed)

data_dir = "drive/MyDrive/エセ関西弁警察/【関西弁】なんでやねん/augmented"
data_dir_ese = "drive/MyDrive/エセ関西弁警察/【エセ関西弁】なんでやねん/augmented"
filenames = tf.random.shuffle(os.listdir(data_dir) + ["ese_" + f for f in os.listdir(data_dir_ese)])
data_count = len(filenames)
labels = np.array(["kansai", "ese"])
print(filenames)
print(data_count)

val_count = data_count // 10
test_count = data_count // 10
train_count = data_count - val_count - test_count
train_files = filenames[:train_count]
val_files = filenames[train_count: train_count + val_count]
test_files = filenames[-test_count:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

"""# 音声ファイルとそのラベルを読む
次にデータセットを前処理し、波形と対応するラベルのデコードされたテンソルを作成します。
ここで注意する点は以下です。

* 各WAVファイルには、1秒あたりのサンプル数が設定された時系列データが含まれています。
* 各サンプルは、その特定の時間における音声信号の振幅を表します。
* ミニ音声コマンドデータセットのWAVファイルのような16ビットシステムでは、振幅値の範囲は-32,768〜32,767です。
* このデータセットのサンプルレートは16kHzです。


tf.audio.decode_wavによって返されるテンソルの形状は[samples, channels]です。ここで、 channelsはモノラルの場合は1 、ステレオの場合は2です。ミニ音声コマンドデータセットには、モノラル録音のみが含まれています。
"""

# 確認用
test_file = tf.io.read_file(os.path.join(data_dir, "augmented0000.wav"))
test_audio, _ = tf.audio.decode_wav(contents=test_file)
test_audio.shape

"""次に、データセットの生のWAV音声ファイルを音声テンソルに前処理する関数を定義します。"""

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)

"""各ファイルの親ディレクトリを使用してラベルを作成する関数を定義します。

ファイルパスをtf.RaggedTensorに分割します（不規則な次元のテンソル-スライスの長さが異なる場合があります）。
"""

def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  label = "ese" if tf.strings.regex_full_match(parts[-1], "^ese_.*") else "kansai"
  return label

"""すべてをまとめる別のヘルパー関数get_waveform_and_labelを定義します。

* 引数はWAV音声ファイル名です。
* 返り値は、教師あり学習の準備ができている音声テンソルとラベルテンソルを含むtupleです。
"""

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  dir_name = data_dir if label == "kansai" else data_dir_ese
  file_name = tf.strings.split(file_path, "_")[-1]
  binary_path = tf.strings.join([dir_name, file_name], separator="/")
  audio_binary = tf.io.read_file(binary_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

"""音声とラベルのペアを抽出するためのトレーニングセットを作成します。

* 前に定義したget_waveform_and_labelを使用して、 Dataset.from_tensor_slicesとDataset.mapを使用してtf.data.Datasetを作成します。


同様に、検証セットとテストセットも作成します。
"""

AUTOTUNE = tf.data.AUTOTUNE

files_ds = tf.data.Dataset.from_tensor_slices(train_files)

waveform_ds = files_ds.map(
    map_func=get_waveform_and_label,
    num_parallel_calls=AUTOTUNE)

"""いくつかの音声波形をプロットしてみます。"""

# 確認用
rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

for i, (audio, label) in enumerate(waveform_ds.take(n)):
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  ax.plot(audio.numpy())
  ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
  label = label.numpy().decode('utf-8')
  ax.set_title(label)

plt.show()

"""# 波形をスペクトログラムに変換する

データセット内の波形は、時間領域で表現されます。次にスペクトログラム（周波数の時間変化を示す2次元画像）へ変換するための短時間フーリエ変換(STFT)により、時間領域信号から時間周波数領域信号へ波形を変換します。スペクトログラムの画像をニューラルネットワークに送り込み、モデルを学習させます。

フーリエ変換（ tf.signal.fft ）は信号をその成分周波数に変換しますが、時間情報は全て失われます。これに対しSTFT（ tf.signal.stft ）は、信号を時間のウィンドウに分割し、各ウィンドウでフーリエ変換を実行して時間情報を保持します。そして標準的な畳み込みを実行できる2Dテンソルを返します。

波形をスペクトログラムに変換するためのユーティリティ関数を作成します。

* 波形は同じ長さである必要があるため、スペクトログラムに変換すると、結果の寸法は同じになります。これは、1秒より短いオーディオクリップをゼロパディングするだけで実行できます（ tf.zerosを使用 ）。
* tf.signal.stftを呼び出すときは、生成されたスペクトログラム「画像」がほぼ正方形になるように、 frame_lengthパラメーターとframe_stepパラメーターを選択します。 
* STFTは、大きさと位相を表す複素数の配列を生成します。ただし今回は、 tf.absの出力にtf.signal.stftを適用することで導出できる大きさのみを使用します。
"""

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 48000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [48000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

"""次に、データの調査を開始します。 1つの例のテンソル化された波形と対応するスペクトログラムの形状を出力し、オリジナルの音声データを再生します。"""

# 確認用
for waveform, label in waveform_ds.take(1):
  label = label.numpy().decode('utf-8')
  spectrogram = get_spectrogram(waveform)

print('Label:', label)
print('Waveform shape:', waveform.shape)
print('Spectrogram shape:', spectrogram.shape)
print('Audio playback')
display.display(display.Audio(waveform, rate=48000))

"""次に、スペクトログラムを表示するための関数を定義します。"""

# 確認用
def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

"""時間の経過に伴う例の波形と対応するスペクトログラム（時間の経過に伴う周波数）をプロットします。"""

# 確認用
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 48000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()

"""【メモ】
ここまで坂本さん担当

次に、波形データセットをスペクトログラムとそれに対応するラベルに整数IDとして変換する関数を定義します。
"""

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == labels)
  return spectrogram, label_id

"""【ここ以降】
宮谷担当
"""

spectrogram_ds = waveform_ds.map(
  map_func=get_spectrogram_and_label_id,
  num_parallel_calls=AUTOTUNE)

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  plot_spectrogram(spectrogram.numpy(), ax)
  ax.set_title(labels[label_id.numpy()])
  ax.axis('off')

plt.show()

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(labels)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=labels,
            yticklabels=labels,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# Augmentation
import wave
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

# augment = Compose([
#     AddGaussianNoise(p=0.4),
#     AddGaussianSNR(p=0.7),
#     AirAbsorption(p=0.6),
#     TimeStretch(p=0.4),
#     PitchShift(min_semitones=-2, max_semitones=2, p=0.7),
# ])
#
# def augmentation(data, n=10):
#   data = np.frombuffer(data, dtype="int16").astype(np.float32)
#   augmented = []
#   for _ in range(n):
#     augmented.append(augment(samples=data, sample_rate=48000).astype("int16").tobytes())
#   return augmented

# テスト用
print("please upload .wav file:")
uploaded = files.upload()
file = list(uploaded.keys())[0]
try:
  # with wave.open(uploaded[file]) as w:
  #   data = w.readframes(w.getnframes())
  # ds = augmentation(data)
  # print(ds)
  waveform = decode_audio(uploaded[file])
  spec = get_spectrogram(waveform)
  pred = model(np.array([spec]))
  result = tf.nn.softmax(pred[0])
  kansaiben_rate = result[0] * 100
  ese_rate = result[1] * 100

  print(f"関西弁率: {kansaiben_rate:3.1f}%")
  print(f"エセ関西弁率: {ese_rate:3.1f}%")
  if kansaiben_rate >= 50.0:
    print("エセ関西弁警察「ホンマごっつええ響きやわ〜！！」")
  else:
    print("エセ関西弁警察「あんたのそれ、エセ関西弁やで」")
finally:
  os.remove(file)

"""pythonのスクリプト単体では並列処理を受け付けないため、同時に5人とかがリクエスト来ると5番目の人がレスポンス待ちに時間かかる可能性あり、
Herokuでも良いが、Lambdaとかでリクエスト分散させる手段でデプロイした方が良さそう。
"""

# モデルの保存
model.save("model")