import os
import shutil

# base_dir = "kansaiben"
base_dir = "ese"
data_dir = "data"

data_dir = os.path.join(base_dir, data_dir)
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.mkdir(data_dir)

for i, fname in enumerate(os.listdir(base_dir)):
    if fname.startswith("_"):
        continue
    if os.path.splitext(fname)[1] != ".m4a":
        continue
    converted = f"data_{i:0>3}.wav"
    print(fname, "->", converted)
    os.system(f"ffmpeg -i '{os.path.join(base_dir, fname)}' -ac 1 '{os.path.join(data_dir, converted)}'")
