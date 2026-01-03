import os
from PIL import Image

data_path = "data/train_data"
valid_extension = (".png", ".jpeg", ".bmp", ".tiff", ".webp")

for root, dirs, files in os.walk(data_path):
    for file in files:
        # print(f"file: {file}")
        if file.lower().endswith(valid_extension):
            src_path = os.path.join(root, file)
            dst_path = os.path.splitext(src_path)[0] + ".jpg"
            # print(f"dst_path: {dst_path}")

            try:
                img = Image.open(src_path).convert("RGB")
                img.save(dst_path, "JPEG", quality=95, subsampling=0)

                print(f"✔ converted: {src_path} -> {dst_path}")
            
            except Exception as e:
                print(f"✖ failed: {src_path} ({e})")
