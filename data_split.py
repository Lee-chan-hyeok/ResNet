import os
import random
import shutil

data_path = "data"
test_set_dir = "test_data"
test_ratio = 0.2

os.makedirs(test_set_dir, exist_ok=True)

class_folder = os.listdir(data_path)
# print(f"class_folder: {class_folder}")    # Ok

for cls in class_folder:
    # cls: Abra
    cls_path = os.path.join(data_path, cls) # "data/Abra"
    if not os.path.isdir(cls_path):
        print("cls_path 안에 데이터 없음")
        continue

    img_list = os.listdir(cls_path)     # img 파일명들이 list로 담김
    num_img = len(img_list)             # cls_path 안에 들어있는 이미지 파일들 개수

    indices = list(range(num_img))      # 0~num 까지 index화 시킴
    random.shuffle(indices)             # index를 shuffle

    num_test = int(num_img * test_ratio)    # 파일들 중 20%에 해당하는 양
    test_indices = indices[:num_test]       # shuffle된 index에서 20% 양만큼만 선택

    test_cls_path = os.path.join(test_set_dir, cls)     # "test_data/Abra"
    os.makedirs(test_cls_path, exist_ok=True)

    for idx in test_indices:
        img_name = img_list[idx]
        src = os.path.join(cls_path, img_name)
        dst = os.path.join(test_cls_path, img_name)

        shutil.move(src, dst)

    print(f"{cls}: # of img: {num_img}, # of test set: {num_test}")