from torch.utils.data import Dataset
import os
import random
from PIL import Image

class ResNetDataset(Dataset):
    def __init__(self, data_path, task="train", val_ratio=0.2, transform=None, seed=123):
        '''
            data_path : "data/train_data"
            task : "train"
            val_ratio : 비율 알아서 조정하기
            transform : 적용할 Aug
            seed : 고정하기
        '''
        self.data_path = data_path
        self.task = task
        self.val_ratio = val_ratio
        self.transform = transform
        self.seed = seed

        self.image_path_list = []
        self.label_list = []
        self.idx_to_class = {}

        extension = (".jpg", ".jpeg", ".png")

        # seed 고정
        random.seed(self.seed)

        # classes_name : 폴더명들
        classes_name = sorted(os.listdir(data_path))   # Abra, ...
        for idx, class_name in enumerate(classes_name):
            self.idx_to_class[idx] = class_name     # 0: Abra, ...
            class_path = os.path.join(data_path, class_name)    # data/train_data/Abra, ...

            images = os.listdir(class_path)
            # extension에 포함되는 확장자만 필터링
            images = [image for image in images if image.endswith(extension)]

            random.shuffle(images)
            # split index
            train_num = int(len(images) * (1-val_ratio))

            if task == "train":
                images = images[:train_num]
            elif task == "val":
                images = images[train_num:]

            for image in images:
                image_path = os.path.join(class_path, image)
                self.image_path_list.append(image_path)
                self.label_list.append(idx)

    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        img_path = self.image_path_list[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.label_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label