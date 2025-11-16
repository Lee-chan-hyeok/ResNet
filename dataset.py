'''
    model: ResNet
    task: Image Classification
'''

from torch.utils.data import Dataset
import os

from PIL import Image

class ResNetDataset(Dataset):
    def __init__(self, data_path, transform):
        '''
            data_path: "data\images"
        '''
        self.data_path = data_path
        self.transform = transform

        self.image_path_list = []
        self.label_list = []
        self.idx_to_class = {}

        extension = (".jpg", ".jpeg", ".png")

        classes_name = sorted(os.listdir(data_path))   # apple, mouse, ...
        for idx, class_name in enumerate(classes_name):
            self.idx_to_class[idx] = class_name     # 0: apple, 1: mouse, ...
            class_path = os.path.join(data_path, class_name)    # data/images/apple

            images = os.listdir(class_path)
            images = [image for image in images if image.lower().endswith(extension)]

            for image in images:
                image_path = os.path.join(class_path, image)
                self.image_path_list.append(image_path)
                self.label_list.append(idx)

    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_path_list[idx])
        label = self.label_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label