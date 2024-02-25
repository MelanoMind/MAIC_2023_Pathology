import torch
import albumentations as A
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
from albumentations.pytorch.transforms import ToTensorV2
import torchstain

class MILDataset(Dataset): # 참고 https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019/blob/master/MIL_train.py    
    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, idx):
        slide_idx = self.slide_idx[idx]
        img = self.patch_list[idx]
        
        transform = A.Compose([
            A.Resize(224, 224),
            A.Rotate(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.ColorJitter(),
            # A.CLAHE(clip_limit=1.0, tile_grid_size=(8,8)),
            A.CLAHE(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        # Apply the transformations
        img = transform(image=img)["image"]
        
        label = self.label_list[slide_idx]
        return img, label
        
    
    def __len__(self):
        return len(self.patch_list)

    
    

def load_dataset_from_pickle(file_name):
    """
    .pkl 파일로부터 데이터셋을 로드합니다.

    :param file_name: 로드할 .pkl 파일의 이름
    :return: 파일에서 로드된 데이터셋
    """
    with open(file_name, 'rb') as file:
        dataset = pickle.load(file)
    return dataset