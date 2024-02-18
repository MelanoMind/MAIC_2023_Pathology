import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__() # openCV 최대 픽셀 오류 관리 코드
# os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**64)

import gc
from glob import glob
import warnings
import random
import easydict
import copy
from collections import defaultdict
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import itertools
# from sklearn.metrics import roc_auc_score
# from tqdm.notebook import tqdm
from tqdm import tqdm
import cv2  # import after setting OPENCV_IO_MAX_IMAGE_PIXELS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms.functional import to_pil_image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import timm

import pickle
import time
import io

from PIL import Image


warnings.filterwarnings(action='ignore')
gc.collect()
torch.cuda.empty_cache()
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams['axes.unicode_minus'] = False
# print('library import')

SEED = 42
PATCH_SIZE = (512, 512) # width, height
OVERLAP_RATIO = 0.1 # 0.5
TISSUE_AREA_RATIO = 0.5

GPU_IDX = 'cuda'
DATE = '240102'


def seed_everything(random_seed: int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])
    
## DEFINE Functions 

tissue_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

def is_inside_tissue(_mask):
    tissue_area = np.sum(_mask) // 255
    mask_area = np.prod(PATCH_SIZE)
    return (tissue_area / mask_area) > TISSUE_AREA_RATIO

def get_otsu_mask(_img):
    v = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)
    v = cv2.medianBlur(v, 15)
    _, timg_th = cv2.threshold(v, -1, 255, cv2.THRESH_OTSU)
    timg_th = cv2.morphologyEx(timg_th, cv2.MORPH_OPEN, tissue_kernel, iterations=2)
    timg_th = cv2.morphologyEx(timg_th, cv2.MORPH_CLOSE, tissue_kernel, iterations=2)
    timg_th = cv2.morphologyEx(timg_th, cv2.MORPH_OPEN, tissue_kernel, iterations=5)
    timg_th = cv2.morphologyEx(timg_th, cv2.MORPH_CLOSE, tissue_kernel, iterations=5)
    return ~timg_th

# def get_augmentation_for_image():
#     _transform = [
#         # A.Rotate(),
#         # A.HorizontalFlip(),
#         # A.VerticalFlip(),
#         # A.ColorJitter(),
#         # A.CLAHE(clip_limit=[5.0, 5.0], p=1),
#         # 새로운 augmentation 기법 필요
#     ]
#     return A.Compose(_transform)

def get_augmentation_for_mask():
    _transform = [
        # A.Rotate(),
        # A.HorizontalFlip(),
        # A.VerticalFlip(),
        # A.ColorJitter(),
        A.CLAHE(clip_limit=[5.0, 5.0], p=1),
    ]
    return A.Compose(_transform)

# def get_preprocessing_before():
#     _transform = [
#         # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 새로운 nomalize도입 필요.
#         # A.Normalize(mean=[0.93116294 0.89953984 0.92485626], std=[0.08036675 0.12898615 0.08541066]), # 직접 구한 새로운 nomalize
#     ]
#     return A.Compose(_transform)


def get_preprocessing_after():
    _transform = [
        A.Resize(PATCH_SIZE[1], PATCH_SIZE[0]),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 새로운 nomalize도입 필요.
        ToTensorV2()
    ]
    return A.Compose(_transform)

class MILDataset(Dataset): # 참고 https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019/blob/master/MIL_train.py
    def __init__(self, slide_list, label_list, augmentation=False, preprocessing=False):
        self.slide_list = slide_list # 받는 슬라이드 이미지 이름. ex) train_001.png
        self.label_list = label_list # 받는 슬라이드 이미지의 재발 여부 라벨.
        self.augmentation = get_augmentation_for_mask() if augmentation else None # 어그멘테이션 적용.
        # self.preprocessing_before = get_preprocessing_before() if preprocessing else None # 전처리 적용
        self.preprocessing_after = get_preprocessing_after() if preprocessing else None

        patch_list = [] 
        slide_idx = []
        for idx, slide_name in enumerate(tqdm(self.slide_list)):
            # slide_path = f'../dataset/train/{slide_name}.png'
            slide_path = f'/data/notebook/dataset/test_public/{slide_name}.png'
            
            try:
                img = cv2.imread(slide_path) # 하나의 슬라이드 이미지 오픈.
                if img is None:
                    raise Exception(f"Failed to read image: {slide_path}")
            except Exception as e:
                print(f"Error processing image: {e}")
                # print('Over size!!!',slide_path)
                print(' (img shape) ', img.shape)
                print(' (CV_IO_MAX_IMAGE_PIXELS) ', os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"])
                continue
                
            # print(' @ (org img) File Size:', convert_size(os.path.getsize(slide_path)), 'bytes')
            
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255 # shape : (12965, 27888, 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # shape : (12965, 27888, 3)
            
            ########################################################################################
            # img = self.preprocessing_before(image=img)['image']
            # img_uint8 = (img * 255).astype(np.uint8)
            # img = self.augmentation(image=img_uint8)['image']            
            ########################################################################################
            
            # tissue_mask = get_otsu_mask(self.augmentation(image=(img * 255).astype(np.uint8))['image'])
            tissue_mask = get_otsu_mask(self.augmentation(image=img)['image'])
            
            h_, w_, _ = img.shape
            
            finded_patch = []
            for i in range(0, h_, int(PATCH_SIZE[1] * (1-OVERLAP_RATIO))):
                if i+PATCH_SIZE[1] > h_:
                    continue
                for j in range(0, w_, int(PATCH_SIZE[0] * (1-OVERLAP_RATIO))):
                    if j+PATCH_SIZE[0] > w_:
                        continue

                    patch_mask = tissue_mask[i:i+PATCH_SIZE[1], j:j+PATCH_SIZE[0]]
                    if not is_inside_tissue(patch_mask):
                        del patch_mask
                        gc.collect()
                        continue
                    
                    patch = img[i:i+PATCH_SIZE[1], j:j+PATCH_SIZE[0], :]
                    finded_patch.append(patch)
                    
            patch_list.extend(finded_patch)
            slide_idx.extend([idx] * len(finded_patch))
            
        # mask_path = f'../img_pickle_mask_unit/dataset_{self.slide_list[0]}_mask.pkl'
        # mask_path = f'../re_pickle_train_mask/{self.slide_list[0]}_mask.pkl'
        mask_path = f'../re_pickle_test_mask/{self.slide_list[0]}_mask.pkl'
        save_dataset_as_pickle(tissue_mask, mask_path)
        
        # print(' @ (mask) File Size:', convert_size(os.path.getsize(mask_path)), 'bytes')
        
        self.patch_list = patch_list
        self.slide_idx = slide_idx

    def __getitem__(self, idx):
        slide_idx = self.slide_idx[idx]
        img = self.patch_list[idx]
        # print('##### slide_idx: ', slide_idx)
        if self.preprocessing_after:
            sample = self.preprocessing_after(image=img)
            img = sample['image']

        label = self.label_list[slide_idx]
        # print('##### label, len(self.slide_list), self.slide_list : ', label, len(self.slide_list), self.slide_list)
        return img, label # img, label, self.slide_list[0]
    
    def __len__(self):
        return len(self.patch_list)

        
def save_dataset_as_pickle(dataset, file_name):
    """
    데이터셋을 .pkl 파일 형식으로 저장합니다.
    """
    with open(file_name, 'wb') as f:
        pickle.dump(dataset, f)
        
if __name__ == "__main__":
    
    seed_everything(SEED)
    
    # df = pd.read_csv('../dataset/train_dataset.csv')
    df = pd.read_csv('./test_dataset.csv')
    train_list = df['Slide_name'].tolist()
    train_label = df['Recurrence'].tolist()
    
    # train_list = train_list[:20]+train_list[-20:]
    # train_label = train_label[:20]+train_label[-20:]
    # train_list = train_list[:2]
    # train_label = train_label[:2]
    
    print(f'[train_list] {type(train_list)}, {len(train_list)}') # , {train_list}')
    print(f'[train_label] {type(train_label)}, {len(train_label)}') # , {train_label}')
    
    # 전체 데이터 이미지 별로 pickle만들기
    for index, (slide_name, label) in enumerate(tqdm(zip(train_list, train_label))):
        # print(index, slide_name, label)
        
        # # 0 ~ 300
        # if index > 300:
        #     break
            
        train_data = MILDataset([slide_name], [label], augmentation=True, preprocessing=True)
        # print(f'    patchs : {type(train_data)}, {train_data}')
        print(f' ## [{index}] slide_name : {slide_name} | label : {label} | patchs : {len(train_data)}') 
        for patch in train_data:
            # print('  # ', len(patch)) # 2 (return img, label)
            # for i in range(len(patch)): # img, label
            #     print(patch[i])
            assert len(patch) == 2, f'[{slide_name}] len(patch) is {len(patch)}'
            assert patch[0].shape == (3, 512, 512), f'[{slide_name}] patch[0].shape is {patch[0].shape}'
            assert isinstance(patch[1], int), f'patch[1] type is {type(patch[1])}'
            break
            
        # patch_path = f'../img_pickle_patch_unit/dataset_{slide_name}_patch.pkl'
        # patch_path = f'../re_pickle_train_patch/{slide_name}_patch.pkl'
        patch_path = f'../re_pickle_test_patch/{slide_name}_patch.pkl'
        save_dataset_as_pickle(train_data, patch_path)
        
        # print(f"  # File Size: (org img) {convert_size(os.path.getsize(f'../dataset/train/{slide_name}.png')), 'bytes'} | (mask) {convert_size(os.path.getsize(f'../img_pickle_mask_unit/dataset_{slide_name}_mask.pkl')), 'bytes'} | (patch) {convert_size(os.path.getsize(patch_path)), 'bytes'}")
        # print(f"  # File Size: (org img) {convert_size(os.path.getsize(f'../dataset/train/{slide_name}.png')), 'bytes'} | (mask) {convert_size(os.path.getsize(f'../re_pickle_train_mask/{slide_name}_mask.pkl')), 'bytes'} | (patch) {convert_size(os.path.getsize(patch_path)), 'bytes'}")
        print(f"  # File Size: (org img) {convert_size(os.path.getsize(f'/data/notebook/dataset/test_public/{slide_name}.png')), 'bytes'} | (mask) {convert_size(os.path.getsize(f'../re_pickle_test_mask/{slide_name}_mask.pkl')), 'bytes'} | (patch) {convert_size(os.path.getsize(patch_path)), 'bytes'}")
        
        
