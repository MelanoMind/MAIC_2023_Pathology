# https://tfimm.readthedocs.io/en/latest/content/efficientnet.html
# https://timm.fast.ai/#How-to-use

import os
import gc
from glob import glob
import warnings
import random
# import easydict
import copy
from collections import defaultdict
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import itertools
# from sklearn.metrics import roc_auc_scorez
from PIL import Image
import pickle
import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms.functional import to_pil_image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts , CyclicLR, ExponentialLR,StepLR, CosineAnnealingLR
import timm
import time
# import libtiff
from tqdm import tqdm
from Loss import myLoss

import datetime as dt
date = dt.datetime.now()
print(f'{date.month,date.day,date.hour}')

warnings.filterwarnings(action='ignore')

gc.collect()
torch.cuda.empty_cache()
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams['axes.unicode_minus'] = False


# 데이터 분포 확인.
class MILDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return len(self.patch_list)
    def __getitem__(self, idx):
        slide_idx = self.slide_idx[idx]
        img = self.patch_list[idx]
        
        transform = A.Compose([
            # A.Resize(512, 512),
            # A.Resize(224, 224),
            A.Rotate(),
            A.RandomCrop(224,224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.CLAHE(p=1,clip_limit=4.0),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # A.Normalize(mean=[0.8770,0.7850,0.8510],std=[0.0924,0.1323,0.0980]), # pkl값을 norm한 결과.
            ToTensorV2(),
        ])
        img = transform(image=img)["image"]
        
        label = self.label_list[slide_idx]
        return img, label

def load_dataset_from_pickle(file_name):
    with open(file_name, 'rb') as file:
        dataset = pickle.load(file)
    return dataset


def train_epoch(model, classifier,train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    y_probs = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device).float(), label.to(device).float()
        
        optimizer.zero_grad()
        y_prob = classifier(model(data))

        label = label.view(-1)
        
        y_prob = y_prob.view(-1)
        
        loss = criterion(y_prob, label)
        loss.backward()
        optimizer.step()

        y_probs +=y_prob.mean().item()
        total_loss += loss.item()
    for i in range(1):
        print(f"Prob: {(y_prob[i].item()):.4f}, Label: {label[i].item()}, Loss: {(loss.item()):.4f}",end=' ')
    print(f"total Loss: {(total_loss/len(train_loader)):.4f}")
    return total_loss / len(train_loader)

def validate_epoch(model,classifier,valid_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(valid_loader):
            data, label = data.to(device).float(), label.to(device).float()
            y_prob = classifier(model(data))
            
            label = label.view(-1)
            y_prob = y_prob.view(-1)
            loss = criterion(y_prob, label.view(-1)).item()
            total_loss += loss
        for i in range(1):
            print(f"Prob: {(y_prob[i].item()):.4f}, Label: {label[i].item()}, Loss: {(loss):.4f}")
    return total_loss / len(valid_loader)#, np.concatenate(all_probs), np.concatenate(all_labels)
    

from scheduler import CosineAnnealingWarmUpRestarts
if __name__ == '__main__':
    ##### HYPER PARAMETER
    
    SEED = 42
    DROP_RATE = 0.25

    EPOCH = 10
    LEARNING_RATE = 6e-6
    WEIGHT_DECAY = 0 #1e-6
    lr_max = 5e-5

    GPU_IDX = 'cuda'
    DATE = '0122'
    EXP_NAME = f'MAIC_{DATE}_{LEARNING_RATE}'
    # seed_everything(SEED)

    # train & valid split
    folder = '../re_pickle_train_patch'
    pkl_list = os.listdir(folder)
    pkl_list.sort()
    
    train_path = '../total_split_train_val/fold_0_train.csv'
    valid_path = '../total_split_train_val/fold_0_val.csv'

    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    
    train_list_0 = list(df_train['Slide_name'])[:49]
    train_list_1 = list(df_train['Slide_name'])[-50:-1]
    
    valid_list = list(df_valid['Slide_name'])[:10]+list(df_valid['Slide_name'])[-10:]

    
    # train_list_0 = list(df_train['Slide_name'])[:1]
    # train_list_1 = list(df_train['Slide_name'])[-2:-1]
    # valid_list = list(df_valid['Slide_name'])[:1]+list(df_valid['Slide_name'])[-1:]
    print(valid_list)
    random.shuffle(train_list_0)
    random.shuffle(train_list_1)
    random.shuffle(valid_list)

    model = timm.create_model('resnet18', pretrained=True, num_classes=1, drop_rate=DROP_RATE)
    model.fc = nn.BatchNorm1d(model.fc.in_features)
    # model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=512)
    # model.classifier = nn.Sequential(
        # nn.Linear(model.classifier.in_features, 512),
        # nn.BatchNorm1d(512),
    # )
    
    classifier = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    for param in model.parameters():
        param.requires_grad = True
        
    model = model.to(GPU_IDX)
    classifier = classifier.to(GPU_IDX)
    
    criterion = nn.CrossEntropyLoss() # 이거는 여러개 클래스 맞추는게 아니라 각 패치마다 하나씩 맞추는거이기때문에 이건 아님.
    criterion = nn.BCELoss() # binary cross entropy를 쓰거나 
    criterion = myLoss() # BCE에서 값을 조절한 이걸 쓰는게 좋아보임.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sigmoid = torch.nn.Sigmoid()
    
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=7, T_mult=1, eta_max=lr_max,  T_up=3, gamma=0.9)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    device= 'cuda'
    for epoch in range(EPOCH):
        time00 = time.time()
        print(f'#######  Epoch {epoch+1} #######') 
        random.shuffle(train_list_0)
        random.shuffle(train_list_1)
        print('Learning Rate :',optimizer.param_groups[0]['lr'])
        for number,(pkl,pkl2) in enumerate(zip(train_list_0,train_list_1)): # 두개의 길이가 같게 해야함..
            if number%2==0:

                PATH = folder + '/' + pkl +'_patch.pkl'
                PATH2 =folder + '/' + pkl2+'_patch.pkl'
                train_dataset_0 = load_dataset_from_pickle(PATH)
                train_dataset_1 = load_dataset_from_pickle(PATH2)
                # print(pkl,pkl2,end=' ')
            if number%2==1:
                PATH = folder + '/' + pkl +'_patch.pkl'
                PATH2 =folder + '/' + pkl2+'_patch.pkl'
                train_dataset_2 = load_dataset_from_pickle(PATH)
                train_dataset_3 = load_dataset_from_pickle(PATH2)
                
                train_data = torch.utils.data.ConcatDataset([train_dataset_0, train_dataset_1,train_dataset_2,train_dataset_3])
                # print(pkl,pkl2,len(train_data))
                max_batch = 77 ################################################################################
                batch_size = min(len(train_data), max_batch) if len(train_data)>max_batch else len(train_data)

                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,pin_memory=True)

                train_loss = train_epoch(model,classifier, train_loader, optimizer, criterion, device)
            torch.cuda.empty_cache() #PyTorch의 캐시 메모리 비우기
            gc.collect()   # 가비지 컬렉터 호출
            
        # print(f"{pkl} Training Loss: {train_loss} : len path {len(train_data)}")
        scheduler.step()
        print()

        # torch.save(model,f'./ENCODER_res50_ep{epoch}_2.pt')
            
        torch.cuda.empty_cache() #PyTorch의 캐시 메모리 비우기
        gc.collect()   # 가비지 컬렉터 호출

        max_batch = 77
        print()
        print('-- Validation --')
        total_valid_loss = 0
        y_true = []
        y_pred = []
        for pkl in valid_list:
            PATH = folder + '/' + pkl +'_patch.pkl'
            valid_data = load_dataset_from_pickle(PATH)
            if len(valid_data) == 0:
                continue
            batch_size = min(len(valid_data), max_batch) if len(valid_data)>max_batch else len(valid_data)
            valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True,pin_memory=True)
            valid_loss = validate_epoch(model, classifier,valid_loader, criterion, device) #valid_prob,valid_label
            total_valid_loss+=valid_loss
        #     y_true.append(valid_label[0])#.cpu().numpy())
        #     y_pred.append(valid_prob[0])#.cpu().numpy())
        # precision, recall, f1, auroc = evaluation(y_true,y_pred)
        
        print(f"Validation Loss: {total_valid_loss/len(valid_list)}")
        # print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUROC: {auroc}")
        print()
        now = time.time()
        print(time00-now)