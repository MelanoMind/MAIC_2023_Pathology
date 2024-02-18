import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__() # openCV 최대 픽셀 오류 관리 코드
# os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**64)

import numpy as np
import pandas as pd
import time
import torch
from copy import deepcopy
import torch.nn as nn
# from dataloader import *
from torch.utils.data import Dataset, DataLoader, RandomSampler
import argparse, os
# from modules import attmil,clam,mhim,dsmil,transmil,mean_max
from modules import transmil, attmil, dsmil, mhim
from torch.nn.functional import one_hot
from torch.cuda.amp import GradScaler
from contextlib import suppress
import time

from timm.utils import AverageMeter, dispatch_clip_grad
from timm.models import  model_parameters
from collections import OrderedDict

from utils import *

from sklearn.metrics import roc_auc_score
# from tqdm.notebook import tqdm
from tqdm import tqdm
# import cv2
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from torch.optim.lr_scheduler import _LRScheduler
# from torchvision.transforms.functional import to_pil_image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import torchstain
import warnings
import pickle
import gc
import random
from datetime import datetime

warnings.filterwarnings(action='ignore')
gc.collect()
torch.cuda.empty_cache()


def seed_everything(random_seed: int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    

class MILDataset(Dataset): # 참고 https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019/blob/master/MIL_train.py
    def __init__(self, slide_list, label_list, augmentation=True, preprocessing=True):        
        # self.slide_list = slide_list # 받는 슬라이드 이미지 이름. ex) train_001.png
        # self.label_list = label_list # 받는 슬라이드 이미지의 재발 여부 라벨.
        # self.augmentation = get_augmentation_for_mask() if augmentation else None # 어그멘테이션 적용.
        # # self.preprocessing_before = get_preprocessing_before() if preprocessing else None # 전처리 적용
        # self.preprocessing_after = get_preprocessing_after() if preprocessing else None
        pass
    
    def __getitem__(self, idx):
        slide_idx = self.slide_idx[idx] # 0
        img = self.patch_list[idx] # torch.Size([3, 512, 512])
        # print(f'{slide_idx} {idx}' )
        # print(f'(before aug) {img.shape} {type(img)} {img}')
        
        transform = A.Compose([
            A.Rotate(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.ColorJitter(),
            # A.CLAHE(p = 0.5),
            A.Resize(512, 512), # (PATCH_SIZE[1], PATCH_SIZE[0]), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imageNet
            # A.Normalize(mean=[0.8770, 0.7850, 0.8510],std=[0.0924, 0.1323, 0.0980]), # maic pkl
            ToTensorV2()
        ])
        img = transform(image=img)["image"]
        # print(f'(after aug) {img.shape} {type(img)} {img}')
        
        label = self.label_list[slide_idx]
        # print(' # ', img.shape, label)
            
        return img, label
    
    def __len__(self):
        return len(self.patch_list)


def load_dataset_from_pickle(file_name):
    with open(file_name, 'rb') as file:
        dataset = pickle.load(file)
    return dataset


import logging
def on_load_checkpoint(self, checkpoint: dict) -> None:
    # https://cchhoo407.tistory.com/37 [전자둥이의 끄적끄적:티스토리]

    state_dict = checkpoint
    model_state_dict = self.state_dict()
    is_changed = False
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                logging.info(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[k].shape}, "
                            f"loaded shape: {state_dict[k].shape}")
                state_dict[k] = model_state_dict[k]
                is_changed = True
        else:
            logging.info(f"Dropping parameter {k}")
            is_changed = True

    if is_changed:
        checkpoint.pop("optimizer_states", None)


def main(args):
    
    # set seed
    seed_torch(args.seed)
    
    # make patch datasets
    df_train = pd.read_csv(os.path.join(args.fold_root, f'fold_{args.fold_num}_train.csv')) # f'../total_split_train_val/fold_3_train.csv'
    df_val = pd.read_csv(os.path.join(args.fold_root, f'fold_{args.fold_num}_val.csv')) # f'../total_split_train_val/fold_3_val.csv'
    df_test = pd.read_csv('../dataset/test_public_dataset.csv')
    if not args.no_log:
        print('Dataset: ' + args.datasets)
    print('- train df path : ', os.path.join(args.fold_root, f'fold_{args.fold_num}_train.csv'))
    print('- val df path : ', os.path.join(args.fold_root, f'fold_{args.fold_num}_val.csv'))
    print('- test df path : ', '../dataset/test_public_dataset.csv')
    
    # print()
    # print(f'[making train/val/test patch dataset...]')

    # if args.cv_fold > 1:
    #     train_p, train_l, test_p, test_l, val_p, val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)

    acs, pre, rec, fs, auc, te_auc, te_fs = [],[],[],[],[],[],[]
    ckc_metric = [acs, pre, rec, fs, auc, te_auc, te_fs]

    # resume
    if args.auto_resume and not args.no_log:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        # args.fold_start = ckp['k']
        if len(ckp['ckc_metric']) == 6:
            acs, pre, rec, fs, auc, te_auc = ckp['ckc_metric']
        elif len(ckp['ckc_metric']) == 7:
            acs, pre, rec, fs, auc, te_auc, te_fs = ckp['ckc_metric']
        else:
            acs, pre, rec,fs,auc = ckp['ckc_metric']
    print(' **[ckc_metric]** ', ckc_metric)

    # for k in range(args.fold_start, args.cv_fold):
    #     if not args.no_log:
    #         print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
    #     ckc_metric = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l)
    
    # ckc_metric = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l)
    ckc_metric = fold_metric(args, ckc_metric, df_train, df_val, df_test) # train_set, val_set, test_set
    print(' **ckc_metric : [acs, pre, rec, fs, auc, te_auc, te_fs]**', ckc_metric) # [acs, pre, rec, fs, auc, te_auc, te_fs]

    if not args.no_log:
        print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))

        
# def one_fold(args, k, ckc_metric, train_p, train_l, test_p, test_l, val_p, val_l):
def fold_metric(args, ckc_metric, df_train, df_val, df_test): # train_set, val_set, test_set
    
    ########################################################################################
    # --->initiation
    seed_torch(args.seed)
    loss_scaler = GradScaler() if args.amp else None
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    acs, pre, rec, fs, auc, te_auc, te_fs = ckc_metric

    mm_sche = None
    if args.teacher_init == 'none':
        _teacher_init =args.teacher_init
    elif not args.teacher_init.endswith('.pt'):
        _str = 'fold_{fold}_model_best_auc.pt'.format(fold=args.fold_num) # fold=k
        # _str = 'fold_{fold}_model_best_auc.pt'.format(fold=0) # fold=k
        _teacher_init = os.path.join(args.teacher_init,_str)
    else:
        _teacher_init =args.teacher_init
    print('- (teacher_init) :', _teacher_init)

    # --->bulid networks
    if args.model == 'mhim':
        if args.mrh_sche:
            mrh_sche = cosine_scheduler(args.mask_ratio_h, 0., epochs=args.num_epoch, niter_per_ep=len(train_loader))
        else:
            mrh_sche = None

        model_params = {
            'baseline': args.baseline,
            'dropout': args.dropout,
            'mask_ratio' : args.mask_ratio,
            'n_classes': args.n_classes,
            'temp_t': args.temp_t,
            'act': args.act,
            'head': args.n_heads,
            'msa_fusion': args.msa_fusion,
            'mask_ratio_h': args.mask_ratio_h,
            'mask_ratio_hr': args.mask_ratio_hr, # 1 or 0
            'mask_ratio_l': args.mask_ratio_l,
            'mrh_sche': mrh_sche,
            'da_act': args.da_act,
            'attn_layer': args.attn_layer,
        }
        
        if args.mm_sche:
            mm_sche = cosine_scheduler(args.mm, args.mm_final, epochs=args.num_epoch, niter_per_ep=len(train_loader), start_warmup_value=1.)

        model = mhim.MHIM(**model_params).to(device)
            
    elif args.model == 'pure':
        model = mhim.MHIM(select_mask=False, n_classes=args.n_classes, act=args.act, head=args.n_heads, da_act=args.da_act, baseline=args.baseline).to(device)
    elif args.model == 'attmil':
        model = attmil.DAttention(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'gattmil':
        model = attmil.AttentionGated(dropout=args.dropout).to(device)
    # # follow the official code
    # # ref: https://github.com/mahmoodlab/CLAM
    # elif args.model == 'clam_sb':
    #     model = clam.CLAM_SB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    # elif args.model == 'clam_mb':
    #     model = clam.CLAM_MB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'transmil':
        model = transmil.TransMIL(n_classes=args.n_classes, dropout=args.dropout, act=args.act).to(device)
    elif args.model == 'dsmil':
        model = dsmil.MILNet(n_classes=args.n_classes, dropout=args.dropout, act=args.act).to(device)
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5
        state_dict_weights = torch.load('./modules/init_cpk/dsmil_init.pth')
        info = model.load_state_dict(state_dict_weights, strict=False)
        if not args.no_log:
            print(info)
    # elif args.model == 'meanmil':
    #     model = mean_max.MeanMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    # elif args.model == 'maxmil':
    #     model = mean_max.MaxMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)

    if args.init_stu_type != 'none':
        if not args.no_log:
            print('######### Model Initializing.....')
        pre_dict = torch.load(_teacher_init)
        new_state_dict ={}
        if args.init_stu_type == 'fc':
        # only patch_to_emb
            for _k, v in pre_dict.items():
                _k = _k.replace('patch_to_emb.','') if 'patch_to_emb' in _k else _k
                new_state_dict[_k]=v
            info = model.patch_to_emb.load_state_dict(new_state_dict,strict=False)
        else:
        # init all
            info = model.load_state_dict(pre_dict,strict=False)
        if not args.no_log:
            print(info)

    # teacher model
    if args.model == 'mhim':
        model_tea = deepcopy(model)
        if not args.no_tea_init and args.tea_type != 'same':
            if not args.no_log:
                print('######### Teacher Initializing.....')
                # pre_dict = torch.load(_teacher_init)
                # print(pre_dict)
                # info = model_tea.load_state_dict(pre_dict,strict=False)
            try:
                pre_dict = torch.load(_teacher_init)
                info = model_tea.load_state_dict(pre_dict,strict=False)
                if not args.no_log:
                    print(info)
            except:
                if not args.no_log:
                    print('########## Init Error')
        if args.tea_type == 'same':
            model_tea = model
    else:
        model_tea = None

    print(' **model** ', model)
    print(' **model_tea** ', model_tea)
    assert model != None, "student model is missing !"
    assert model_tea != None, "teacher model is missing !"

    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()

    # optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, 
                                       stop_epoch=args.max_epoch if args.datasets=='camelyon16' else 70, 
                                       save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None

    optimal_ac, opt_pre, opt_re, opt_fs, opt_auc, opt_epoch = 0, 0, 0, 0, 0, 0
    opt_te_auc, opt_tea_auc, opt_te_fs, opt_te_tea_auc, opt_te_tea_fs  = 0., 0., 0., 0., 0.
    epoch_start = 0

    if args.fix_train_random:
        seed_torch(args.seed)

    # resume
    if args.auto_resume and not args.no_log:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        epoch_start = ckp['epoch']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['lr_sche'])
        early_stopping.load_state_dict(ckp['early_stop'])
        optimal_ac, opt_pre, opt_re, opt_fs, opt_auc, opt_epoch = ckp['val_best_metric']
        opt_te_auc = ckp['te_best_metric'][0]
        if len(ckp['te_best_metric']) > 1:
            opt_te_fs = ckp['te_best_metric'][1]
        opt_te_tea_auc, opt_te_tea_fs = ckp['te_best_metric'][2:4]
        np.random.set_state(ckp['random']['np'])
        torch.random.set_rng_state(ckp['random']['torch'])
        random.setstate(ckp['random']['py'])
        if args.fix_loader_random:
            train_loader.sampler.generator.set_state(ckp['random']['loader'])
        args.auto_resume = False
    ########################################################################################
    
    train_time_meter = AverageMeter()
    
    for epoch in range(epoch_start, args.num_epoch):
        print()
        print(f"Training... (epoch {epoch+1})")
        for i in range(20): # 0 ~ 717
            idx = i * 5 # 40
            train_slide_patches = []
            patch_counts = []

            try:
                for index, slide_name in enumerate(tqdm(list(df_train['Slide_name'])[idx:idx+5], disable=len(list(df_train['Slide_name'])[idx:idx+40])==0)):
                    temp = load_dataset_from_pickle(os.path.join(args.dataset_train_root, f'{slide_name}_patch.pkl'))
                    # print(type(temp), len(temp), type(temp[0])) # <class '__main__.MILDataset'> 677 <class 'tuple'>
                    patch_counts.append(len(temp))
                    train_slide_patches.append(temp)
                    # train_slide_patches.append(load_dataset_from_pickle(os.path.join(args.dataset_train_root, f'{slide_name}_patch.pkl')))
                if train_slide_patches:
                    print(f'# patch slice {i}, {idx} -> slide_name range [{idx}:{idx+40}] (slide_count : {len(train_slide_patches)} | patch_count : {sum(patch_counts)}')
                    train_set = torch.utils.data.ConcatDataset(train_slide_patches)
                    # print(f'  train_set : {type(train_set)}')
                    print(f'  >> train_set : {len(train_set)}')
                
            except IndexError:
                continue
            
            if train_slide_patches:
                if args.fix_loader_random:
                    # generated by int(torch.empty((), dtype=torch.int64).random_().item())
                    big_seed_list = 7784414403328510413
                    generator = torch.Generator()
                    generator.manual_seed(big_seed_list)  
                    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,generator=generator)
                else:
                    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=args.num_workers)
                    
                train_loss, start, end = train_loop(args, model, model_tea, train_loader, optimizer, device, amp_autocast, criterion, loss_scaler, scheduler, mm_sche, epoch)

        del train_slide_patches
        gc.collect()
        del patch_counts
        gc.collect()

        train_time_meter.update(end-start)
        
        print(f"Validation... (epoch {epoch+1})")
        val_accuracy, val_auc_value, val_precision, val_recall, val_fscore, val_test_loss = [],[],[],[],[],[] ## added
        teacher_accuracy, teacher_auc_value, teacher_precision, teacher_recall, teacher_fscore, teacher_test_loss = [],[],[],[],[],[] ## added
        for i in range(20): # 0 ~ 717
            idx = i * 1 # 40
            val_slide_patches = []
            patch_counts = []

            try:
                for index, slide_name in enumerate(tqdm(list(df_val['Slide_name'])[idx:idx+1], disable=len(list(df_val['Slide_name'])[idx:idx+40])==0)):
                    temp = load_dataset_from_pickle(os.path.join(args.dataset_train_root, f'{slide_name}_patch.pkl'))
                    # print(type(temp), len(temp), type(temp[0])) # <class '__main__.MILDataset'> 677 <class 'tuple'>
                    patch_counts.append(len(temp))
                    val_slide_patches.append(temp)
                    # val_slide_patches.append(load_dataset_from_pickle(os.path.join(args.dataset_val_root, f'{slide_name}_patch.pkl')))
                if val_slide_patches:
                    print(f'# patch slice {i}, {idx} -> slide_name range [{idx}:{idx+40}] (slide_count : {len(val_slide_patches)} | patch_count : {sum(patch_counts)}')
                    val_set = torch.utils.data.ConcatDataset(val_slide_patches)
                    # print(f'  val_set : {type(val_set)}')
                    print(f'  >> val_set : {len(val_set)}')
                
            except IndexError:
                continue

            if val_slide_patches:
                val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                accuracy, auc_value, precision, recall, fscore, test_loss = val_loop(args, model, val_loader, device, criterion, early_stopping, epoch, model_tea)
                val_accuracy.append(accuracy) 
                val_auc_value.append(auc_value) 
                val_precision.append(precision) 
                val_recall.append(recall) 
                val_fscore.append(fscore) 
                val_test_loss.append(test_loss)

                if model_tea is not None:
                    accuracy_tea, auc_value_tea, precision_tea, recall_tea, fscore_tea, test_loss_tea = val_loop(args, model_tea, val_loader, device, criterion, None, epoch, model_tea)
                    teacher_accuracy.append(accuracy) 
                    teacher_auc_value.append(auc_value) 
                    teacher_precision.append(precision) 
                    teacher_recall.append(recall) 
                    teacher_fscore.append(fscore) 
                    teacher_test_loss.append(test_loss)
        
        accuracy, auc_value, precision, recall, fscore, test_loss = np.mean(torch.tensor(val_accuracy).detach().cpu().numpy()), np.mean(torch.tensor(val_auc_value).detach().cpu().numpy()), np.mean(torch.tensor(val_precision).detach().cpu().numpy()), np.mean(torch.tensor(val_recall).detach().cpu().numpy()), np.mean(torch.tensor(val_fscore).detach().cpu().numpy()), np.mean(torch.tensor(val_test_loss).detach().cpu().numpy())
        accuracy_tea, auc_value_tea, precision_tea, recall_tea, fscore_tea, test_loss_tea = np.mean(torch.tensor(teacher_accuracy).detach().cpu().numpy()), np.mean(torch.tensor(teacher_auc_value).detach().cpu().numpy()), np.mean(torch.tensor(teacher_precision).detach().cpu().numpy()), np.mean(torch.tensor(teacher_recall).detach().cpu().numpy()), np.mean(torch.tensor(teacher_fscore).detach().cpu().numpy()), np.mean(torch.tensor(teacher_test_loss).detach().cpu().numpy())

        if auc_value_tea > opt_tea_auc:
            opt_tea_auc = auc_value_tea
        print(' *(opt_tea_auc)* ', opt_tea_auc)
        
        # early stop
        if early_stopping is not None:
            early_stopping(epoch, -auc_value, model)
            stop = early_stopping.early_stop
        else:
            stop = False
            
        del val_slide_patches
        gc.collect()
        del patch_counts
        gc.collect()

        # val_slide_patches = []
        # for index, slide_name in enumerate(tqdm(list(df_val['Slide_name']), disable=len(list(df_val['Slide_name']))==0)):
        #     val_slide_patches.append(load_dataset_from_pickle(os.path.join(args.dataset_train_root, f'{slide_name}_patch.pkl')))
        # val_set = torch.utils.data.ConcatDataset(val_slide_patches)
        
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        # stop, accuracy, auc_value, precision, recall, fscore, test_loss = val_loop(args, model, val_loader, device, criterion, early_stopping, epoch, model_tea)

        # if model_tea is not None:
        #     _, accuracy_tea, auc_value_tea, precision_tea, recall_tea, fscore_tea, test_loss_tea = val_loop(args, model_tea, val_loader, device, criterion, None, epoch, model_tea)

        #     if auc_value_tea > opt_tea_auc:
        #         opt_tea_auc = auc_value_tea
        #     print(' *(opt_tea_auc)* ', opt_tea_auc)

        # del val_slide_patches
        # gc.collect()
                        
        # if not args.no_log:
        #     print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' % 
        # (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))

        if not args.no_log:
            print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' % 
        (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))
        
        if auc_value > opt_auc and epoch >= args.save_best_model_stage*args.num_epoch:
            print(f' **Reached Best auc_value ! (epoch {epoch+1})** ')
            optimal_ac = accuracy
            opt_pre = precision
            opt_re = recall
            opt_fs = fscore
            opt_auc = auc_value
            opt_epoch = epoch

            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            if not args.no_log:
                best_pt = {
                    'model': model.state_dict(),
                    'teacher': model_tea.state_dict() if model_tea is not None else None,
                }
                torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=args.fold_num))) # fold=k
        
        # save checkpoint
        random_state = {
            'np': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
            'py': random.getstate(),
            'loader': train_loader.sampler.generator.get_state() if args.fix_loader_random else '',
        }
        ckp = {
            'model': model.state_dict(),
            'lr_sche': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'k': args.fold_num, # k,
            'early_stop': early_stopping.state_dict() if args.early_stopping else None,
            'random': random_state,
            'ckc_metric': [acs, pre, rec, fs, auc, te_auc, te_fs],
            'val_best_metric': [optimal_ac, opt_pre, opt_re, opt_fs, opt_auc, opt_epoch],
            'te_best_metric': [opt_te_auc, opt_te_fs, opt_te_tea_auc, opt_te_tea_fs],
        }
        if not args.no_log:
            torch.save(ckp, os.path.join(args.model_path, 'ckp.pt'))
        if stop:
            break
        print(" **ckp['ckc_metric']** ", ckp['ckc_metric'])
        print(" **ckp['val_best_metric']** ", ckp['val_best_metric'])
        print(" **ckp['te_best_metric']** ", ckp['te_best_metric'])
        
            
    del train_loader
    gc.collect()
    del val_loader
    gc.collect()
    
    print()
    print(f"Testing...")
    # test (inference)
    if not args.no_log:
        best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=args.fold_num))) # fold=k
        info = model.load_state_dict(best_std['model'])
        print(info)
        if model_tea is not None and best_std['teacher'] is not None:
            info = model_tea.load_state_dict(best_std['teacher'])
            print(info)
    
    test_slide_patches = []
    for index, slide_name in enumerate(tqdm(list(df_test['Slide_name']), disable=len(list(df_test['Slide_name']))==0)):
        test_slide_patches.append(load_dataset_from_pickle(os.path.join(args.dataset_test_root, f'{slide_name}_patch.pkl')))
    test_set = torch.utils.data.ConcatDataset(test_slide_patches)
    del test_slide_patches
    gc.collect()
    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    accuracy, auc_value, precision, recall, fscore, test_loss_log = test(args, model, test_loader, device, criterion, model_tea)
    # stop, accuracy, auc_value, precision, recall, fscore, test_loss

    if not args.no_log:
        print('\n Optimal accuracy: %.3f, Optimal auc: %.3f, Optimal precision: %.3f, Optimal recall: %.3f, Optimal fscore: %.3f (optimal epoch {%d})' % (optimal_ac, opt_auc, opt_pre, opt_re, opt_fs, opt_epoch))
    acs.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    fs.append(fscore)
    auc.append(auc_value)
    
    del test_loader
    gc.collect()
        
    return [acs, pre, rec, fs, auc, te_auc, te_fs]


def train_loop(args, model, model_tea, loader, optimizer, device, amp_autocast, criterion, loss_scaler, scheduler, mm_sche, epoch):
    start = time.time()
    loss_cls_meter = AverageMeter()
    loss_cl_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    keep_num_meter = AverageMeter()
    mm_meter = AverageMeter()
    train_loss_log = 0.
    model.train()
    
    if model_tea is not None:
        model_tea.train()

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        
        # print(type(data), data[0], data[1]) # <class 'torch.Tensor'> torch.Size([1, 3, 512, 512]) <class 'torch.Tensor'>
        # if isinstance(data[0], (list, tuple)):
        #     for i in range(len(data[0])):
        #         data[0][i] = data[0][i].to(device)
        #     bag = data[0]
        #     batch_size = data[0][0].size(0)
        # else:
        #     bag = data[0].to(device)  # b*n*1024
        #     batch_size = bag.size(0)
            
        bag = data[0].to(device)  # b*n*1024
        batch_size = bag.size(0)
            
        label = data[1].to(device)
        
        # print(' ** ', type(bag), bag.shape, type(label))
        
        with amp_autocast():
            if args.patch_shuffle:
                bag = patch_shuffle(bag, args.shuffle_group)
                # print(' *** ', type(bag), bag.shape, type(label))
            elif args.group_shuffle:
                bag = group_shuffle(bag, args.shuffle_group)
                # print(' *** ', type(bag), bag.shape, type(label))

            if args.model == 'mhim':
                if model_tea is not None:
                    cls_tea, attn = model_tea.forward_teacher(bag, return_attn=True)
                else:
                    attn, cls_tea = None,None

                cls_tea = None if args.cl_alpha == 0. else cls_tea
                # print('** ', cls_tea)

                train_logits, cls_loss, patch_num, keep_num = model(bag, attn, cls_tea, i=epoch*len(loader)+i)
                # print(' ** ', train_logits, cls_loss, patch_num, keep_num)
                # print(' ** ', cls_loss, patch_num, keep_num)

            elif args.model == 'pure':
                train_logits, cls_loss, patch_num, keep_num = model.pure(bag)
            elif args.model in ('clam_sb', 'clam_mb', 'dsmil'):
                train_logits, cls_loss, patch_num = model(bag, label, criterion)
                keep_num = patch_num
            else:
                train_logits = model(bag)
                cls_loss, patch_num, keep_num = 0.,0.,0.

            if args.loss == 'ce':
                logit_loss = criterion(train_logits.view(batch_size,-1), label)
            elif args.loss == 'bce':
                logit_loss = criterion(train_logits.view(batch_size,-1), one_hot(label.view(batch_size,-1).float(), num_classes=2))

        train_loss = args.cls_alpha * logit_loss +  cls_loss * args.cl_alpha

        train_loss = train_loss / args.accumulation_steps
        if args.clip_grad > 0.:
            dispatch_clip_grad(
                model_parameters(model),
                value = args.clip_grad, mode='norm')

        if (i+1) % args.accumulation_steps == 0:
            train_loss.backward()
            optimizer.step()
            if args.lr_supi and scheduler is not None:
                scheduler.step()
            if args.model == 'mhim':
                if mm_sche is not None:
                    mm = mm_sche[epoch*len(loader)+i]
                else:
                    mm = args.mm
                if model_tea is not None:
                    if args.tea_type == 'same':
                        pass
                    else:
                        ema_update(model, model_tea, mm)
            else:
                mm = 0.

        loss_cls_meter.update(logit_loss, 1)
        loss_cl_meter.update(cls_loss, 1)
        patch_num_meter.update(patch_num, 1)
        keep_num_meter.update(keep_num, 1)
        mm_meter.update(mm, 1)
        # print(' * ', loss_cl_meter, loss_cl_meter.avg)

        if i % args.log_iter == 0 or i == len(loader) - 1:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            rowd = OrderedDict([
                ('cls_loss', loss_cls_meter.avg),
                ('lr', lr),
                ('cl_loss', loss_cl_meter.avg),
                ('patch_num', patch_num_meter.avg),
                ('keep_num', keep_num_meter.avg),
                ('mm', mm_meter.avg),
            ])
            if not args.no_log:
                print('[{}/{}] logit_loss:{}, cls_loss:{},  patch_num:{}, keep_num:{} '.format(i, len(loader)-1, loss_cls_meter.avg, loss_cl_meter.avg, patch_num_meter.avg, keep_num_meter.avg))
            rowd = OrderedDict([ (str(args.fold_num)+'-fold/'+_k,_v) for _k, _v in rowd.items()])

        train_loss_log = train_loss_log + train_loss.item()

    end = time.time()
    train_loss_log = train_loss_log/len(loader)
    if not args.lr_supi and scheduler is not None:
        scheduler.step()
    
    return train_loss_log, start, end


def val_loop(args, model, loader, device, criterion, early_stopping, epoch, model_tea=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    loss_cls_meter = AverageMeter()
    bag_logit, bag_labels=[], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())

            # print(type(data), data[0], data[1]) # <class 'torch.Tensor'> torch.Size([1, 3, 512, 512]) <class 'torch.Tensor'>
            # if isinstance(data[0], (list, tuple)):
            #     for i in range(len(data[0])):
            #         data[0][i] = data[0][i].to(device)
            #     bag = data[0]
            #     batch_size = data[0][0].size(0)
            # else:
            #     bag = data[0].to(device)  # b*n*1024
            #     batch_size = bag.size(0)
            
            bag = data[0].to(device)  # b*n*1024
            batch_size = bag.size(0)

            label=data[1].to(device)
            
            # print(' ** ', type(bag), bag.shape, type(label))
            
            if args.model in ('mhim', 'pure'):
                test_logits = model.forward_test(bag)
            elif args.model == 'dsmil':
                test_logits,_ = model(bag)
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
                    test_loss = criterion(test_logits[0].view(batch_size, 1), label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1], dim=-1) + 0.5*torch.softmax(test_logits[0], dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1), label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits.view(batch_size,-1), label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1]) + 0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits[0].view(batch_size,-1), label.view(batch_size,-1).float())
                    
                    bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            loss_cls_meter.update(test_loss,1)
    
    # save the log file
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit)
    
    # # early stop
    # if early_stopping is not None:
    #     early_stopping(epoch, -auc_value, model)
    #     stop = early_stopping.early_stop
    # else:
    #     stop = False
    # return stop, accuracy, auc_value, precision, recall, fscore, loss_cls_meter.avg
    return accuracy, auc_value, precision, recall, fscore, loss_cls_meter.avg


def test(args, model, loader, device, criterion, model_tea=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    test_loss_log = 0.
    bag_logit, bag_labels = [], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())

            # print(type(data), data[0], data[1]) # <class 'torch.Tensor'> torch.Size([1, 3, 512, 512]) <class 'torch.Tensor'>    
            # if isinstance(data[0], (list, tuple)):
            #     for i in range(len(data[0])):
            #         data[0][i] = data[0][i].to(device)
            #     bag = data[0]
            #     batch_size = data[0][0].size(0)
            # else:
            #     bag = data[0].to(device)  # b*n*1024
            #     batch_size = bag.size(0)
            
            bag = data[0].to(device)  # b*n*1024
            batch_size = bag.size(0)

            label=data[1].to(device)
            
            # print(' ** ', type(bag), bag.shape, type(label))
            
            if args.model in ('mhim', 'pure'):
                test_logits = model.forward_test(bag)
            elif args.model == 'dsmil':
                test_logits, _ = model(bag)
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
                    test_loss = criterion(test_logits[0].view(batch_size, -1), label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1], dim=-1) + 0.5*torch.softmax(test_logits[0], dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size, -1), label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits, dim=-1)[:,1].cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits, dim=-1)[:,1].cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits[0].view(batch_size,-1), label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1]) + 0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1), label.view(1,-1).float())
                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            test_loss_log = test_loss_log + test_loss.item()
    
    # save the log file
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit)
    test_loss_log = test_loss_log / len(loader)

    return accuracy, auc_value, precision, recall, fscore, test_loss_log

if __name__ == '__main__':
    
    # SEED = 42 (O)
    # PATCH_SIZE = (512, 512) # width, height (O)
    # OVERLAP_RATIO = 0.1 # 0.5
    # TISSUE_AREA_RATIO = 0.5
    # DROP_RATE = 0.5
    # EPOCH = 10
    # LEARNING_RATE = 1e-5
    # WEIGHT_DECAY = 1e-5 (O)
    # TRAIN_BATCH_SIZE = 64
    # EVAL_BATCH_SIZE = 256
    # TOPK = 5 # MIL top K
    # GPU_IDX = 0
    # DATE = '240102'
    # EXP_NAME = f'MAIC_{DATE}_{LEARNING_RATE}'

    parser = argparse.ArgumentParser(description='MIL Training Script')
    now = datetime.now()
    
    # Dataset 
    parser.add_argument('--datasets', default='MILDataset', type=str, help='[camelyon16, tcga, maic]') ## camelyon16
    parser.add_argument('--dataset_train_root', default='../re_pickle_train_patch', type=str, help='Dataset(patch) root path for train/val dataset')
    parser.add_argument('--dataset_test_root', default='../re_pickle_test_patch', type=str, help='Dataset(patch) root path for test dataset')
    parser.add_argument('--fold_num', default=2, type=int, help='[0, 1, 2, 3, 4]') # camelyon16 3
    parser.add_argument('--fold_root', default='../total_split_train_val', type=str, help='fold root path')
    
    # parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    # parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    # parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    # parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')
    
    # Train
    parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    parser.add_argument('--num_epoch', default=100, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--early_stopping', action='store_false', help='Early stopping') ## store_false
    parser.add_argument('--max_epoch', default=130, type=int, help='Number of max training epochs in the earlystopping [130]')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size') ## 1
    parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce]')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--model', default='mhim', type=str, help='Model name') # mhim
    parser.add_argument('--seed', default=2021, type=int, help='random number [2021, 42]') ## 2021
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
    # parser.add_argument('--always_test', action='store_true', help='Test model in the training phase')

    # Model
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # Our
    parser.add_argument('--baseline', default='selfattn', type=str, help='Baselin model [attn,selfattn]')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the MSA')
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # (trainsmil_C16) 
    # --mask_ratio=0. --mask_ratio_l=0.8 --mask_ratio_h=0.03 --mask_ratio_hr=0.5 
    # --cl_alpha=0.1 --init_stu_type=fc --attn_layer=0 --seed=2021
    # --teacher_init=./modules/init_ckp/c16_3fold_init_transmil_seed2021 
    # --mrh_sche --mm_sche 
    
    # MHIM
    # Mask ratio
    parser.add_argument('--mask_ratio', default=0., type=float, help='Random mask ratio')
    parser.add_argument('--mask_ratio_l', default=0.8, type=float, help='Low attention mask ratio') # 0.
    parser.add_argument('--mask_ratio_h', default=0.03, type=float, help='High attention mask ratio') # 0.
    parser.add_argument('--mask_ratio_hr', default=0.5, type=float, help='Randomly high attention mask ratio') # 1.
    parser.add_argument('--mrh_sche', action='store_true', help='Decay of HAM')
    parser.add_argument('--msa_fusion', default='vote', type=str, help='[mean, vote]')
    parser.add_argument('--attn_layer', default=0, type=int)
    
    # Siamese(Teacher-Student) framework
    parser.add_argument('--cl_alpha', default=0.1, type=float, help='Auxiliary loss alpha') # 0.
    parser.add_argument('--temp_t', default=0.1, type=float, help='Temperature')
    parser.add_argument('--teacher_init', default='/data/notebook/hyena/modules/init_ckp/c16_3fold_init_transmil_seed2021', type=str, help='Path to initial teacher model') # none
    parser.add_argument('--no_tea_init', action='store_true', help='Without teacher initialization')
    parser.add_argument('--init_stu_type', default='none', type=str, help='Student initialization [none, fc, all]') # none
    parser.add_argument('--tea_type', default='same', type=str, help='[none, same]') # none
    parser.add_argument('--mm', default=0.9999, type=float, help='Ema decay [0.9997]')
    parser.add_argument('--mm_final', default=1., type=float, help='Final ema decay [1.]')
    parser.add_argument('--mm_sche', action='store_true', help='Cosine schedule of ema decay')

    # Misc
    parser.add_argument('--title', default='transmil', type=str, help='Title of exp') ## default
    parser.add_argument('--project', default=now.date(), type=str, help='Project name of exp') ## mil_new_c16
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--no_log', action='store_true', help='Without log') ## store_true
    parser.add_argument('--model_path', default='paths', type=str, help='Output path')

    args = parser.parse_args()
    
    if not os.path.exists(args.model_path): # os.path.join(args.model_path, args.project)
        os.mkdir(args.model_path)
    args.model_path = os.path.join(args.model_path, f'{args.project}_{args.title}')

    if args.model == 'pure':
        args.cl_alpha=0.
    # # follow the official code
    # # ref: https://github.com/mahmoodlab/CLAM
    # elif args.model == 'clam_sb':
    #     args.cls_alpha= .7
    #     args.cl_alpha = .3
    # elif args.model == 'clam_mb':
    #     args.cls_alpha= .7
    #     args.cl_alpha = .3
    elif args.model == 'dsmil':
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5

    # if args.datasets == 'camelyon16':
    #     args.fix_loader_random = True
    #     args.fix_train_random = True
    # if args.datasets == 'tcga':
    #     args.num_workers = 0
    #     args.always_test = True
    
    if args.auto_resume:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
    print(args)
    
    print()
    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)
    
    main(args=args)
