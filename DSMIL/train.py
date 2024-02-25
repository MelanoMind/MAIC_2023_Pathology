import torch
import numpy as np
import torch.nn.functional as F
from backbone import BackboneResNet
from dsmil import IClassifier, BClassifier, MILNet
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from utils import get_slides_name, seed_everything, get_image_path, get_pos_weight
from tqdm.auto import tqdm
import torch.nn as nn
from Dataset import load_dataset_from_pickle, MILDataset
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.cluster import KMeans

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 하나의 WSI의 patch들을 batch단위로 뜯어서 학습.
def wsi_dsmil_train(dataloader, model, criterion, kl_criterion, optimizer) :
    model.train()
    batch_loss = 0
    for batch_index, (wsi_patch, wsi_label) in enumerate(dataloader) :
        batch_size = wsi_patch.size(0)
        if (batch_index > 0) & (batch_size < 200) :
            break    
        # label = tensor([[0 or 1]]) 이렇게 만드는 이유는, criterion에 input tensor와 맞춰주기 위해
        label = torch.tensor([[wsi_label[0]]]) 
        wsi_patch, label = wsi_patch.to(device).float(), label.to(device).float()
        optimizer.zero_grad()
        
        # instance prediction :tensor.Size(1, batch_size) -> 각 patch의 prob 존재 
        # bag prediction : tensor([prob]) -> bag의 prob  
        instance_prediction, bag_prediction, attention_score, features = model(wsi_patch)
        
        cluster = KMeans(n_clusters=11, n_init=13)
        features_cpu = features.cpu().detach().numpy()
        cluster.fit(features_cpu)
        
        bag_prob = torch.sigmoid(bag_prediction)
        max_prediction, index = torch.max(instance_prediction, 0)  # patch의 prob들 중에 max값만 구해 loss 
        loss_bag = criterion(bag_prediction.view(1,-1), label)
        loss_instance = criterion(max_prediction.view(1,-1), label)
        kl_loss = kl_criterion(attention_score, cluster.labels_)
        instance_prob = torch.sigmoid(max_prediction)
        loss_total = 1.0*loss_bag+0.01*loss_instance+0.1*kl_loss
        loss_total = loss_total.mean()
        batch_loss += loss_total.item()
        loss_total.backward()
        optimizer.step()
        
    wsi_loss = batch_loss/(len(train_wsi_data_loader))
    return wsi_loss, wsi_label[0], instance_prob, bag_prob, max_prediction

def dsmil_validation(valid_list,model,batch_size) :
    print('##########--- START TEST ---###########')
    bag_labels = []
    bag_predictions = []
    model.eval()
    with torch.no_grad() :
        #validation WSI load
        for valid_wsi_path in valid_list :
            valid_wsi = load_dataset_from_pickle(valid_wsi_path)
            valid_loader=DataLoader(valid_wsi,batch_size=batch_size, shuffle=False, drop_last=False)
            batch_result = [] # batch로 나눠진 하나의 이미지를 종합해서 판단하기위해 
            batch_label = []
            for data, label in valid_loader:
                data = data.to(device).float()
                batch_label.append(int(label[0]))
                instance_prediction, bag_logit, _, _ = model(data)
                bag_prob = torch.sigmoid(bag_logit)
                
                bag_prediction = (bag_prob > 0.5).float()
                bag_prediction = int(bag_prediction.item())
                batch_result.append(bag_prediction)
                print(f'Bag Prob : {bag_prob} | Label : {int(label[0])} | Predicted Label : {bag_prediction}')
            bag_predictions.append(int(any(batch_result)))  # batch_result값중에 하나라도 positive라면 True(1)를 넣어준다
            bag_labels.append(int(any(batch_label)))
        # 종합하여 값을 구한다.
        if all(value == 1 for value in bag_predictions) or all(value == 0 for value in bag_predictions):
                print("모두 1 또는 모두 0입니다.")
        else :
            try :
                precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions)
                auc_value = roc_auc_score(bag_labels, bag_predictions)
                accuracy = accuracy_score(bag_labels, bag_predictions)
                print('accuracy : ', accuracy, 'precision : ', precision, 'auc_value : ',auc_value)
            except :
                print(f'bag_label : {bag_labels}, bag_prediction : {bag_predictions}')


                    
if __name__ == '__main__': 
    from backbone import BackboneResNet
    from dsmil import IClassifier, BClassifier, MILNet
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
    from loss import KLDLoss
    
    SEED = 12
    pkl_path='../re_pickle_train_patch'
    EPOCH=10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRAIN_CSV_PATH = '../total_split_train_val/fold_4_train.csv'
    VALID_CSV_PATH = '../total_split_train_val/fold_4_val.csv'
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 3e-5
    BATCH_SIZE = 700
    
    seed_everything(SEED)
    train_slide_name_list = get_slides_name(TRAIN_CSV_PATH, filter_image=False)
    valid_slide_name_list = get_slides_name(VALID_CSV_PATH, filter_image=False)
    
    train_pkl_list = get_image_path(train_slide_name_list, pkl_path)
    valid_pkl_list = get_image_path(valid_slide_name_list, pkl_path)
    random.shuffle(train_pkl_list)
    
    # positive 비율을 고려하여 label이 positive인 경우, loss에서 추가 weight를 부여
    pos_weight = get_pos_weight(TRAIN_CSV_PATH)
    pos_weight = pos_weight.to(device)
    
    feature_extractor = BackboneResNet('resnet18')
    
    instance_classifier = IClassifier(backbone=feature_extractor, freeze=True, out_dim=1)
    bag_classifier = BClassifier(input_size=64, output_class=1, nonlinear=True)
    
    milnet = MILNet(instance_classifier, bag_classifier).to(device)
    criterion = nn.BCEWithLogitsLoss()
    kl_criterion = KLDLoss()
    optimizer = torch.optim.Adam(milnet.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9), weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH, 0)

    for epoch in tqdm(range(EPOCH)) :
        print(f'-------###### EPOCH : {epoch} start######------')
        epoch_loss=0
        for train_wsi_path in train_pkl_list:
            train_wsi = load_dataset_from_pickle(train_wsi_path)

            if len(train_wsi) <= 10:
                print("SKIP Dataset len:", len(train_wsi))
                continue
            
            train_wsi_data_loader = DataLoader(train_wsi, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
            if len(train_wsi_data_loader)==0:
                continue
            else :
                wsi_loss, label, max_prediction, bag_prediction, max_p = wsi_dsmil_train(dataloader=train_wsi_data_loader, model=milnet, criterion=criterion, kl_criterion=kl_criterion,  optimizer=optimizer)
                print(f'Label : {label} : | Instance Prob : {max_prediction.item():.5f} | Bag Prob : {bag_prediction.item():.5f} |Bag_max : {max_p.item():.5f} | Loss : {wsi_loss:.5f}')
        dsmil_validation(valid_list=valid_pkl_list, model=milnet, batch_size=BATCH_SIZE)
            
        scheduler.step()
        random.shuffle(train_pkl_list)
        if(epoch+1)>=1:
            torch.save(milnet, f'./dsmilnet{epoch}.pt')

    
    
    