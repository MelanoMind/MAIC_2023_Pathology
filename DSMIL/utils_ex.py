import torch
import numpy as np
import random
import pandas as pd
import os

def seed_everything(random_seed: int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    print('seed setting complete')

    
def get_slides_name(path, filter_image=False) :
    df = pd.read_csv(path)
    df_pkl_list = df['Slide_name'].tolist()
    
    if filter_image :
        filtered_name=[]
        for slide_name in df_pkl_list :
            if int(slide_name.split('_')[1]) > 345 :
                filtered_name.append(slide_name)
        print('Filtering Below 345')
        return filtered_name
    return df_pkl_list

def get_image_path(slide_name_list, image_path) :
    path_list = []
    for name in slide_name_list :
        path_list.append(image_path+f'/{name}_patch.pkl')
    return path_list

def get_pos_weight(path) :
    df = pd.read_csv(path)
    recurrence_counts = df['Recurrence'].value_counts()
    pos_weight = len(df)/recurrence_counts.get(1, 0)
    return torch.tensor([pos_weight])