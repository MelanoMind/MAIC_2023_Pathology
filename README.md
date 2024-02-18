## 💻 **Medical Image AI Challenge 2023 : Pathology data**

<img src="https://github.com/MelanoMind/MAIC_2023_Pathology/assets/70469008/a08d5c4d-9bec-440e-870c-3417ccbdad13">

#### Intro

- This competition is a medical artificial intelligence competition hosted by Seoul National University Hospital
- its purpose is to predict disease risk (survival/recurrence) using pathological images and clinical information of malignant melanoma patients.
- 12 out of 49 teams got advanced to the finals, and unfortunately our team did not make it into the rankings in private data, despite we ranked 4th in public data.

#### Team Members

| <img src="" width="80"> | <img src="https://avatars.githubusercontent.com/u/70469008?v=4" width="80"> | <img src="https://avatars.githubusercontent.com/u/126538370?v=4" width="80"> | <img src="https://avatars.githubusercontent.com/u/78690390?v=4" width="80"> | <img src="https://avatars.githubusercontent.com/u/7111986?v=4" width="80"> |
| :---------------------: | :-------------------------------------------------------------------------: | :--------------------------------------------------------------------------: | :-------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
|       [박성현]()        |                  [박혜나](https://github.com/hyenagatha02)                  |                    [신건희](https://github.com/Rigel0718)                    |                  [신중현](https://github.com/Blackeyes0u0)                  |                    [한상준](https://github.com/jphan32)                    |

<br/>

#### main concept

- Whole Slide Image (WSI)
  - The entire shape of the tissue contained on a glass slide is converted into a high-resolution digital image in a short period of time.
  - In general, the size of the image is 100,000 x 100,000 pixels or more, so in this competition we use images reduced and saved at 100x magnification for smooth processiong.
- Multiple Instance Learning (MIL)
  - One of the weakly supervised learning methods, the deep learning model automatically selects k patches that are deemed useful for label prediction.
  - If even one lesion exists in WSI, it is classified as a positive slide, and if none of the lesions exist, it is classified as a negative slide

#### Process

1. EDA (image data / table data)
2. baseline code construction & modulation
3. search references (models and WSI techniques)
4. patch extraction experiment
5. dataloader experiment (save patch to pickle)
6. train model (AB_MIL / DSMIL / SimCLR + tabNet / AC_MIL / MHIM_MIL)
7. Ensemble

#### Dataset

- number of images : 894
- number of patientID : 217
- Image
  - H&E stained pathology whole slide images(WSI)
  - width : (avg) 28,440 / (max) 54,945 / (min) 8,963
  - height : (avg) 19,106 / (max) 24,538 / (min) 5,656
- Table
  - data of each WSI images composed by 18 columns
  - columns : 'Slide_name', 'Patient_ID', 'Recurrence', 'Location', 'Diagnosis',
    'Growth phase', 'Size of tumor', 'Depth of invasion', 'Level of invasion', 'Mitosis',
    'Histologic subtype', 'Tumor cell type', 'Surgical margin', 'Lymph node', 'Breslow thickness',
    'Precursor lesion', 'Date_of_diagnosis', 'Date_of_recurrence'
  - Recurrence : 0 or 1 (nonrecurrent or recurrent)
    - (WSI proportion) recurrent : 688 | nonrrent : 206
    - (patientID proportion)recurrent : 48 | nonrrent : 169

#### File

```
    ├─── EDA
    │   ├── EDA.ipynb
    │─── data_process
    │   ├── split_dataset.ipynb
    │   ├── dataset2pkl_train.py
    │   ├── dataset2pkl_test_public.py
    │─── check_status
    │   ├── check_capacity.ipynb
    │   ├── check_file_status.ipynb
    │   ├── check_fold_patches.ipynb
    │   ├── check_gpu.ipynb
    ├─── AB_MIL
    │─── DSMIL
    ├─── TabNet
    │─── AC_MIL
    │─── MHIM_MIL
    │   ├── camelyon16
    │   ├── modules
    │   ├── ...

```

#### Wrap-up Report

- (need to be added)

#### 평가 Metric

- AUROC for recurrence prediction on a per-patient basis

### Reference papers

- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712)
- [Attention-Challenging Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/abs/2311.07125)
- [Interpretable Prediction of Lung Squamous Cell Carcinoma Recurrence With Self-supervised Learning](https://arxiv.org/abs/2203.12204)
- [Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939)
- [Multiple Instance Learning Framework with Masked Hard Instance Mining for Whole Slide Image Classification](https://arxiv.org/abs/2307.15254)
- [TransMIL_Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/abs/2106.00908)
