#### MHIM_MIL

- All files related to model(modules and else) are all from official MHIM_MIL [paper code](https://github.com/dearcaat/mhim-mil).
  - We only tested on some of the images from [CAMELYON16](https://camelyon16.grand-challenge.org/) open dataset which is simlilar with MAIC competition images.
  - Originally 'init_ckp' in 'modules' contains trained models (offered by paper code), but I removed them since not using those pretrained weights were better on results.
- Example datasets are in 'camleyon16' folder
  - pickle folders are masks/patches created by our MAIC dataset format (using dataset2pkl_train.py, dataset2pkl_test_public.py).
  - 'total_split_train_val' folders contain train/val fold csv files splitted by recurrence.
