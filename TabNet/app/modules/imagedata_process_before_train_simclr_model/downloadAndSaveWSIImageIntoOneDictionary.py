# 박성현
# 작성일 2024-02-25
# 실제 MAIC 데이터셋은 제공되었지만, github에 올리기 위해선 가상의 이미지 데이터셋을 사용. ham10000 데이터를 사용함.
# ham10000의 하나의 이미지를, 마치 n(3~5)개의 patch들로 이루어진 것처럼 구성. 실제로는 n은 더욱 큼.
# 팀원이 WSI 이미지 1개를 여러 개의 patch로 분할한 뒤, 그것들을 MILDataset 이라는 클래스로 객체화하여 pickle로 저장하기로 하였기에, 그에 맞추어 임시 데이터셋을 구성.
# TODO 모듈명이 충분히 그 기능을 포함하지 못하는 것 같은데.
import deeplake
import os
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Any
from multiprocessing import Pool, cpu_count
import random
import pickle
import pandas as pd
from app.lib.hydra_setup import setup_config
from functools import partial


# 이미지 처리 함수 정의
def patch_image_dimension_transform(indices, ds):
    return [np.transpose(np.array(ds[i]["images"]), (0, 1, 2)) for i in indices]


# 작업 완료 상황을 로깅하기 위한 콜백 함수를 수정합니다.
def log_result(result, results, tasks):
    results.append(result)
    print(f"Processed {len(results)}/{len(tasks)} tasks")


def save_objectified_wsi():
    """
    각각의 WSI 이미지를 여러 개의 patch로 분할하여 train_001, train_002,... 등의 wsi 이미지를
    dictionary로 저장함. 해당 dictionary의 key는 train_001, train_002....이며
    type(dictionary[train_001])은 dictionary type이다. 이때
    dictionary[train_001]은 다시
    dict_keys(['patch_image_array_list', 'slide_list', 'slide_idx', 'label_list'])라는 key를 가짐.
    """
    _, path_config, _, _ = setup_config()
    ds = deeplake.load("hub://activeloop/ham10000")

    objectified_wsi_dict_path = path_config["image_dataset"]
    if not os.path.exists(objectified_wsi_dict_path):
        os.makedirs(objectified_wsi_dict_path)

    # load하는 부분
    with open(path_config["slide_list_by_patient"], "r") as f:
        slide_list_by_patient = json.load(f)

    with open(path_config["recurrence_by_patient"], "r") as f:
        recurrence_dict = json.load(f)

    # load된 데이터 처리하는 부분
    for key, values in recurrence_dict.items():
        if "train" in key:
            # NaN 값을 처리하기 위해 np.nan을 None으로 변환하고, float를 int로 변환
            recurrence_dict[key] = [
                int(value) if value == value else None for value in values
            ]

    inverse_dict = {}
    for pid, test_list in slide_list_by_patient.items():
        for test_id in test_list:
            inverse_dict[test_id] = pid

    patch_dict = {}
    for key in inverse_dict.keys():
        patch_dict[key] = {
            "patch_image_array_list": None,
            "slide_list": None,
            "slide_idx": None,
            "label_list": None,
        }

    # 이미지를 처리하는 부분

    # 총 이미지 수
    total_images = len(ds)
    num_processes = min(12, cpu_count())

    # multiprocessing Pool 초기화와 함께 결과와 작업 목록을 관리
    results = []
    tasks = []
    with Pool(processes=num_processes) as pool:
        for pid in patch_dict.keys():
            num_images = random.randint(3, 5)
            selected_indices = random.sample(range(total_images), num_images)
            tasks.append(selected_indices)

        # functools.partial을 사용하여 log_result에 results와 tasks를 전달
        log_result_with_args = partial(log_result, results=results, tasks=tasks)
        for task in tasks:
            pool.apply_async(
                patch_image_dimension_transform,
                args=(task, ds),
                callback=log_result_with_args,
            )

        pool.close()
        pool.join()

    # 결과를 올바른 PID에 할당
    for pid, result in zip(patch_dict.keys(), results):
        patch_dict[pid]["patch_image_array_list"] = result

    for k, v in patch_dict.items():
        patch_dict[k]["slide_list"] = [k]
        patch_dict[k]["slide_idx"] = [
            i for i in range(len(patch_dict[k]["patch_image_array_list"]))
        ]
        patch_dict[k]["label_list"] = [
            list(set(recurrence_dict[inverse_dict[k]]))[0]
            for _ in range(len(patch_dict[k]["patch_image_array_list"]))
        ]

    # 결과를 pickle 파일로 저장
    with open(
        os.path.join(objectified_wsi_dict_path, "patch_dictionary_by_wsi.pkl"), "wb"
    ) as f:
        pickle.dump(patch_dict, f)


if __name__ == "__main__":
    save_objectified_wsi()
