# metadata_process.py
# 작성자 : 박성현

import pandas as pd
import numpy as np
import re
import json
import os
from typing import Dict
from app.lib.metadata_processor import *


def paitent_comparison(df: pd.DataFrame) -> Dict:
    """
    Input : df (원본 DataFrame)
    Output : 한 명의 환자에 대해 여러 Row가 있을텐데, 그 Row가 모두 동일한지 확인 (하나로 동일시해도 되는지 파악하기 위함)
    """
    # 'Slide_name' 칼럼이 있는지 확인하고 비교를 위해 임시로 제거
    if "Slide_name" in df.columns:
        df_comparison = df.drop(columns="Slide_name")
    else:
        df_comparison = df.copy()

    # 각 Patient_ID별로 행들을 저장할 딕셔너리 생성
    patient_rows = {}

    # 각 행을 반복하며 해당 Patient_ID의 리스트에 추가
    for _, row in df_comparison.iterrows():
        patient_id = row["Patient_ID"]
        if patient_id not in patient_rows:
            patient_rows[patient_id] = []
        patient_rows[patient_id].append(row)

    # 같은 Patient_ID를 가진 모든 행이 동일한지 확인
    # 비교 결과를 저장할 딕셔너리 생성
    patient_comparison_results = {}

    for patient_id, rows in patient_rows.items():
        # 비교를 위해 행 리스트를 데이터프레임으로 변환
        rows_df = pd.DataFrame(rows)

        # drop_duplicates 후 모든 행이 첫 행과 동일한지 all() 메소드로 비교
        all_equal = rows_df.drop_duplicates().shape[0] == 1

        # 결과를 딕셔너리에 저장
        patient_comparison_results[patient_id] = all_equal

    return patient_comparison_results


def select_row_based_on_filter(group, patient_comparison_results: Dict):
    # patient_comparison_results 딕셔너리를 활용하여 각 Patient_ID에 대한 조건을 설정합니다.
    # False이면 가장 NaN이 적은 행을, True이면 첫 번째 행을 선택합니다.
    pid = group.name
    if patient_comparison_results.get(pid, True):
        return group.iloc[0]
    else:
        return group.loc[group.isnull().sum(axis=1).idxmin()]


def metadata_process():
    df = load_metadata()
    df = clean_metadata(df)

    # 환자별 WSI 리스트 만들기
    slide_list_by_patient = df.groupby("Patient_ID")["Slide_name"].apply(list).to_dict()
    save_metadata(slide_list_by_patient, "slide_list_by_patient.json")
    # 환자별로 Recurrence를 리스트로 모음
    recurrence_by_patient = df.groupby("Patient_ID")["Recurrence"].apply(list).to_dict()
    save_metadata(recurrence_by_patient, "recurrence_by_patient.json")

    # TODO : grouping_
    # 그룹화를 통해 각 Patient_ID 별로 'Slide_name' 리스트를 만들고, 나머지 열에 대해서는 조건에 따라 행을 선택 (2차 전처리 - 여러 행으로 되어 있는 테이블을 1 Patient 1 Row로 변경)
    patient_comparison_results = paitent_comparison(df)
    df_grouped = df.groupby("Patient_ID").apply(
        lambda group: select_row_based_on_filter(group, patient_comparison_results)
    )
    # 'Slide_name' 열의 개수를 통해 1 Patient 1 Slide_List로 변경
    df_grouped["Slide_List"] = df.groupby("Patient_ID")["Slide_name"].apply(list)
    df_grouped = df_grouped.drop(columns=["Slide_name"]).reset_index(drop=True)

    df_grouped["Date_of_diagnosis"] = pd.to_datetime(df_grouped["Date_of_diagnosis"])
    df_grouped["Date_of_recurrence"] = pd.to_datetime(df_grouped["Date_of_recurrence"])
    df_grouped["time_to_recurrence"] = (
        df_grouped["Date_of_recurrence"] - df_grouped["Date_of_diagnosis"]
    ).dt.days

    # Categorizing the numeric columns into 10 equal groups and creating new columns for each
    numeric_columns = [
        "tumor_length",
        "tumor_width",
        "tumor_height",
        "Area of tumor",
        "Volume of tumor",
        "Breslow thickness",
    ]
    for col in numeric_columns:
        df_grouped[f"{col}_category"] = pd.qcut(
            df_grouped[col], q=10, labels=range(1, 11)
        )
    # 1. Remove 'Slide_List' column but create a new 'Slide_Count' column based on its length
    # 'Date_of_diagnosis', 'Date_of_recurrence'
    df_grouped["Slide_Count"] = df_grouped["Slide_List"].apply(
        lambda x: len(str(x).split(", "))
    )
    df_grouped.drop("Date_of_diagnosis", axis=1, inplace=True)
    df_grouped.drop("Date_of_recurrence", axis=1, inplace=True)
    # Patient_ID 별로 Slide_name을 리스트로 모으는 딕셔너리 생성

    df_grouped["Surgical margin"] = df_grouped["Surgical margin"].apply(
        lambda record: (
            None
            if pd.isna(record)
            else (
                "Free from Tumor"
                if "free from tumor" in record
                else (
                    "Involved by Tumor"
                    if "involved by tumor" in record
                    or "involvement of deep margin by tumor" in record
                    else "Other"
                )
            )
        )
    )

    # TODO 3차 전처리?
    df_grouped.drop("Slide_List", axis=1, inplace=True)
    df_grouped.drop("Size of tumor", axis=1, inplace=True)

    df_grouped["Mitosis_Value"] = (
        df_grouped["Mitosis"].str.extract(r"(\d+)/10HPF").astype(float)
    )

    df_grouped.drop("Mitosis", axis=1, inplace=True)

    df_grouped["Mitosis_category"] = pd.cut(
        df_grouped["Mitosis_Value"], bins=20, labels=False
    )

    recurrence = df_grouped.pop("Recurrence")
    df_grouped["Recurrence"] = recurrence

    # Categorical columns
    category_columns = [
        "Location",
        "Diagnosis",
        "Growth phase",
        "Level of invasion",
        "Histologic subtype",
        "Tumor cell type",
        "Surgical margin",
        "Lymph node",
        "Precursor lesion",
        "tumor_length_category",
        "tumor_width_category",
        "tumor_height_category",
        "Area of tumor_category",
        "Volume of tumor_category",
        "Breslow thickness_category",
        "Mitosis_category",
    ]

    # Discrete columns
    discrete_columns = ["Slide_Count"]

    # Continuous columns (remaining ones except 'Patient_ID')
    continuous_columns = [
        col
        for col in df_grouped.columns
        if col not in category_columns + discrete_columns + ["Patient_ID"]
    ]

    # 타입 맞춰주기
    # category_columns의 각 열을 str로 변환

    for col in df_grouped.columns:
        if col in set(category_columns):
            df_grouped[col] = df_grouped[col].astype(str)

    # New column order: Patient_ID -> Categorical -> Discrete -> Continuous
    new_column_order = (
        ["Patient_ID"] + category_columns + discrete_columns + continuous_columns
    )

    # Reordering the DataFrame
    df_grouped = df_grouped[new_column_order]
    df_grouped.drop(["time_to_recurrence", "Slide_Count"], axis=1, inplace=True)

    # 결과를 'cleaned_dataset.csv' 파일로 저장
    df_grouped.to_csv(
        os.path.join(path_config["processed_dataset"], "cleaned_dataset.csv"),
        encoding="cp949",
        index=False,
    )


if __name__ == "__main__":
    metadata_process()
