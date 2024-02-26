import pandas as pd
import numpy as np
import re
import json
import os
from typing import Optional, List
from app.lib.hydra_setup import setup_config

model_config, path_config, dataset_config, _ = setup_config()


def extract_first_number(text: str) -> Optional[float]:
    # 문자열에서 첫 번째로 발견되는 숫자를 추출하는 함수
    # re.findall은 일치하는 모든 것을 찾아 리스트로 반환
    # 리스트가 비어있지 않으면 첫 번째 요소를 실수로 변환하여 반환합니다.
    numbers = re.findall(r"[\d.]+", str(text))
    return float(numbers[0]) if numbers else None


def preprocess_size_of_tumor(value: str) -> List[Optional[float]]:
    # NaN 확인
    if pd.isna(value):
        return [np.nan, np.nan, np.nan]

    # 'at least' or 'about' 같은 문자열을 제거함
    value = re.sub(r"\(.*?\)|at least|about", "", value)

    # 앞에 나온 최대 세 개의 숫자만 추출
    numbers = re.findall(r"[\d.]+", value)
    return [float(num) if num else np.nan for num in numbers[:3]]


def replace_roman_numerals(text: str) -> str:
    def roman_to_arabic(roman: str) -> int:
        roman_numerals = {
            "I": 1,
            "II": 2,
            "III": 3,
            "IV": 4,
            "V": 5,
            "VI": 6,
            "VII": 7,
            "VIII": 8,
            "IX": 9,
            "X": 10,
        }
        return roman_numerals.get(roman, roman)

    if pd.isna(text):
        return text
    # 로마 숫자와 해당 위치 찾기 (대문자 및 소문자 모두 고려)
    roman_numeral_match = re.finditer(r"\b(level [ivxlcdm]+)\b", text, re.IGNORECASE)
    for match in roman_numeral_match:
        roman_numeral = match.group(1)
        # 대문자로 변환하여 로마 숫자를 아라비아 숫자로 변환
        arabic_numeral = roman_to_arabic(roman_numeral.split()[1].upper())
        # 원래 문자열에서 로마 숫자를 아라비아 숫자로 교체
        text = text.replace(roman_numeral, f"level {arabic_numeral}", 1)
        return text


def load_metadata() -> pd.DataFrame:
    train_df = pd.read_csv(
        os.path.join(path_config["metadata_dir"], "train_dataset.csv")
    )
    test_df = pd.read_csv(
        os.path.join(path_config["metadata_dir"], "test_public_dataset.csv")
    )
    df = pd.concat([train_df, test_df], axis=0)
    return df


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    # Recurrence 열의 위치 변경
    recurrence_column = df.pop("Recurrence")

    # 소문자 변환 및 문자열 정제 (같은 카테고리로 처리될만한 것들을 모두 처리)
    df["Location"] = df["Location"].str.lower()
    # (see note)같은 문자열은 제거
    df["Diagnosis"] = (
        df["Diagnosis"]
        .str.lower()
        .replace("(invasion)", "invasion")
        .replace("melanoma, invasion", "melanoma invasion")
        .replace("(see note)", "")
    )
    df["Diagnosis"] = df["Diagnosis"].str.replace(
        "malignant melanoma (invasive)", "malignant melanoma, invasive"
    )
    df["Diagnosis"] = df["Diagnosis"].str.replace(
        "malignant melanoma, invasive \(see note\)",
        "malignant melanoma, invasive",
        regex=True,
    )
    df["Diagnosis"] = df["Diagnosis"].str.replace(
        "malignant melanoma, invasive, residual",
        "malignant melanoma, invasive, residual",
    )
    df["Diagnosis"] = df["Diagnosis"].str.replace(
        "malignant melanoma \(invasive\), residual",
        "malignant melanoma, invasive, residual",
        regex=True,
    )
    df["Diagnosis"] = df["Diagnosis"].str.replace(
        "malignant melanoma invasive", "malignant melanoma, invasive"
    )
    df["Diagnosis"] = df["Diagnosis"].str.replace(
        "melanoma in situ \(see note\)", "melanoma in situ", regex=True
    )
    df["Diagnosis"] = df["Diagnosis"].str.replace(
        "atypical melanocytic proliferative lesion,",
        "atypical melanocytic proliferative,",
    )

    # radial, vertical 과 radial and vertical은 같은 것으로 처리
    df["Growth phase"] = df["Growth phase"].replace(
        "radial, vertical", "radial and vertical"
    )

    df["Histologic subtype"] = df["Histologic subtype"].str.replace(
        "acaral lentiginous", "acral lentiginous"
    )
    df["Histologic subtype"] = df["Histologic subtype"].str.replace(
        "nodular melanoma", "nodular"
    )

    # Level III, Level 3 등으로 혼용되어서 일괄처리
    # 'Level of invasion'에서 로마 숫자를 찾아 아라비아 숫자로 변환하고 원래 문자열에 삽입
    df["Level of invasion"] = df["Level of invasion"].str.lower()
    df["Level of invasion"] = df["Level of invasion"].apply(replace_roman_numerals)
    df["Precursor lesion"] = df["Precursor lesion"].str.lower()

    # Size of tumor 처리
    df[["tumor_length", "tumor_width", "tumor_height"]] = (
        pd.DataFrame(df["Size of tumor"].apply(preprocess_size_of_tumor).tolist()) * 10
    )

    # # cm 단위의 크기를 mm 단위로 변환
    # df['tumor_length_mm'] = df['tumor_length'] * 10
    # df['tumor_width_mm'] = df['tumor_width'] * 10
    # df['tumor_height_mm'] = df['tumor_height'] * 10

    # 종양의 사이즈를 넓이와 부피로 추가
    df["Area of tumor"] = df["tumor_length"] * df["tumor_width"]
    df["Volume of tumor"] = df.apply(
        lambda row: (
            row["tumor_length"] * row["tumor_width"] * row["tumor_height"]
            if pd.notna(row["tumor_height"])
            else np.nan
        ),
        axis=1,
    )

    # 'Depth of invasion'과 'Breslow thickness' 열을 처리하기 위해 위에서 정의한 함수 사용
    df["Depth of invasion"] = df["Depth of invasion"].apply(extract_first_number)
    df["Breslow thickness"] = df["Breslow thickness"].apply(extract_first_number)

    # Mitosis 처리 ( <, > 등의 문자를 지우고 공백 처리)
    df["Mitosis"] = (
        df["Mitosis"]
        .str.replace("<", "")
        .replace(">", "")
        .replace(" HPF", "HPF")
        .replace(" HPF", "HPF")
        .replace("< ", "")
        .replace("> ", "")
    )
    # 불필요한 공백과 특수문자 제거
    df["Mitosis"] = (
        df["Mitosis"]
        .str.replace("<", "")
        .replace(">", "")
        .replace(" HPF", "HPF")
        .replace(" HPF", "")
        .replace("< ", "")
        .replace("> ", "")
        .replace(" /", "/")
        .replace("/ ", "/")
    )
    # 숫자만 추출
    df["Mitosis"] = df["Mitosis"].str.extract(r"(\d+/\d+HPF)")

    # Tumor cell type 정제
    df["Tumor cell type"] = df["Tumor cell type"].replace(
        ["epithelioid, spindle", "spindle and epithelioid"], "epithelioid and spindle"
    )

    # Lymph node 처리
    number_words = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ]
    number_mapping = {
        r"\b" + word + r"\b": str(i + 1) for i, word in enumerate(number_words)
    }
    df["Lymph node"] = df["Lymph node"].replace(number_mapping, regex=True)
    # None과 'none'을 같은 값으로 처리
    df["Lymph node"] = df["Lymph node"].replace("none", None)

    df["Recurrence"] = recurrence_column

    return df


def save_metadata(data, file_name: str):
    """
    data : 데이터
    file_name : 저장될 파일 이름

    ---
    경로는 함수 안에서 정의를 하고 있음.
    """
    processed_metadata_path = path_config["processed_dataset"]
    if not os.path.exists(processed_metadata_path):
        os.makedirs(processed_metadata_path, exist_ok=True)

    path_to_save = os.path.join(processed_metadata_path, file_name)

    with open(path_to_save, "w") as f:
        json.dump(data, f)
