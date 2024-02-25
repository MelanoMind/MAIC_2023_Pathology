import pandas as pd
import numpy as np
import re
import json

def extract_first_number(text):
    # 문자열에서 첫 번째로 발견되는 숫자를 추출하는 함수입니다.
    # re.findall은 일치하는 모든 것을 찾아 리스트로 반환합니다.
    # 리스트가 비어있지 않으면 첫 번째 요소를 실수로 변환하여 반환합니다.
    numbers = re.findall(r"[\d.]+", str(text))
    return float(numbers[0]) if numbers else None

def roman_to_arabic(roman):
    roman_numerals = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
    return roman_numerals.get(roman, roman)

def preprocess_size_of_tumor(value):
    # NaN 확인
    if pd.isna(value):
        return [np.nan, np.nan, np.nan]
    
    # 'at least' or 'about' 같은 문자열을 제거함
    value = re.sub(r'\(.*?\)|at least|about', '', value)
    
    # 앞에 나온 최대 세 개의 숫자만 추출
    numbers = re.findall(r'[\d.]+', value)
    return [float(num) if num else np.nan for num in numbers[:3]]

def data_cleaning(df):
    # Recurrence 열의 위치 변경
    recurrence_column = df.pop('Recurrence')
    
    # 소문자 변환 및 문자열 정제 (같은 카테고리로 처리될만한 것들을 모두 처리)
    df['Location'] = df['Location'].str.lower()
    #(see note)같은 문자열은 제거
    df['Diagnosis'] = df['Diagnosis'].str.lower().replace('(invasion)', 'invasion').replace('melanoma, invasion', 'melanoma invasion').replace('(see note)', '')
    df['Diagnosis'] = df['Diagnosis'].str.replace('malignant melanoma (invasive)', 'malignant melanoma, invasive')
    df['Diagnosis'] = df['Diagnosis'].str.replace('malignant melanoma, invasive \(see note\)', 'malignant melanoma, invasive', regex=True)
    df['Diagnosis'] = df['Diagnosis'].str.replace('malignant melanoma, invasive, residual', 'malignant melanoma, invasive, residual')
    df['Diagnosis'] = df['Diagnosis'].str.replace('malignant melanoma \(invasive\), residual', 'malignant melanoma, invasive, residual', regex=True)
    df['Diagnosis'] = df['Diagnosis'].str.replace('malignant melanoma invasive', 'malignant melanoma, invasive')
    df['Diagnosis'] = df['Diagnosis'].str.replace('melanoma in situ \(see note\)', 'melanoma in situ', regex=True)
    df['Diagnosis'] = df['Diagnosis'].str.replace('atypical melanocytic proliferative lesion,', 'atypical melanocytic proliferative,')

    # radial, vertical 과 radial and vertical은 같은 것으로 처리
    df['Growth phase'] = df['Growth phase'].replace('radial, vertical', 'radial and vertical')
    
    df['Histologic subtype'] = df['Histologic subtype'].str.replace('acaral lentiginous', 'acral lentiginous')
    df['Histologic subtype'] = df['Histologic subtype'].str.replace('nodular melanoma', 'nodular')
    
    # Level III, Level 3 등으로 혼용되어서 일괄처리
    # 'Level of invasion'에서 로마 숫자를 찾아 아라비아 숫자로 변환하고 원래 문자열에 삽입
    df['Level of invasion'] = df['Level of invasion'].str.lower()
    def replace_roman_numerals(text):
        if pd.isna(text):
            return text
        # 로마 숫자와 해당 위치 찾기 (대문자 및 소문자 모두 고려)
        roman_numeral_match = re.finditer(r'\b(level [ivxlcdm]+)\b', text, re.IGNORECASE)
        for match in roman_numeral_match:
            roman_numeral = match.group(1)
            # 대문자로 변환하여 로마 숫자를 아라비아 숫자로 변환
            arabic_numeral = roman_to_arabic(roman_numeral.split()[1].upper())
            # 원래 문자열에서 로마 숫자를 아라비아 숫자로 교체
            text = text.replace(roman_numeral, f"level {arabic_numeral}", 1)
        return text
    df['Level of invasion'] = df['Level of invasion'].apply(replace_roman_numerals)

    df['Precursor lesion'] = df['Precursor lesion'].str.lower()

    # Size of tumor 처리
    df[['tumor_length', 'tumor_width', 'tumor_height']] = pd.DataFrame(df['Size of tumor'].apply(preprocess_size_of_tumor).tolist()) * 10
    
    # # cm 단위의 크기를 mm 단위로 변환
    # df['tumor_length_mm'] = df['tumor_length'] * 10
    # df['tumor_width_mm'] = df['tumor_width'] * 10
    # df['tumor_height_mm'] = df['tumor_height'] * 10
    
    # 종양의 사이즈를 넓이와 부피로 추가
    df['Area of tumor'] = df['tumor_length'] * df['tumor_width']
    df['Volume of tumor'] = df.apply(lambda row: row['tumor_length'] * row['tumor_width'] * row['tumor_height'] if pd.notna(row['tumor_height']) else np.nan, axis=1)
    
    # 'Depth of invasion'과 'Breslow thickness' 열을 처리하기 위해 위에서 정의한 함수 사용
    df['Depth of invasion'] = df['Depth of invasion'].apply(extract_first_number)
    df['Breslow thickness'] = df['Breslow thickness'].apply(extract_first_number)
    
    # Mitosis 처리 ( <, > 등의 문자를 지우고 공백 처리)
    df['Mitosis'] = df['Mitosis'].str.replace('<', '').replace('>', '').replace(' HPF', 'HPF').replace(' HPF', 'HPF').replace('< ', '').replace('> ','')
    # 불필요한 공백과 특수문자 제거
    df['Mitosis'] = df['Mitosis'].str.replace('<', '').replace('>', '').replace(' HPF', 'HPF').replace(' HPF', '').replace('< ', '').replace('> ','').replace(' /', '/').replace('/ ', '/')
    # # 숫자만 추출
    df['Mitosis'] = df['Mitosis'].str.extract(r'(\d+/\d+HPF)')

    # Tumor cell type 정제
    df['Tumor cell type'] = df['Tumor cell type'].replace(['epithelioid, spindle', 'spindle and epithelioid'], 'epithelioid and spindle')

    # Lymph node 처리
    number_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 
                    'eighteen', 'nineteen', 'twenty']
    number_mapping = {r'\b' + word + r'\b': str(i+1) for i, word in enumerate(number_words)}
    df['Lymph node'] = df['Lymph node'].replace(number_mapping, regex=True)
    # None과 'none'을 같은 값으로 처리
    df['Lymph node'] = df['Lymph node'].replace('none', None)
    
    df['Recurrence'] = recurrence_column

    return df

def paitent_comparison(df):
    """
    Input : df (원본 DataFrame)
    Output : 한 명의 환자에 대해 여러 Row가 있을텐데, 그 Row가 모두 동일한지 확인 (하나로 동일시해도 되는지 파악하기 위함)
    """
    # 'Slide_name' 칼럼이 있는지 확인하고 비교를 위해 임시로 제거
    if 'Slide_name' in df.columns:
        df_comparison = df.drop(columns='Slide_name')
    else:
        df_comparison = df.copy()

    # 각 Patient_ID별로 행들을 저장할 딕셔너리 생성
    patient_rows = {}

    # 각 행을 반복하며 해당 Patient_ID의 리스트에 추가
    for index, row in df_comparison.iterrows():
        patient_id = row['Patient_ID']
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

def select_row_with_least_nans(group):
    # 각 Patient_ID에 대해 가장 NaN이 적은 행을 반환하는 함수를 정의
    return group.loc[group.isnull().sum(axis=1).idxmin()]


def select_row_based_on_filter(group, patient_comparison_results):
    # patient_comparison_results 딕셔너리를 활용하여 각 Patient_ID에 대한 조건을 설정합니다.
    # False이면 가장 NaN이 적은 행을, True이면 첫 번째 행을 선택합니다.
    pid = group.name
    if patient_comparison_results.get(pid, True):
        return group.iloc[0]
    else:
        return select_row_with_least_nans(group)


def main():
    # 원본 train, test data_set을 concatenate 시키기 (Recurrence, Date_of recurrence 등을 제외하고 추후에 결측치를 채울 것)
    # 이때, 어차피 train dataset, test dataset은 각종 칼럼에 결측치가 있으므로 서로 관계가 있음을 가정하고 결측치를 채울 예정
    train_dataset_table_path = 'dataset/train_dataset.csv'
    test_dataset_table_path = 'dataset/test_public_dataset.csv'
    train_df = pd.read_csv(train_dataset_table_path)
    test_df = pd.read_csv(test_dataset_table_path)
    df = pd.concat([train_df, test_df], axis=0)

    # 1차 전처리 (data cleaning)
    df = data_cleaning(df)

    patient_comparison_results = paitent_comparison(df)

    # 그룹화를 통해 각 Patient_ID 별로 'Slide_name' 리스트를 만들고, 나머지 열에 대해서는 조건에 따라 행을 선택 (2차 전처리 - 여러 행으로 되어 있는 테이블을 1 Patient 1 Row로 변경)
    df_grouped = df.groupby('Patient_ID').apply(lambda group: select_row_based_on_filter(group, patient_comparison_results))

    # 'Slide_name' 열의 개수를 통해 1 Patient 1 Slide_List로 변경
    df_grouped['Slide_List'] = df.groupby('Patient_ID')['Slide_name'].apply(list)

    # 'Slide_name' 열을 제거하고, 인덱스를 리셋
    df_grouped = df_grouped.drop(columns=['Slide_name']).reset_index(drop=True)

    df_grouped['Date_of_diagnosis'] = pd.to_datetime(df_grouped['Date_of_diagnosis'])
    df_grouped['Date_of_recurrence'] = pd.to_datetime(df_grouped['Date_of_recurrence'])
    df_grouped['time_to_recurrence'] = (df_grouped['Date_of_recurrence'] - df_grouped['Date_of_diagnosis']).dt.days

    # Categorizing the numeric columns into 10 equal groups and creating new columns for each
    numeric_columns = ['tumor_length', 'tumor_width', 'tumor_height', 'Area of tumor', 'Volume of tumor', 'Breslow thickness']
    for col in numeric_columns:
        df_grouped[f'{col}_category'] = pd.qcut(df_grouped[col], q=10, labels=range(1, 11))
    # 1. Remove 'Slide_List' column but create a new 'Slide_Count' column based on its length
    # 'Date_of_diagnosis', 'Date_of_recurrence'
    df_grouped['Slide_Count'] = df_grouped['Slide_List'].apply(lambda x: len(str(x).split(', ')))
    df_grouped.drop('Date_of_diagnosis', axis=1, inplace=True)
    df_grouped.drop('Date_of_recurrence', axis=1, inplace=True)

    # Patient_ID 별로 Slide_name을 리스트로 모으는 딕셔너리 생성
    slide_list_dict = df.groupby('Patient_ID')['Slide_name'].apply(list).to_dict()
    with open('slide_list_dict.json', 'w') as f:
        json.dump(slide_list_dict, f)

    # Patient_ID 별로 Recurrence를 리스트로 모으는 딕셔너리 생성
    recurrence_dict = df.groupby('Patient_ID')['Recurrence'].apply(list).to_dict()
    with open('recurrence_dict.json', 'w') as f:
        json.dump(recurrence_dict, f)

    df_grouped.drop('Slide_List', axis=1, inplace=True)
    df_grouped.drop('Size of tumor', axis=1, inplace=True)


    df_grouped['Mitosis_Value'] = df_grouped['Mitosis'].str.extract(r'(\d+)/10HPF').astype(float)

    df_grouped.drop('Mitosis', axis=1, inplace=True)

    df_grouped['Mitosis_category'] = pd.cut(df_grouped['Mitosis_Value'], bins=20, labels=False)


    df_grouped['Surgical margin'] = df_grouped['Surgical margin'].apply(lambda record: 
        None if pd.isna(record) 
        else 'Free from Tumor' if 'free from tumor' in record 
        else 'Involved by Tumor' if 'involved by tumor' in record or 'involvement of deep margin by tumor' in record 
        else 'Other'
    )

    recurrence = df_grouped.pop('Recurrence')
    df_grouped['Recurrence'] = recurrence

    # Categorical columns (as identified earlier)
    category_columns = [
        'Location', 'Diagnosis', 'Growth phase', 'Level of invasion', 
        'Histologic subtype', 'Tumor cell type', 'Surgical margin', 'Lymph node', 
        'Precursor lesion', 'tumor_length_category', 'tumor_width_category',
        'tumor_height_category', 'Area of tumor_category',
        'Volume of tumor_category', 'Breslow thickness_category',
        'Mitosis_category'
    ]

    # Discrete columns
    discrete_columns = ['Slide_Count']

    # Continuous columns (remaining ones except 'Patient_ID')
    continuous_columns = [col for col in df_grouped.columns if col not in category_columns + discrete_columns + ['Patient_ID']]

    # 타입 맞춰주기
    # category_columns의 각 열을 str로 변환
    # print(df_grouped.columns)
    # print(set(category_columns))
    for col in df_grouped.columns:
        if col in set(category_columns):
            df_grouped[col] = df_grouped[col].astype(str)

    # New column order: Patient_ID -> Categorical -> Discrete -> Continuous
    new_column_order = ['Patient_ID'] + category_columns + discrete_columns + continuous_columns

    # Reordering the DataFrame
    df_grouped = df_grouped[new_column_order]
    df_grouped.drop(['time_to_recurrence', 'Slide_Count'], axis=1, inplace=True)
    
    

    # 결과 저장
    df_grouped.to_csv("dataset.csv", encoding='cp949', index=False)

if __name__ == '__main__':
    main()