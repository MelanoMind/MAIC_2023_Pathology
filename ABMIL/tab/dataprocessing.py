from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from fancyimpute import SoftImpute
import pandas as pd

# 타입 바꾸는 것을 그냥 클래스로 만들어버리자.

class TypeCastingForCategoricalColumns():
    def __new__(cls, dataFrame, category_columns, type_to_casting):
        for col in dataFrame.columns:
            if col in category_columns:
                dataFrame[col] = dataFrame[col].astype(type_to_casting)
        return dataFrame

class FillMissingColumns():
    def __init__(self, dataFrame, category_columns, numerical_columns):
        self.dataFrame = dataFrame
        self.category_columns = category_columns
        self.numerical_columns = numerical_columns
        
    def fill_numerical_columns(self, method=None):
        
        # method에 따라서 수치형 칼럼을 채우는 방법을 바꿀 수 있다.
        if method == 'means':
            for col in self.numerical_columns:
                if col in self.dataFrame.columns:
                    mean_value = self.dataFrame[col].mean()
                    self.dataFrame[col].fillna(mean_value, inplace=True)
            return self.dataFrame
        
        elif method == 'impute':
            id_cols = self.dataFrame['Patient_ID']
            recurrence_cols = self.dataFrame['Recurrence']
            self.dataFrame.drop(columns=['Patient_ID', 'Recurrence'], inplace=True)
            label_encoders = {}
            for col in self.category_columns:
                self.dataFrame[col] = self.dataFrame[col].fillna('nan').astype(str)
                self.dataFrame[col] = self.dataFrame[col].astype(str)
                if self.dataFrame[col].dtype == 'object':
                    le = LabelEncoder()
                    self.dataFrame[col] = le.fit_transform(self.dataFrame[col].astype(str))
                    label_encoders[col] = le
                
            soft_impute = SoftImpute(verbose=False)
            self.dataFrame[:] = soft_impute.fit_transform(self.dataFrame.values)

            for col, le in label_encoders.items():
                self.dataFrame[col] = le.inverse_transform(self.dataFrame[col].astype(int))
            
            # id_cols와 recurrence_cols가 DataFrame 형태인 경우 Series로 변환
            # tolist() 호출 제거
            self.dataFrame['Patient_ID'] = id_cols
            self.dataFrame['Recurrence'] = recurrence_cols

            columns_to_reorder = [col for col in self.dataFrame.columns if col not in ['Patient_ID', 'Recurrence']]

            self.dataFrame = self.dataFrame[['Patient_ID'] + columns_to_reorder + ['Recurrence']]
            return self.dataFrame
        elif method == 'None':
            return self.dataFrame
    
class TabularDataSet():
    def __init__(self, dataFrame, category_columns, numerical_columns, fill_numerical_method='means'):
        """
        In -
        dataframe : pd.DataFrame
        category_columns = List
        numerical_columns : List 
        fill_numerical_type : Optional['means', 'impute', 'None'], default: means
        """
        # df = self.typeCastingForOriginalDataFrame(dataFrame, category_columns)
        df = TypeCastingForCategoricalColumns(dataFrame, category_columns, str)
        self.numerical_columns = numerical_columns
        self.dataFrame = FillMissingColumns(df, category_columns, numerical_columns).fill_numerical_columns(method=fill_numerical_method)
        self.category_columns = category_columns
        self.category_dims = {}
        self.label_encoders = {}
        self.encoded_dataFrame = self.convertEncodedDataFrame(category_columns)
    """
    def typeCastingForOriginalDataFrame(self, dataFrame, category_columns):
        for col in category_columns:
            dataFrame[col] = dataFrame[col].fillna('nan').astype(str)
        return dataFrame
    """
    
    def convertEncodedDataFrame(self, category_columns):
        encoded_df = self.dataFrame.copy()
        for col in category_columns:
            encoded_df[col] = encoded_df[col].fillna('nan').astype(str)
            encoded_df[col] = encoded_df[col].astype(str)
            if encoded_df[col].dtype == 'object':
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                self.label_encoders[col] = le
                # 카테고리별 고유값의 개수 저장
                self.category_dims[col] = len(le.classes_)
        return encoded_df
    
class DatasetSplitter:
    def __init__(self, TabularDataSet):
        """
        TabularDataSet 객체를 입력으로 받는다.
        """
        self.tabular_data_set = TabularDataSet
        self.category_dims = self.tabular_data_set.category_dims
        self.category_columns = self.tabular_data_set.category_columns
        self.label_encoders = self.tabular_data_set.label_encoders

    def split(self):
        df = self.tabular_data_set.dataFrame
        train_data = df[df['Patient_ID'].str.contains('train', na=False)]
        test_data = df[df['Patient_ID'].str.contains('test', na=False)]
        train_data = train_data.drop(columns=['Patient_ID'])
        test_data = test_data.drop(columns=['Patient_ID'])

        return train_data, test_data
    
class GetIndicesForCategoricalColumn:
    def __init__(self, dataFrame, categorical_columns):
        self.dataFrame = dataFrame
        self.categorical_columns = categorical_columns

    def get_indices(self):
        indices = [self.dataFrame.columns.get_loc(col) for col in self.categorical_columns if col in self.dataFrame.columns]
        return indices 

class TransformDataToTabNet():
    def __init__(self, dataFrame, category_columns, numerical_columns, method):
        """
        Parameters
        --------------
        dataFrame : pd.DataFrame
        category_columns : List[str]
        numerical_columns : List[str]
        method: Optional[str]= means, impute, None
        """
        df = TabularDataSet(dataFrame, category_columns, numerical_columns, method).convertEncodedDataFrame(category_columns)
        test = TabularDataSet(df, category_columns, numerical_columns, 'None')
        test_splitter = DatasetSplitter(test)
        train_data, test_data = test_splitter.split()
        self.categorical_columns = test_splitter.category_columns
        self.categorical_dims = [v for v in test_splitter.category_dims.values()]
        self.categorical_indices = GetIndicesForCategoricalColumn(train_data, category_columns).get_indices()
        train_data = TypeCastingForCategoricalColumns(train_data, category_columns, int)
        test_data = TypeCastingForCategoricalColumns(test_data, category_columns, int)
        self.train = train_data.values
        self.test = test_data.values
    def get_data(self):
        return self.train, self.test, self.categorical_columns, self.categorical_dims, self.categorical_indices 