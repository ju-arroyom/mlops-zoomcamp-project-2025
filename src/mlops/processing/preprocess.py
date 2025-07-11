
import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, data:pd.DataFrame, target:str):
        self.data = data
        self.target = target

    def identify_categorical_encoded_vars(self, maxcat=8):
        self.categorical_vars = []
        for c in self.data.columns:
            unique_values = self.data[c].unique()
            if (len(unique_values) < maxcat) & (c!=self.target):
                self.categorical_vars.append(c)

    def identify_numerical_vars(self):
        numerical_vars = self.data.select_dtypes(include='number').columns.to_list()
        self.numerical_vars = [x for x in numerical_vars if x not in self.categorical_vars]

    def split_datasets(self):
        df_full_train, df_test = train_test_split(self.data, test_size=0.20, random_state=40)
        df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=40)

        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        y_train = df_train[self.target].values
        y_val = df_val[self.target].values

        del df_train[self.target]
        del df_val[self.target]

        for col in self.categorical_vars:
            df_train[col] = df_train[col].astype('category')
            df_val[col] = df_val[col].astype('category')
            df_test[col] = df_test[col].astype('category')
        
        self.full_df = df_full_train
        self.df_test = df_test

        return {
                'x_train': df_train,
                'y_train': y_train,
                'x_valid': df_val,
                'y_valid': y_val,}
    
    def build_datasets(self):
        self.identify_categorical_encoded_vars()
        self.identify_numerical_vars()
        self.data_dict = self.split_datasets()
        