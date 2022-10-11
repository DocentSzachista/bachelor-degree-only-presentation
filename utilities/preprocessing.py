import pandas as pd 

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer



def find_columns_with_null_values(df: pd.DataFrame, treshold = 0.85):
    """
        Find columns, which have null values
        ===============

        ### Description:
         
         Finds columns which have null values in it and later checks if theres any 
         of the columns which have more null values than given treshold
        
        ### Params: 
         - df (DataFrame): Our dataset with data
         - threshold (float) default = 0.85: treshold on which we should stop considering column as usable
        
        ### Returns:
         - list of column names with null values
         - list of column names which cells has more null values than given threshold 
    """
    null_columns_with_threshold  = [col for col in df.columns if df[col].isnull().sum() > treshold * df.size]
    null_columns_in_general = [col for col in df.columns if df[col].isnull().any() > 0]
    print(null_columns_with_threshold, null_columns_in_general)
    return null_columns_in_general, null_columns_with_threshold
    
def encode_object_values(df: pd.DataFrame, obj_cols):
    """
        ## Encode values of object column to int values
        
        -----------

        ### Descritpion
        Use OrdinalEncoder class to transform string data in given columns into numerical values, 
        so they can be better used in dataset training proccess.
        
        ### Params:
         - df (DataFrame): Our dataset with data
         - obj_cols (list): list with column names that are to convert
        
        ### Returns:
         - dataframe object with changed columns
    """
    encoder = OrdinalEncoder()
    df[obj_cols] = encoder.fit_transform(df[obj_cols])
    return df


def impute_columns(df: pd.DataFrame, columns_to_impute, impute_strategy = "mean"):
    columns = df.columns
    imputer  = SimpleImputer(strategy=impute_strategy )
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
    df.columns = columns
    return df