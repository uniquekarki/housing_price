import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def feature_encode(df):
    # categorical feature encoding
    cat_columns = df.select_dtypes(include = ['object']).columns.to_list()
    cat_encoder = {}
    for i in cat_columns:
        en = LabelEncoder()
        df[i] = en.fit_transform(df[i])
        cat_encoder[i] = en
    
    file = open('../models/data-cleaning-models/cat_encoder.pkl', 'wb')
    pickle.dump(cat_encoder, file)
    return df

def null_columns(df):
    # null values
    to_remove = []
    null_val = df.isnull().sum()
    null_val = null_val[null_val.values > df.shape[0]*0.2]
    to_remove = np.append(to_remove, null_val.index.to_list())
    return to_remove

def poor_corr_columns(df):
    # poor correlation from numeric data
    to_remove = []
    num_df = df.select_dtype(include = [np.number])
    corr_mat = num_df.corr()
    corr_mat = corr_mat.SalePrice.sort_values(ascending = False)
    poor_corr = corr_mat[(corr_mat.values < 0.35) & (corr_mat.values > -0.03)]
    to_remove = np.append(to_remove, poor_corr.index.to_list())
    return to_remove

def data_scale(df):
    numeric_col = df.select_dtypes(includes = [np.number]).columns.to_list()
    num_col_x = numeric_col[:-1]
    num_col_y = numeric_col[-1]
    scaler_x = StandardScaler()
    scaled_x_df = scaler_x.fit_transform(df[num_col_x])
    scaler_y = StandardScaler()
    scaled_y_df = scaler_y.fit_tramsform(df[num_col_y])
    
    file1 = open('../models/data-cleaning-models/scaler_x.pkl', 'wb')
    pickle.dump(cat_encoder, file1)
    
    file2 = open('../models/data-cleaning-models/scaler_y.pkl', 'wb')
    pickle.dump(cat_encoder, file2)
    
    return scaled_x_df, scaled_y_df
    
def clean_train(df):
    
    # Removing the outliers from 
    Q1 = df.SalePrice.quantile(0.25)
    Q3 = df.SalePrice.quamtile(0.75)
    IQR = Q3 - Q1
    bound = Q3 + 3 * IQR
    df.drop(df[df.Saleprice > bound].index, axis = 0, inplace = True)
    
    # multicollinearity
    to_remove = ['GarageCars','1stFlrSF','GrLivArea','FullBath'] # Observation from data visualization
    
    #poor correlation
    to_remove = np.append(to_remove, poor_corr_columns(df))
    
    # null handling
    to_remove = np.append(to_remove, null_columns(df))
    
    # removing columns
    df.drop(to_remove, axis = 1, inplace = True)
    
    # Dropping null values
    df.dropna(inplace = True)
    
    # Scaling data
    numeric_col = df.select_dtypes(includes = [np.number]).columns.to_list()
    num_col_x = numeric_col[:-1]
    num_col_y = numeric_col[-1]
    df[num_col_x], df[num_col_y] = data_scale(df)
    
    return df