import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

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

def test_encode(df_test):
    picklefile_enc = open('../models/data-cleaning-models/cat_encoder.pkl', 'rb')
    cat_encoder = pickle.load(picklefile_enc)
    
    for i in cat_col:
        encoder = cat_encoder[i]
        df_test[i]= encoder.fit_transform(df_test[i])
        
    return df_test

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
    num_df = df.select_dtypes(include = [np.number])
    corr_mat = num_df.corr()
    corr_mat = corr_mat['SalePrice'].sort_values(ascending = False)
    poor_corr = corr_mat[(corr_mat.values < 0.35) & (corr_mat.values > -0.035)]
    to_remove = np.append(to_remove, poor_corr.index.to_list())
    return to_remove

def data_scale(df):
    numeric_col = df.select_dtypes(include = [np.number]).columns.to_list()
    num_col_x = numeric_col[:-1]
    num_col_y = numeric_col[-1]
    scaler_x = StandardScaler()
    scaled_x_df = scaler_x.fit_transform(df[num_col_x])
    scaler_y = StandardScaler()
    scaled_y_df = scaler_y.fit_transform(df[num_col_y].values.reshape(-1, 1))
    
    file1 = open('../models/data-cleaning-models/scaler_x.pkl', 'wb')
    pickle.dump(scaler_x, file1)
    
    file2 = open('../models/data-cleaning-models/scaler_y.pkl', 'wb')
    pickle.dump(scaler_y, file2)
    
    return num_col_x, num_col_y, scaled_x_df, scaled_y_df
    
def clean_train(df):
    
    # Removing the outliers from 
    Q1 = df.SalePrice.quantile(0.25)
    Q3 = df.SalePrice.quantile(0.75)
    IQR = Q3 - Q1
    bound = Q3 + 3 * IQR
    df.drop(df[df.SalePrice > bound].index, axis = 0, inplace = True)
    
    # multicollinearity
    to_remove = np.array(['Id','GarageCars','1stFlrSF','GrLivArea','FullBath'], dtype = object) # Observation from data visualization
    
    #poor correlation
    to_remove = np.append(to_remove, poor_corr_columns(df))
    
    # null handling
    to_remove = np.append(to_remove, null_columns(df))
    
    # Print to remove
    print(to_remove)
    
    # removing columns
    df.drop(to_remove, axis = 1, inplace = True)
    
    # Dropping null values
    df.dropna(inplace = True)
    
    # # categorical feature encoding
    # df = feature_encode(df)
    
    return df, to_remove

def clean_test(df_test, to_remove):
    df_test.drop(to_remove, axis = 1, inplace = True)
    
    picklefile_scale = open('../models/data-cleaning-models/scaler_x.pkl', 'rb')
    scaler_x = pickle.load(picklefile_scale)
    
    cat_col = df.select_dtypes(include = ['object']).columns.to_list()
    numeric_col = df_test.select_dtypes(include = [np.number]).columns.to_list()
    scaled_x_df_test = scaler_x.fit_transform(df_test[num_col_x])
    df_test[num_col_x]= scaled_x_df_test
    
    df_test.dropna(inplace = True)
    
    df_test = test_encode(df_test)
        
    return df_test
        
    