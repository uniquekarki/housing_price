import pandas as pd
import numpy as np

from dataCleaning import clean_train, data_scale, train_encode, clean_test
from modelTraining import model_test

if __name__ == '__main__':
    df_train = pd.read_csv('../data/raw/train.csv')
    df_test = pd.read_csv('../data/raw/test.csv')
    
    df_train, to_remove = clean_train(df_train)
    
    num_col_x, num_col_y, scaled_x_df, scaled_y_df = data_scale(df_train)
    df_train[num_col_x],df_train[num_col_y] = scaled_x_df, scaled_y_df
    
    df_train = train_encode(df_train)
    
    df_test = clean_test(df_test, to_remove)
    
    pred = model_test(df_test)
    
    df_train.to_csv("../data/processed/clean_train_data.csv", index = False)
    df_train.to_csv("../data/processed/clean_test_data.csv", index = False)
    pred.to_csv('../data/prediction/predictions.csv', index=False)
    
    print('Completed Successfully!')
    
    
    