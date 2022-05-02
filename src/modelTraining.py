import pandas as pd
import numpy as np
from pycaret.regression import *
import pickle

def model_test(df_test):
    model = load_model('../models/ml-models/final_gbr')
    
    picklefile_scale = open('../models/data-cleaning-models/scaler_y.pkl', 'rb')
    scaler_y = pickle.load(picklefile_scale)
    
    predictions = predict_model(model, data = df_test)
    
    pred = pd.DataFrame({
        "SalePrice": scaler_y.inverse_transform(predictions['Label'])
    })
    return pred