import numpy as np
import pandas as pd
import pickle

# Function to predict results
def predict_results(model, encoder, sample):
    result = model.predict(sample.reshape(1, -1))
    result = encoder.inverse_transform(result)
    if result == 'R':
        return 'Normal'
    else:
        return 'Abnormal'



with open('../models/label_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('../models/model.pkl', 'rb') as file:
    model = pickle.load(file)




attack_types = ['Abnormal', 'Normal']
risk_levels = {
    'Normal': 'Low',
    'Abnormal': 'High'
}