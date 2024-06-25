import streamlit as st
import numpy as np
import pandas as pd
import pickle  

def predict_results(model, encodedr, sample):
    result = model.predict(sample.reshape(1,-1))
    result = encoder.inverse_transform(result)
    if result == 'R':
        return 'Normal'
    else:
        return 'Abnormal'
    
with open('label_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

data = pd.read_csv('random_sample.csv')

attack_types = ['dos', 'rpm', 'fuzzy', 'gear', 'r']
risk_levels = {
    'dos': 'High',
    'rpm': 'Medium',
    'fuzzy': 'Low',
    'gear': 'Medium',
    'r': 'High'
}


st.set_page_config(page_title="Attack Classification Demo", page_icon=":shield:", layout="wide")

st.title('🛡️ Attack Classification Demo')
st.markdown("""
Welcome to the Attack Classification Demo! This tool allows you to explore and classify different types of network attacks using a pre-trained model. Select an attack sample from the dropdown menu below and see the model's prediction.
""")

attack_choice = st.selectbox('Select an attack sample:', attack_types)
sample = data[data['flag']==attack_types].sample()
sample = sample.squeeze()


st.subheader('Sample Data:')
st.dataframe(sample.to_frame().T)

features = sample.iloc[:-1].values.reshape(1, -1)  


prediction = predict_results(model, encoder, features)

col1, col2 = st.columns(2)
with col1:
    st.subheader('Predicted Attack Type:')
    if prediction == sample['label']:
        st.success(f'{prediction}')
    else:
        st.error(f'{prediction}')

with col2:
    st.subheader('True Attack Type:')
    st.info(f'{sample["label"]}')

risk_level = risk_levels[prediction]
if risk_level == 'High':
    st.error(f'Risk Level: {risk_level}')
elif risk_level == 'Medium':
    st.warning(f'Risk Level: {risk_level}')
else:
    st.success(f'Risk Level: {risk_level}')

