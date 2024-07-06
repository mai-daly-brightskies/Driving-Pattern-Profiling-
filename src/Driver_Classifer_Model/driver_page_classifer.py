import streamlit as st
import numpy as np
import pandas as pd
from Driver_Classifer_Model.helper_functions import *

def driver_model():

    test = pd.read_csv("../data/preprocessed/train_preprocessed.csv")

    st.title('Driver Classifier Model')

    options = list(test['Class'].unique())
    selected_option = st.selectbox('Choose an option:', options)
    st.write('You selected:', selected_option)

    seq_length = 60
    driver = driver_dataframe(test, selected_option)

    counter = 0
    anomalies = [0] * 5
    is_anomaly = False
    while not is_anomaly and ((counter + seq_length) < len(driver)):
        sample_driver = driver.iloc[counter:counter + seq_length, :]
        model = load_model_and_encoder(sample_driver.shape[1], 20, 60)
        criterion = nn.MSELoss()
        loss_pred = predict_loss(model, sample_driver, 60, criterion, threshold=5.58727)
        anomalies[counter % 5] = loss_pred
        is_anomaly = np.array(anomalies).all()
        counter += 1

    st.subheader('Predicted Anomalies:')
    st.write(f'Anomaly: {is_anomaly}')
    st.write(f"Detected the anamolies after: {120 + counter} seconds" )

    if st.button("Back to Home"):
        st.session_state.page = 'home'
        st.experimental_rerun()

