import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

data = pd.read_csv('../data/preprocessed/random_sample.csv')
test = pd.read_csv("../data/preprocessed/train_preprocessed.csv")

def driver_dataframe(df, driver):
    driver_df = df[df['PathOrder'] == 1]
    driver_df = driver_df[driver_df['Class'] == driver]
    driver_df.drop(columns=['Class', 'PathOrder', 'Unnamed: 0'], inplace=True)
    return driver_df

class ComplexTimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, sequence_length):
        super(ComplexTimeSeriesAutoencoder, self).__init__()

        # Define the CNN Encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Define the LSTM Encoder
        self.lstm_encoder = nn.LSTM(input_size=64, hidden_size=latent_dim, batch_first=True)

        # Define the LSTM Decoder
        self.lstm_decoder = nn.LSTM(input_size=latent_dim, hidden_size=64, batch_first=True)

        # Define the CNN Decoder
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=input_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Pass the input through the CNN Encoder
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, input_dim, sequence_length) for CNN
        x = self.cnn_encoder(x)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, sequence_length, num_channels) for LSTM

        # Pass through the LSTM Encoder
        x, _ = self.lstm_encoder(x)

        # Pass through the LSTM Decoder
        x, _ = self.lstm_decoder(x)

        # Pass through the CNN Decoder
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, num_channels, sequence_length) for CNN
        x = self.cnn_decoder(x)
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, sequence_length, input_dim)

        return x

# Define the dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx + self.sequence_length], dtype=torch.float)

# Function to load model and encoder
def load_model_and_encoder(input_dim, latent_dim, sequence_length):
    model = ComplexTimeSeriesAutoencoder(input_dim, latent_dim, sequence_length)
    model.load_state_dict(torch.load('../models/complex_time_series_autoencoder.pth', map_location=torch.device('cpu')))
    return model

# Function to predict loss
def predict_loss(model, data, sequence_length, criterion, threshold=11):
    dataset = TimeSeriesDataset(data.values, sequence_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    anomalies_cnt = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.float()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            if loss.item() > threshold:
                return True
            else:
                return False

attack_types = ['Abnormal', 'Normal']
risk_levels = {
    'Normal': 'Low',
    'Abnormal': 'High'
}

# Define page navigation variable
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Homepage
if st.session_state.page == 'home':
    st.title("Welcome to the Classification Demo")
    st.markdown("""
    This application provides two main functionalities:
    - Attack Classification
    - Driver Classification
    """)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Attack Model"):
            st.session_state.page = 'attack_model'
            st.experimental_rerun()
    with col2:
        if st.button("Go to Driver Classifier"):
            st.session_state.page = 'driver_classifier'
            st.experimental_rerun()

# Attack Model page
if st.session_state.page == 'attack_model':
    st.title('🛡️ Attack Classification Demo')
    st.markdown("""
    Welcome to the Attack Classification Demo! This tool allows you to explore and classify different types of network attacks using a pre-trained model. Select an attack sample from the dropdown menu below and see the model's prediction.
    """)

    attack_choice = st.selectbox('Select an attack sample:', attack_types)
    sample = data[data['flag'] == attack_choice].sample(1).squeeze()
    st.subheader('Sample Data:')
    st.dataframe(sample.to_frame().T)

    features = sample.drop(labels='flag').values.reshape(1, -1)
    prediction = predict_results(model, encoder, features)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Predicted Attack Type:')
        if prediction == sample['flag']:
            st.success(f'{prediction}')
        else:
            st.error(f'{prediction}')

    with col2:
        st.subheader('True Attack Type:')
        st.info(f'{sample["flag"]}')

    risk_level = risk_levels[prediction]
    if risk_level == 'High':
        st.error(f'Risk Level: {risk_level}')
    else:
        st.success(f'Risk Level: {risk_level}')

    if st.button("Back to Home"):
        st.session_state.page = 'home'
        st.experimental_rerun()

# Driver Classifier page
if st.session_state.page == 'driver_classifier':
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
