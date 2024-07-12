import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
class ComplexTimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, sequence_lengt):
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
def load_model_and_encoder():

    model = torch.load('./models/complex_time_series_autoencoder.pth', map_location=torch.device('cpu'))
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
                anomalies_cnt += 1

    return anomalies_cnt / len(dataset)

# Function to predict results
def predict_results(model, encoder, sample):
    result = model.predict(sample.reshape(1, -1))
    result = encoder.inverse_transform(result)
    if result == 'R':
        return 'Normal'
    else:
        return 'Abnormal'

# Load encoder and model for attack model
with open('./models/label_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('./models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load data
data = pd.read_csv('./data/random_sample.csv')
test = pd.read_csv("./data/df_train_preprocessed.csv")

# Function to create a DataFrame for a specific driver with consecutive rows
def driver_dataframe(df, driver, seq_length):
    driver_df = df[df['Class'] == driver]
    driver_df = driver_df[driver_df['PathOrder'] == 1]
    driver_df.drop(columns=['Class', 'PathOrder'], inplace=True)
    # Ensure the sample consists of consecutive rows
    if len(driver_df) >= seq_length:
        start_idx = np.random.randint(0, len(driver_df) - seq_length + 1)
        driver_sample = driver_df.iloc[start_idx:start_idx + seq_length]
    else:
        driver_sample = driver_df
    return driver_sample

# Streamlit sidebar navigation
st.sidebar.header('Navigation')
page = st.sidebar.selectbox('Select Page', ['Attack Model', 'Driver Classifier'])

# Attack Model page
if page == 'Attack Model':
    st.title('🛡️ Attack Classification Demo')
    st.markdown("""
    Welcome to the Attack Classification Demo! This tool allows you to explore and classify different types of network attacks using a pre-trained model. Select an attack sample from the dropdown menu below and see the model's prediction.
    """)

    attack_types = ['Abnormal', 'Normal']
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

    risk_levels = {'Normal': 'Low', 'Abnormal': 'High'}
    risk_level = risk_levels[prediction]
    if risk_level == 'High':
        st.error(f'Risk Level: {risk_level}')
    else:
        st.success(f'Risk Level: {risk_level}')

else:
    st.title('Driver Classifier Model')

    options = list(test['Class'].unique())
    selected_option = st.selectbox('Choose an option:', options)
    st.write('You selected:', selected_option)

    seq_length = 60
    driver_sample = driver_dataframe(test, selected_option, seq_length)
    
    st.subheader('Driver Sample Data:')
    st.dataframe(driver_sample)

    # Load the model and encoder
    model = load_model_and_encoder()
    criterion = torch.nn.MSELoss()

    # Predict loss
    loss = predict_loss(model, driver_sample, seq_length, criterion)
    
    st.subheader('Predicted Loss:')
    st.write(f'{loss}')
