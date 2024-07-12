import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle

def driver_dataframe(df, driver):
    driver_df = df[df['PathOrder'] == 1]
    driver_df = driver_df[driver_df['Class'] == driver]
    driver_df.drop(columns=['Class', 'PathOrder', 'Unnamed: 0'], inplace=True)
    scaler = pickle.load(open('../models/scaler.pkl', 'rb'))
    Scaled_driver_df = scaler.transform(driver_df)
    driver_df = pd.DataFrame(Scaled_driver_df, columns=driver_df.columns)
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
    model.load_state_dict(torch.load('../models/complex_time_series_autoencoder_rolling_features.pkl', map_location=torch.device('cpu')))
    return model

# Function to predict loss
def predict_loss(model, data, sequence_length, criterion, threshold=0.03):
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