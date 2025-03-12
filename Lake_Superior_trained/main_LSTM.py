import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from garage_LSTM import LSTModel
from garage_LSTM import generate_dataset, reframeDF, cal_moving_ave
from garage_LSTM import final_test, test_analysis, create_date_dic, generate_prediction_data

# Configuration class for LSTM model parameters
class config():
    def __init__(self):
        self.cycle_num = 12  # Number of historical time steps used in training
        self.cycle_add = 5  # Number of future time steps used in training
        self.day_num = 30  # Time interval (in days) between successive historical input data
        self.moving_length = 30  # Moving average window length
        self.is_mean = False  # Whether to use mean values for input data
        self.train_pro = 1  # Training data proportion (1 means full dataset used for training)
        self.input_size = 7  # Number of input features
        self.hidden_size = 512  # LSTM hidden layer size
        self.output_size = 1  # Output dimension
        self.batch_size = 256  # Training batch size
        self.num_layers = 4  # Number of LSTM layers
        self.path = './model/lstm_' + str(self.num_layers) + '.pt'  # Path to save the model
        self.intervel = 150  # Parameter to predict the prediction points
        self.start_year = 1981  # Starting year of the dataset
        self.total_year = 100  # Total years used for training (fake number)
        self.test_length = 730  # Length of test dataset (2 years)
        self.predict_long = 2  # Prediction length in years
        self.predict_year_start = 2022  # Starting year for predictions

# Initialize LSTM configuration
wl_p = config()

# Initialize the MinMaxScaler for data normalization
scaler = MinMaxScaler(feature_range=(0, 1))

# Dataset feature mapping:
# water_level (index 0), temp_lake (index 1), temp_land (index 2), 
# wspd_lake (index 3), wspd_land (index 4), lst (index 5), precipitation (index 6)
# Load the training dataset
dataset = generate_dataset('./dataset/finaldata.csv')
dataset_original = generate_dataset('./dataset/finaldata.csv')

# Load testing datasets for 2022 and 2023
dataset_1 = generate_dataset('./dataset/test_2022.csv')
dataset_1[:, 1] = dataset_1[:, 1] - 273.15  # Convert temperature from Kelvin to Celsius
dataset_1[:, 2] = dataset_1[:, 2] - 273.15  # Convert temperature from Kelvin to Celsius
dataset_2 = generate_dataset('./dataset/test_2023.csv')

# Combine datasets to create the full dataset
dataset_whole = np.concatenate((dataset, dataset_1))
dataset_whole_original = np.concatenate((dataset_whole, dataset_2))
dataset_whole = np.concatenate((dataset_whole, dataset_2))

# Normalize and compute the moving average for the datasets
dataset_whole = reframeDF(dataset_whole, scaler)
dataset_whole = cal_moving_ave(dataset_whole, wl_p.moving_length)
dataset = reframeDF(dataset, scaler)
dataset = cal_moving_ave(dataset, wl_p.moving_length)

# Initialize the LSTM model
model = LSTModel(wl_p.input_size, wl_p.hidden_size, wl_p.output_size, 
                 wl_p.batch_size, wl_p.cycle_num + wl_p.cycle_add, wl_p.num_layers)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()

# Load the trained LSTM model
model.load_state_dict(torch.load(wl_p.path))
model.eval()  # Set model to evaluation mode

# Perform final testing on the complete dataset
results, targets = final_test(dataset_whole, wl_p, model, criterion)

# Generate test dataset date range
date = pd.date_range('19810101', '20231231').strftime("%Y-%m-%d").tolist()
date_test, date_original = date[-wl_p.test_length:], date[:-wl_p.test_length]
dataset_test = dataset_whole_original[-wl_p.test_length:]

# Generate prediction dataset
date, dataset_original, dataset_new = generate_prediction_data(date_test, date_original, dataset_test, dataset_original, wl_p)
target = dataset_new[:, 0][-wl_p.test_length:]
dataset_new, dataset_daily_max, dataset_daily_min, max_dic, min_dic = create_date_dic(dataset_new, wl_p, date, dataset_original)
dataset_new = reframeDF(dataset_new, scaler)

# Perform test analysis
error, plot_targets, plot_preds = test_analysis(date, dataset_new, dataset_original, max_dic, min_dic, results, targets, wl_p, scaler)

# Save the prediction results
np.save('./results/lstm_' + str(wl_p.num_layers) + '_results.npy', plot_preds)

# Compute RMSE error
test_error = plot_preds - target
test_rmse = np.sqrt(sum(np.square(test_error)) / wl_p.test_length)
print(f"Test RMSE: {test_rmse:.4f}")