import torch
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from garage_single_Transformer import cal_moving_ave, generate_prediction_data
from garage_single_Transformer import generate_dataset, reframeDF, create_date_dic, final_test, test_analysis
from model_single_transformer import Transformer


# Configuration class for Transformer model training and testing
class config():
    def __init__(self):
        self.cycle_num = 12           # Number of historical time steps used in training
        self.cycle_add = 5            # Number of future time steps used in training
        self.day_num = 150            # Parameter to predict the prediction points
        self.moving_length = 30       # Moving average window size
        self.interval = 30            # Time interval (in days) between successive historical input data
        self.is_mean = False          # Whether to use mean values
        self.is_train = True          # Flag for training mode
        self.train_pro = 1            # Proportion of training data
        self.input_size = 7           # Number of input features
        self.batch_size = 256         # Batch size for training
        self.path = './model/single_transformer.pt'  # Model save path
        self.data_path = './dataset/finaldata.csv'   # Path to main dataset
        self.test_path_1 = './dataset/test_2022.csv' # Path to 2022 test dataset
        self.test_path_2 = './dataset/test_2023.csv' # Path to 2023 test dataset
        self.test_length = 365 * 2    # Length of testing dataset
        self.start_year = 1981        # Start year of dataset
        self.total_year = 100         # Total years used for training (fake number)
        self.predict_long = 2         # Prediction length (years)
        self.predict_year_start = 2022  # Year to start prediction


# Initialize configuration
wl_p = config()

# Initialize data scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Dataset feature mapping:
# water_level (index 0), temp_lake (index 1), temp_land (index 2), 
# wspd_lake (index 3), wspd_land (index 4), lst (index 5), precipitation (index 6)
# Load and preprocess testing datasets (2022 and 2023)
date_test_1, dataset_test_1 = generate_dataset(wl_p.test_path_1)
dataset_test_1[:, 1] = dataset_test_1[:, 1] - 273.15  # Convert temperature from Kelvin to Celsius
dataset_test_1[:, 2] = dataset_test_1[:, 2] - 273.15
date_test_2, dataset_test_2 = generate_dataset(wl_p.test_path_2)
date_test = np.concatenate((date_test_1, date_test_2), axis=0)
dataset_test = np.concatenate((dataset_test_1, dataset_test_2), axis=0)

# Load and preprocess the training dataset
date_original, dataset_original = generate_dataset(wl_p.data_path)
date, dataset = generate_dataset(wl_p.data_path)
dataset = cal_moving_ave(dataset, wl_p.moving_length)  # Apply moving average
dataset = reframeDF(dataset, scaler)  # Normalize dataset
print(dataset.shape, len(date))  # Print dataset shape and length

# Combine dataset with test data
dataset_whole = np.concatenate((dataset, dataset_test_1))
dataset_whole_original = np.concatenate((dataset_whole, dataset_test_2))

# Initialize Transformer model, loss function, and optimizer
transformer = Transformer()
criterion = torch.nn.MSELoss()         

# Load pre-trained Transformer model
transformer.load_state_dict(torch.load(wl_p.path))
transformer.eval()
wl_p.is_train = False  # Set to evaluation mode

# Generate prediction dataset
date_1, dataset_original_1, dataset_new_1 = generate_prediction_data(
    date_test, date_original, dataset_test, dataset_original, wl_p
)
dataset_new_2 = reframeDF(dataset_new_1, scaler)
results, targets = final_test(dataset_new_2, wl_p, transformer, criterion)

# Define date range for test dataset
date = pd.date_range('19810101', '20231231').strftime("%Y-%m-%d").tolist()
date_test, date_original = date[-wl_p.test_length:], date[:-wl_p.test_length]
dataset_test = dataset_whole_original[-wl_p.test_length:]

# Generate prediction results for test analysis
date_3, dataset_original_3, dataset_new_3 = generate_prediction_data(
    date_test, date_original, dataset_test, dataset_original, wl_p
)
target = dataset_new_3[:, 0][-wl_p.test_length:]

dataset_new_4, dataset_daily_max, dataset_daily_min, max_dic, min_dic = create_date_dic(
    dataset_new_3, wl_p, date_3, dataset_original_3
)
dataset_new_5 = reframeDF(dataset_new_4, scaler)
# Evaluate the Transformer model performance on testing dataset
error, plot_targets, plot_preds = test_analysis(
    date, dataset_new_5, dataset_original_3, max_dic, min_dic, results, targets, wl_p, scaler
)

# Save prediction results
np.save('./results/transformer_results.npy', plot_preds)

# Compute Root Mean Square Error (RMSE)
test_error = np.array(plot_preds) - np.array(target)
test_rmse = np.sqrt(sum(np.square(test_error)) / wl_p.test_length)
print(test_rmse)  # Print final test RMSE