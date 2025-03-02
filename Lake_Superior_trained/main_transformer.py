import torch
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from garage_Transformer import cal_moving_ave, reframeDF, create_date_dic, final_test
from garage_Transformer import generate_dataset, final_test, test_analysis, generate_prediction_data
from model_base import Transformer_half_base
from model_modify import Transformer_half_modify
from model_weighted import auto_weighted
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# Configuration class for setting model parameters
class config():
    def __init__(self):
        self.cycle_num_before = 12  # Number of steps before the current time (historical time steps)
        self.cycle_num_later = 5  # Number of steps after the current time (future time steps)
        self.moving_length = 30  # Moving average window length
        self.interval_1 = 30  # Interval for historical dataset splitting
        self.interval_2 = 30  # Another interval for future dataset splitting
        self.is_mean = True  # Whether to use moving-average operation
        self.is_train = True  # Flag indicating training mode
        self.train_pro = 1  # Proportion of training data
        self.input_size = 7  # Input feature size
        self.batch_size = 256  # Batch size for training
        self.output_size = 1 # Output size
        
        # Model file paths
        self.path_base = './model/transformer_base_6.pt'
        self.path_modify = './model/transformer_modify_6.pt'
        self.path_weight = './model/transformer_aw_6.pt'
        
        # Data file paths
        self.data_path = './dataset/finaldata.csv'
        self.test_path_1 = './dataset/test_2022.csv'
        self.test_path_2 = './dataset/test_2023.csv'
        
        self.test_length = 365 * 2  # Length of test data (two years)
        self.start_year = 1981  # Start year for training data
        self.total_year = 100  # Total years in dataset (fake)
        self.predict_year_start = 2022  # Start year for prediction
        self.plot_start_year = self.predict_year_start - 1  # Start year for visualization
        self.predict_long = 2  # Number of years for prediction

# Initialize configuration
wl_p = config()

# Initialize scaler for normalization
scaler = MinMaxScaler(feature_range=(0, 1))

# Load original dataset and preprocess it
date_original, dataset_original = generate_dataset(wl_p.data_path)
date, dataset = generate_dataset(wl_p.data_path)
np.save('./results/dataset_train.npy', dataset)  # Save training dataset

# Convert date strings to datetime objects
date = [datetime.strptime(d, '%Y-%m-%d').date() for d in date]
print(dataset.shape)

# Apply moving average filtering
dataset = cal_moving_ave(dataset, wl_p.moving_length)
print(dataset.shape)

# Normalize and reframe dataset
dataset = reframeDF(dataset, scaler)

# Load test datasets
date_test_1, dataset_test_1 = generate_dataset(wl_p.test_path_1)
dataset_test_1[:, 1] = dataset_test_1[:, 1] - 273.15  # Convert temperature from Kelvin to Celsius
dataset_test_1[:, 2] = dataset_test_1[:, 2] - 273.15

date_test_2, dataset_test_2 = generate_dataset(wl_p.test_path_2)

# Concatenate test datasets
date_test = np.concatenate((date_test_1, date_test_2), axis=0)
dataset_test = np.concatenate((dataset_test_1, dataset_test_2), axis=0)
np.save('./results/dataset_test.npy', dataset_test)  # Save test dataset

# Load Transformer models
transformer_base = Transformer_half_base(wl_p)
transformer_modify = Transformer_half_modify(wl_p)
aw = auto_weighted(wl_p)
criterion = torch.nn.MSELoss()  # Define loss function

# Load pre-trained model weights
transformer_base.load_state_dict(torch.load(wl_p.path_base))
transformer_base.eval()

transformer_modify.load_state_dict(torch.load(wl_p.path_modify))
transformer_modify.eval()

aw.load_state_dict(torch.load(wl_p.path_weight))
aw.eval()

# Switch to evaluation mode
wl_p.is_train = False

# Generate prediction dataset
date, dataset_original, dataset_new = generate_prediction_data(date_test, date_original, dataset_test, dataset_original, wl_p)

# Create dataset dictionary with min/max/average values
dataset_new, dataset_daily_max, dataset_daily_min, max_dic, min_dic = create_date_dic(dataset_new, wl_p, date, dataset_original)
np.save('./results/dataset_whole.npy', dataset_new)  # Save the entire dataset

# Normalize dataset for prediction
dataset_new = reframeDF(dataset_new, scaler)
print(dataset_new.shape)

# Perform final testing using trained models
results, targets, weights_1, weights_2 = final_test(dataset_new, wl_p, transformer_base, transformer_modify, aw, criterion)
np.save('./results/results.npy', results)  # Save predicted results
np.save('./results/targets.npy', targets)  # Save target values

# Analyze test results
error, error_pearson, plot_targets, plot_preds, plot_average, plot_original = test_analysis(date, dataset_new, dataset_original, max_dic, min_dic, results, targets, wl_p, scaler)
print(error)

# Convert lists to numpy arrays
plot_targets = np.array(plot_targets)
plot_preds = np.array(plot_preds)

# Compute test error and RMSE
test_error = plot_targets - plot_preds
print(test_error.shape)
test_rmse = np.sqrt(sum(np.square(test_error)) / wl_p.test_length)
print(test_rmse)

# Compute Pearson correlation coefficient
pearson_corr, _ = pearsonr(plot_targets, plot_preds)
print(pearson_corr)

# Compute R-squared score
r2 = r2_score(plot_targets, plot_preds)
print(r2)
