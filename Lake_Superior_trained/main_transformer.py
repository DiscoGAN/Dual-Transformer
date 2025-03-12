import torch
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from garage_Transformer import cal_moving_ave, reframeDF, create_date_dic, final_test
from garage_Transformer import generate_dataset, final_test, test_analysis, generate_prediction_data
from model_Prophet import Transformer_half_base
from model_Critic import Transformer_half_modify
from model_weighted import auto_weighted
from scipy.stats import pearsonr

# Define a configuration class to store hyperparameters and file paths
class config():
    def __init__(self):
        self.cycle_num_before = 12  # Number of historical time steps used as input (specific to Prophet model)
                                    # Default setting corresponds to a 6-month-ahead period
        self.cycle_num_later = 5  # Number of future time steps used as input (specific to Critic model)
                                  # Default setting corresponds to a 6-month-ahead period
        self.moving_length = 30  # Moving average window size
        self.interval_1 = 30  # Time interval (in days) between successive historical input data time steps
        self.interval_2 = 30  # Time interval (in days) between successive future input data time steps
        self.is_mean = True  # Whether to use mean values
        self.is_train = False  # Training mode flag
        self.train_pro = 1  # Training proportion
        self.input_size = 7  # Number of input features
        self.batch_size = 256  # Batch size for training
        self.n = 6  # Parameter index for model configuration 
                    # Default setting corresponds to a 6-month-ahead period
        # Paths for saving and loading model weights
        self.path_base = './model/Dual_Transformer_Prophet_'+str(self.n)+'.pt'
        self.path_modify = './model/Dual_Transformer_Critic_'+str(self.n)+'.pt'
        self.path_weight = './model/Dual_Transformer_aw_'+str(self.n)+'.pt'
        # Dataset file paths
        self.data_path = './dataset/finaldata.csv'
        self.test_path_1 = './dataset/test_2022.csv'
        self.test_path_2 = './dataset/test_2023.csv'
        # Other parameters
        self.test_length = 365 * 2  # Total test length (two years)
        self.start_year = 1981  # Start year for dataset
        self.total_year = 100  # Total number of years in the dataset (fake number for plotting)
        self.predict_year_start = 2022  # Start year for predictions
        self.plot_start_year = self.predict_year_start - 1  # Year for plotting
        self.predict_long = 2  # Prediction length in years

# Initialize the configuration object
wl_p = config()

# Initialize a Min-Max Scaler for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))

# Dataset feature mapping:
# water_level (index 0), temp_lake (index 1), temp_land (index 2), 
# wspd_lake (index 3), wspd_land (index 4), lst (index 5), precipitation (index 6)
# Load training dataset
date_original, dataset_original = generate_dataset(wl_p.data_path)
date, dataset = generate_dataset(wl_p.data_path)
date = [datetime.strptime(d, '%Y-%m-%d').date() for d in date]
# Apply moving average smoothing
dataset = cal_moving_ave(dataset, wl_p.moving_length)
# Normalize and reframe dataset
dataset = reframeDF(dataset, scaler)

# Load testing datasets for evaluation
date_test_1, dataset_test_1 = generate_dataset(wl_p.test_path_1)
dataset_test_1[:, 1] -= 273.15  # Convert temperature from Kelvin to Celsius
dataset_test_1[:, 2] -= 273.15
date_test_2, dataset_test_2 = generate_dataset(wl_p.test_path_2)

# Merge testing datasets
date_test = np.concatenate((date_test_1, date_test_2), axis=0)
dataset_test = np.concatenate((dataset_test_1, dataset_test_2), axis=0)

# Predefined hyperparameter configurations: 
# Each entry represents (cycle_num_before, cycle_num_later, interval_1, interval_2),
# corresponding to different time durations (in days):
# {0: 7 days, 1: 32 days, 2: 63 days, 3: 91 days, 4: 119 days, 5: 147 days, 6: 180 days}
parameter_dict = {
    0: [7, 6, 1, 1],  
    1: [12, 7, 5, 4],  
    2: [12, 6, 10, 9],  
    3: [12, 6, 20, 13],  
    4: [12, 6, 25, 17],  
    5: [12, 6, 25, 21],  
    6: [12, 5, 30, 30]   
}

# Update configuration with selected parameters
wl_p.cycle_num_before = parameter_dict[wl_p.n][0]
wl_p.cycle_num_later = parameter_dict[wl_p.n][1]
wl_p.interval_1 = parameter_dict[wl_p.n][2]
wl_p.interval_2 = parameter_dict[wl_p.n][3]

# Initialize Transformer models
transformer_base = Transformer_half_base(parameter_dict[wl_p.n])
transformer_modify = Transformer_half_modify(parameter_dict[wl_p.n])
aw = auto_weighted(wl_p)

# Define loss function and optimizers
criterion = torch.nn.MSELoss()  # Mean Squared Error loss

# Load pre-trained model weights
transformer_base.load_state_dict(torch.load(wl_p.path_base))
transformer_base.eval()
transformer_modify.load_state_dict(torch.load(wl_p.path_modify))
transformer_modify.eval()
aw.load_state_dict(torch.load(wl_p.path_weight))
aw.eval()

# Generate prediction data
date, dataset_original, dataset_new = generate_prediction_data(
    date_test, date_original, dataset_test, dataset_original, wl_p
)

# Create dataset dictionary and normalize
dataset_new, dataset_daily_max, dataset_daily_min, max_dic, min_dic = create_date_dic(dataset_new, wl_p, date, dataset_original)
dataset_new = reframeDF(dataset_new, scaler)

# Perform final model testing and evaluation
results, targets = final_test(
    dataset_new, wl_p, transformer_base, transformer_modify, aw, criterion, parameter_dict[wl_p.n]
)

# Analyze and plot prediction results on the testing dataset
error, error_cos, error_pearson, plot_targets, plot_preds, plot_average, plot_original = test_analysis(
    date, dataset_new, dataset_original, max_dic, min_dic, results, targets, wl_p, scaler
)

# Save final predictions
np.save('./results/dual_results.npy', plot_preds)

# Compute RMSE error
plot_targets = np.array(plot_targets)
plot_preds = np.array(plot_preds)
test_error = plot_targets - plot_preds
test_rmse = np.sqrt(sum(np.square(test_error)) / wl_p.test_length)
print(f"Test RMSE error: {test_rmse}")

# Compute correlation coefficient
pearson_corr, _ = pearsonr(plot_targets, plot_preds)
print(f"correlation coefficient: {pearson_corr}")