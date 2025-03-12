import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from garage_Transformer import cal_moving_ave, reframeDF, create_date_dic, final_test
from garage_Transformer import final_test, test_analysis, generate_prediction_data
from model_Prophet import Transformer_half_base
from model_Critic import Transformer_half_modify
from model_weighted import auto_weighted
from scipy.stats import pearsonr


# Configuration class for setting model parameters
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
        self.input_size = 8  # Number of input features
        self.batch_size = 256  # Batch size for training
        self.n = 6  # Parameter index for model configuration 
                    # Default setting corresponds to a 6-month-ahead period
        self.path_base = './model/Dual_Transformer_Prophet_'+str(self.n)+'.pt'  # Path to Prophet model
        self.path_modify = './model/Dual_Transformer_Critic_'+str(self.n)+'.pt'  # Path to Critic model
        self.path_weight = './model/Dual_Transformer_aw_'+str(self.n)+'.pt'  # Path to weighted model
        self.dataset_path = './results/dataset_whole.npy'  # Path to dataset
        self.date_path = './results/date.npy'  # Path to date information
        # Other parameters
        self.test_length = 365 * 2  # Total test length (two years)
        self.start_year = 1981  # Start year for dataset
        self.total_year = 100  # Total number of years in the dataset (fake number for plotting)
        self.predict_year_start = 2022  # Start year for predictions
        self.plot_start_year = self.predict_year_start - 1  # Year for plotting
        self.predict_long = 2  # Prediction length in years


# Initialize configuration
wl_p = config()

# Load precomputed prediction water level results for Lake Superior
wl_sup = np.load('./dataset/superior_prediction_results.npy')

# Dataset feature mapping:
# water_level_Erie (index 0), water_level_Superior (index 1) 
# temp_lake (index 2), temp_land (index 3), 
# wspd_lake (index 4), wspd_land (index 5), lst (index 6), precipitation (index 7)

# Load date and dataset
date_whole = list(np.load(wl_p.date_path))
dataset_whole = np.load(wl_p.dataset_path)

# Initialize MinMaxScaler for data normalization
scaler = MinMaxScaler(feature_range=(0, 1))

# Split training dataset
date_original, date = date_whole[:-wl_p.test_length], date_whole[:-wl_p.test_length]
dataset_original, dataset = dataset_whole[:-wl_p.test_length], dataset_whole[:-wl_p.test_length]

# Calculate the moving average and replace the observed Lake Superior water levels with predicted values in training dataset
dataset = cal_moving_ave(dataset, wl_p.moving_length)
dataset[-len(wl_sup[:-wl_p.test_length]):, 1] = wl_sup[:-wl_p.test_length]
# Normalize dataset
dataset = reframeDF(dataset, scaler)

# Split testing dataset
date_test, dataset_test = date_whole[-wl_p.test_length:], dataset_whole[-wl_p.test_length:]

# Predefined hyperparameter configurations: 
# Each entry represents (cycle_num_before, cycle_num_later, interval_1, interval_2),
# corresponding to different time durations (in days):
# {0: 7 days, 1: 32 days, 2: 63 days, 3: 91 days, 4: 119 days, 5: 147 days, 6: 180 days}
parameter_dict = {
    6: [12, 5, 30, 30]
}

# Update model parameters based on selected configuration
wl_p.cycle_num_before, wl_p.cycle_num_later, wl_p.interval_1, wl_p.interval_2 = parameter_dict[wl_p.n]

# Initialize models
transformer_base = Transformer_half_base(parameter_dict[wl_p.n], wl_p)
transformer_modify = Transformer_half_modify(parameter_dict[wl_p.n], wl_p)
aw = auto_weighted(wl_p)

# Define loss function and optimizers
criterion = torch.nn.MSELoss()

# Load pre-trained model weights
transformer_base.load_state_dict(torch.load(wl_p.path_base))
transformer_base.eval()
transformer_modify.load_state_dict(torch.load(wl_p.path_modify))
transformer_modify.eval()
aw.load_state_dict(torch.load(wl_p.path_weight))
aw.eval()

# Generate prediction dataset
date, dataset_original, dataset_new = generate_prediction_data(date_test, date_original, dataset_test, dataset_original, wl_p, wl_sup)

# Compute max/min values and normalize dataset
dataset_new, dataset_daily_max, dataset_daily_min, max_dic, min_dic = create_date_dic(dataset_new, wl_p, date, dataset_original)
dataset_new = reframeDF(dataset_new, scaler)

# Perform final test and save prediction results
results, targets, weights_1, weights_2 = final_test(dataset_new, wl_p, transformer_base, transformer_modify, aw, criterion, parameter_dict[wl_p.n])
np.save('./results/results.npy', results)
np.save('./results/targets.npy', targets)

# testing analysis
error, error_pearson, plot_targets, plot_preds = test_analysis(date, dataset_new, dataset_original, max_dic, min_dic, results, targets, wl_p, scaler)

# Compute RMSE error
plot_targets = np.array(plot_targets)
plot_preds = np.array(plot_preds)
test_error = plot_targets - plot_preds
test_rmse = np.sqrt(sum(np.square(test_error)) / wl_p.test_length)
print(f"Test RMSE error: {test_rmse}")

# Compute correlation coefficient
pearson_corr, _ = pearsonr(plot_targets, plot_preds)
print(f"correlation coefficient: {pearson_corr}")