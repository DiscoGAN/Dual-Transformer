import torch

# Configuration class for the Auto-weighted model
class auto_weighted_config():
    def __init__(self):
        self.transform_layer_1 = 128  # First transformation layer size
        self.integrate_1 = 128  # First integration layer size
        self.integrate_2 = 256  # Second integration layer size
        self.integrate_3 = 16   # Third integration layer size
        self.weighted_1 = 512   # First weighted layer size
        self.weighted_2 = 128   # Second weighted layer size
        self.weighted_3 = 16    # Third weighted layer size
        self.output_size = 2    # Output size for final weighting

# Initialize the configuration
awc = auto_weighted_config()

# Auto-weighted model class
class auto_weighted(torch.nn.Module):
    def __init__(self, parameter):
        super(auto_weighted, self).__init__()
        self.feature_size = parameter.input_size  # Number of input features
        self.cycle_num_before = parameter.cycle_num_before  # Historical time steps
        self.cycle_num_later = parameter.cycle_num_later  # Future time steps
        
        # Feature transformation layer
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, awc.transform_layer_1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.transform_layer_1, self.feature_size - 1, bias=False)
        )
        
        # Integration layers
        self.integrate = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size - 1, awc.integrate_1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.integrate_1, awc.integrate_2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.integrate_2, awc.integrate_3, bias=False)
        )
        
        # Weighted layers for final output computation
        self.weighted = torch.nn.Sequential(
            torch.nn.Linear((self.cycle_num_before + self.cycle_num_later) * awc.integrate_3, awc.weighted_1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.weighted_1, awc.weighted_2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.weighted_2, awc.weighted_3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.weighted_3, awc.output_size, bias=False)
        )
    
    def forward(self, dataset):
        dataset_size = dataset.shape[0]  # Get batch size
        dataset_history = dataset[:, :self.cycle_num_before, :]  # Extract historical data
        dataset_future = dataset[:, self.cycle_num_before:self.cycle_num_before + self.cycle_num_later, 1:]  # Extract future data (excluding water level)
        
        # Apply transformation to historical data
        history_transform = self.transform(dataset_history)
        
        # Concatenate transformed history with future data
        dataset_integrate = torch.cat([history_transform, dataset_future], 1)
        
        # Apply integration layers
        dataset_integrate = self.integrate(dataset_integrate)
        
        # Reshape for final weighting
        dataset_to_weight = dataset_integrate.view(dataset_size, -1)
        
        # Compute softmax weights
        weights = torch.softmax(self.weighted(dataset_to_weight), dim=1)
        
        return weights
