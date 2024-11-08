import torch

class auto_weighted_config():
    def __init__(self):
        self.transform_layer_1 = 128
        self.integrate_1 = 128
        self.integrate_2 = 256
        self.integrate_3 = 16
        self.weighted_1 = 512
        self.weighted_2 = 128
        self.weighted_3 = 16
        self.output_size = 2

    
awc = auto_weighted_config()


class auto_weighted(torch.nn.Module):
    def __init__(self, parameter):
        super(auto_weighted, self).__init__()
        self.feature_size = parameter.input_size
        self.cycle_num_before = parameter.cycle_num_before
        self.cycle_num_later = parameter.cycle_num_later
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, awc.transform_layer_1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.transform_layer_1, self.feature_size-1, bias=False))
        self.integrate = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size-1, awc.integrate_1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.integrate_1, awc.integrate_2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.integrate_2, awc.integrate_3, bias=False))
        self.weighted = torch.nn.Sequential(
            torch.nn.Linear((self.cycle_num_before+self.cycle_num_later)*awc.integrate_3, awc.weighted_1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.weighted_1, awc.weighted_2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.weighted_2, awc.weighted_3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.weighted_3, awc.output_size, bias=False))
        
    
    def forward(self, dataset):
        dataset_size = dataset.shape[0]
        dataset_history = dataset[:, :self.cycle_num_before, :]
        dataset_future = dataset[:, self.cycle_num_before:self.cycle_num_before+self.cycle_num_later, 1:]
        history_transform = self.transform(dataset_history)
        dataset_integrate = torch.cat([history_transform, dataset_future], 1)
        dataset_integrate = self.integrate(dataset_integrate)
        dataset_to_weight = dataset_integrate.view(dataset_size, -1)
        weights = torch.softmax(self.weighted(dataset_to_weight), 1)

        return weights    