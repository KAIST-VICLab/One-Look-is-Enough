import torch

def median_norm(tensor):
    flattened_tensor = tensor.flatten(start_dim=-2)
    
    median = flattened_tensor.median(dim=-1, keepdim=True).values.unsqueeze(-1)
    
    mad = torch.mean(torch.abs(tensor - median), dim=(-1, -2), keepdim=True)
    
    normalized_tensor = (tensor - median) / (mad + 1e-6)
    
    return normalized_tensor, median, mad

def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())

def min_max_norm_per_sample(tensor):
    n, c, h, w = tensor.shape

    min_vals = tensor.view(n, -1).min(dim=1, keepdim=True).values.view(n, 1, 1, 1)
    max_vals = tensor.view(n, -1).max(dim=1, keepdim=True).values.view(n, 1, 1, 1)

    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals + 1e-8)
    
    normalized_tensor = torch.where((max_vals - min_vals) == 0, tensor, normalized_tensor)

    return normalized_tensor