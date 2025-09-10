import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricDilation(nn.Module):
    def __init__(self, kernel_size_h: int, kernel_size_w: int):
        """
        Args:
            kernel_size_h (int): height of the kernel
            kernel_size_w (int): width of the kernel
        """
        super().__init__()
        self.kernel_h = kernel_size_h
        self.kernel_w = kernel_size_w

        kernel = torch.ones((1, 1, self.kernel_h, self.kernel_w))

        # This allows the kernel to be part of the model's state without being a parameter
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dilation using asymmetric kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, H, W)

        Returns:
            torch.Tensor: Dilated binary mask of shape (B, 1, H, W)
        """
        # Padding
        pad_h = self.kernel_h // 2
        pad_w = self.kernel_w // 2

        # Apply Dilation
        x_dilated = F.conv2d(x, self.kernel, padding=(pad_h, pad_w))

        # Convert to Binary mask 
        return (x_dilated > 0).float()

