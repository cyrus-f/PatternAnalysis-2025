import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """A single ConvNeXt block as described in the ConvNeXt paper.
    Args:
        num_channels (int): The number of input and output channels for the block (same).  
    """
    def __init__(self, num_channels):         
        super().__init__()
        hidden_channels = num_channels * 4    # inverted bottleneck ratio of 4

        
        self.conv0 = nn.Conv2d(in_channels=num_channels,         # number of input and output channels are the same in depthwise conv
                               out_channels=num_channels,        
                               kernel_size=7,    # 7x7 convolution
                               stride=1,
                               padding=3,        # to maintain spatial dimensions
                               groups=num_channels)              # depthwise convolution
        
        self.norm = nn.LayerNorm(normalized_shape=num_channels)  
        
        self.conv1 = nn.Conv2d(in_channels=num_channels,         # increase the number of channels
                               out_channels=hidden_channels, 
                               kernel_size=1, # 1x1 convolution
                               stride=1, 
                               padding=0)
        
        self.gelu = nn.GELU()  #Gaussian Error Linear Unit activation function to replace batchnorm + relu
        
        self.conv2 = nn.Conv2d(in_channels=hidden_channels,      # decrease the number of channels back to original
                               out_channels=num_channels, 
                               kernel_size=1, # 1x1 convolution
                               stride=1, 
                               padding=0)
        
    def forward(self, x):
        """
        Forward pass of the ConvNeXt block.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W)
        """
        residual = x                 # residual for skip connection like in ResNet
        x = self.conv0(x)
        x = x.permute(0, 2, 3, 1)    # adjust dimensions for layer norm to (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)    # adjust dimensions back to (N, C, H, W)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = x + residual             # skip connection addition of the residual
        return x
