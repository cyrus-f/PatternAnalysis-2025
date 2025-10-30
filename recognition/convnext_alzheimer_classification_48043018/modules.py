"""modules.py
Implementation of the ConvNeXt architecture for image classification.
This module defines the ConvNeXt model and its building blocks, including ConvNeXtBlock and
ConvNeXtBlockTransition.

key classes:
- ConvNeXtBlock: A single ConvNeXt block.
- ConvNeXtBlockTransition: A ConvNeXt block that performs downsampling and channel expansion between stages.
- ConvNeXt: The complete ConvNeXt architecture for image classification.
"""
import torch
import torch.nn as nn

# CONSTANTS
IN_CHANNELS  = 3     
IMAGE_SIZE   = 224   

NUM_BLOCKS   = [3, 3, 9, 3]       # implementing the ConvNeXt-T variant architecture
OUT_CHANNELS = [96, 192, 384, 768]  
NUM_CLASSES  = 2 # number of output classes for classification namely Alzheimer’s disease (AD) and Cognitive Normal (CN)


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
    
class ConvNeXtBlockTransition(nn.Module):
    """
    A ConvNeXt block that performs downsampling and channel expansion between stages.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden_channels = out_channels * 4  # inverted bottleneck ratio of 4

        # Downsample and project input to new channel dimension
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,    # single stride-2 downsample
            padding=0
        )

        # Depthwise convolution (spatial mixing)
        self.conv0 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=out_channels
        )

        # LayerNorm operates on channels-last format, so we permute before/after
        self.norm = nn.LayerNorm(normalized_shape=out_channels)

        # Pointwise expansion, GELU ,projection back
        self.conv1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1
        )

        self.gelu = nn.GELU()

        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H/2, W/2)
        """
        # Downsample and change number of channels
        residual = self.projection(x)

        # Depthwise convolution + normalization + MLP-style projection
        x = self.conv0(residual)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)

        # Skip connection
        x = x + residual
        return x
class ConvNeXt(nn.Module):
    """"
    The ConvNeXt architecture for image classification.
    """
    def __init__(self):
        super().__init__()
        # stem layer convolution with 4x4 kernel and stride 4, reducing spatial dimensions to 1/4 of original
        self.stem = nn.Conv2d(in_channels=IN_CHANNELS,    
                              out_channels=OUT_CHANNELS[0],
                              kernel_size=4,
                              stride=4,
                             )

        self.normstem = nn.LayerNorm(normalized_shape=OUT_CHANNELS[0])  # layer normalization after stem
        
        self.res2 = nn.ModuleList()
        for _ in range(NUM_BLOCKS[0]):
            self.res2.append(ConvNeXtBlock(num_channels=OUT_CHANNELS[0]))
        
        self.res3 = nn.ModuleList([ConvNeXtBlockTransition(in_channels=OUT_CHANNELS[0], 
                                                           out_channels=OUT_CHANNELS[1])])
        for _ in range(NUM_BLOCKS[1]-1):
            self.res3.append(ConvNeXtBlock(num_channels=OUT_CHANNELS[1]))

        self.res4 = nn.ModuleList([ConvNeXtBlockTransition(in_channels=OUT_CHANNELS[1], 
                                                           out_channels=OUT_CHANNELS[2])])
        for _ in range(NUM_BLOCKS[2]-1):
            self.res4.append(ConvNeXtBlock(num_channels=OUT_CHANNELS[2]))

        self.res5 = nn.ModuleList([ConvNeXtBlockTransition(in_channels=OUT_CHANNELS[2], 
                                                           out_channels=OUT_CHANNELS[3])])
        for _ in range(NUM_BLOCKS[3]-1):
            self.res5.append(ConvNeXtBlock(num_channels=OUT_CHANNELS[3]))
    
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))  # reduce to 1x1 by averaging over channel dimensions
        self.normpool = nn.LayerNorm(normalized_shape=OUT_CHANNELS[3])  # layer normalization after pooling
        self.fc = nn.Linear(in_features=OUT_CHANNELS[3],        # output layer for classification
                            out_features=NUM_CLASSES)
        
        self.relu = nn.ReLU()

    def forward(self, x):
            """
            Forward pass of the ConvNeXt model.
            Args:
                x (torch.Tensor): Input tensor of shape (N, C, H, W)
            Returns:
                torch.Tensor: Output tensor of shape (N, NUM_CLASSES)   
            """
            x = self.relu(self.stem(x))
            x = x.permute(0, 2, 3, 1) # adjust dimensions for layer norm to (N, H, W, C)
            x = self.normstem(x)
            x = x.permute(0, 3, 1, 2) # adjust dimensions back to (N, C, H, W)
            for i, block in enumerate(self.res2):    
                x = block(x)

            for i, block in enumerate(self.res3):    
                x = block(x)

            for i, block in enumerate(self.res4):    
                x = block(x)

            for i, block in enumerate(self.res5):    
                x = block(x)

            x = self.avgpool(x)
            x = x.permute(0, 2, 3, 1) # adjust dimensions for layer norm to (N, H, W, C)
            x = self.normpool(x)
            x = x.permute(0, 3, 1, 2) # adjust dimensions back to (N, C, H, W)
            x = x.reshape(x.shape[0], -1)             
            x = self.fc(x)
            return x
