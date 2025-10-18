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
    """A ConvNeXt block with transition to downsample and change number of channels.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.  
    """
    def __init__(self, in_channels, out_channels):  # different number of input and output channels
        super().__init__()
        hidden_channels = out_channels * 4
         
        self.projection = nn.Conv2d(in_channels=in_channels,      
                                    out_channels=out_channels, 
                                    kernel_size=1, 
                                    stride=2,
                                    padding=0)
        
        self.conv0 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=7,
                               stride=1,
                               padding=3,
                               groups=in_channels)
        
        self.norm0 = nn.LayerNorm(normalized_shape=out_channels)
        
        self.conv1 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=hidden_channels, 
                               kernel_size=1, 
                               stride=1, 
                               padding=0)
        
        self.gelu = nn.GELU()
        
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, 
                               out_channels=out_channels, 
                               kernel_size=1, 
                               stride=1,
                               padding=0)
        
        self.norm1 = nn.LayerNorm(normalized_shape=out_channels)  

        self.downsample = nn.Conv2d(in_channels=out_channels,     
                                    out_channels=out_channels, 
                                    kernel_size=2, 
                                    stride=2)
    def forward(self, x):
        """
        Forward pass of the ConvNeXt block with transition.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H/2, W/2)
        """
        residual = self.projection(x)  #(1)
        x = self.conv0(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm0(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)  #(2)
        x = x + residual  #(3)
        
        return x
    
class ConvNeXt(nn.Module):
    """"The ConvNeXt architecture for image classification.
    Args:"""
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
