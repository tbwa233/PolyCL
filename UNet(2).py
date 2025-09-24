from torch import nn
from torch.nn import functional as F
import torch
import torchvision.models
from typing import Type, Union, Optional, Callable, List  
    
#Convolution block taken from MultiMix code
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(in_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.25)
    ) 

#A convolutional block that can be used in place of double_conv
#Adds a simple skip connection from the input to the output with a 1x1 convolution in the middle for resizing
#Otherwise is the same as double_conv
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.instNorm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(0.25)

        self.downsample = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, X):
        result = self.conv1(X)
        result = self.instNorm(result)
        result = self.relu(result)

        result = self.conv2(result)
        result = self.instNorm(result)

        xSkip = self.downsample(X)
        result += xSkip

        return self.dropout(self.relu(result))

#Standard UNet implementation using a block specified in initialization
#Includes code for true multitasking with a classification and segmentation performance threshold
class UNet(nn.Module):
    def __init__(self, device, n_class = 1, encoder=None, multiTask=False, classThreshold=0, segmentThreshold=0, block=double_conv):
        super().__init__()

        if encoder == None:
            self.encoder = Encoder(n_class=n_class, block=block)
        else:
            self.encoder = encoder

        self.decoder = Decoder(n_class=n_class)
        self.multiTask = multiTask

        self.classThreshold = classThreshold
        self.segmentThreshold = segmentThreshold

        self.device = device
        

    def forward(self, x, classThresholdReached=True, segmentThresholdReached=True):
        #Gets output from the classification branch and each block of the encoder
        outC, conv5, conv4, conv3, conv2, conv1 = self.encoder(x)
        #Passes this output to the decoder to get the segmentation mask
        outSeg = self.decoder(x, conv5, conv4, conv3, conv2, conv1)

        finalSeg = None
        finalC = None

        #Enables true multitasking between classification and segmentation if performance thresholds are reached
        #If the classification label is 0, the segmentation mask is set to all 0's
        #If the segmentation mask is all 0's, the classification label is set to 0
        if self.multiTask:
            if classThresholdReached:
                for i, classLabel in enumerate(outC):
                    if classLabel < 0.5:
                        result = (outSeg[i] * 0).unsqueeze(dim=0)
                    else:
                        result = outSeg[i].unsqueeze(dim=0)

                    if finalSeg == None:
                        finalSeg = result
                    else:
                        finalSeg = torch.cat((finalSeg, result), dim=0)
            else:
                finalSeg = outSeg

            if segmentThresholdReached:
                for i, segMap in enumerate(outSeg):
                    if torch.count_nonzero(torch.round(segMap)) == 0:
                        result = outC[i]
                    else:
                        result = outC[i]

                    if finalC == None:
                        finalC = result
                    else:
                        finalC = torch.cat((finalC, result), dim=0)
            else:
                finalC = outC

            return finalSeg, finalC

        # return outSeg, outC
        return outSeg, outC

    #Freezes all the weights of the encoder for fine-tuning if needed
    def freezeEncoder(self, state=False):
        for param in self.encoder.parameters():
            param.requires_grad = state

#Standard UNet encoder taken from MultiMix paper
class Encoder(nn.Module):
    def __init__(self, n_class = 1, block=double_conv):
        super().__init__()
    
        self.dconv_down1 = block(1, 16)
        self.dconv_down2 = block(16, 32)
        self.dconv_down3 = block(32, 64)
        self.dconv_down4 = block(64, 128)
        self.dconv_down5 = block(128, 256)    
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))       
        self.fc = nn.Linear(256, 1) 
        self.sigm = nn.Sigmoid()

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        x1 = self.maxpool(conv5)
        
        avgpool = self.avgpool(x1)
        avgpool = avgpool.view(avgpool.size(0), -1)
        outC = self.fc(avgpool)
        
        return self.sigm(outC), conv5, conv4, conv3, conv2, conv1
    
#Same as the above encoder with a projection head
class ContrastiveEncoder(nn.Module):

    def __init__(self, n_class = 1, block=double_conv):
        super().__init__()
    
        self.dconv_down1 = block(1, 16)
        self.dconv_down2 = block(16, 32)
        self.dconv_down3 = block(32, 64)
        self.dconv_down4 = block(64, 128)
        self.dconv_down5 = block(128, 256)
        
        self.maxpool = nn.MaxPool2d(2)

        self.avgPool = nn.AdaptiveAvgPool2d((1,1))
        self.projHead = nn.Linear(256, 256)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        x1 = self.maxpool(conv5)
        
        projection = self.avgPool(x1)
        projection = projection.view(projection.size(0), -1)
        projection = self.projHead(projection)
        
        return projection, conv5, conv4, conv3, conv2, conv1

#Standard UNet decoder taken from MultiMix paper
class Decoder(nn.Module):

    def __init__(self, n_class = 1, block=double_conv):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = block(256 + 128, 128)
        self.dconv_up3 = block(128 + 64, 64)
        self.dconv_up2 = block(64 + 32, 32)
        self.dconv_up1 = block(32 + 16, 16)
        self.conv_last = nn.Conv2d(16, n_class, 1)

        self.sigm = nn.Sigmoid()
        
        
    def forward(self, input, conv5, conv4, conv3, conv2, conv1):
  
        x = self.upsample(conv5)        
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)       

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1) 

        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return self.sigm(out)
    
#Decoder meant to be used with a ResNet encoder
#Very similar to the normal decoder with a higher dimensionality to match ResNet
class ResDecoder(nn.Module):
    def __init__(self, n_class = 1, block=double_conv):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = block(512 + 256, 256)
        self.dconv_up3 = block(256 + 128, 128)
        self.dconv_up2 = block(128 + 64, 64)
        self.dconv_up1 = block(64 + 32, 32)
        self.conv_last = nn.Conv2d(32, n_class, 1)

        self.sigm = nn.Sigmoid()
        
        
    def forward(self, input, conv5, conv4, conv3, conv2, conv1):
        x = self.upsample(conv5)        
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)       

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1) 

        x = self.dconv_up1(x)

        x = self.upsample(x)

        out = self.conv_last(x)
        
        return self.sigm(out)

#UNet model combining a ResNet18 encoder and a decoder
#Very similar to the standard UNet model defined below, using a separate encoder
class ResUNet(nn.Module):
    def __init__(self, num_classes, encoder=None) -> None:
        super().__init__()

        self.classThreshold = 0
        self.segmentThreshold = 0
        self.multiTask = False

        if encoder == None:
            self.encoder = ResNetEncoder(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        else:
            self.encoder = encoder

        self.decoder = ResDecoder()

    def forward(self, x, temp1=0, temp2=0):
        outC, conv5, conv4, conv3, conv2, conv1 = self.encoder(x)

        outSeg = self.decoder(x, conv5, conv4, conv3, conv2, conv1)

        return outSeg, outC

#From here to line 532, all code was taken from the torchvision ResNet implementation at https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html
#The only changes made are modifying the number of input channels from 3 to 1 for CT slices and returning the output from all blocks in the forward function
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512 * block.expansion, num_classes), nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        out1 = self.relu(x)
        x = self.maxpool(out1)

        out2 = self.layer1(x)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        x = self.avgpool(out5)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, out5, out4, out3, out2, out1