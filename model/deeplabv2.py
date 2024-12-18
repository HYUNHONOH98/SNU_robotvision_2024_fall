#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>, parts of the model from https://github.com/ZJULearning/MaxSquareLoss
#
# SPDX-License-Identifier: MIT

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

affine_par = True

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

# model = ReNetMulti()
# output, feature = model(input, return_features=True)
from torch.utils.checkpoint import checkpoint


class ResNetMultiBayes(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNetMultiBayes, self).__init__()
        self.resnet = ResNetMulti(block, layers, num_classes)
        self.thresholdnet = ThresholdNet(2048 ,256) # input size, output size
        self.avgpool = nn.AdaptiveAvgPool2d((19, 1)) # class num?
        self.rf_classifier = RFClassifierNet(2048)

    def forward(self, x, return_features= False):
        if return_features:
            output, feature = self.resnet(x, return_features=return_features)
            
            # Discriminator
            rf_pred = self.rf_classifier(feature)
            # BPL
            feature = self.avgpool(feature) # feature.shape is (2, 2048, 1, 1)
            # mu, logvar = self.thresholdnet(torch.flatten(feature, start_dim=1)) # ORIGINAL
            mu, logvar = self.thresholdnet(torch.squeeze(feature, dim=-1).permute(0,2,1)) # CHANGED (multi-channel, 19)
            
            return {"output": output, "mu": mu, "logvar": logvar, "rf_pred" :rf_pred}
        else:
            return {"output": self.resnet(x)}

class RFClassifierNet(nn.Module):
    def __init__(self, input_channels=2048):
        super(RFClassifierNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(input_channels, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256,1)
        )
    
    def forward(self, x):
        return self.classifier(x)

class ThresholdNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ThresholdNet, self).__init__()
        self.logvar_layer = nn.Linear(input_channels, output_channels)
        self.mu_layer = nn.Linear(input_channels, output_channels)

    def forward(self, x):
        return self.mu_layer(x), self.logvar_layer(x)

class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.dropout = nn.Dropout2d(p=0.3)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
    
    def forward(self, x, return_features=False):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.layer1(x)
        x = checkpoint(self.custom(self.layer1), x)

        # x = self.layer2(x)
        x = checkpoint(self.custom(self.layer2), x)

        # x = self.layer3(x)
        x = checkpoint(self.custom(self.layer3), x)
        # x1 = self.layer5(x)
        # x1 = F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)

        x = checkpoint(self.custom(self.layer4), x)
        # x = self.layer4(x)
        feats = x
        # x1 = self.layer6(x)
        x1 = checkpoint(self.custom(self.layer6), x)
        x1 = F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)

        if return_features:
            return x1, feats
    
        return x1

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def freeze_encoder_bn(self):
        modules = []
        modules.extend(self.layer1.modules())
        modules.extend(self.layer2.modules())
        modules.extend(self.layer3.modules())
        modules.extend(self.layer4.modules())
        for module in modules:
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                for i in module.parameters():
                    i.requires_grad = False

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def DeeplabMulti(num_classes=19, pretrained=True, backbone="resnet50"):
    """
    

    Args:
        num_classes (int, optional): _description_. Defaults to 19.
        pretrained (bool, optional): _description_. Defaults to True.
        backbone (str, optional): _description_. Defaults to "resnet50".

    Returns:
        _type_: _description_
    """
    if backbone == "resnet101":
        layers = [3, 4, 23, 3]
    elif backbone == "resnet50":
        layers = [3, 4, 6, 3]
    model = ResNetMulti(Bottleneck, layers, num_classes)

    if pretrained:
        cwd = os.getcwd()
        restore_from = f'{cwd}/model/pretrained/resnet50-19c8e357.pth'
        saved_state_dict = torch.load(restore_from, weights_only=True)
        model.load_state_dict(saved_state_dict, strict=False)
    return model


def DeeplabMultiBayes(num_classes=19, pretrained=True, backbone="resnet50"):
    """
    

    Args:
        num_classes (int, optional): _description_. Defaults to 19.
        pretrained (bool, optional): _description_. Defaults to True.
        backbone (str, optional): _description_. Defaults to "resnet50".

    Returns:
        _type_: _description_
    """
    if backbone == "resnet101":
        layers = [3, 4, 23, 3]
    elif backbone == "resnet50":
        layers = [3, 4, 6, 3]
    model = ResNetMultiBayes(Bottleneck, layers, num_classes)

    return model