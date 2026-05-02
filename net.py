import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import pdb

from CBAM_attention import cbam
from WAttention import WindowAttention2
from MCrossAttention import SwinFusion as net

EPSILON = 1e-10

def var(x, dim=0):
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)

class MultConst(nn.Module):
    def forward(self, input):
        return 255 * input

class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right % 2 == 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot % 2 == 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2

class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
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
    layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                        base_width=self.base_width, dilation=previous_dilation,
                        norm_layer=norm_layer, kernel_size=kernel_size))
    self.inplanes = planes * block.expansion
    if stride != 1:
        kernel_size = kernel_size // 2

    for _ in range(1, blocks):
        layers.append(block(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))

    return nn.Sequential(*layers)

class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        # out_channels_def = 16
        out_channels_def = int(in_channels / 2)
        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out

# Convolution operation
class f_ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(f_ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = F.relu(out, inplace=True)
        return out

class FusionBlock_res(torch.nn.Module):
    expansion = 2

    def __init__(self, channels, img_size, index, stride=1, downsample=None,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(FusionBlock_res, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.f1 = MCABlock2(channels, channels // 2, kernel_size=img_size)
        self.wattent = WindowBlock(channels, channels // 2, kernel_size=img_size)

        self.axial_fusion = nn.Sequential(f_ConvLayer(2 * channels, channels, 1, 1))
        self.conv_fusion = nn.Sequential(f_ConvLayer(channels, channels, 1, 1))

        width = int(channels * (base_width / 64.))

        self.bn1 = norm_layer(width)

        self.cbam = cbam(width)

        self.conv_up = conv1x1(width, channels * self.expansion)
        self.bn2 = norm_layer(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


        self.fc1 = f_ConvLayer(2 * channels, channels, 1, 1)
        self.fc11 = f_ConvLayer(3 * channels, channels, 1, 1)
        self.fc2 = f_ConvLayer(channels, channels, 3, 1)
        self.fc3 = f_ConvLayer(channels, channels, 3, 1)

    def forward(self, x_mri, x_spect):

        f1 = self.f1(x_mri, x_spect)

        a_cat = torch.cat([self.wattent(x_mri), self.wattent(x_spect)], 1)
        awt = self.axial_fusion(a_cat)

        x_cspect = self.conv_fusion(x_spect)

        out1 = self.cbam(x_cspect)
        out1 = self.cbam(out1)
        out1 = self.cbam(out1)

        out1 = self.bn1(out1)
        out = self.relu(out1)
        out6 = x_spect + out


        x_cmri = self.conv_fusion(x_mri)

        out1 = self.cbam(x_cmri)
        out1 = self.cbam(out1)
        out1 = self.cbam(out1)

        out1 = self.bn1(out1)
        out = self.relu(out1)
        out7 = x_mri + out

        out = torch.cat([out6, out7], 1)

        # RDB1
        outs = self.fc1(out)

        out = self.fc2(outs)
        out1 = self.fc3(out)
        out11 = out + out1

        out = self.fc2(out11)
        out2 = self.fc3(out)
        out22 = out + out1 + out2

        out = self.fc2(out22)
        out3 = self.fc3(out)
        out33 = out + out1 + out2 + out3

        s1 = out + out33

        # RDB2
        out = self.fc2(s1)
        out1 = self.fc3(out)
        out11 = out + out1

        out = self.fc2(out11)
        out2 = self.fc3(out)
        out22 = out + out1 + out2

        out = self.fc2(out22)
        out3 = self.fc3(out)
        out33 = out + out1 + out2 + out3

        s2 = out + out33

        out = f1 + s2
        out = torch.cat([out,awt], 1)
        out = self.axial_fusion(out)

        return out

# Fusion network, 4 groups of features
class Fusion_network(nn.Module):
    def __init__(self, nC, fs_type):
        super(Fusion_network, self).__init__()
        self.fs_type = fs_type
        img_size = [256, 128, 64, 32]

        self.fusion_block1 = FusionBlock_res(nC[0], img_size[0], 0)
        self.fusion_block2 = FusionBlock_res(nC[1], img_size[1], 1)
        self.fusion_block3 = FusionBlock_res(nC[2], img_size[2], 2)
        self.fusion_block4 = FusionBlock_res(nC[3], img_size[3], 3)

    def forward(self, en_mri, en_spect):
        f1_0 = self.fusion_block1(en_mri[0], en_spect[0])
        f2_0 = self.fusion_block2(en_mri[1], en_spect[1])
        f3_0 = self.fusion_block3(en_mri[2], en_spect[2])
        f4_0 = self.fusion_block4(en_mri[3], en_spect[3])

        return [f1_0, f2_0, f3_0, f4_0]


class Fusion_ADD(torch.nn.Module):
    def forward(self, en_mri, en_spect):
        temp = en_mri + en_spect
        return temp


class Fusion_AVG(torch.nn.Module):
    def forward(self, en_mri, en_spect):
        temp = (en_mri + en_spect) / 2
        return temp


class Fusion_MAX(torch.nn.Module):
    def forward(self, en_mri, en_spect):
        temp = torch.max(en_mri, en_spect)
        return temp


class Fusion_SPA(torch.nn.Module):
    def forward(self, en_mri, en_spect):
        shape = en_mri.size()
        spatial_type = 'mean'
        # calculate spatial attention
        spatial1 = spatial_attention(en_mri, spatial_type)
        spatial2 = spatial_attention(en_spect, spatial_type)
        # get weight map, soft-max
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * en_mri + spatial_w2 * en_spect
        return tensor_f


def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial

class Fusion_Nuclear(torch.nn.Module):
    def forward(self, en_mri, en_spect):
        shape = en_mri.size()
        # calculate channel attention
        global_p1 = nuclear_pooling(en_mri)
        global_p2 = nuclear_pooling(en_spect)

        # get weight map
        global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
        global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

        global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
        global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

        tensor_f = global_p_w1 * en_mri + global_p_w2 * en_spect
        return tensor_f


def nuclear_pooling(tensor):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors

class Fusion_strategy(nn.Module):
    def __init__(self, fs_type):
        super(Fusion_strategy, self).__init__()
        self.fs_type = fs_type
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_spa = Fusion_SPA()
        self.fusion_nuc = Fusion_Nuclear()

    def forward(self, en_mri, en_spect):
        if self.fs_type == 'add':
            fusion_operation = self.fusion_add
        elif self.fs_type == 'avg':
            fusion_operation = self.fusion_avg
        elif self.fs_type == 'max':
            fusion_operation = self.fusion_max
        elif self.fs_type == 'spa':
            fusion_operation = self.fusion_spa
        elif self.fs_type == 'nuclear':
            fusion_operation = self.fusion_nuc

        f1_0 = fusion_operation(en_mri[0], en_spect[0])
        f2_0 = fusion_operation(en_mri[1], en_spect[1])
        f3_0 = fusion_operation(en_mri[2], en_spect[2])
        f4_0 = fusion_operation(en_mri[3], en_spect[3])
        return [f1_0, f2_0, f3_0, f4_0]

class NestFuse_light2_nodense(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(NestFuse_light2_nodense, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)

        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)  # 64+112=176,64
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)  # 112+160=272,112
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)  # 160+208=368,160

        # short connection
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)

        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)


    def encoder(self, input):

        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))

        return [x1_0, x2_0, x3_0, x4_0]

    def decoder_train(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))
        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)

            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

    def decoder_eval(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)

            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

class RFN_decoder(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(RFN_decoder, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        # short connection
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)

        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def decoder_train(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))
        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)

            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

    def decoder_eval(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)

            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

class MCABlock2(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(MCABlock2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.swt = nn.ModuleList([])
        for i in range(4):
            if i == 0:
                aaa = net(upscale=1, in_chans=1, img_size=256, window_size=8,
                          img_range=1., depths=[1, 1, 1, 1], embed_dim=64, num_heads=[8, 8, 8, 8],
                          mlp_ratio=2, upsampler=None, resi_connection='1conv')
            elif i == 1:
                aaa = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                          img_range=1., depths=[1, 1, 1, 1], embed_dim=112, num_heads=[8, 8, 8, 8],
                          mlp_ratio=2, upsampler=None, resi_connection='1conv')
            elif i == 2:
                aaa = net(upscale=1, in_chans=1, img_size=64, window_size=8,
                          img_range=1., depths=[1, 1, 1, 1], embed_dim=160, num_heads=[8, 8, 8, 8],
                          mlp_ratio=2, upsampler=None, resi_connection='1conv')
            elif i == 3:
                aaa = net(upscale=1, in_chans=1, img_size=32, window_size=8,
                          img_range=1., depths=[1, 1, 1, 1], embed_dim=208, num_heads=[8, 8, 8, 8],
                          mlp_ratio=2, upsampler=None, resi_connection='1conv')
                pass
            self.swt.append(aaa)

    def forward(self, x, y):

        if x.shape[1] == 64:
            out = self.swt[0](x, y)
        if x.shape[1] == 112:
            out = self.swt[1](x, y)
        if x.shape[1] == 160:
            out = self.swt[2](x, y)
        if x.shape[1] == 208:
            out = self.swt[3](x, y)

        return out

class WindowBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(WindowBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.WindowAttention_list = nn.ModuleList([])
        for i in range(4):
            if i == 0:
                aWindowAttention = WindowAttention2(in_channels=64, window_size=3)
            elif i == 1:
                aWindowAttention = WindowAttention2(in_channels=112, window_size=3)
            elif i == 2:
                aWindowAttention = WindowAttention2(in_channels=160, window_size=3)
            elif i == 3:
                aWindowAttention = WindowAttention2(in_channels=208, window_size=3)
                pass
            self.WindowAttention_list.append(aWindowAttention)

    def forward(self, x):

        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if x.shape[1] == 64:
            out = self.WindowAttention_list[0](out)
        if x.shape[1] == 112:
            out = self.WindowAttention_list[1](out)
        if x.shape[1] == 160:
            out = self.WindowAttention_list[2](out)
        if x.shape[1] == 208:
            out = self.WindowAttention_list[3](out)

        out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out
