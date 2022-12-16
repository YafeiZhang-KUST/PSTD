import math
from torch import nn
import torch
from net.att import Atention
from net.transformermodel import Transformers
# from net.Graph import SG
import cv2
import numpy as np
import torch.nn.functional as F
from math import sqrt
import numpy

def value_rotation(num):
   value_list = np.zeros((8), np.uint8)
   temp = int(num)
   value_list[0] = temp
   for i in range(7):
       temp = ((temp << 1) | int((temp / 128)) % 256)
       value_list[i + 1] = temp
   return np.min(value_list)


def rotation_invariant_LBP(src):
   height = src.shape[0]
   width = src.shape[1]
   # dst = np.zeros([height, width], dtype=np.uint8)
   dst = src.copy()

   lbp_value = np.zeros((1, 8), dtype=np.uint8)
   neighbours = np.zeros((1, 8), dtype=np.uint8)
   for x in range(1, width - 1):
       for y in range(1, height - 1):
           neighbours[0, 0] = src[y - 1, x - 1]
           neighbours[0, 1] = src[y - 1, x]
           neighbours[0, 2] = src[y - 1, x + 1]
           neighbours[0, 3] = src[y, x - 1]
           neighbours[0, 4] = src[y, x + 1]
           neighbours[0, 5] = src[y + 1, x - 1]
           neighbours[0, 6] = src[y + 1, x]
           neighbours[0, 7] = src[y + 1, x + 1]

           center = src[y, x]

           for i in range(8):
               if neighbours[0, i] > center:
                   lbp_value[0, i] = 1
               else:
                   lbp_value[0, i] = 0

           lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                 + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

           dst[y, x] = value_rotation(lbp)

   return dst


# class ResBlock(nn.Module):
#
#     def __init__(self, channels):
#         super(ResBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=5, stride=1,
#                                padding=2, bias=False)
#         # self.bn1 = nn.BatchNorm2d(channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1,
#                                padding=2, bias=False)
#         # self.bn2 = nn.BatchNorm2d(channels)
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         # out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         # out = self.bn2(out)
#
#         out += residual
#         # out = self.relu(out)
#
#         return out
#
#
# class Fusion(nn.Module):
#     """ Head consisting of convolution layers
#     Extract features from corrupted images, mapping N3HW images into NCHW feature map.
#     """
#
#     def __init__(self, in_channels=112, out_channels=56):
#         super(Fusion, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         # self.bn1 = nn.BatchNorm2d(out_channels) if task_id in [0, 1, 5] else nn.Identity()
#         # self.relu = nn.ReLU(inplace=True)
#         self.resblock1 = ResBlock(out_channels)
#         self.resblock2 = ResBlock(out_channels)
#
#     def forward(self, x):
#         out = self.conv1(x)
#
#         out = self.resblock1(out)
#         # out = self.resblock2(out)
#
#         return out


class Fusion(nn.Module):
    '''Bi-directional Gated Feature Fusion.'''

    def __init__(self, in_channels=56, out_channels=56):
        super(Fusion, self).__init__()

        self.structure_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.Sigmoid()
        )
        self.texture_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.Sigmoid()
        )
        self.structure_gamma = nn.Parameter(torch.zeros(1))
        self.texture_gamma = nn.Parameter(torch.zeros(1))
        self.conv = nn.Conv2d(in_channels=112, out_channels=56, kernel_size=1)

    def forward(self, texture_feature, structure_feature):
        energy = torch.cat((texture_feature, structure_feature), dim=1)

        gate_structure_to_texture = self.structure_gate(energy)
        gate_texture_to_structure = self.texture_gate(energy)

        texture_feature = texture_feature + self.texture_gamma * (gate_structure_to_texture * structure_feature)
        structure_feature = structure_feature + self.structure_gamma * (gate_texture_to_structure * texture_feature)
        out = torch.cat((texture_feature, structure_feature), dim=1)
        out = self.conv(out)

        return out


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 4)
        self.input = nn.Conv2d(in_channels=28, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=28, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out


# class Att(nn.Module):
#     def __init__(self):
#         super(Att, self).__init__()
#         self.conv3 = nn.Conv2d(56, 56, kernel_size=3, padding=1, bias=False)
#
#     def forward(self, x):
#         second_c = F.sigmoid(torch.mean(x, 1).unsqueeze(1))
#         second_c = second_c * x
#         xh = x.permute(0, 2, 1, 3)
#         second_h = F.sigmoid(torch.mean(xh, 1).unsqueeze(1))
#         second_h = (second_h * xh).permute(0, 2, 1, 3)
#         xw = x.permute(0, 3, 2, 1)
#         second_w = F.sigmoid(torch.mean(xw, 1).unsqueeze(1))
#         second_w = (second_w * xw).permute(0, 3, 2, 1)
#
#         y = second_c + second_h + second_w + x
#         top_b = F.sigmoid(self.conv3(y))
#         return x * top_b
def Get_gradient_nopaddind(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    # 定义sobel算子参数
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    # 对图像进行卷积操作
    edge_detect = conv_op(im)
    # 将输出转换为图片格式
    edge_detect = edge_detect.detach()
    return edge_detect


# class Gradient_Map(nn.Module):
#     def __init__(self):
#         super(Gradient_Map, self).__init__()
#         kernel_v = [[0, -1, 0],
#                     [0, 0, 0],
#                     [0, 1, 0]]
#         kernel_h = [[0, 0, 0],
#                     [-1, 0, 1],
#                     [0, 0, 0]]
#         kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
#         kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
#         self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
#         self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()
#
#     def forward(self, x):
#         x0 = x[:, 0]
#         x1 = x[:, 1]
#         x2 = x[:, 2]
#         x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
#         x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)
#
#         x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
#         x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)
#
#         x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
#         x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)
#
#         x00 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
#         x11 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
#         x22 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)
#
#         x3 = torch.cat([x00, x11, x22], dim=1)
#
#         return x3


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=5, padding=5 // 2),
            nn.PReLU(28)
        )
        self.first_part11 = nn.Sequential(
            nn.Conv2d(6, 28, kernel_size=5, padding=5 // 2),
            nn.PReLU(28)
        )
        self.mid_part = Net()
        self.mid_part1 = Net()
        self.fusion1 = Fusion()
        self.fusion2 = Fusion()
        self.fusion3 = Fusion()
        self.fusion4 = Fusion()
        # self.fusion5 = Fusion()
        # self.Gradient_Map=Gradient_Map()
        self.Net_conv = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=5, padding=5 // 2),
            nn.PReLU(28)
        )
        self.Net2 = Net()
        self.Net3 = Net()
        self.Net4 = Net()
        self.Net5 = Net()

        self.Transformer = Transformers()
        self.Transformer1 = Transformers()

        self.Transformer2 = Transformers()
        self.Transformer3 = Transformers()
        self.Transformer4 = Transformers()

        self.mid_part3 = Net()
        self.mid_part4 = Net()
        self.mid_part5 = Net()
        self.mid_part6 = Net()
        self.mid_part7 = Net()
        self.mid_part8 = Net()
        self.mid_part9 = Net()
        self.mid_part10 = Net()
        self.last_part0 = nn.ConvTranspose2d(28, num_channels, kernel_size=9, stride=scale_factor, padding=9 // 2,
                                             output_padding=scale_factor - 1)
        self.last_part = nn.ConvTranspose2d(28, num_channels, kernel_size=9, stride=scale_factor, padding=9 // 2,
                                            output_padding=scale_factor - 1)
        self.last_part1 = nn.ConvTranspose2d(56, num_channels, kernel_size=9, stride=scale_factor, padding=9 // 2,
                                             output_padding=scale_factor - 1)
        self._initialize_weights()

        self.Atention0 = Atention()
        self.Atention1 = Atention()
        self.Atention2 = Atention()
        self.Atention3 = Atention()
        self.Atention4 = Atention()

        self.convc = nn.Sequential(
            nn.Conv2d(56, 28, kernel_size=3, padding=3 // 2),
            nn.PReLU(28)
        )
        self.convd = nn.Sequential(
            nn.Conv2d(56, 28, kernel_size=3, padding=3 // 2),
            nn.PReLU(28))
        self.conve = nn.Sequential(
            nn.Conv2d(56, 28, kernel_size=3, padding=3 // 2),
            nn.PReLU(28))
        self.convf = nn.Sequential(
            nn.Conv2d(56, 28, kernel_size=3, padding=3 // 2),
            nn.PReLU(28))
        self.convg = nn.Sequential(
            nn.Conv2d(56, 28, kernel_size=3, padding=3 // 2),
            nn.PReLU(28))
        self.refine = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.convfirst = nn.Conv2d(in_channels=1, out_channels=28, kernel_size=3, stride=1, padding=1)
        self.refine = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)

        # self.gra_conv1 = Gra_conv()
        # self.gra_conv2 = Gra_conv()
        # self.gra_conv3 = Gra_conv()
        # self.gra_conv4 = Gra_conv()
        # self.gra_conv5 = Gra_conv()

        self.conv1 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=1)
        self.conv7 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=1)
        self.conv8 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=1)

        self.conv9 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=1)
        self.conv10 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=1)

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                # nn.init.zeros_(m.bias.data)
        # for m in self.mid_part:
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight.data, mean=0.0,
        #                         std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
        #         # nn.init.zeros_(m.bias.data)
        # nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        # # nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x,lr_texture):
       # a, b, c, d = x.shape
       # x = x.cpu().numpy()
       # y2 = numpy.zeros((a,1,c,d))
       #
       # for i in range(a):
       #     x_out = x[i, :, :, :]
       #     x_out = x_out * 255
       #     x_out=x_out.transpose(1,2,0)
       #     gray = cv2.cvtColor(x_out, cv2.COLOR_RGB2GRAY)
       #     x_out = rotation_invariant_LBP(gray)
       #     x1 = torch.from_numpy(x_out)
       #     y2[i, :, :, :] = x1
       #     y2 = y2 / 255.0
       #
       # y2 = torch.from_numpy(y2)
       # y2 = y2.type(torch.FloatTensor)
       # y2 = y2.cuda()
       #
       # x = torch.from_numpy(x).cuda()

        # x_grat =Get_gradient_nopaddind(x)
        #
        # x_grat1 = self.Net_conv(x_grat)
        # x_grat2 = self.Net2(x_grat1)
        # x_grat3 = self.Net3(x_grat2)
        # x_grat4 = self.Net4(x_grat3)
        # x_grat5 = self.Net5(x_grat4)

        out_1 = self.first_part(x)
        x_t = torch.cat((x, lr_texture), 1)
        # out_1 = self.convc(out_1)
        out_2 = self.first_part11(x_t)
        out = torch.cat((out_1, out_2), 1)
        att0 = self.Atention0(out)

        out_1 = self.mid_part(out_1)
        # out_1 = torch.cat((out_1, x_grat2), 1)
        # out_1 = self.convd(out_1)
        out_2 = self.mid_part1(out_2)

        out_a1 = self.conv1(out_1)
        out_a2 = self.conv2(out_2)
        out_11 = att0 * out_a1
        out_12 = (1 - att0) * out_a2
        # out1 = torch.cat((out_11, out_12), 1)
        out1 = self.fusion1(out_11, out_12)

        out_1 = self.mid_part3(out_1)
        # out_1 = torch.cat((out_1, x_grat3), 1)
        # out_1 = self.conve(out_1)
        out_2 =self.mid_part4(out_2)

        out_a3 = self.conv3(out_1)
        out_a4 = self.conv4(out_2)
        att1 = self.Atention1(out1)
        out_13 = att1 * out_a3
        out_14 = (1 - att1) * out_a4
        # out2 = torch.cat((out_13, out_14), 1)
        out2 = self.fusion2(out_13, out_14)

        out_1 = self.mid_part5(out_1)
        # out_1 = torch.cat((out_1, x_grat4), 1)
        # out_1 = self.convf(out_1)
        out_2 = self.mid_part6(out_2)

        out_a5 = self.conv5(out_1)
        out_a6 = self.conv6(out_2)
        att2 = self.Atention2(out2)
        out_15 = att2 * out_a5

        out_16 = (1 - att2) * out_a6
        # out3= torch.cat((out_15, out_16), 1)
        out3 = self.fusion3(out_15, out_16)

        out_1 = self.mid_part7(out_1)
        # out_1 = torch.cat((out_1, x_grat5), 1)
        # out_1 = self.convg(out_1)
        out_2 = self.mid_part8(out_2)
#        out_2 = self.Transformer1(out_2)
        out_a7 = self.conv7(out_1)
        out_a8 = self.conv8(out_2)
        att3 = self.Atention3(out3)
        out_17 = att3 * out_a7
        out_18 = (1 - att3) * out_a8
        # out4= torch.cat((out_17, out_18), 1)
        out4 = self.fusion4(out_17, out_18)

        out_1 = self.mid_part9(out_1)
        # out_1 = torch.cat((out_1, x_grat5), 1)
        # out_1 = self.convg(out_1)
        out_2 = self.mid_part10(out_2)
#        out_2 = self.Transformer(out_2)
        out_a7 = self.conv9(out_1)
        out_a8 = self.conv10(out_2)
        att4 = self.Atention4(out4)
        out_19 = att4 * out_a7
        out_20 = (1 - att4) * out_a8
        # out5 = torch.cat((out_19, out_20), 1)
        out5 = self.fusion4(out_19, out_20)

        out_1 = self.last_part0(out_1)
        out_2 = self.last_part(out_2)
        out_s = self.last_part1(out5)

        output = torch.cat((out_2, out_s), 1)
        output = self.refine(output)

        return out_1, out_2, output


