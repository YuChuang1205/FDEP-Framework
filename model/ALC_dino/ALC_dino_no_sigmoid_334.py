from __future__ import division
import os
from torch.nn.modules import module
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
from .fusion import AsymBiChaFuse

# from mxnet import nd
from torchvision import transforms
from  torchvision.models.resnet import BasicBlock
import torch.nn.functional as F
from .dinov2 import vit_large
class _FCNHead(nn.Module):
    # pylint: disable=redefined-outer-name
    def __init__(self, in_channels, channels, momentum, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,kernel_size=3, padding=1, bias=False),
        norm_layer(inter_channels, momentum=momentum),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(in_channels=inter_channels, out_channels=channels,kernel_size=1)
        )
    # pylint: disable=arguments-differ
    def forward(self, x):
        return self.block(x)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Dinov2_feature(nn.Module):
    def __init__(self, amp=False, dinov2_weights=None, amp_dtype=torch.float16):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        vit_kwargs = dict(img_size=518,
                          patch_size=14,
                          init_values=1.0,
                          ffn_layer="mlp",
                          block_chunks=0,
                          )

        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        # self.amp = amp
        # self.amp_dtype = amp_dtype
        # if self.amp:
        #     dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14]  # ugly hack to not show parameters to DDP

    def forward(self, x, upsample=False):
        B, C, H, W = x.shape
        if not upsample:
            with torch.no_grad():
                if self.dinov2_vitl14[0].device != x.device:
                    self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device)
                ##return_class_token为true时将是 tuple(zip(outputs, class_tokens))
                dinov2_features_multi_stage = self.dinov2_vitl14[0].get_intermediate_layers(x, n=24, reshape=True, return_class_token=False)


        return dinov2_features_multi_stage

class ALC_dino_No_Sigmoid(nn.Module):
    def __init__(self, in_channels=3, layers=[4,4,4], channels=[8,16,32,64], fuse_mode='AsymBi', act_dilation=16, classes=1, tinyFlag=False,
                 norm_layer=BatchNorm2d,groups=1,norm_kwargs=None, **kwargs):
        super(ALC_dino_No_Sigmoid, self).__init__()
        self.dinov2_feature = Dinov2_feature(amp=True)

        self.layer_num = len(layers)
        self.tinyFlag = tinyFlag
        self.groups = groups
        self._norm_layer = norm_layer
        stem_width = int(channels[0])
        self.momentum=0.9
        if tinyFlag:
            self.stem = nn.Sequential(
                norm_layer(in_channels, self.momentum),
                nn.Conv2d(in_channels, out_channels=stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width * 2, momentum=self.momentum),
                nn.ReLU(inplace=True)
            )

        else:
            self.stem = nn.Sequential(
                # self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=2,
                #                          padding=1, use_bias=False))
                # self.stem.add(norm_layer(in_channels=stem_width*2))
                # self.stem.add(nn.Activation('relu'))
                # self.stem.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
                norm_layer(in_channels, momentum=self.momentum),
                nn.Conv2d(in_channels=in_channels, out_channels=stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=stem_width, out_channels=stem_width, kernel_size=3, stride=1, padding=1,
                          bias=False),
                norm_layer(stem_width, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=stem_width, out_channels=stem_width * 2, kernel_size=3, stride=1, padding=1,
                          bias=False),
                norm_layer(stem_width * 2, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )


            # self.head1 = _FCNHead(in_channels=channels[1], channels=classes)
            # self.head2 = _FCNHead(in_channels=channels[2], channels=classes)
            # self.head3 = _FCNHead(in_channels=channels[3], channels=classes)
            # self.head4 = _FCNHead(in_channels=channels[4], channels=classes)

            self.head = _FCNHead(in_channels=channels[0], channels=classes, momentum=self.momentum)

            self.layer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                           out_channels=channels[1],
                                           in_channels=channels[1], stride=1)

            self.layer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                           out_channels=channels[2], stride=2,
                                           in_channels=channels[1])
            #
            self.layer3 = self._make_layer(block=BasicBlock, blocks=layers[2],
                                           out_channels=channels[3], stride=2,
                                           in_channels=channels[2])
            self.deconv2 = nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2], kernel_size=(4, 4),
                                              ##channels: 8 16 32 64
                                              stride=2, padding=1)
            self.uplayer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                             out_channels=channels[2], stride=1,
                                             in_channels=channels[2])


            self.deconv1 = nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[1], kernel_size=(4, 4),
                                              stride=2, padding=1)

            self.deconv0 = nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[0], kernel_size=(4, 4),
                                              stride=2, padding=1)

            self.uplayer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                             out_channels=channels[1], stride=1,
                                             in_channels=channels[1])


            if self.layer_num == 4:
                self.layer4 = self._make_layer(block=BasicBlock, blocks=layers[3],
                                               out_channels=channels[3], stride=2,
                                               in_channels=channels[3])

            if self.layer_num == 4:
                self.fuse34 = self._fuse_layer(fuse_mode, channels=channels[3])  # channels[4]

            self.fuse23 = self._fuse_layer(fuse_mode, channels=channels[2])  # 64
            self.fuse12 = self._fuse_layer(fuse_mode, channels=channels[1])  # 32

            # if fuse_order == 'reverse':
            #     self.fuse12 = self._fuse_layer(fuse_mode, channels=channels[2])  # channels[2]
            #     self.fuse23 = self._fuse_layer(fuse_mode, channels=channels[3])  # channels[3]
            #     self.fuse34 = self._fuse_layer(fuse_mode, channels=channels[4])  # channels[4]
            # elif fuse_order == 'normal':
            # self.fuse34 = self._fuse_layer(fuse_mode, channels=channels[4])  # channels[4]
            # self.fuse23 = self._fuse_layer(fuse_mode, channels=channels[4])  # channels[4]
            # self.fuse12 = self._fuse_layer(fuse_mode, channels=channels[4])  # channels[4]

            ##########################
            self.layer_convert_4_mul = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm2d(32),
                # nn.ReLU(inplace=True),
                nn.Sigmoid(),
            )
            self.layer_convert_4_mul.apply(weights_init)
            self.layer_convert_4_add = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_convert_4_add.apply(weights_init)

            self.layer_convert_3_mul = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm2d(32),
                # nn.ReLU(inplace=True),
                nn.Sigmoid(),
            )
            self.layer_convert_3_mul.apply(weights_init)
            self.layer_convert_3_add = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_convert_3_add.apply(weights_init)

            self.layer_convert_2_mul = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm2d(32),
                # nn.ReLU(inplace=True),
                nn.Sigmoid(),
            )
            self.layer_convert_2_mul.apply(weights_init)
            self.layer_convert_2_add = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_convert_2_add.apply(weights_init)

            self.layer_convert_1_mul = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm2d(32),
                # nn.ReLU(inplace=True),
                nn.Sigmoid(),
            )
            self.layer_convert_1_mul.apply(weights_init)
            self.layer_convert_1_add = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_convert_1_add.apply(weights_init)

            self.layer_convert_0_mul = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm2d(32),
                # nn.ReLU(inplace=True),
                nn.Sigmoid(),
            )
            self.layer_convert_0_mul.apply(weights_init)
            self.layer_convert_0_add = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_convert_0_add.apply(weights_init)
            ##########################

            ##########################
            self.layer_conv_4 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer_conv_4.apply(weights_init)

            self.layer_conv_3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer_conv_3.apply(weights_init)

            self.layer_conv_2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer_conv_2.apply(weights_init)

            self.layer_conv_1 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer_conv_1.apply(weights_init)

            self.layer_conv_0 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer_conv_0.apply(weights_init)
            ##########################

            ##########################
            self.layer_branch_4 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_4.apply(weights_init)

            self.layer_branch_4_2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_4_2.apply(weights_init)

            self.layer_branch_4_3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_4_3.apply(weights_init)

            self.layer_branch_3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_3.apply(weights_init)

            self.layer_branch_3_2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_3_2.apply(weights_init)

            self.layer_branch_3_3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_3_3.apply(weights_init)

            self.layer_branch_2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_2.apply(weights_init)

            self.layer_branch_2_2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_2_2.apply(weights_init)

            self.layer_branch_2_3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_2_3.apply(weights_init)

            self.layer_branch_1 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_1.apply(weights_init)

            self.layer_branch_1_2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_1_2.apply(weights_init)

            self.layer_branch_1_3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_1_3.apply(weights_init)

            self.layer_branch_0 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_0.apply(weights_init)

            self.layer_branch_0_2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_0_2.apply(weights_init)

            self.layer_branch_0_3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer_branch_0_3.apply(weights_init)
            ##########################

    def _make_layer(self, block, out_channels, in_channels, blocks, stride):

        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or out_channels != in_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels , stride),
                norm_layer(out_channels * block.expansion, momentum=self.momentum),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, self.groups, norm_layer=norm_layer))
        self.inplanes = out_channels  * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, self.groups, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _fuse_layer(self, fuse_mode, channels):

        if fuse_mode == 'AsymBi':
          fuse_layer = AsymBiChaFuse(channels=channels)
        else:
            raise ValueError('Unknown fuse_mode')
        return fuse_layer

    def forward(self, x):
        dino_out = self.dinov2_feature(x)

        dino_out_4 = dino_out[23]
        dino_out_3 = dino_out[17]
        dino_out_2 = dino_out[11]
        dino_out_1 = dino_out[5]

        _, _, hei, wid = x.shape# 1024 1024

        x_1 = self.stem(x)      #torch.Size([8, 16, 256, 256])
        c1 = self.layer1(x_1)   # torch.Size([8, 16, 256, 256])
        c2 = self.layer2(c1)  # torch.Size([8, 32, 128, 128])

        c4_out = self.layer3(c2)  # (8,64, 64, 64)

        if self.layer_num == 4:
            c4 = self.layer4(c4_out)  # torch.Size([8,64, 32, 32])
            if self.tinyFlag:
                c4 = transforms.Resize([hei // 4, wid // 4])(c4)  # down 4
            else:
                c4 = transforms.Resize([hei // 16, wid // 16])(c4)  # down 16 torch.Size([8, 64, 64, 64])
            c4_out = self.fuse34(c4, c4_out)  # torch.Size([8, 64, 128, 128])`

        if self.tinyFlag:
            c4_out = transforms.Resize([hei // 2, wid // 2])(c4_out)  # down 16 torch.Size([8, 64, 64, 64])
        else:
            c4_out = transforms.Resize([hei // 16, wid // 16])(c4_out)  # down 8, 128 torch.Size([8, 64, 64, 64])


        out_b_4, out_c_4, out_h_4, out_w_4 = c4_out.shape

        dino_out_4 = F.interpolate(dino_out_4, size=(out_h_4, out_w_4), mode='bilinear', align_corners=False)
        dino_out_4_convert_mul = self.layer_convert_4_mul(dino_out_4)
        dino_out_4_convert_add = self.layer_convert_4_add(dino_out_4)
        dino_out_3 = F.interpolate(dino_out_3, size=(out_h_4, out_w_4), mode='bilinear', align_corners=False)
        dino_out_3_convert_mul = self.layer_convert_3_mul(dino_out_3)
        dino_out_3_convert_add = self.layer_convert_3_add(dino_out_3)
        dino_out_2 = F.interpolate(dino_out_2, size=(out_h_4, out_w_4), mode='bilinear', align_corners=False)
        dino_out_2_convert_mul = self.layer_convert_2_mul(dino_out_2)
        dino_out_2_convert_add = self.layer_convert_2_add(dino_out_2)
        dino_out_1 = F.interpolate(dino_out_1, size=(out_h_4, out_w_4), mode='bilinear', align_corners=False)
        dino_out_1_convert_mul = self.layer_convert_1_mul(dino_out_1)
        dino_out_1_convert_add = self.layer_convert_1_add(dino_out_1)
        # dino_out_0 = F.interpolate(dino_out_0, size=(out_h_4, out_w_4), mode='bilinear', align_corners=False)
        # dino_out_0_convert_mul = self.layer_convert_0_mul(dino_out_0)
        # dino_out_0_convert_add = self.layer_convert_0_add(dino_out_0)

        # out_layer_4_branch1 = outs[4][:,0:32,:,:] * dino_out_0_convert_mul
        # out_layer_4_branch1 = self.layer_branch_4(out_layer_4_branch1)
        # out_layer_4_branch2 = outs[4][:, 32:64, :, :] + dino_out_0_convert_add
        # out_layer_4_branch2 = self.layer_branch_4_2(out_layer_4_branch2)
        # out_layer_4 = torch.cat((out_layer_4_branch1, out_layer_4_branch2), dim=1)
        # out_layer_4 = out_layer_4 + outs[4]
        # out_layer_4 = self.layer_conv_4(out_layer_4)

        out_layer_3_branch1 = c4_out[:, 0:32, :, :] * dino_out_1_convert_mul
        out_layer_3_branch1 = self.layer_branch_3(out_layer_3_branch1)
        out_layer_3_branch2 = c4_out[:, 32:64, :, :] + dino_out_1_convert_add
        out_layer_3_branch2 = self.layer_branch_3_2(out_layer_3_branch2)
        out_layer_3 = torch.cat((out_layer_3_branch1, out_layer_3_branch2), dim=1)
        out_layer_3 = out_layer_3 + c4_out
        out_layer_3 = self.layer_conv_3(out_layer_3)

        out_layer_2_branch1 = out_layer_3[:, 0:32, :, :] * dino_out_2_convert_mul
        out_layer_2_branch1 = self.layer_branch_2(out_layer_2_branch1)
        out_layer_2_branch2 = out_layer_3[:, 32:64, :, :] + dino_out_2_convert_add
        out_layer_2_branch2 = self.layer_branch_2_2(out_layer_2_branch2)
        out_layer_2 = torch.cat((out_layer_2_branch1, out_layer_2_branch2), dim=1)
        out_layer_2 = out_layer_2 + out_layer_3
        out_layer_2 = self.layer_conv_2(out_layer_2)

        out_layer_1_branch1 = out_layer_2[:, 0:32, :, :] * dino_out_3_convert_mul
        out_layer_1_branch1 = self.layer_branch_1(out_layer_1_branch1)
        out_layer_1_branch2 = out_layer_2[:, 32:64, :, :] + dino_out_3_convert_add
        out_layer_1_branch2 = self.layer_branch_1_2(out_layer_1_branch2)
        out_layer_1 = torch.cat((out_layer_1_branch1, out_layer_1_branch2), dim=1)
        out_layer_1 = out_layer_1 + out_layer_2
        out_layer_1 = self.layer_conv_1(out_layer_1)

        out_layer_0_branch1 = out_layer_1[:, 0:32, :, :] * dino_out_4_convert_mul
        out_layer_0_branch1 = self.layer_branch_0(out_layer_0_branch1)
        out_layer_0_branch2 = out_layer_1[:, 32:64, :, :] + dino_out_4_convert_add
        out_layer_0_branch2 = self.layer_branch_0_2(out_layer_0_branch2)
        out_layer_0 = torch.cat((out_layer_0_branch1, out_layer_0_branch2), dim=1)
        out_layer_0 = out_layer_0 + out_layer_1
        out_layer_0 = self.layer_conv_0(out_layer_0)



        de1_out = self.deconv2(out_layer_0) # torch.Size([8, 32, 128, 128])
        de1_out = self.fuse23(de1_out, c2) # torch.Size([8, 32, 128, 128])
        if self.tinyFlag:
            de1_out =  transforms.Resize([hei, wid])(de1_out)  # down 1
        else:
            de1_out =  transforms.Resize( [hei//8, wid//8])(de1_out)  # (4,16,120,120)

        de2_out = self.deconv1(de1_out)  # torch.Size([8, 16, 256, 256])
        de2_out = self.fuse12(de2_out, c1) # torch.Size([8, 16, 256, 256])

        de3_out = self.deconv0(de2_out)  # torch.Size([8, 8, 512, 512])
        pred = self.head(de3_out) # torch.Size([8, 8, 512, 512])


        if self.tinyFlag:
            out = pred
        else:
            out = transforms.Resize( [hei, wid])(pred)  # down 4

        ######### reverse order ##########
        # up_c2 = F.contrib.BilinearResize2D(c2, height=hei//4, width=wid//4)  # down 4
        # fuse2 = self.fuse12(up_c2, c1)  # down 4, channels[2]
        #
        # up_c3 = F.contrib.BilinearResize2D(c3, height=hei//4, width=wid//4)  # down 4
        # fuse3 = self.fuse23(up_c3, fuse2)  # down 4, channels[3]
        #
        # up_c4 = F.contrib.BilinearResize2D(c4, height=hei//4, width=wid//4)  # down 4
        # fuse4 = self.fuse34(up_c4, fuse3)  # down 4, channels[4]
        #

        ######### normal order ##########
        # out = F.contrib.BilinearResize2D(c4, height=hei//16, width=wid//16)
        # out = self.fuse34(out, c3)
        # out = F.contrib.BilinearResize2D(out, height=hei//8, width=wid//8)
        # out = self.fuse23(out, c2)
        # out = F.contrib.BilinearResize2D(out, height=hei//4, width=wid//4)
        # out = self.fuse12(out, c1)
        # out = self.head(out)
        # out = F.contrib.BilinearResize2D(out, height=hei, width=wid)


        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):  # bn需要初始化的前提是affine=True
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    return