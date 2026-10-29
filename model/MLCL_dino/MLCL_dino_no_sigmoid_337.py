"""
@author: yuchuang,zhaojinmiao
@time: 
@desc: 这个版本是MLCL-Net的基础版本（论文一致）。即每阶段用了3个block      paper:"Infrared small target detection based on multiscale local contrast learning networks"
"""
import torch
import torch.nn as nn
from .dinov2 import vit_large
import torch.nn.functional as F

class Resnet1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resnet1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU(inplace=True)
        #  网络初始化
        self.layer.apply(weights_init)

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out += identity
        return self.relu(out)


class Resnet2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resnet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        #  网络初始化
        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        identity = self.layer2(identity)
        out += identity
        return self.relu(out)


class Stage(nn.Module):
    def __init__(self):
        super(Stage, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.resnet1_1 = Resnet1(in_channel=16, out_channel=16)
        self.resnet1_2 = Resnet1(in_channel=16, out_channel=16)
        self.resnet1_3 = Resnet1(in_channel=16, out_channel=16)
        self.resnet2_1 = Resnet2(in_channel=16, out_channel=32)
        self.resnet2_2 = Resnet1(in_channel=32, out_channel=32)
        self.resnet2_3 = Resnet1(in_channel=32, out_channel=32)
        self.resnet3_1 = Resnet2(in_channel=32, out_channel=64)
        self.resnet3_2 = Resnet1(in_channel=64, out_channel=64)
        self.resnet3_3 = Resnet1(in_channel=64, out_channel=64)

    def forward(self, x):
        outs = []
        out = self.layer1(x)
        out = self.resnet1_1(out)
        out = self.resnet1_2(out)
        out = self.resnet1_3(out)
        outs.append(out)
        out = self.resnet2_1(out)
        out = self.resnet2_2(out)
        out = self.resnet2_3(out)
        outs.append(out)
        out = self.resnet3_1(out)
        out = self.resnet3_2(out)
        out = self.resnet3_3(out)
        outs.append(out)
        return outs


class MLCL(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLCL, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=3, stride=1, dilation=3),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=5, stride=1, dilation=5),
            nn.ReLU(inplace=True)
        )
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=7, padding=2, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=7, stride=1,dilation=7),
        #     nn.ReLU(inplace=True)
        # )
        self.conv = nn.Conv2d(in_channels=out_channel * 3, out_channels=out_channel, kernel_size=1)
        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)
        self.layer3.apply(weights_init)
        self.conv.apply(weights_init)
    def forward(self, x):
        x1 = x
        x2 = x
        x3 = x
        out1 = self.layer1(x1)
        out2 = self.layer2(x2)
        out3 = self.layer3(x3)
        outs = torch.cat((out1, out2, out3), dim=1)
        return self.conv(outs)



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


class MLCL_dino_No_Sigmoid(nn.Module):
    def __init__(self):
        super(MLCL_dino_No_Sigmoid, self).__init__()
        self.dinov2_feature = Dinov2_feature(amp=True)
        self.stage = Stage()
        self.mlcl3 = MLCL(64, 64)
        self.mlcl2 = MLCL(32, 32)
        self.mlcl1 = MLCL(16, 16)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            #nn.Sigmoid()
        )

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
        self.loss_fn = torch.nn.MSELoss()


    def forward(self, x):
        dino_out = self.dinov2_feature(x)
        outs = self.stage(x)

        dino_out_4 = dino_out[23]
        dino_out_3 = dino_out[17]
        dino_out_2 = dino_out[11]
        dino_out_1 = dino_out[5]

        out_b_4, out_c_4, out_h_4, out_w_4 = outs[2].shape

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

        out_layer_3_branch1 = outs[2][:, 0:32, :, :] * dino_out_1_convert_mul
        out_layer_3_branch1 = self.layer_branch_3(out_layer_3_branch1)
        out_layer_3_branch2 = outs[2][:, 32:64, :, :] + dino_out_1_convert_add
        out_layer_3_branch2 = self.layer_branch_3_2(out_layer_3_branch2)
        out_layer_3 = torch.cat((out_layer_3_branch1, out_layer_3_branch2), dim=1)
        out_layer_3 = out_layer_3 + outs[2]
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

        out3 = self.mlcl3(out_layer_0)
        out2 = self.mlcl2(outs[1])
        out1 = self.mlcl1(outs[0])
        out3 = self.conv3(out3)  # 128*128 64
        out3 = self.up3(out3)    # 256*256 64
        out2 = self.conv2(out2)  # 256*256 64
        out = out3 + out2
        out = self.up2(out)
        out1 = self.conv1(out1)
        out = out + out1
        out = self.layer(out)

        ##############################
        outs_2 = self.stage(x)

        out3_22 = self.mlcl3(outs_2[2])
        out2_22 = self.mlcl2(outs_2[1])
        out1_22 = self.mlcl1(outs_2[0])
        out3_22 = self.conv3(out3_22)  # 128*128 64
        out3_22 = self.up3(out3_22)  # 256*256 64
        out2_22 = self.conv2(out2_22)  # 256*256 64
        out_22 = out3_22 + out2_22
        out_22 = self.up2(out_22)
        out1_22 = self.conv1(out1_22)
        out_22 = out_22 + out1_22
        out_22 = self.layer(out_22)

        distillation_loss = self.loss_fn(out, out_22)

        return out_22, out, distillation_loss

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

# if __name__ == '__main__':
#     model = MLCL_No_Sigmoid()
#     x = torch.rand(8, 3, 512, 512)
#     outs = model(x)
#     print(outs.size())

