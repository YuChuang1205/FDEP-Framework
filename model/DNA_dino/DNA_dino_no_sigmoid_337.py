import torch
import torch.nn as nn
from .dinov2 import vit_large
import torch.nn.functional as F

class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


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

class DNA_dino_No_Sigmoid(nn.Module):
    def __init__(self, num_classes=1,input_channels=3, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256], deep_supervision=True, mode='test'):
        super(DNA_dino_No_Sigmoid, self).__init__()
        self.dinov2_feature = Dinov2_feature(amp=True)
        self.mode = mode
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

        ##########################
        self.layer_convert_4_mul = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.layer_convert_4_mul.apply(weights_init)
        self.layer_convert_4_add = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_convert_4_add.apply(weights_init)

        self.layer_convert_3_mul = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.layer_convert_3_mul.apply(weights_init)
        self.layer_convert_3_add = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_convert_3_add.apply(weights_init)

        self.layer_convert_2_mul = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.layer_convert_2_mul.apply(weights_init)
        self.layer_convert_2_add = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_convert_2_add.apply(weights_init)

        self.layer_convert_1_mul = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.layer_convert_1_mul.apply(weights_init)
        self.layer_convert_1_add = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_convert_1_add.apply(weights_init)

        self.layer_convert_0_mul = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.layer_convert_0_mul.apply(weights_init)
        self.layer_convert_0_add = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_convert_0_add.apply(weights_init)
        ##########################

        ##########################
        self.layer_conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_conv_4.apply(weights_init)

        self.layer_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_conv_3.apply(weights_init)

        self.layer_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_conv_2.apply(weights_init)

        self.layer_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_conv_1.apply(weights_init)

        self.layer_conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_conv_0.apply(weights_init)
        ##########################

        ##########################
        self.layer_branch_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_4.apply(weights_init)

        self.layer_branch_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_4_2.apply(weights_init)

        self.layer_branch_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_4_3.apply(weights_init)

        self.layer_branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_3.apply(weights_init)

        self.layer_branch_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_3_2.apply(weights_init)

        self.layer_branch_3_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_3_3.apply(weights_init)

        self.layer_branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_2.apply(weights_init)

        self.layer_branch_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_2_2.apply(weights_init)

        self.layer_branch_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_2_3.apply(weights_init)

        self.layer_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_1.apply(weights_init)

        self.layer_branch_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_1_2.apply(weights_init)

        self.layer_branch_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_1_3.apply(weights_init)

        self.layer_branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_0.apply(weights_init)

        self.layer_branch_0_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_0_2.apply(weights_init)

        self.layer_branch_0_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_0_3.apply(weights_init)
        ##########################
        self.loss_fn = torch.nn.MSELoss()

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        dino_out = self.dinov2_feature(input)
        dino_out_4 = dino_out[23]
        dino_out_3 = dino_out[17]
        dino_out_2 = dino_out[11]
        dino_out_1 = dino_out[5]

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))

        out_b_4, out_c_4, out_h_4, out_w_4 = x4_0.shape

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

        out_layer_3_branch1 = x4_0[:, 0:128, :, :] * dino_out_1_convert_mul
        out_layer_3_branch1 = self.layer_branch_3(out_layer_3_branch1)
        out_layer_3_branch2 = x4_0[:, 128:256, :, :] + dino_out_1_convert_add
        out_layer_3_branch2 = self.layer_branch_3_2(out_layer_3_branch2)
        out_layer_3 = torch.cat((out_layer_3_branch1, out_layer_3_branch2), dim=1)
        out_layer_3 = out_layer_3 + x4_0
        out_layer_3 = self.layer_conv_3(out_layer_3)

        out_layer_2_branch1 = out_layer_3[:, 0:128, :, :] * dino_out_2_convert_mul
        out_layer_2_branch1 = self.layer_branch_2(out_layer_2_branch1)
        out_layer_2_branch2 = out_layer_3[:, 128:256, :, :] + dino_out_2_convert_add
        out_layer_2_branch2 = self.layer_branch_2_2(out_layer_2_branch2)
        out_layer_2 = torch.cat((out_layer_2_branch1, out_layer_2_branch2), dim=1)
        out_layer_2 = out_layer_2 + out_layer_3
        out_layer_2 = self.layer_conv_2(out_layer_2)

        out_layer_1_branch1 = out_layer_2[:, 0:128, :, :] * dino_out_3_convert_mul
        out_layer_1_branch1 = self.layer_branch_1(out_layer_1_branch1)
        out_layer_1_branch2 = out_layer_2[:, 128:256, :, :] + dino_out_3_convert_add
        out_layer_1_branch2 = self.layer_branch_1_2(out_layer_1_branch2)
        out_layer_1 = torch.cat((out_layer_1_branch1, out_layer_1_branch2), dim=1)
        out_layer_1 = out_layer_1 + out_layer_2
        out_layer_1 = self.layer_conv_1(out_layer_1)

        out_layer_0_branch1 = out_layer_1[:, 0:128, :, :] * dino_out_4_convert_mul
        out_layer_0_branch1 = self.layer_branch_0(out_layer_0_branch1)
        out_layer_0_branch2 = out_layer_1[:, 128:256, :, :] + dino_out_4_convert_add
        out_layer_0_branch2 = self.layer_branch_0_2(out_layer_0_branch2)
        out_layer_0 = torch.cat((out_layer_0_branch1, out_layer_0_branch2), dim=1)
        out_layer_0 = out_layer_0 + out_layer_1
        out_layer_0 = self.layer_conv_0(out_layer_0)



        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(out_layer_0),self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(out_layer_0)),self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))

        ##################################################################################

        x0_0_di = self.conv0_0(input)
        x1_0_di = self.conv1_0(self.pool(x0_0_di))
        x0_1_di = self.conv0_1(torch.cat([x0_0_di, self.up(x1_0_di)], 1))

        x2_0_di = self.conv2_0(self.pool(x1_0_di))
        x1_1_di = self.conv1_1(torch.cat([x1_0_di, self.up(x2_0_di), self.down(x0_1_di)], 1))
        x0_2_di = self.conv0_2(torch.cat([x0_0_di, x0_1_di, self.up(x1_1_di)], 1))

        x3_0_di = self.conv3_0(self.pool(x2_0_di))
        x2_1_di = self.conv2_1(torch.cat([x2_0_di, self.up(x3_0_di), self.down(x1_1_di)], 1))
        x1_2_di = self.conv1_2(torch.cat([x1_0_di, x1_1_di, self.up(x2_1_di), self.down(x0_2_di)], 1))
        x0_3_di = self.conv0_3(torch.cat([x0_0_di, x0_1_di, x0_2_di, self.up(x1_2_di)], 1))

        x4_0_di = self.conv4_0(self.pool(x3_0_di))

        x3_1_di = self.conv3_1(torch.cat([x3_0_di, self.up(x4_0_di), self.down(x2_1_di)], 1))
        x2_2_di = self.conv2_2(torch.cat([x2_0_di, x2_1_di, self.up(x3_1_di), self.down(x1_2_di)], 1))
        x1_3_di = self.conv1_3(torch.cat([x1_0_di, x1_1_di, x1_2_di, self.up(x2_2_di), self.down(x0_3_di)], 1))
        x0_4_di = self.conv0_4(torch.cat([x0_0_di, x0_1_di, x0_2_di, x0_3_di, self.up(x1_3_di)], 1))

        Final_x0_4_di = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0_di)), self.up_8(self.conv0_3_1x1(x3_1_di)),
                       self.up_4(self.conv0_2_1x1(x2_2_di)), self.up(self.conv0_1_1x1(x1_3_di)), x0_4_di], 1))

        if self.deep_supervision:
            # output1 = self.final1(x0_1).sigmoid()
            # output2 = self.final2(x0_2).sigmoid()
            # output3 = self.final3(x0_3).sigmoid()
            # output4 = self.final4(Final_x0_4).sigmoid()
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)

            output1_di = self.final1(x0_1_di)
            output2_di = self.final2(x0_2_di)
            output3_di = self.final3(x0_3_di)
            output4_di = self.final4(Final_x0_4_di)

            distillation_loss = self.loss_fn(output4, output4_di)

            if self.mode == 'train':
                return [output4_di, output3_di, output2_di, output1_di, output4, output3, output2, output1,
                        distillation_loss]
            else:
                return output4_di
        else:
            # output = self.final(Final_x0_4).sigmoid()
            output = self.final(Final_x0_4)
            output_di = self.final(Final_x0_4_di)
            distillation_loss = self.loss_fn(output, output_di)
            return output_di, output, distillation_loss



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):  # bn需要初始化的前提是affine=True
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    # elif isinstance(m, nn.Linear):
    #     nn.init.xavier_uniform_(m.weight)
    #     nn.init.constant_(m.bias, 0)
    return
