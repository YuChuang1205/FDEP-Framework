import torch
import torch.nn as nn
from .dinov2 import vit_large
import torch.nn.functional as F
#############################################
from .fusion import *
##################################################

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

def _upsample_like(src,tar):

    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin

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


##### UIU-net ####
class UIU_dino_No_Sigmoid(nn.Module):

    def __init__(self, in_ch=3, out_ch=1,mode='test'):
        super(UIU_dino_No_Sigmoid, self).__init__()
        self.dinov2_feature = Dinov2_feature(amp=True)
        self.mode = mode
        
        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

        #self.fuse6 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse5 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse4 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse3 = self._fuse_layer(256, 256, 256, fuse_mode='AsymBi')
        self.fuse2 = self._fuse_layer(128, 128, 128, fuse_mode='AsymBi')

        ##########################
        self.layer_convert_4_mul = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.layer_convert_4_mul.apply(weights_init)
        self.layer_convert_4_add = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_convert_4_add.apply(weights_init)

        self.layer_convert_3_mul = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.layer_convert_3_mul.apply(weights_init)
        self.layer_convert_3_add = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_convert_3_add.apply(weights_init)

        self.layer_convert_2_mul = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.layer_convert_2_mul.apply(weights_init)
        self.layer_convert_2_add = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_convert_2_add.apply(weights_init)

        self.layer_convert_1_mul = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.layer_convert_1_mul.apply(weights_init)
        self.layer_convert_1_add = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_convert_1_add.apply(weights_init)

        self.layer_convert_0_mul = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.layer_convert_0_mul.apply(weights_init)
        self.layer_convert_0_add = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_convert_0_add.apply(weights_init)
        ##########################

        ##########################
        self.layer_conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer_conv_4.apply(weights_init)

        self.layer_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer_conv_3.apply(weights_init)

        self.layer_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer_conv_2.apply(weights_init)

        self.layer_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer_conv_1.apply(weights_init)

        self.layer_conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer_conv_0.apply(weights_init)
        ##########################

        ##########################
        self.layer_branch_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_4.apply(weights_init)

        self.layer_branch_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_4_2.apply(weights_init)

        self.layer_branch_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_4_3.apply(weights_init)

        self.layer_branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_3.apply(weights_init)

        self.layer_branch_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_3_2.apply(weights_init)

        self.layer_branch_3_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_3_3.apply(weights_init)

        self.layer_branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_2.apply(weights_init)

        self.layer_branch_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_2_2.apply(weights_init)

        self.layer_branch_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_2_3.apply(weights_init)

        self.layer_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_1.apply(weights_init)

        self.layer_branch_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_1_2.apply(weights_init)

        self.layer_branch_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_1_3.apply(weights_init)

        self.layer_branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_0.apply(weights_init)

        self.layer_branch_0_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_0_2.apply(weights_init)

        self.layer_branch_0_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer_branch_0_3.apply(weights_init)
        ##########################
        self.loss_fn = torch.nn.MSELoss()

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels,fuse_mode='AsymBi'):#fuse_mode='AsymBi'
        # assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        # if fuse_mode == 'BiLocal':
        #     fuse_layer = BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        # el
        if fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        # elif fuse_mode == 'BiGlobal':
        #     fuse_layer = BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer


    def forward(self, x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)

        dino_out = self.dinov2_feature(x)
        dino_out_4 = dino_out[23]
        dino_out_3 = dino_out[17]
        dino_out_2 = dino_out[11]
        dino_out_1 = dino_out[5]

        out_b_4, out_c_4, out_h_4, out_w_4 = hx6.shape

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

        out_layer_3_branch1 = hx6[:, 0:256, :, :] * dino_out_1_convert_mul
        out_layer_3_branch1 = self.layer_branch_3(out_layer_3_branch1)
        out_layer_3_branch2 = hx6[:, 256:512, :, :] + dino_out_1_convert_add
        out_layer_3_branch2 = self.layer_branch_3_2(out_layer_3_branch2)
        out_layer_3 = torch.cat((out_layer_3_branch1, out_layer_3_branch2), dim=1)
        out_layer_3 = out_layer_3 + hx6
        out_layer_3 = self.layer_conv_3(out_layer_3)

        out_layer_2_branch1 = out_layer_3[:, 0:256, :, :] * dino_out_2_convert_mul
        out_layer_2_branch1 = self.layer_branch_2(out_layer_2_branch1)
        out_layer_2_branch2 = out_layer_3[:, 256:512, :, :] + dino_out_2_convert_add
        out_layer_2_branch2 = self.layer_branch_2_2(out_layer_2_branch2)
        out_layer_2 = torch.cat((out_layer_2_branch1, out_layer_2_branch2), dim=1)
        out_layer_2 = out_layer_2 + out_layer_3
        out_layer_2 = self.layer_conv_2(out_layer_2)

        out_layer_1_branch1 = out_layer_2[:, 0:256, :, :] * dino_out_3_convert_mul
        out_layer_1_branch1 = self.layer_branch_1(out_layer_1_branch1)
        out_layer_1_branch2 = out_layer_2[:, 256:512, :, :] + dino_out_3_convert_add
        out_layer_1_branch2 = self.layer_branch_1_2(out_layer_1_branch2)
        out_layer_1 = torch.cat((out_layer_1_branch1, out_layer_1_branch2), dim=1)
        out_layer_1 = out_layer_1 + out_layer_2
        out_layer_1 = self.layer_conv_1(out_layer_1)

        out_layer_0_branch1 = out_layer_1[:, 0:256, :, :] * dino_out_4_convert_mul
        out_layer_0_branch1 = self.layer_branch_0(out_layer_0_branch1)
        out_layer_0_branch2 = out_layer_1[:, 256:512, :, :] + dino_out_4_convert_add
        out_layer_0_branch2 = self.layer_branch_0_2(out_layer_0_branch2)
        out_layer_0 = torch.cat((out_layer_0_branch1, out_layer_0_branch2), dim=1)
        out_layer_0 = out_layer_0 + out_layer_1
        out_layer_0 = self.layer_conv_0(out_layer_0)


        hx6up = _upsample_like(out_layer_0,hx5)

        #-------------------- decoder --------------------

        fusec51,fusec52 = self.fuse5(hx6up, hx5)
        hx5d = self.stage5d(torch.cat((fusec51, fusec52),1))
        hx5dup = _upsample_like(hx5d,hx4)

        fusec41,fusec42 = self.fuse4(hx5dup, hx4)
        hx4d = self.stage4d(torch.cat((fusec41,fusec42),1))
        hx4dup = _upsample_like(hx4d,hx3)


        fusec31,fusec32 = self.fuse3(hx4dup, hx3)
        hx3d = self.stage3d(torch.cat((fusec31,fusec32),1))
        hx3dup = _upsample_like(hx3d,hx2)

        fusec21, fusec22 = self.fuse2(hx3dup, hx2)
        hx2d = self.stage2d(torch.cat((fusec21, fusec22), 1))
        hx2dup = _upsample_like(hx2d,hx1)


        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d22 = self.side2(hx2d)
        d2 = _upsample_like(d22,d1)

        d32 = self.side3(hx3d)
        d3 = _upsample_like(d32,d1)

        d42 = self.side4(hx4d)
        d4 = _upsample_like(d42,d1)

        d52 = self.side5(hx5d)
        d5 = _upsample_like(d52,d1)

        d62 = self.side6(out_layer_0)
        d6 = _upsample_like(d62,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        #############################################
        hx_22 = x

        # stage 1
        hx1_22 = self.stage1(hx_22)
        hx_22 = self.pool12(hx1_22)

        # stage 2
        hx2_22 = self.stage2(hx_22)
        hx_22 = self.pool23(hx2_22)

        # stage 3
        hx3_22 = self.stage3(hx_22)
        hx_22 = self.pool34(hx3_22)

        # stage 4
        hx4_22 = self.stage4(hx_22)
        hx_22 = self.pool45(hx4_22)

        # stage 5
        hx5_22 = self.stage5(hx_22)
        hx_22 = self.pool56(hx5_22)

        # stage 6
        hx6_22 = self.stage6(hx_22)

        hx6up_22 = _upsample_like(hx6_22, hx5_22)

        # -------------------- decoder --------------------

        fusec51_22, fusec52_22 = self.fuse5(hx6up_22, hx5_22)
        hx5d_22 = self.stage5d(torch.cat((fusec51_22, fusec52_22), 1))
        hx5dup_22 = _upsample_like(hx5d_22, hx4_22)

        fusec41_22, fusec42_22 = self.fuse4(hx5dup_22, hx4_22)
        hx4d_22 = self.stage4d(torch.cat((fusec41_22, fusec42_22), 1))
        hx4dup_22 = _upsample_like(hx4d_22, hx3_22)

        fusec31_22, fusec32_22 = self.fuse3(hx4dup_22, hx3_22)
        hx3d_22 = self.stage3d(torch.cat((fusec31_22, fusec32_22), 1))
        hx3dup_22 = _upsample_like(hx3d_22, hx2_22)

        fusec21_22, fusec22_22 = self.fuse2(hx3dup_22, hx2_22)
        hx2d_22 = self.stage2d(torch.cat((fusec21_22, fusec22_22), 1))
        hx2dup_22 = _upsample_like(hx2d_22, hx1_22)

        hx1d_22 = self.stage1d(torch.cat((hx2dup_22, hx1_22), 1))

        # side output
        d1_22 = self.side1(hx1d_22)

        d22_22 = self.side2(hx2d_22)
        d2_22 = _upsample_like(d22_22, d1_22)

        d32_22 = self.side3(hx3d_22)
        d3_22 = _upsample_like(d32_22, d1_22)

        d42_22 = self.side4(hx4d_22)
        d4_22 = _upsample_like(d42_22, d1_22)

        d52_22 = self.side5(hx5d_22)
        d5_22 = _upsample_like(d52_22, d1_22)

        d62_22 = self.side6(hx6_22)
        d6_22 = _upsample_like(d62_22, d1_22)

        d0_22 = self.outconv(torch.cat((d1_22, d2_22, d3_22, d4_22, d5_22, d6_22), 1))

        distillation_loss = self.loss_fn(d0, d0_22)

        if self.mode == 'train':
            return d0_22, d1_22, d2_22, d3_22, d4_22, d5_22, d6_22, d0, d1, d2, d3, d4, d5, d6, distillation_loss
        else:
            return d0_22


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
