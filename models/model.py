import torch.nn as nn
import torch.nn.functional as F
from models.MobileNetV2 import mobilenet_v2
from thop import profile
import torch
class REBnConv(nn.Module):
    def __init__(self, in_ch=3, out_channel=3, dilation_rate=1):
        super(REBnConv, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_channel, kernel_size=(3, 3), padding=1 * dilation_rate,
                                 dilation=1 * dilation_rate)
        self.bn_s1 = nn.BatchNorm2d(out_channel)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, midd_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.REBnConvin = REBnConv(in_ch, out_ch, dilation_rate=1)

        self.REBnConv1 = REBnConv(out_ch, midd_ch, dilation_rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv2 = REBnConv(midd_ch, midd_ch, dilation_rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv3 = REBnConv(midd_ch, midd_ch, dilation_rate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv4 = REBnConv(midd_ch, midd_ch, dilation_rate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv5 = REBnConv(midd_ch, midd_ch, dilation_rate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.REBnConv6 = REBnConv(midd_ch, midd_ch, dilation_rate=1)

        self.REBnConv7 = REBnConv(midd_ch, midd_ch, dilation_rate=2)

        self.REBnConv6d = REBnConv(midd_ch * 2, midd_ch, dilation_rate=1)
        self.REBnConv5d = REBnConv(midd_ch * 2, midd_ch, dilation_rate=1)
        self.REBnConv4d = REBnConv(midd_ch * 2, midd_ch, dilation_rate=1)
        self.REBnConv3d = REBnConv(midd_ch * 2, midd_ch, dilation_rate=1)
        self.REBnConv2d = REBnConv(midd_ch * 2, midd_ch, dilation_rate=1)
        self.REBnConv1d = REBnConv(midd_ch * 2, out_ch, dilation_rate=1)

    def forward(self, x):
        hx = x
        hx_input = self.REBnConvin(hx)

        hx1 = self.REBnConv1(hx_input)
        hx = self.pool1(hx1)

        hx2 = self.REBnConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.REBnConv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.REBnConv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.REBnConv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.REBnConv6(hx)

        hx7 = self.REBnConv7(hx6)

        hx6d = self.REBnConv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.REBnConv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.REBnConv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.REBnConv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.REBnConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.REBnConv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hx_input

def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

    return src

class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))

        return c_out


class SCFHM_RSU_1(nn.Module):
    def __init__(self,in_d,out_d):
        super(SCFHM_RSU_1,self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)
        self.stage_1 =RSU7(64,32,64)
        self.pool1 =nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)

    def forward(self,x1,x2):
        hx = torch.abs(x1 - x2)
        hx =self.stage_1(hx)
        hx = F.interpolate(hx, scale_factor=(2, 2), mode='bilinear')
        hx =self.pool1(hx)

        return hx


class MFNFM(nn.Module):
    def __init__(self,in_ch=None,out_channel=64):
        super(MFNFM,self).__init__()
        if in_ch is None:
            in_ch = [16,24,32,96,320]
        self.in_ch = in_ch
        self.mid_ch = out_channel // 2
        self.out_channel = out_channel

        self.conv_22 = nn.Sequential(
            nn.Conv2d(self.in_ch[1],self.mid_ch,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )

        self.conv_23 = nn.Sequential(
            nn.Conv2d(self.in_ch[2],self.mid_ch,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )

        self.conv_a2 = FeatureFusionModule(self.mid_ch*2,self.in_ch[1],self.out_channel)

        self.conv_32 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_ch[1], self.mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_33 = nn.Sequential(
            nn.Conv2d(self.in_ch[2], self.mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_34 = nn.Sequential(
            nn.Conv2d(self.in_ch[3], self.mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_a3 = FeatureFusionModule(self.mid_ch * 6, self.in_ch[2], self.out_channel)

        self.conv_43 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_ch[2], self.mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_44 = nn.Sequential(
            nn.Conv2d(self.in_ch[3], self.mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_45 = nn.Sequential(
            nn.Conv2d(self.in_ch[4], self.mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_a4 = FeatureFusionModule(self.mid_ch * 6, self.in_ch[3], self.out_channel)

        self.conv_54 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_ch[3], self.mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_55 = nn.Sequential(
            nn.Conv2d(self.in_ch[4], self.mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_a5 = FeatureFusionModule(self.mid_ch * 2, self.in_ch[4], self.out_channel)

    def forward(self, c2, c3, c4, c5):
        # scale 2
        c2_s2 = self.conv_22(c2)

        c3_s2 = self.conv_23(c3)
        c3_s2 = F.interpolate(c3_s2, scale_factor=(2, 2), mode='bilinear')

        s2 = self.conv_a2(torch.cat([c2_s2, c3_s2], dim=1), c2)
        # scale 3
        c2_s3 = self.conv_32(c2)

        c3_s3 = self.conv_33(c3)

        c4_s3 = self.conv_34(c4)
        c4_s3 = F.interpolate(c4_s3, scale_factor=(2, 2), mode='bilinear')

        c2_c3_s3 = torch.cat([c2_s3,c3_s3],dim=1)
        c3_c4_s3 = torch.cat([c3_s3,c4_s3],dim=1)
        c2_c4_s3 = torch.cat([c2_s3,c4_s3],dim=1)

        s3 = self.conv_a3(torch.cat([c2_c3_s3,c3_c4_s3,c2_c4_s3], dim=1), c3)
        # scale 4
        c3_s4 = self.conv_43(c3)

        c4_s4 = self.conv_44(c4)

        c5_s4 = self.conv_45(c5)
        c5_s4 = F.interpolate(c5_s4, scale_factor=(2, 2), mode='bilinear')

        c3_c4_s4 = torch.cat([c3_s4,c4_s4],dim=1)
        c4_c5_s4 = torch.cat([c4_s4,c5_s4],dim=1)
        c3_c5_s4 = torch.cat([c3_s4,c5_s4],dim=1)

        s4 = self.conv_a4(torch.cat([c3_c4_s4,c4_c5_s4,c3_c5_s4], dim=1), c4)
        # scale 5
        c4_s5 = self.conv_54(c4)

        c5_s5 = self.conv_55(c5)

        s5 = self.conv_a5(torch.cat([c4_s5, c5_s5], dim=1), c5)

        return s2, s3, s4, s5

class SCFHM_R1(nn.Module):
    def __init__(self, in_d=32, out_d=32):
        super(SCFHM_R1, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        # fusionvb
        self.SCFHM1= SCFHM_RSU_1(self.in_d, self.out_d)
        self.SCFHM2= SCFHM_RSU_1(self.in_d, self.out_d)
        self.SCFHM3 = SCFHM_RSU_1(self.in_d, self.out_d)
        self.SCFHM4= SCFHM_RSU_1(self.in_d, self.out_d)


    def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        # temporal fusion
        c2 = self.SCFHM1(x1_2, x2_2)
        c3 = self.SCFHM2(x1_3, x2_3)
        c4 = self.SCFHM3(x1_4, x2_4)
        c5 = self.SCFHM4(x1_5, x2_5)

        return c2, c3, c4, c5

class SCFHM_RSU_2(nn.Module):
    def __init__(self,mid_ch):
        super(SCFHM_RSU_2,self).__init__()
        self.mid_ch = mid_ch
        self.cls = nn.Conv2d(self.mid_ch, 1, kernel_size=1)
        self.stage_1 = RSU7(64, 32, 64)

    def forward(self,x):
        mask_map = self.cls(x)
        x_out = self.stage_1(x)
        return x_out,mask_map






class Decoder(nn.Module):
    def __init__(self, mid_ch=320):
        super(Decoder, self).__init__()
        self.mid_ch = mid_ch

        self.SR2 = SCFHM_RSU_2(self.mid_ch)
        self.SR3 = SCFHM_RSU_2(self.mid_ch)
        self.SR4 = SCFHM_RSU_2(self.mid_ch)
        self.conv_p4 = nn.Sequential(
            nn.Conv2d(self.mid_ch, self.mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(self.mid_ch, self.mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(self.mid_ch, self.mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(self.mid_ch, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5):
        # high-level
        p5, mask_map_p5 = self.SR2(d5)
        p4 = self.conv_p4(d4 + F.interpolate(p5, scale_factor=(2, 2), mode='bilinear'))

        p4, mask_map_p4 = self.SR3(p4)
        p3 = self.conv_p3(d3 + F.interpolate(p4, scale_factor=(2, 2), mode='bilinear'))

        p3, mask_map_p3 = self.SR4(p3)
        p2 = self.conv_p2(d2 + F.interpolate(p3, scale_factor=(2, 2), mode='bilinear'))
        mask_map_p2 = self.cls(p2)

        return p2, p3, p4, p5, mask_map_p2, mask_map_p3, mask_map_p4, mask_map_p5



class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1):
        super(BaseNet, self).__init__()
        self.BackBone = mobilenet_v2(pretrained=True)
        channles = [16, 24, 32, 96, 320]
        self.end_ch = 32
        self.mid_ch = self.end_ch * 2
        self.MFNFM = MFNFM(channles, self.mid_ch)
        self.SR1 = SCFHM_R1(self.mid_ch, self.end_ch * 2)
        self.decoder = Decoder(self.end_ch * 2)

    def forward(self, x1, x2):
        # forward BackBone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.BackBone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.BackBone(x2)


        # aggregation
        x1_2, x1_3, x1_4, x1_5 = self.MFNFM(x1_2, x1_3, x1_4, x1_5)
        x2_2, x2_3, x2_4, x2_5 = self.MFNFM(x2_2, x2_3, x2_4, x2_5)
        # temporal fusion
        c2, c3, c4, c5 = self.SR1(x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)
        # fpn
        p2, p3, p4, p5, mask_map_p2, mask_map_p3, mask_map_p4, mask_map_p5 = self.decoder(c2, c3, c4, c5)

        mask_map_p2 = F.interpolate(mask_map_p2, scale_factor=(4, 4), mode='bilinear')
        mask_map_p2 = torch.sigmoid(mask_map_p2)
        mask_map_p3 = F.interpolate(mask_map_p3, scale_factor=(8, 8), mode='bilinear')
        mask_map_p3 = torch.sigmoid(mask_map_p3)
        mask_map_p4 = F.interpolate(mask_map_p4, scale_factor=(16, 16), mode='bilinear')
        mask_map_p4 = torch.sigmoid(mask_map_p4)
        mask_map_p5 = F.interpolate(mask_map_p5, scale_factor=(32, 32), mode='bilinear')
        mask_map_p5 = torch.sigmoid(mask_map_p5)

        return mask_map_p2, mask_map_p3, mask_map_p4, mask_map_p5


if __name__ =='__main__':
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    model = BaseNet(3, 1)
    flops, params = profile(model, inputs=(x1, x2))
    print('flops:{:.2f}G, params:{:.2f}M'.format(flops / 1e9, params / 1e6))
