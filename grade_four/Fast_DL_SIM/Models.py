import math
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F


def get_model(config):
    if config['MODEL']['NAME'].lower() == 'edsr':
        net = EDSR(config)
    elif config['MODEL']['NAME'].lower() == 'rcan':
        net = RCAN(config)
    elif config['MODEL']['NAME'].lower() == 'unet':
        net = UNet(config['TRAIN']['IN_CHANNEL'],
                   config['TRAIN']['OUT_CHANNEL'],
                   config)
    else:
        print("model undefined")
        return None

    net.cuda()
    net = nn.DataParallel(net)

    return net


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, n_feats, kernel_size,
            bias=True, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = [conv(n_feats, n_feats, kernel_size, bias=bias),
             act,
             conv(n_feats, n_feats, kernel_size, bias=bias)]

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class UpSampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'p_relu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'p_relu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(UpSampler, self).__init__(*m)

# ----------------------------------- EDSR------------------------------------------


class EDSR(nn.Module):
    def __init__(self, config):
        super(EDSR, self).__init__()

        n_res_blocks = config['MODEL']['NUM_RES_BLOCK']  # originally 16
        n_feats = config['MODEL']['NUM_FEATS']  # originally 64
        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(config['MODEL']['IN_CHANNELS'], n_feats, kernel_size)]

        # define body module
        m_body: list = [
            ResBlock(
                n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_res_blocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        if config['TRAIN']['SCALE'] == 1:
            if config['TRAIN']['TASK'] == 'segment':
                m_tail = [nn.Conv2d(n_feats, 2, 1)]
            else:
                m_tail = [conv(n_feats, config['MODEL']['OUT_CHANNELS'], kernel_size)]
        else:
            m_tail = [
                UpSampler(config['SCALE'], n_feats, act=False),
                conv(n_feats, config['MODEL']['OUT_CHANNELS'], kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

    # ----------------------------------- RCAN ------------------------------------------


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, res_scale, n_res_blocks):
        super(ResidualGroup, self).__init__()
        # modules_body = []
        modules_body: list = [
            RCAB(n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale)
            for _ in range(n_res_blocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, config):
        super(RCAN, self).__init__()
        n_res_groups = config['MODEL']['NUM_RES_GROUPS']
        n_res_blocks = config['MODEL']['NUM_RES_BLOCKS']
        n_feats = config['MODEL']['NUM_FEATS']
        kernel_size = 3
        reduction = config['MODEL']['REDUCTION']
        act = nn.ReLU(True)
        self.n_arch = config['MODEL']['NUM_ARCH']

        # define head module
        if self.n_arch == 0:
            modules_head = [conv(config['MODEL']['IN_CHANNELS'], n_feats, kernel_size)]
            self.head = nn.Sequential(*modules_head)
        else:
            self.head0 = conv(1, n_feats, kernel_size)
            self.head02 = conv(n_feats, n_feats, kernel_size)
            self.head1 = conv(1, n_feats, kernel_size)
            self.head12 = conv(n_feats, n_feats, kernel_size)
            self.head2 = conv(1, n_feats, kernel_size)
            self.head22 = conv(n_feats, n_feats, kernel_size)
            self.head3 = conv(1, n_feats, kernel_size)
            self.head32 = conv(n_feats, n_feats, kernel_size)
            self.head4 = conv(1, n_feats, kernel_size)
            self.head42 = conv(n_feats, n_feats, kernel_size)
            self.head5 = conv(1, n_feats, kernel_size)
            self.head52 = conv(n_feats, n_feats, kernel_size)
            self.head6 = conv(1, n_feats, kernel_size)
            self.head62 = conv(n_feats, n_feats, kernel_size)
            self.head7 = conv(1, n_feats, kernel_size)
            self.head72 = conv(n_feats, n_feats, kernel_size)
            self.head8 = conv(1, n_feats, kernel_size)
            self.head82 = conv(n_feats, n_feats, kernel_size)
            self.combineHead = conv(9 * n_feats, n_feats, kernel_size)

        # define body module
        modules_body: list = [
            ResidualGroup(n_feats, kernel_size, reduction, act=act, res_scale=1, n_res_blocks=n_res_blocks)
            for _ in range(n_res_groups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        if config['TRAIN']['SCALE'] == 1:
            if config['TRAIN']['TASK'] == 'segment':
                modules_tail = [nn.Conv2d(n_feats, config['MODEL']['OUT_CHANNELS'], 1)]
            else:
                modules_tail = [conv(n_feats, config['MODEL']['OUT_CHANNELS'], kernel_size)]
        else:
            modules_tail = [
                UpSampler(config['TRAIN']['SCALE'], n_feats, act=False),
                conv(n_feats, config['MODEL']['OUT_CHANNELS'], kernel_size)]

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        if self.n_arch == 0:
            x = self.head(x)
        else:
            x0 = self.head02(self.head0(x[:, 0:0 + 1, :, :]))
            x1 = self.head12(self.head1(x[:, 1:1 + 1, :, :]))
            x2 = self.head22(self.head2(x[:, 2:2 + 1, :, :]))
            x3 = self.head32(self.head3(x[:, 3:3 + 1, :, :]))
            x4 = self.head42(self.head4(x[:, 4:4 + 1, :, :]))
            x5 = self.head52(self.head5(x[:, 5:5 + 1, :, :]))
            x6 = self.head62(self.head6(x[:, 6:6 + 1, :, :]))
            x7 = self.head72(self.head7(x[:, 7:7 + 1, :, :]))
            x8 = self.head82(self.head8(x[:, 8:8 + 1, :, :]))
            x = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8), 1)
            x = self.combineHead(x)

        res = self.body(x)
        res += x  # long skip connection

        x = self.tail(res)

        return x


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel, stride, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


# ----------------------------------- Unet------------------------------------------


class DoubleConv(nn.Module):
    # (conv => BN => ReLU) * 2
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.Conv2d(in_ch,in_ch, 2, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mp_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bi_linear=False):
        super(Up, self).__init__()

        #  would be a nice idea if the up_sampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bi_linear:
            self.up = nn.Upsample(scale_factor=2, mode='bi_linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, config):
        super(UNet, self).__init__()
        self.inc = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        if config['TRAIN']['TASK'] == 'segment':
            self.outc = OutConv(64, 2)
        else:
            self.outc = OutConv(64, n_classes)

        # Initialize weights
        # self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)
