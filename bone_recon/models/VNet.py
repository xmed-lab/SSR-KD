from torch import nn



class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, normalization='batchnorm'):
        super(VNet, self).__init__()

        # downsampling
        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization) # 1 -> 16 
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization) # 16 -> 32

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization) # 2x: 32 -> 32
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization) # 32 -> 64

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization) # 3x: 64 -> 64
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization) # 64 -> 128

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization) # 3x: 128 -> 128
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization) # 128 -> 256

        # upsampling
        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization) # 3x: 256 -> 256
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization) # 256 -> 128

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization) # 3x: 128 -> 128
        self.head_6 = ConvBlock(1, n_filters * 8, n_filters * 4, normalization=normalization) # <-- head
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization) # 128 -> 64

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization) # 3x: 64 -> 64
        self.head_7 = ConvBlock(1, n_filters * 4, n_filters * 4, normalization=normalization) # <-- head
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization) # 64 -> 32

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization) # 2x: 32 -> 32
        self.head_8 = ConvBlock(1, n_filters * 2, n_filters * 4, normalization=normalization) # <-- head
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization) # 1x: 16 -> 16
        self.head_9 = ConvBlock(1, n_filters, n_filters * 4, normalization=normalization) # <-- head: 16 -> 64

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        return x1, x2, x3, x4, x5

    def decoder(self, features):
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up) # =>
        o6 = self.head_6(x6)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up) # =>
        o7 = self.head_7(x7)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up) # =>
        o8 = self.head_8(x8)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up) # =>
        o9 = self.head_9(x9)
        return o6, o7, o8, o9

    def forward(self, image):
        features = self.encoder(image)
        out = self.decoder(features)
        return out
