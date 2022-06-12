import math

import mlconfig
import torch
from torch import nn
import torch.nn.functional as F
from .utils import load_state_dict_from_url
from revgrad_master.src.pytorch_revgrad.module import RevGrad
model_urls = {
    'efficientnet_b0': 'https://www.dropbox.com/s/9wigibun8n260qm/efficientnet-b0-4cfa50.pth?dl=1',
    'efficientnet_b1': 'https://www.dropbox.com/s/6745ear79b1ltkh/efficientnet-b1-ef6aa7.pth?dl=1',
    'efficientnet_b2': 'https://www.dropbox.com/s/0dhtv1t5wkjg0iy/efficientnet-b2-7c98aa.pth?dl=1',
    'efficientnet_b3': 'https://www.dropbox.com/s/5uqok5gd33fom5p/efficientnet-b3-bdc7f4.pth?dl=1',
    'efficientnet_b4': 'https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1',
    'efficientnet_b5': 'https://www.dropbox.com/s/qxonlu3q02v9i47/efficientnet-b5-4c7978.pth?dl=1',
    'efficientnet_b6': None,
    'efficientnet_b7': None,
}

params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.62),
    'efficientnet_b1': (1.0, 1.1, 240, 0.52),
    'efficientnet_b2': (1.1, 1.2, 260, 0.85),
    'efficientnet_b3': (1.2, 1.4, 300, 0.1),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}

att = {
    'efficientnet_b0': 1280,
    'efficientnet_b1': 1280,
    'efficientnet_b2': 1408,
    'efficientnet_b3': 1536,
    'efficientnet_b4': 1536,
    'efficientnet_b5': 1536,
    'efficientnet_b6': 1536,
    'efficientnet_b7': 1536,
}

sigmoid = torch.nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        #ctx.sigmoid_i
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)

swish_layer = Swish_module()

class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish_layer(x)# x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1,Affined=True):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, affine=Affined),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x,y=None):
        if y is None:
            return x * self.se(x)
        return x * self.se(y)


class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 padding=0,
                 inplanss=0,
                 reduction_ratio=4,
                 drop_connect_rate=0.2,
                 affine=True):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        self.padding=padding
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw

        if padding>0:
            hidden_dim = inplanss * expand_ratio
            reduced_dim = max(1, int(inplanss / reduction_ratio))
            self.firstx=ConvBNReLU(in_planes, hidden_dim, 1,Affined=affine)
            self.firsts = ConvBNReLU(inplanss, hidden_dim, 1,Affined=affine)
            self.secoundx = ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim,Affined=affine)
            self.secounds = ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim,Affined=affine)
            self.squeezex = SqueezeExcitation(hidden_dim, reduced_dim)
            self.squeezes = SqueezeExcitation(hidden_dim, reduced_dim)
            layers =[nn.Conv2d(2*hidden_dim, out_planes, 1, bias=False, padding=0),
            nn.BatchNorm2d(out_planes)]
        else:
            padding=0
            if in_planes != hidden_dim:
                if self.padding==0:
                    layers += [ConvBNReLU(in_planes, hidden_dim, 1,Affined=affine)]
                else:
                    self.beresize=ConvBNReLU(in_planes, hidden_dim, 1,Affined=affine)

            layers += [
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim,Affined=affine),
                # se
                SqueezeExcitation(hidden_dim, reduced_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, out_planes, 1, bias=False, padding=padding)
            ]
            if self.padding==0:
                layers += [nn.BatchNorm2d(out_planes,affine=affine)]


        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if  self.padding>0:
            x, skip = x
            x=self.firstx(x)
            skip=self.firsts(skip)
            x=self.secoundx(x)
            skip=self.secounds(skip)
            x=self.squeezex(x)
            skip = self.squeezes(skip,x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            return self.conv(x)
        else:
            if self.padding<0:
                x=self.beresize(x)
                x=F.interpolate(x, scale_factor=2, mode='bilinear')
                return self.conv(x)
            else:
                if self.use_residual:
                    return x + self._drop_connect(self.conv(x))
                else:
                    return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


@mlconfig.register
class EfficientNet(nn.Module):

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.5, num_classes=1000, efficent_Name='efficientnet_b3'):
        super(EfficientNet, self).__init__()

        # yapf: disable
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]
        # yapf: enable

        out_channels = _round_filters(32, width_mult)
        features = [ConvBNReLU(3, out_channels, 3, stride=2)]

        in_channels = out_channels
        chanels=[]
        #chanels.append(in_channels)
        for t, c, n, s, k in settings:
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                if stride>1:
                    chanels.append(in_channels)
                in_channels = out_channels
        #chanels.append(in_channels)
        last_channels = _round_filters(1280, width_mult)

        firstDecChann = 2 * chanels[-1]
        decoder=[
            nn.Sequential(MBConvBlock(last_channels , firstDecChann,inplanss=chanels[-1], expand_ratio=6, stride=1, kernel_size=3,padding=1)
                           ,MBConvBlock(firstDecChann , firstDecChann, expand_ratio=6, stride=1, kernel_size=3)
                           ,MBConvBlock(firstDecChann, firstDecChann,  expand_ratio=6, stride=1,kernel_size=3)
                          )]

        for chann in reversed(chanels[:-1]):
            decoder += [nn.Sequential(MBConvBlock(firstDecChann, 2 * chann,inplanss=chann, expand_ratio=6, stride=1, kernel_size=3,padding=1,affine=False)
                                       ,MBConvBlock( 2 * chann, 2 * chann,  expand_ratio=6, stride=1,
                                                   kernel_size=3,affine=False)
                                 ,MBConvBlock(2 * chann, 2 * chann,  expand_ratio=6, stride=1,
                                             kernel_size=3,affine=False)
                                      )]
            firstDecChann = 2 * chann

        features += [ConvBNReLU(in_channels, last_channels, 1)]
        self.Decoder = nn.Sequential(*decoder)
        self.segclassifier=MBConvBlock(2*chanels[0], 12, expand_ratio=6, stride=1, kernel_size=5,padding=-1,affine=False)
        self.features = nn.Sequential(*features)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.inplanes = att[efficent_Name] #1536☺

        self.maskLayerVec=[]
        self.classifierVec=[]
        self.classifierHiddenVec=[]
        # self.classifierReverseHiddenVec=[]
        # self.classifierVecModal=[]
        lentotl=0
        for chann in reversed(chanels):
            self.maskLayerVec.append(nn.Conv2d(2*chann, 1, kernel_size=1, stride=1, padding=0,
                                       bias=True).cuda())
            lentotl+=last_channels
            # self.classifierReverseHiddenVec.append(nn.Sequential(RevGrad(),
            #     nn.Linear(2 * chann, last_channels),
            #     nn.ReLU(inplace=True)
            # ).cuda())
            self.classifierHiddenVec.append(nn.Sequential(
            nn.Linear(2 * chann, last_channels),
            nn.ReLU(inplace=True)
            ).cuda())

            # self.classifierVecModal.append(nn.Sequential(
            #     nn.Linear(last_channels, num_classes)
            # ).cuda())

            self.classifierVec.append(nn.Sequential(

                nn.Dropout(dropout_rate),
                nn.Linear(last_channels, num_classes)
            ).cuda())

        self.maskLayer = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.alpha=nn.Parameter(torch.Tensor(len(chanels)+1)) #nn.Parameter(torch.zeros(len(chanels)+1))

        self.MaskNorm = nn.BatchNorm2d(1,affine=False)
        self.relu6 = nn.Sigmoid()#ReLU6(inplace=True)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )
        lentotl+=last_channels
        self.AttentionExtractor=nn.Linear(last_channels, len(chanels)+1)
        self.classifierTotal = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )
        # self.maskFeatLayer = nn.Conv2d(self.inplanes, 136, kernel_size=1, stride=1, padding=0,
        #                                 bias=True)
        # self.maskLayerBiger = nn.Conv2d(272, 1, kernel_size=1, stride=1, padding=0,
        #                                  bias=True)
        # self.classifier2 = nn.Sequential(
        #      nn.Dropout(dropout_rate),
        #      nn.Linear(last_channels+136, num_classes),
        #  )
        # self.classifier3 = nn.Sequential(
        #      nn.Dropout(dropout_rate),
        #      nn.Linear( 136, num_classes),
        #  )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.reset_parameters()

    # def forward(self, x):
    #     y = self.features(x)
    #     Masker = self.maskLayer(y)
    #     Masker = self.MaskNorm(Masker)
    #     Masker = self.relu6(Masker)
    #     x = self._avg_pooling(y * Masker) / (self._avg_pooling(Masker) + 1e-10)
    #     x = torch.flatten(x, 1)
    #     x = self.classifier(x)
    #     return x
    def reset_parameters(self):

        nn.init.constant_(self.alpha,0)

    def forward(self, x, LivMask=None):
        count=0
        incode=[]
        #print(self.alpha)
        Maskervec=[]
        for featch in self.features:
            if count%2==0:
                next=featch(x)
            else:
                x=featch(next)
            if next.shape[2]!= x.shape[2]:
                if next.shape[2]> x.shape[2]:
                    incode.append(next)
                else:
                    incode.append(x)
            count+=1   #     y = self.features(x)
        y = self.Decoder[0]([x, incode[-1]*self.alpha[-1]])
        alphavec=[]
        alphavec.append(self.alpha[-1])
        decoders=self.Decoder[1:]
        if 1:
            Masker = self.maskLayer(x)
            Masker = self.MaskNorm(Masker)
            Masker = self.relu6(Masker)
            if LivMask is not None:
                # .resize_as_(Masker)
                LivMask = torch.nn.functional.interpolate(LivMask, size=8, mode='bilinear')
                Masker2 = Masker * LivMask
                x = self._avg_pooling(x * Masker2) / (self._avg_pooling(Masker2) + 1e-10)
            else:
                x = self._avg_pooling(x * Masker) / (self._avg_pooling(Masker) + 1e-10)
                Masker2 = Masker
            Maskervec.append(Masker2)
            x = torch.flatten(x, 1)
            # outcut=torch.cat((outcut, x), 1)
            scalar = self.AttentionExtractor(x)
            scalar=self.relu6(scalar)
            outcut = scalar[:,-1].view(-1,1)*x

        ind=-2
        out=[]
        outm=[]
        indmask = 0
        #outt =
        #out.append( self.classifier(x))
    #    outt = self.classifierHiddenVec[indmask](x)
    #    out.append(self.classifierVec[indmask](outt))
    #    indmask = 1
        #outcut=0
        #ylast= F.interpolate(y, scale_factor=2, mode='nearest')
        for featch in decoders:
            Maskerer = self.maskLayerVec[indmask](y)
            Maskerer = self.MaskNorm(Maskerer)
            Maskerer = self.relu6(Maskerer)
            outt = self._avg_pooling(y * Maskerer) / (self._avg_pooling(Maskerer) + 1e-10)
            Maskervec.append(Maskerer)
            outt = torch.flatten(outt, 1)
            # outmo=self.classifierReverseHiddenVec[indmask](outt)
            #outm.append(self.classifierVecModal[indmask](outmo))
            outt = self.classifierHiddenVec[indmask](outt)
            # if outcut is None:
            #     outcut=outt
            # else:
            outcut += scalar[:,indmask].view(-1,1)*outt
                #outcut=torch.cat((outcut, outt), 1)

            out.append(self.classifierVec[indmask](outt))
            indmask += 1
            alphavec.append(self.alpha[ind])
            accinc=incode[ind]+1
            for al in  alphavec:
                accinc *= al
            y=featch([y, accinc])
            ind-=1

        Maskerer = self.maskLayerVec[indmask](y)
        Maskerer = self.MaskNorm(Maskerer)
        Maskerer = self.relu6(Maskerer)
        Maskervec.append(Maskerer)
        outt = self._avg_pooling(y * Maskerer) / (self._avg_pooling(Maskerer) + 1e-10)
        outt = torch.flatten(outt, 1)
        outt = self.classifierHiddenVec[indmask](outt)
        outcut +=  scalar[:,indmask].view(-1,1)*outt
        #outcut = torch.cat((outcut, outt), 1)
        out.append(self.classifierVec[indmask](outt))
        indmask += 1

        segout=self.segclassifier(y)
        x = self.classifier(x)
        out.append(x)

        outcut = self.classifierTotal(outcut)
        out.append(outcut)
        out.append(segout)
        return out,Maskervec,Masker2 #,outm

    # def forward(self, x):
    #      count=0
    #      for featu in self.features:
    #          if count==0:
    #              y = featu(x)
    #          else:
    #              if count<19:
    #                  y = featu(y)
    #              else:
    #                  if count==19:
    #                      z=featu(y)
    #                  else:
    #                      z = featu(z)
    #          count+=1
    #      Masker = self.maskLayer(z)
    #      masker2=self.maskFeatLayer(z)
    #      masker2=torch.nn.functional.interpolate(masker2,scale_factor=2)
    #      masker2=torch.cat([masker2,y],dim=1)
    #      masker2=self.maskLayerBiger(masker2)
    #      masker2 = self.MaskNorm(masker2)
    #      masker2=self.relu6(masker2)
    #      Masker = self.MaskNorm(Masker)
    #      Masker = self.relu6(Masker)
    #      x2=self._avg_pooling(y * masker2) / (self._avg_pooling(masker2) + 1e-10)
    #      x2 = torch.flatten(x2, 1)
    #      x = self._avg_pooling(z * Masker) / (self._avg_pooling(Masker) + 1e-10)
    #      x = torch.flatten(x, 1)
    #      xxx = self.classifier2(torch.cat([x,x2],dim=1))
    #      x = self.classifier(x)
    #      x2 = self.classifier3(x2)
    #      return xxx,x,x2



def _efficientnet(arch, pretrained, progress, **kwargs):
    width_mult, depth_mult, _, dropout_rate = params[arch]
    model = EfficientNet(width_mult, depth_mult, dropout_rate, **kwargs,efficent_Name=arch)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            del state_dict['classifier.1.weight']
            del state_dict['classifier.1.bias']

        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        #model.load_state_dict(state_dict, strict=False)
    return model


@mlconfig.register
def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b0', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b1(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b1', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b2(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b2', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b3(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b3', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b4(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b4', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b5(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b5', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b6(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b6', pretrained, progress, **kwargs)


@mlconfig.register
def efficientnet_b7(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b7', pretrained, progress, **kwargs)
