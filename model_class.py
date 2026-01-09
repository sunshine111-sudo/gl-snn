import numpy as np
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode

from sympy.physics.units.systems.si import dimex
# from lif_layers import LIFSpike
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model




#qkformer
class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8,Threshold=1.,qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.Threshold=Threshold
        self.dim = dim
        self.num_heads = num_heads
        self.scale=0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0,v_threshold=Threshold, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0,v_threshold=Threshold, detach_reset=True, backend='cupy')
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, v_threshold=Threshold, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0,v_threshold=Threshold, detach_reset=True, backend='cupy')


    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_lif(x)
        x = x.flatten(3)
        T, B, C, N = x.shape

        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        attn = torch.mul(k,v)
        attn = torch.sum(attn, dim=3, keepdim=True)
        attn = self.attn_lif(attn)

        x = torch.mul(q,attn)

        x = x.flatten(2, 3)

        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        # x = self.proj_lif(x)

        return x

class GRToken_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8,Threshold=1., qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0,v_threshold=Threshold, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, v_threshold=Threshold,detach_reset=True, backend='cupy')
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, v_threshold=Threshold, detach_reset=True, backend='cupy')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, v_threshold=Threshold,detach_reset=True, backend='cupy')


    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)
        x = x.flatten(3)
        T, B, C, N = x.shape

        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)
        #
        attn=torch.mul(k, v)
        attn=torch.sum(attn, dim = 4, keepdim = True)
        attn=self.attn_lif(attn)
        x=torch.mul(q,attn)
        #
        # q = torch.sum(q, dim = 4, keepdim = True)
        # attn = self.attn_lif(q)
        # x = torch.mul(attn, k)

        x = x.flatten(2, 3)

        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        # x = self.proj_lif(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.,Threshold=1.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, v_threshold=Threshold,detach_reset=True, backend='cupy')

        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = MultiStepLIFNode(tau=2.0, v_threshold=Threshold,detach_reset=True, backend='cupy')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_lif(x)

        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)

        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)

        return x

#
class ChanelSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,Threshold=1.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = GRToken_QK_Attention(dim, num_heads,Threshold=Threshold)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop,Threshold=Threshold)

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)
        # print(x)
        return x

class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,Threshold=1.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads,Threshold=Threshold)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop,Threshold=Threshold)

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256,Threshold=1.):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif = MultiStepLIFNode(tau=2.0,v_threshold=Threshold, detach_reset=True, backend='cupy')

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 1)
        self.proj1_lif = MultiStepLIFNode(tau=2.0, v_threshold=Threshold,detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims //1, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj2_lif = MultiStepLIFNode(tau=2.0,v_threshold=Threshold, detach_reset=True, backend='cupy')
    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1)

        x_feat = x


        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H, W)


        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H, W).contiguous()

        x = x + x_feat # shortcut

        return x

class SSCFM(nn.Module):
    def __init__(self,embed_dims=128,Threshold=1.):
        super().__init__()
        self.proj_conv = nn.Conv2d(embed_dims*2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims)
        self.proj_lif = MultiStepLIFNode(tau=2.0,v_threshold=Threshold, detach_reset=True, backend='cupy')
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, v_threshold=Threshold,detach_reset=True, backend='cupy')
        self.proj_conv1 = nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, bias=False)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, v_threshold=Threshold, detach_reset=True, backend='cupy')
        self.sig = nn.Sigmoid()
    def forward(self, x, y):

        fuse=torch.cat((x, y), dim=2)

        T, B, C, H, W = fuse.shape

        fuse = self.proj_lif2(fuse)

        fuse = self.proj_conv(fuse.flatten(0, 1))
        fuse = self.proj_bn(fuse)


        fuse = fuse.reshape(T, B, -1, H, W).contiguous()
        #
        fuse=self.proj_conv1(fuse.flatten(0, 1))
        fuse = fuse.reshape(T, B, -1, H, W).contiguous()
        w=self.sig(fuse)
        return x*w+y*(1-w)
        # return  fuse

class attentionFusion(nn.Module):
    def __init__(self,embed_dims=128,Threshold=1.):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(embed_dims, embed_dims // 4, kernel_size=1, bias=False)
        self.linear1=nn.Linear(embed_dims // 4, embed_dims)
        self.linear2=nn.Linear(embed_dims // 4, embed_dims)
        self.bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.bn = nn.BatchNorm2d(embed_dims )
        self.conv2 = nn.Conv2d(embed_dims // 4,embed_dims, kernel_size=1, bias=False)
        self.linear3 = nn.Linear(embed_dims, embed_dims)
        self.linear4 = nn.Linear(embed_dims, embed_dims)
        self.bn2 = nn.BatchNorm2d(embed_dims)
        self.sigmoid = nn.Sigmoid()
        self.proj_lif = MultiStepLIFNode(tau=2.0,v_threshold=0.5, detach_reset=True, backend='cupy')
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, v_threshold=0.5,detach_reset=True, backend='cupy')
        #

    def forward(self, x, y):


        # fuse=x+y
        T, B, C, H, W = x.shape
        fuse=x+y
        fuse1=fuse

        fuse=fuse.flatten(0, 1)
        fuse = self.bn(fuse)


        avg = self.avg_pool(fuse)
        max = self.max_pool(fuse)
        avg = self.proj_lif(avg)
        max = self.proj_lif1(max)
        #
        #

        avg_out = self.conv2(self.bn1(self.conv1(avg)))
        max_out = self.conv2(self.bn1(self.conv1(max)))

        out = avg_out + max_out
        out=out.reshape(T, B, -1, 1, 1).contiguous()
        w=self.sigmoid(out)
        return x*w+y*(1-w)




class localBlock(nn.Module):
    def __init__(self,embed_dims=128,in_channels=2,Threshold=1.):
        super().__init__()

        self.proj_bn = nn.BatchNorm2d(embed_dims)
        self.proj_lif = MultiStepLIFNode(tau=2.0,v_threshold=Threshold, detach_reset=True, backend='cupy')

        self.dwconv1 = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, groups=embed_dims, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dims)

        self.proj_lif1 = MultiStepLIFNode(tau=2.0,v_threshold=Threshold, detach_reset=True, backend='cupy')
        self.conv3d = nn.Conv3d(embed_dims, embed_dims, kernel_size=(3, 1, 1), padding=(1, 0, 0),bias=False)
        self.bn2 = nn.BatchNorm3d(embed_dims)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0,v_threshold=Threshold, detach_reset=True, backend='cupy')
        self.proj_conv = nn.Conv2d(embed_dims*2,embed_dims , kernel_size=1, stride=1,  bias=False)
        # self.proj_conv = nn.Conv2d(embed_dims*2, embed_dims, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, v_threshold=Threshold, detach_reset=True, backend='cupy')


        if in_channels != embed_dims:
            self.shortcut = nn.Conv2d(in_channels,embed_dims, kernel_size=1, bias=False)
    def forward(self, x):
        T, B, C, H, W = x.shape

        feat = x
        # x = self.proj_conv_head(x.flatten(0, 1))
        # x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x)



        x1 = self.dwconv1(x.flatten(0, 1))
        x1 = self.bn1(x1).reshape(T, B, -1, H, W).contiguous()
        # x1 =self.proj_lif1(x1)

        x_spectral = x.unsqueeze(3)  # (batch_size, in_channels, 1, height, width)

        # Process the spectral branch

        x2 = self.conv3d(x_spectral.flatten(0, 1))
        x2 = self.bn2(x2)
        x2 = x2.squeeze(2)  # (batch_size, out_channels, height, width)
        x2=x2.reshape(T, B, -1, H, W).contiguous()
        # x2 =self.proj_lif2(x2)




        # Concatenate the outputs of both branches
        x_cat = torch.cat((x1, x2), dim=2)  # (batch_size, in_channels + out_channels, height, width)
        x_cat=self.proj_lif1(x_cat)

        # print("x_cat",x_cat)
        x_cat =x_cat.flatten(0, 1)
        # Apply pointwise convolution

        # x_out = self.pointwise_conv(x_cat)
        x_out = self.proj_conv(x_cat)
        x_out = self.bn3(x_out).reshape(T, B, -1, H, W).contiguous()


        x_out = x_out+feat
        # print(x_out)
        return x_out


class spiking_transformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=103, num_classes=11,
                 embed_dims=128, num_heads=[8, 8, 8], mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,Threshold=1.,
                 depths=4, sr_ratios=1, T=4,pretrained_cfg= None,pretrained_cfg_overlay=None
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T=T
        self.Threshold=Threshold
        num_heads = [8, 8, 8]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.patch_embed1 = PatchEmbedInit(
                                 img_size_h= img_size_h, img_size_w= img_size_w,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims,
                                    Threshold=Threshold)


        self.stage1 = nn.ModuleList([ChanelSpikingTransformer(
            dim=embed_dims, num_heads=num_heads[0], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios,  Threshold=Threshold)
          ])

        self.stage2 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims, num_heads=num_heads[2], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, sr_ratio=sr_ratios,  Threshold=Threshold)
          ])
        self.fuse = SSCFM(embed_dims=embed_dims, Threshold=Threshold)
        self.local = localBlock(embed_dims=embed_dims, in_channels=embed_dims, Threshold=Threshold)
        self.atfuse=attentionFusion(embed_dims=embed_dims,  Threshold=Threshold)
        # self.atfuse1=attentionFusion(embed_dims=embed_dims)




        # self.CrossGateFusion=CrossGateFusion(dim=embed_dims)
        self.bn= nn.BatchNorm2d(embed_dims)
        self.flatten = nn.Flatten(1)
        self.cls_head =   nn.Linear(embed_dims, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        T, B, C, H, W = x.shape

        x = self.patch_embed1(x)
        #
        local = x
        # # feat = x
        for blk in self.stage1:
            x_1 = blk(x)
        #
        for blk in self.stage2:
            x_2 = blk(x)
        # x=x_1
        x=self.fuse(x_1, x_2)

        local=self.local(local)

        x = self.atfuse(x, local)

        x=x.mean(0)

        return  self.flatten(self.avgpool(x))

    def forward(self, x):
        T=self.T
        x= x.squeeze(dim=1)
        x = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)

        x = self.forward_features(x)
        x = self.cls_head(x)
        # print(len(power_list))
        return x

@register_model
def QKFormer(pretrained=False, **kwargs):
    model = spiking_transformer(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model






if __name__ == '__main__':
    # model =create_model(
    #     'QKFormer',
    #
    #     drop_rate=0,
    #     drop_path_rate=0.1,
    #
    #     img_size_h=4, img_size_w=4,
    #     patch_size=4, embed_dims=128, num_heads=8, mlp_ratios=4,
    #     in_channels=30, num_classes=9, qkv_bias=False,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
    #     T=4,
    # ).cuda()

    model =spiking_transformer (

        drop_rate=0,
        drop_path_rate=0.1,

        img_size_h=4, img_size_w=4,
        patch_size=4, embed_dims=64, num_heads=8, mlp_ratios=4,
        in_channels=30, num_classes=9, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,Threshold=1.,
        T=5,
    ).cuda()

    # model.eval()
    # print(model)
    input = torch.randn(1, 1, 30, 13, 13).cuda()
    # y = model(input)
    # print(y.size())
    # from ptflops import get_model_complexity_info

    # macs, params = get_model_complexity_info(
    #     model, ( 1, 30, 13, 13),
    #     print_per_layer_stat=True,  # 逐层打印
    #     as_strings=True,
    #     verbose=True  # 显示卷积核尺寸等细节
    # )
    from thop import profile
    flops, params = profile(model, input)
    print("FLOPs: ", flops / 1e9, "params: ", params / 1e6)

    # analyze_model(model,input)
