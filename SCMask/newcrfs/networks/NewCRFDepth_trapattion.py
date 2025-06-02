import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP,DAPPM

from .trap_block.trap_head import BottleNeck,LayerNorm2d,trapped_inter
#from .trap_block.block_selection import BlockSelectionNeck
########################################################################################################################


class TrapDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture.+Trap-Attention
    """
    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()
        #三个全flase的结构参数
        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False

        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)
    
        # 从这以下为backone架构配置，首先判断swin架构大小，然后根据版本号选择对应的配置
        window_size = int(version[-2:])
        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]
        
        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )
        self.backbone = SwinTransformer(**backbone_cfg)

        #为了实现psp上下文聚合的decoder
        embed_dim = 768 #512,为适配trap最后一层改为
        in_channels = [192, 384, 768, 1536]
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        #crf的输出维度？
        #crf_dims = [128, 256, 512, 1024]
        '''trap 为了进行数据融合的head层'''
        decode_head=dict(
        type='TrappedHead',
        in_channels=[192, 384, 768, 1536],
        post_process_channels=[96 ,192, 384, 768, 1536],#out channels,512为ppm层
        final_norm=False,
        scale_up=True,
        drop_path_rate=0.3,
        align_corners=False,
        min_depth=0.001,
        max_depth=80,
        loss_decode=dict(type='SigLoss', valid_mask=True, loss_weight=10))
        '''traphead init'''
        self.channels = decode_head['post_process_channels']
        dpr = [x.item() for x in torch.linspace(0, decode_head['drop_path_rate'], 
                                                len(decode_head['post_process_channels'])+1)]
        #5个stage
        #第一层像素太小还没交互，不额外进行trap操作和残差连接，通道减半操作
        #将                        in channlea     通过 hidden channels 变为 out channles
        self.block1 = BottleNeck(self.channels[-1], self.channels[-1]*2, self.channels[-2], trap=False, drop_path=dpr[0])
        self.block2 = BottleNeck(self.channels[-2], self.channels[-2]*2, self.channels[-3], drop_path=dpr[1])
        self.block3 = BottleNeck(self.channels[-3], self.channels[-3]*2, self.channels[-4], drop_path=dpr[2])
        self.block4 = BottleNeck(self.channels[-4], self.channels[-4]*2, self.channels[-5], drop_path=dpr[3])
        self.block5 = BottleNeck(self.channels[-5], self.channels[-5]*2, self.channels[-5], drop_path=dpr[4])
        self.block6 = BottleNeck(self.channels[-5]//2, self.channels[-5]*2, self.channels[-5]//2,drop_path=dpr[5]
                                 )

        #通过fusion进行图片大小放缩
        self.fusion1 = nn.Sequential(
            LayerNorm2d(self.channels[-2]),#对当前通道数进行归一化
            nn.Conv2d(self.channels[-2], self.channels[-5], kernel_size=1),#通过一维卷积更改通道数为指定数值
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)#对图像进行上采样实现scale倍的放大
        )

        self.fusion2 = nn.Sequential(
            LayerNorm2d(self.channels[-3]),
            nn.Conv2d(self.channels[-3], self.channels[-5], kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

        self.fusion3 = nn.Sequential(
            LayerNorm2d(self.channels[-4]),
            nn.Conv2d(self.channels[-4], self.channels[-5], kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.fusion4 = nn.Sequential(
            LayerNorm2d(self.channels[-5]),
            nn.Conv2d(self.channels[-5], self.channels[-5], kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.act = nn.GELU()
        self.sig = nn.Sigmoid()
        #self.decoder = PSP(**decoder_cfg)
        planes = 64
        spp_planes = 512
        # in = 1536 , 中层卷积的out = 512 , 最终特征压缩+残差连接的out = 512
        self.decoder = DAPPM(planes * 24, planes * 12, planes * 12)
        #进行最后深度预测分类的prediect层,对传入通道为指定的特征进行为参数倍数的上采样
        self.disp_head1 = DispHead(96)

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(self.channels[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))

        self.min_depth = min_depth
        self.max_depth = max_depth
        #应用backbon和trap的参数初始化
        self.apply(self.init_weight_trap)
        self.use_checkpoint = True
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')

        self.backbone.init_weights(pretrained=pretrained)
        #self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def upsample_mask(self, disp, mask):
        """
        对深度图进行上采样操作，从 [H/4, W/4, 1] 变换到 [H, W, 1]，使用凸组合的方式。
        Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination
        Args:
            disp (torch.Tensor): 输入的深度图，形状为 [N, 1, H/4, W/4]，其中 N 为 batch_size，H 和 W 为原图像的高度和宽度。
            mask (torch.Tensor): 输入的掩码，形状为 [N, 9, 4, 4, H/4, W/4]，表示每个像素点周围的 3x3 网格中各个点的权重。
        
        Returns:
            torch.Tensor: 上采样后的深度图，形状为 [N, 1, H, W]。
        
        """
        
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, 4*H, 4*W)

    def init_weight_trap(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.kaiming_normal_(m.weight)
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
    
    #将模型输出的tensor转化为可视的.png图像
    def forward(self, imgs):

        #feats 
        feats = self.backbone(imgs) #img = [batch, 3, H, W]
        
        if self.with_neck:
            feats = self.neck(feats)

        ppm_out = self.decoder(feats) # 1 512 11 35


        #trap forward
        f4, f8, f16, f32 = feats
        '''batch
            2   64  176 560

            4   192 88 280
            8    384 44 140
            16   768 22 70
            32   1536 11 35
        '''
        #将ppm聚合的特征作为第5stage进行融合，注意大小，效果一般
        #直接将ppm作为第四特征，4个stage融合？
        
        f16 = f16 + trapped_inter(ppm_out)
        f16 = self.block2(f16) 
        f8 = f8 + trapped_inter(f16)
        f8 = self.block3(f8) 
        f4 = f4 + trapped_inter(f8) 
        f4 = self.block4(f4) #96 88 280
        #f4 = self.block4(f4) 
        #f2 = f2 + trapped_inter(f4)
        #f2 = self.block5(f2) 

        f32 = self.fusion1(ppm_out)
        f16 = self.fusion2(f16)
        f8 = self.fusion3(f8)

        fusion = f32 + f16 + f8 + f4

        fusion = self.block5(fusion) # 96 88 280
        #采用crf的预测方式，判断最后融合的大小，决定上采样的尺度，
        if self.up_mode == 'mask':
            mask = self.mask_head(fusion)
            d1 = self.disp_head1(fusion, 1)
            d1 = self.upsample_mask(d1, mask)
        else:
            d1 = self.disp_head1(fusion, 4) #4倍的放大太过容易模糊边界，但是否适合iebins的尺度？

        depth = d1 * self.max_depth # batch，1,352,1120
        '''
        import cv2
        import numpy as np
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Spectral_r')
        #.cm 是Matplotlib中用于处理颜色映射（colormap）的模块
        depth = depth[0].squeeze()
        depth = depth.detach().cpu().numpy() # 如果depth是GPU张量
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth = np.uint8(depth)
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        cv2.imwrite("depth.png", depth) #[2,1,3,1120,4]
        '''

        return depth

#将最终特征转化为depth图的预测方式
class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x

#没用到的上采样？
class DispUnpack(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128):
        super(DispUnpack, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x, output_size):
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x)) # [b, 16, h/4, w/4]
        # x = torch.reshape(x, [x.shape[0], 1, x.shape[2]*4, x.shape[3]*4])
        x = self.pixel_shuffle(x)

        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)