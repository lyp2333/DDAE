import torch
import math
import pytorch_lightning as pl
from torch import nn
from torch.nn import init
from typing import Dict, Any
from omegaconf.dictconfig import DictConfig
from collections import OrderedDict

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def torch_checkpoint(func, args, flag, preserve_rng_state=False):
    # torch's gradient checkpoint works with automatic mixed precision, given torch >= 1.8
    if flag:
        return torch.utils.checkpoint.checkpoint(
            func, *args, preserve_rng_state=preserve_rng_state)
    else:
        return func(*args)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(32, channels), channels)


class ResidualBlock(nn.Module):
    """Residual Block with group normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(dim_in, dim_out, bias=False),
            nn.GroupNorm(32, dim_out, affine=True),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out, bias=False),
            nn.GroupNorm(32, dim_out, affine=True),
        )

    def forward(self, x):
        return x + self.main(x)


class MlpBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MlpBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(dim_in, dim_out, bias=False),
            nn.GroupNorm(32, dim_out, affine=True),
            nn.SiLU(),
        )
        self.skip_connection = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x):
        return self.main(x) + self.skip_connection(x)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % self.num_heads == 0
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return torch_checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class GAE(nn.Module):
    """GAE's Generator network, with fully connected layers to get latent Z"""

    def __init__(self,
                 in_channels=512,
                 base_dim=512,
                 out_channels=512,
                 nums_skip_blocks=4,
                 repeat_num=2,
                 z_dim=100,
                 dropout=0.1,
                 num_groups=32,
                 ):
        super(GAE, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_dim
        self.out_channels = out_channels
        self.num_skip_blocks = nums_skip_blocks
        self.repeat_num = repeat_num
        self.z_dim = z_dim
        self.dropout_rate = dropout
        self.num_groups = num_groups
        self.skip_connection_down = nn.ModuleList()
        self.skip_connection_up = nn.ModuleList()
        '''
        encoder
        '''
        self.start_part = MlpBlock(self.in_channels, self.base_channels)

        # skip-MLP layers.
        self.skip_mlp_down_layers = []
        curr_dim = self.base_channels
        for i in range(self.num_skip_blocks):
            if i <= 1:
                self.skip_mlp_down_layers.append(nn.Sequential(
                    nn.Linear(curr_dim, curr_dim * 2, bias=False),
                    nn.GroupNorm(self.num_groups, curr_dim * 2, affine=True),
                    nn.SiLU(),
                    nn.Dropout(self.dropout_rate)
                ))
                self.skip_connection_down.append(
                    nn.Sequential(nn.Linear(curr_dim, curr_dim * 2))
                )
                curr_dim *= 2
            else:
                self.skip_mlp_down_layers.append(
                    nn.Sequential(
                        nn.Linear(curr_dim, curr_dim, bias=False),
                        nn.GroupNorm(self.num_groups, curr_dim, affine=True),
                        nn.SiLU(),
                        nn.Dropout(self.dropout_rate)
                    )
                )
                self.skip_connection_down.append(
                    nn.Sequential(nn.Identity())
                )

        # Encoder Bottleneck layers.
        self.bottle_encoder_layers = []
        for i in range(repeat_num):
            self.bottle_encoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.encoder = nn.Sequential(self.start_part, *self.skip_mlp_down_layers, *self.bottle_encoder_layers)
        self.start_part, self.down_part, self.bottleneck_down = self.encoder[0], self.encoder[1: -self.repeat_num], self.encoder[-self.repeat_num:]

        # fc layers
        self.fc_encoder = nn.Sequential(
            nn.Linear(curr_dim, curr_dim),
            nn.ReLU(),
            nn.Linear(curr_dim, self.z_dim),
        )

        '''
        decoder
        '''
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.z_dim, curr_dim),
            nn.ReLU(),
            nn.Linear(curr_dim, curr_dim),
        )
        # Decoder Bottleneck layers.
        self.bottle_decoder_layers = []
        for i in range(repeat_num):
            self.bottle_decoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.bottle_decoder_part = nn.Sequential(*self.bottle_decoder_layers)

        # Up-sampling layers.
        self.skip_mlp_up_layers = nn.ModuleList()
        for i in range(self.num_skip_blocks):
            if i <= self.num_skip_blocks - 3:
                self.skip_mlp_up_layers.append(
                    nn.Sequential(
                        nn.Linear(curr_dim, curr_dim, bias=False),
                        nn.GroupNorm(self.num_groups, curr_dim, affine=True),
                        nn.SiLU(),
                        nn.Dropout(self.dropout_rate),
                    )
                )
                self.skip_connection_up.append(nn.Sequential(nn.Identity()))
            else:
                self.skip_mlp_up_layers.append(
                    nn.Sequential(
                        nn.Linear(curr_dim, curr_dim // 2, bias=False),
                        nn.GroupNorm(self.num_groups, curr_dim // 2, affine=True),
                        nn.SiLU(),
                        nn.Dropout(self.dropout_rate),
                    )
                )
                self.skip_connection_up.append(
                    nn.Sequential(nn.Linear(curr_dim, curr_dim // 2))
                )
                curr_dim = curr_dim // 2

        #
        self.end_part = MlpBlock(self.base_channels, self.out_channels)

        self.decoder = nn.Sequential(*self.bottle_decoder_layers, *self.skip_mlp_up_layers, self.end_part)
        self.bottleneck_up, self.up_part, self.end_part = self.decoder[:self.repeat_num], self.decoder[self.repeat_num: -1], self.decoder[-1]

    def forward(self, x):
        h = self.start_part(x)
        for idx, layer in enumerate(self.down_part):
            x = layer(h)
            skip_out = self.skip_connection_down[idx](h)
            h = x + skip_out
        h = self.bottleneck_down(h)
        z = self.fc_encoder(h)

        h = self.fc_decoder(z)
        h = self.bottleneck_up(h)
        for idx, layer in enumerate(self.up_part):
            x = layer(h)
            skip_out = self.skip_connection_up[idx](h)
            h = x + skip_out
        h = self.end_part(h)
        return h, z

    def get_latent_code(self, x):
        h = self.start_part(x)
        for idx, layer in enumerate(self.down_part):
            x = layer(h)
            skip_out = self.skip_connection_down[idx](h)
            h = x + skip_out
        h = self.bottleneck_down(h)
        z = self.fc_encoder(h)
        return z

    def get_recon(self, z):
        h = self.fc_decoder(z)
        h = self.bottleneck_up(h)
        for idx, layer in enumerate(self.up_part):
            x = layer(h)
            skip_out = self.skip_connection_up[idx](h)
            h = x + skip_out
        h = self.end_part(h)
        return h


class GAE_PL(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super(GAE_PL, self).__init__()
        self.save_hyperparameters()
        DataType = ['fonts', 'ilab']
        LossType = ['l1', 'l2']
        flag = False
        for data_type in DataType:
            if data_type in config.data.name:
                self.data_type = data_type
                flag = True
        assert flag is True, 'data type not found in config file.'
        self.config = config
        # model attributes
        self.in_channels = config.model.in_channels
        self.base_dim = config.model.base_dim
        self.num_skip_blocks = config.model.num_skip_blocks
        self.out_channels = config.model.out_channels
        self.repeat_num = config.model.repeat_num
        self.dropout_rate = config.model.dropout
        self.z_dim = config.model.z_dim
        self.loss_type = 'l1'
        assert self.loss_type in LossType
        if 'training' in config:
            self.optimizer = self.config.training.optimizer
            self.lr = self.config.training.training_lr
            self.beta1 = self.config.training.beta_1
            self.beta2 = self.config.training.beta_2
            self.lambda_recon = self.config.training.lambda_recon
            self.lambda_combine = self.config.training.lambda_combine
            self.lambda_unsup = self.config.training.lambda_unsup

        self.Gae = GAE(
            in_channels=self.in_channels,
            base_dim=self.base_dim,
            out_channels=self.out_channels,
            nums_skip_blocks=self.num_skip_blocks,
            repeat_num=self.repeat_num,
            z_dim=self.z_dim,
            dropout=self.dropout_rate,
        )

        if self.data_type == 'fonts':
            self.z_content_dim = config.model.z_content_dim
            self.z_size_dim = config.model.z_size_dim
            self.z_font_color_dim = config.model.z_font_color_dim
            self.z_back_color_dim = config.model.z_back_color_dim
            self.z_style_dim = config.model.z_style_dim
            self.z_content_start_dim = 0
            self.z_size_start_dim = self.z_content_start_dim + self.z_content_dim
            self.z_font_color_start_dim = self.z_size_start_dim + self.z_size_dim
            self.z_back_color_start_dim = self.z_font_color_start_dim + self.z_font_color_dim
            self.z_style_start_dim = self.z_back_color_start_dim + self.z_back_color_dim

        elif self.data_type == 'ilab':
            self.z_pose_dim = config.model.z_pose_dim
            self.z_back_dim = config.model.z_back_dim
            self.z_id_dim = self.z_dim - self.z_pose_dim - self.z_back_dim

        else:
            raise NotImplementedError("Not NotImplemented")

    def training_step(self, batch, bacth_idx):
        terms = {}
        if self.data_type == 'fonts':
            A_img = batch[:, 0, ...]
            B_img = batch[:, 1, ...]
            C_img = batch[:, 2, ...]
            D_img = batch[:, 3, ...]
            E_img = batch[:, 4, ...]
            F_img = batch[:, 5, ...]

            ## 1. A B C seperate(first400: id last600 background)
            A_recon, A_z = self.Gae(A_img)
            B_recon, B_z = self.Gae(B_img)
            C_recon, C_z = self.Gae(C_img)
            D_recon, D_z = self.Gae(D_img)
            E_recon, E_z = self.Gae(E_img)
            F_recon, F_z = self.Gae(F_img)
            ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''
            terms['z_original'] = batch
            terms['z_recon'], terms['z_dis'] = torch.stack([A_recon, B_recon, C_recon, D_recon, E_recon, F_recon], dim=1), \
                                               torch.stack([A_z, B_z, C_z, D_z, E_z, F_z], dim=1)

            A_z_1 = A_z[:, 0:self.z_size_start_dim]  # 0-20
            A_z_2 = A_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
            A_z_3 = A_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
            A_z_4 = A_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
            A_z_5 = A_z[:, self.z_style_start_dim:]  # 80-100
            B_z_1 = B_z[:, 0:self.z_size_start_dim]  # 0-20
            B_z_2 = B_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
            B_z_3 = B_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
            B_z_4 = B_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
            B_z_5 = B_z[:, self.z_style_start_dim:]  # 80-100
            C_z_1 = C_z[:, 0:self.z_size_start_dim]  # 0-20
            C_z_2 = C_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
            C_z_3 = C_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
            C_z_4 = C_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
            C_z_5 = C_z[:, self.z_style_start_dim:]  # 80-100
            D_z_1 = D_z[:, 0:self.z_size_start_dim]  # 0-20
            D_z_2 = D_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
            D_z_3 = D_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
            D_z_4 = D_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
            D_z_5 = D_z[:, self.z_style_start_dim:]  # 80-100
            E_z_1 = E_z[:, 0:self.z_size_start_dim]  # 0-20
            E_z_2 = E_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
            E_z_3 = E_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
            E_z_4 = E_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
            E_z_5 = E_z[:, self.z_style_start_dim:]  # 80-100
            F_z_1 = F_z[:, 0:self.z_size_start_dim]  # 0-20
            F_z_2 = F_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
            F_z_3 = F_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
            F_z_4 = F_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
            F_z_5 = F_z[:, self.z_style_start_dim:]  # 80-100

            ## 2. combine with strong supervise
            ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''
            # C A same content 1
            A1Co_combine_2C = torch.cat((A_z_1, C_z_2, C_z_3, C_z_4, C_z_5), dim=1)
            mid_A1Co = self.Gae.fc_decoder(A1Co_combine_2C)
            A1Co_2C = self.Gae.decoder(mid_A1Co)

            AoC1_combine_2A = torch.cat((C_z_1, A_z_2, A_z_3, A_z_4, A_z_5), dim=1)
            mid_AoC1 = self.Gae.fc_decoder(AoC1_combine_2A)
            AoC1_2A = self.Gae.decoder(mid_AoC1)

            # C B same size 2
            B2Co_combine_2C = torch.cat((C_z_1, B_z_2, C_z_3, C_z_4, C_z_5), dim=1)
            mid_B2Co = self.Gae.fc_decoder(B2Co_combine_2C)
            B2Co_2C = self.Gae.decoder(mid_B2Co)

            BoC2_combine_2B = torch.cat((B_z_1, C_z_2, B_z_3, B_z_4, B_z_5), dim=1)
            mid_BoC2 = self.Gae.fc_decoder(BoC2_combine_2B)
            BoC2_2B = self.Gae.decoder(mid_BoC2)

            # C D same font_color 3
            D3Co_combine_2C = torch.cat((C_z_1, C_z_2, D_z_3, C_z_4, C_z_5), dim=1)
            mid_D3Co = self.Gae.fc_decoder(D3Co_combine_2C)
            D3Co_2C = self.Gae.decoder(mid_D3Co)

            DoC3_combine_2D = torch.cat((D_z_1, D_z_2, C_z_3, D_z_4, D_z_5), dim=1)
            mid_DoC3 = self.Gae.fc_decoder(DoC3_combine_2D)
            DoC3_2D = self.Gae.decoder(mid_DoC3)

            # C E same back_color 4
            E4Co_combine_2C = torch.cat((C_z_1, C_z_2, C_z_3, E_z_4, C_z_5), dim=1)
            mid_E4Co = self.Gae.fc_decoder(E4Co_combine_2C)
            E4Co_2C = self.Gae.decoder(mid_E4Co)

            EoC4_combine_2E = torch.cat((E_z_1, E_z_2, E_z_3, C_z_4, E_z_5), dim=1)
            mid_EoC4 = self.Gae.fc_decoder(EoC4_combine_2E)
            EoC4_2E = self.Gae.decoder(mid_EoC4)

            # C F same style 5
            F5Co_combine_2C = torch.cat((C_z_1, C_z_2, C_z_3, C_z_4, F_z_5), dim=1)
            mid_F5Co = self.Gae.fc_decoder(F5Co_combine_2C)
            F5Co_2C = self.Gae.decoder(mid_F5Co)

            FoC5_combine_2F = torch.cat((F_z_1, F_z_2, F_z_3, F_z_4, C_z_5), dim=1)
            mid_FoC5 = self.Gae.fc_decoder(FoC5_combine_2F)
            FoC5_2F = self.Gae.decoder(mid_FoC5)

            # combine_2C
            A1B2D3E4F5_combine_2C = torch.cat((A_z_1, B_z_2, D_z_3, E_z_4, F_z_5), dim=1)
            mid_A1B2D3E4F5 = self.Gae.fc_decoder(A1B2D3E4F5_combine_2C)
            A1B2D3E4F5_2C = self.Gae.decoder(mid_A1B2D3E4F5)

            # '''  need unsupervise '''
            A2B3D4E5F1_combine_2N = torch.cat((F_z_1, A_z_2, B_z_3, D_z_4, E_z_5), dim=1)
            mid_A2B3D4E5F1 = self.Gae.fc_decoder(A2B3D4E5F1_combine_2N)
            A2B3D4E5F1_2N = self.Gae.decoder(mid_A2B3D4E5F1)

            '''
            optimize for Gae
            '''
            # 1. recon_loss
            A_recon_loss = torch.mean(torch.abs(A_img - A_recon))
            B_recon_loss = torch.mean(torch.abs(B_img - B_recon))
            C_recon_loss = torch.mean(torch.abs(C_img - C_recon))
            D_recon_loss = torch.mean(torch.abs(D_img - D_recon))
            E_recon_loss = torch.mean(torch.abs(E_img - E_recon))
            F_recon_loss = torch.mean(torch.abs(F_img - F_recon))
            recon_loss = A_recon_loss + B_recon_loss + C_recon_loss + D_recon_loss + E_recon_loss + F_recon_loss

            # 2. sup_combine_loss
            A1Co_2C_loss = torch.mean(torch.abs(C_img - A1Co_2C))
            AoC1_2A_loss = torch.mean(torch.abs(A_img - AoC1_2A))
            B2Co_2C_loss = torch.mean(torch.abs(C_img - B2Co_2C))
            BoC2_2B_loss = torch.mean(torch.abs(B_img - BoC2_2B))
            D3Co_2C_loss = torch.mean(torch.abs(C_img - D3Co_2C))
            DoC3_2D_loss = torch.mean(torch.abs(D_img - DoC3_2D))
            E4Co_2C_loss = torch.mean(torch.abs(C_img - E4Co_2C))
            EoC4_2E_loss = torch.mean(torch.abs(E_img - EoC4_2E))
            F5Co_2C_loss = torch.mean(torch.abs(C_img - F5Co_2C))
            FoC5_2F_loss = torch.mean(torch.abs(F_img - FoC5_2F))
            A1B2D3E4F5_2C_loss = torch.mean(torch.abs(C_img - A1B2D3E4F5_2C))
            combine_sup_loss = A1Co_2C_loss + AoC1_2A_loss + B2Co_2C_loss + BoC2_2B_loss + D3Co_2C_loss + DoC3_2D_loss + E4Co_2C_loss + EoC4_2E_loss + F5Co_2C_loss + FoC5_2F_loss + A1B2D3E4F5_2C_loss

            # 3. unsup_combine_loss
            _, A2B3D4E5F1_z = self.Gae(A2B3D4E5F1_2N)
            combine_unsup_loss = torch.mean(torch.abs(F_z_1 - A2B3D4E5F1_z[:, 0:self.z_size_start_dim])) \
                                 + torch.mean(torch.abs(A_z_2 - A2B3D4E5F1_z[:, self.z_size_start_dim: self.z_font_color_start_dim])) \
                                 + torch.mean(torch.abs(B_z_3 - A2B3D4E5F1_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim])) \
                                 + torch.mean(torch.abs(D_z_4 - A2B3D4E5F1_z[:, self.z_back_color_start_dim: self.z_style_start_dim])) \
                                 + torch.mean(torch.abs(E_z_5 - A2B3D4E5F1_z[:, self.z_style_start_dim:]))

            # whole loss
            whole_loss = self.lambda_recon * recon_loss + self.lambda_combine * combine_sup_loss + self.lambda_unsup * combine_unsup_loss
            terms['loss'] = whole_loss
        elif self.data_type == 'ilab':

            A_img = batch[:, 0, ...]
            B_img = batch[:, 1, ...]
            C_img = batch[:, 2, ...]
            D_img = batch[:, 3, ...]
            # 1. A B C seperate(first400: id last600 background)
            A_recon, A_z = self.Gae(A_img)
            B_recon, B_z = self.Gae(B_img)
            C_recon, C_z = self.Gae(C_img)
            D_recon, D_z = self.Gae(D_img)
            terms['z_original'] = batch
            terms['z_recon'], terms['z_dis'] = torch.stack([A_recon, B_recon, C_recon, D_recon], dim=1), \
                                               torch.stack([A_z, B_z, C_z, D_z], dim=1)
            # 2. A B C fuse
            A_z_id = A_z[:, 0:self.z_id_dim]  # 0-60
            A_z_back = A_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]  # 60-80
            A_z_pose = A_z[:, self.z_id_dim + self.z_back_dim:]  # 80-100
            B_z_id = B_z[:, 0:self.z_id_dim]
            B_z_back = B_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]
            B_z_pose = B_z[:, self.z_id_dim + self.z_back_dim:]
            C_z_id = C_z[:, 0:self.z_id_dim]
            C_z_back = C_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]
            C_z_pose = C_z[:, self.z_id_dim + self.z_back_dim:]
            D_z_id = D_z[:, 0:self.z_id_dim]
            D_z_back = D_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]
            D_z_pose = D_z[:, self.z_id_dim + self.z_back_dim:]

            ## 2. combine with strong supervise
            # C A same pose diff id, back
            ApCo_combine_2C = torch.cat((C_z_id, C_z_back), dim=1)
            ApCo_combine_2C = torch.cat((ApCo_combine_2C, A_z_pose), dim=1)
            ApCo_2C = self.Gae.get_recon(ApCo_combine_2C)

            AoCp_combine_2A = torch.cat((A_z_id, A_z_back), dim=1)
            AoCp_combine_2A = torch.cat((AoCp_combine_2A, C_z_pose), dim=1)
            AoCp_2A = self.Gae.get_recon(AoCp_combine_2A)

            # C B same id diff pose, back
            BaCo_combine_2C = torch.cat((B_z_id, C_z_back), dim=1)
            BaCo_combine_2C = torch.cat((BaCo_combine_2C, C_z_pose), dim=1)
            BaCo_2C = self.Gae.get_recon(BaCo_combine_2C)

            BoCa_combine_2B = torch.cat((C_z_id, B_z_back), dim=1)
            BoCa_combine_2B = torch.cat((BoCa_combine_2B, B_z_pose), dim=1)
            BoCa_2B = self.Gae.get_recon(BoCa_combine_2B)

            # C D same background diff id, pose
            DbCo_combine_2C = torch.cat((C_z_id, D_z_back), dim=1)
            DbCo_combine_2C = torch.cat((DbCo_combine_2C, C_z_pose), dim=1)
            DbCo_2C = self.Gae.get_recon(DbCo_combine_2C)

            DoCb_combine_2D = torch.cat((D_z_id, C_z_back), dim=1)
            DoCb_combine_2D = torch.cat((DoCb_combine_2D, D_z_pose), dim=1)
            DoCb_2D = self.Gae.get_recon(DoCb_combine_2D)

            # combine_2C
            ApBaDb_combine_2C = torch.cat((B_z_id, D_z_back), dim=1)
            ApBaDb_combine_2C = torch.cat((ApBaDb_combine_2C, A_z_pose), dim=1)
            ApBaDb_2C = self.Gae.get_recon(ApBaDb_combine_2C)

            # '''  need unsupervise '''
            AaBpDb_combine_2N = torch.cat((A_z_id, D_z_back), dim=1)
            AaBpDb_combine_2N = torch.cat((AaBpDb_combine_2N, B_z_pose), dim=1)
            AaBpDb_2N = self.Gae.get_recon(AaBpDb_combine_2N)

            '''
            optimize for Gae
            '''

            if self.loss_type == 'l1':
                # 1. recon_loss
                A_recon_loss = torch.mean(torch.abs(A_img - A_recon))
                B_recon_loss = torch.mean(torch.abs(B_img - B_recon))
                C_recon_loss = torch.mean(torch.abs(C_img - C_recon))
                D_recon_loss = torch.mean(torch.abs(D_img - D_recon))
                # 2. sup_combine_loss
                ApCo_2C_loss = torch.mean(torch.abs(C_img - ApCo_2C))
                AoCp_2A_loss = torch.mean(torch.abs(A_img - AoCp_2A))
                BaCo_2C_loss = torch.mean(torch.abs(C_img - BaCo_2C))
                BoCa_2B_loss = torch.mean(torch.abs(B_img - BoCa_2B))
                DbCo_2C_loss = torch.mean(torch.abs(C_img - DbCo_2C))
                DoCb_2D_loss = torch.mean(torch.abs(D_img - DoCb_2D))
                ApBaDb_2C_loss = torch.mean(torch.abs(C_img - ApBaDb_2C))
                # 3. unsup_combine_loss
                AaBpDb_img, AaBpDb_z = self.Gae(AaBpDb_2N)
                combine_unsup_loss = torch.mean(torch.abs(A_z_id - AaBpDb_z[:, 0:self.z_id_dim])) + \
                                     torch.mean(torch.abs(D_z_back - AaBpDb_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim])) + \
                                     torch.mean(torch.abs(B_z_pose - AaBpDb_z[:, self.z_id_dim + self.z_back_dim:]))

            elif self.loss_type == 'l2':
                A_recon_loss = torch.mean((A_img - A_recon) ** 2)
                B_recon_loss = torch.mean((B_img - B_recon) ** 2)
                C_recon_loss = torch.mean((C_img - C_recon) ** 2)
                D_recon_loss = torch.mean((D_img - D_recon) ** 2)
                # 2. sup_combine_loss
                ApCo_2C_loss = torch.mean((C_img - ApCo_2C) ** 2)
                AoCp_2A_loss = torch.mean((A_img - AoCp_2A) ** 2)
                BaCo_2C_loss = torch.mean((C_img - BaCo_2C) ** 2)
                BoCa_2B_loss = torch.mean((B_img - BoCa_2B) ** 2)
                DbCo_2C_loss = torch.mean((C_img - DbCo_2C) ** 2)
                DoCb_2D_loss = torch.mean((D_img - DoCb_2D) ** 2)
                ApBaDb_2C_loss = torch.mean((C_img - ApBaDb_2C) ** 2)
                # 3. unsup_combine_loss
                AaBpDb_img, AaBpDb_z = self.Gae(AaBpDb_2N)
                combine_unsup_loss = torch.mean((A_z_id - AaBpDb_z[:, 0:self.z_id_dim]) ** 2) + \
                                     torch.mean((D_z_back - AaBpDb_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]) ** 2) + \
                                     torch.mean((B_z_pose - AaBpDb_z[:, self.z_id_dim + self.z_back_dim:]) ** 2)
            else:
                raise NotImplementedError()

            recon_loss = A_recon_loss + B_recon_loss + C_recon_loss + D_recon_loss
            combine_sup_loss = ApCo_2C_loss + AoCp_2A_loss + BaCo_2C_loss + BoCa_2B_loss + DbCo_2C_loss + DoCb_2D_loss + ApBaDb_2C_loss
            # Whole loss
            whole_loss = self.lambda_recon * recon_loss + self.lambda_combine * combine_sup_loss + self.lambda_unsup * combine_unsup_loss
            terms['loss'] = whole_loss
        else:
            raise NotImplementedError()
        if bacth_idx != -1:
            self.log('whole_loss', whole_loss, prog_bar=True)
            self.log('recon_loss', recon_loss, prog_bar=True)
            self.log('sup_loss', combine_sup_loss, prog_bar=True)
            self.log('unsup_loss', combine_unsup_loss, prog_bar=True)
        return terms

    def state_dict(self, *args, **kwargs):
        # don't save sampler
        out = OrderedDict()
        for k, v in super().state_dict(*args, **kwargs).items():
            if k.startswith('sampler.'):
                pass
            else:
                out[k] = v
        return out

    def forward(self, x):
        return self.Gae(x)

    def get_latent_code(self, x):
        return self.Gae.get_latent_code(x)

    def get_recon(self, z):
        return self.Gae.get_recon(z)

    def configure_optimizers(self):
        out = {}
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
            out['optimizer'] = optimizer
        else:
            raise NotImplementedError('Not Implemented')
        return out


if __name__ == '__main__':
    model = GAE()
    emb = torch.ones(128, 512)
    # z_dis = model.get_latent_code(emb)
    # z = model.get_recon(z_dis)
    # print(z_dis.shape)
    # print(z.shape)
    print(model(emb)[0].shape)
    print(model(emb)[1].shape)
    # print(type(model.up_part))
    # for name, pars in model.up_part.named_parameters():
    #     print(name)
