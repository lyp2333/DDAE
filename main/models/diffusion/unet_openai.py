import math
import warnings
import copy
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import init
from abc import abstractmethod
from typing import Union
from numbers import Number
from enum import Enum


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def torch_checkpoint(func, args, flag, preserve_rng_state=False):
    # torch's gradient checkpoint works with automatic mixed precision
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


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(32, channels), channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Activation(Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        activation: Activation,
        dropout: float = 0,
    ):
        super().__init__()
        self.activation = activation

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = activation.get_act()
        if norm:
            self.norm = nn.GroupNorm(32, out_channels, affine=True)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == Activation.relu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == Activation.lrelu:
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == Activation.silu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, t_emb, z_emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, t_emb=None, z_emb=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t_emb=t_emb, z_emb=z_emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(self.dims, kernel_size=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class MappingNet(nn.Module):
    """
    Mapping latent code to another space, not used
    """

    def __init__(self,
                 num_layers=2,
                 in_channels=100,
                 emb_channels=512,
                 activation: Activation = Activation.silu,
                 use_norm=True,
                 dropout=0.1,
                 ):
        super(MappingNet, self).__init__()
        assert num_layers > 0, f'num_layer in MappingNet should > 0'
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.activation = activation
        self.use_norm = use_norm
        self.dropout = dropout
        self.in_layers = nn.ModuleList([])
        self.extra_in_layers = nn.ModuleList([])
        for i in range(self.num_layers):
            if i == 0:
                act = self.activation
                norm = self.use_norm
                a, b = self.in_channels, self.emb_channels
                dropout = self.dropout
                skip_connection = nn.Linear(a, b)
            elif i == self.num_layers - 1:
                act = Activation.none
                norm = False
                a, b = self.emb_channels, self.emb_channels
                dropout = 0.
                skip_connection = nn.Identity()
            else:
                act = self.activation
                norm = self.use_norm
                a, b = self.emb_channels, self.emb_channels
                dropout = self.dropout
                skip_connection = nn.Identity()

            self.extra_in_layers.append(skip_connection)
            self.in_layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    dropout=dropout,
                ))

    def forward(self, input: torch.Tensor):
        h: torch.Tensor = input
        for i in range(len(self.in_layers)):
            skip_connection_out = self.extra_in_layers[i](h)
            h = self.in_layers[i].forward(x=h)
            h += skip_connection_out
        return h


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        dropout,
        emb_channels=512,
        cond_emb_channels=None,
        out_channels=None,
        use_conv=True,
        use_scale_shift_norm=True,
        dims=2,
        use_checkpoint=False,
        use_z=True,
        use_t=True,
        resblock_up: bool = False,
        resblock_down: bool = False,
    ):
        super().__init__()
        self.dims = dims
        self.channels = channels
        self.t_emb_channels = int(emb_channels)
        self.cond_emb_channels = cond_emb_channels if cond_emb_channels is not None else self.t_emb_channels
        self.out_channels = out_channels if out_channels is not None else channels
        self.dropout = dropout
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_z = use_z
        self.use_t = use_t
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_up = resblock_up
        self.resblock_down = resblock_down
        self.updown = resblock_up or resblock_down

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        if self.use_t:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    self.t_emb_channels,
                    int(2 * self.out_channels) if use_scale_shift_norm else self.out_channels,
                ),
            )
        if self.use_z:
            self.z_emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(self.cond_emb_channels,
                       int(self.out_channels) if use_scale_shift_norm else self.out_channels,
                       ),
            )
        if self.resblock_up:
            self.h_upd = Upsample(self.channels, False, self.dims)
            self.x_upd = Upsample(self.channels, False, self.dims)
        elif self.resblock_down:
            self.h_upd = Downsample(self.channels, False, self.dims)
            self.x_upd = Downsample(self.channels, False, self.dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        self._init_weights()

    def _init_weights(self):
        """
        initialize weights
        :return: None
        """

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # zero-out adaLN modulation layers in ResBlock
        if self.use_t:
            nn.init.constant_(self.emb_layers[-1].weight, 0)
        if self.use_z:
            nn.init.constant_(self.z_emb_layers[-1].weight, 0)

    def forward(self, x, t_emb=None, z_emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding and condition z.
        :param x:
        :param t_emb:
        :param z_emb:
        :return:
        """
        return torch_checkpoint(
            self._forward, (x, t_emb, z_emb), self.use_checkpoint
        )

    def _forward(self, x, t_emb=None, z_emb=None):

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        t_emb_out, z_emb_out = None, None
        if self.use_t:
            t_emb_out = self.emb_layers(t_emb).type(h.dtype)
        if self.use_z:
            z_emb_out = self.z_emb_layers(z_emb).type(h.dtype)

        h = self._apply_conditions(
            h=h,
            t_emb=t_emb_out,
            z_emb=z_emb_out,
            emb_dim=self.out_channels,
            layers=self.out_layers,
            scale_bias=1.,
        )
        return self.skip_connection(x) + h

    def _apply_conditions(self,
                          h,
                          t_emb=None,
                          z_emb=None,
                          emb_dim=512,
                          layers=None,
                          scale_bias: Union[float, list] = 1.,
                          ):
        if self.use_t:
            while len(t_emb.shape) < len(h.shape):
                t_emb = t_emb[..., None]
        if self.use_z:
            while len(z_emb.shape) < len(h.shape):
                z_emb = z_emb[..., None]
        scale_shifts = [t_emb, z_emb]
        for i, each in enumerate(scale_shifts):
            if each is None:
                a = None
                b = None
            else:
                if each.shape[1] == emb_dim * 2:
                    a, b = torch.chunk(each, 2, dim=1)
                else:
                    a = each
                    b = None
            scale_shifts[i] = (a, b)
        # condition scale bias could be a list
        if isinstance(scale_bias, Number):
            biases = [scale_bias] * len(scale_shifts)
        else:
            # a list
            biases = scale_bias

        # default, the scale & shift are applied after the group norm but BEFORE SiLU
        pre_layers, post_layers = layers[0], layers[1:]

        h = pre_layers(h)
        # scale and shift for each condition
        for i, (scale, shift) in enumerate(scale_shifts):
            if scale is not None:
                h *= (biases[i] + scale)
            if shift is not None:
                h += shift
        h = post_layers(h)
        return h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self,
                 channels,
                 num_heads=1,
                 num_head_channels=-1,
                 use_checkpoint=False
                 ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return torch_checkpoint(self._forward, (x,), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


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
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        resblock_updown=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
    ):
        super(Encoder, self).__init__()
        self.dtype = th.float32

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.dims = dims
        self.resblock_updown = resblock_updown

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_z=False,
                        use_t=False,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.dropout,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            resblock_down=True,
                            use_z=False,
                            use_t=False,
                        ) if self.resblock_updown
                        else Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_z=False,
                use_t=False,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels),
            ResBlock(
                ch,
                self.dropout,
                dims=dims,
                use_z=False,
                use_t=False,
                use_checkpoint=use_checkpoint,
            ),
        )

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            conv_nd(dims, ch, self.out_channels, 1),
            nn.Flatten(),
        )

    def forward(self, x):
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
        h = self.middle_block(h)
        h = h.type(x.dtype)
        h = self.out(h)
        return h


class AdaptiveLayer(TimestepBlock):
    """
    TODO
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        emb_channels=512,
        cond_emb_channels=None,
        dropout=0,
        dims=2,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        use_z=True,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.t_emb_channels = int(emb_channels)
        self.cond_emb_channels = cond_emb_channels if cond_emb_channels is not None else self.t_emb_channels

        self.blocks_list = nn.ModuleList([
            ResBlock(
                in_channels,
                dropout,
                emb_channels=self.t_emb_channels,
                out_channels=model_channels,
                dims=dims,
                use_z=use_z,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        ])
        for _ in self.num_blocks - 2:
            self.blocks_list.append(
                ResBlock(
                    model_channels,
                    dropout,
                    emb_channels=self.t_emb_channels,
                    out_channels=model_channels,
                    dims=dims,
                    use_z=use_z,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )
        self.blocks_list.append(
            ResBlock(
                model_channels,
                dropout,
                emb_channels=self.t_emb_channels,
                out_channels=out_channels,
                dims=dims,
                use_z=use_z,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            )
        )

    def forward(self, x, t_emb=None, z_emb=None):
        h = x.type(self.dtype)
        for module in self.blocks_list:
            h = module(h, t_emb, z_emb)
        return h


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        emb_channels=512,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        z_dim=None,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        use_z=True,
        resblock_updown=True,
        use_mappingnet=False,
        use_x_hat=True,
        pred_mode="eps",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.pred_mode = pred_mode
        self.use_adaptive = "adaptive" in self.pred_mode
        self.use_ada_model = False
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels if not self.use_adaptive else out_channels * 2 + 1
        self.emb_channels = emb_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_z = use_z
        self.resblock_updown = resblock_updown
        self.use_mappingnet = use_mappingnet

        self.use_x_hat = use_x_hat
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = self.emb_channels
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if use_z:
            ...
            # todo
            # conf_MappingNet = conf.model_MappingNet
            # self.z_proj = MappingNet(
            #     num_layers=conf_MappingNet.num_layers,
            #     in_channels=conf_MappingNet.in_channels,
            #     emb_channels=time_embed_dim,
            #     use_norm=conf_MappingNet.use_norm,
            #     dropout=conf_MappingNet.dropout,
            # ) if self.use_mappingnet else nn.Identity()

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        emb_channels=time_embed_dim,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_z=use_z,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.dropout,
                            emb_channels=time_embed_dim,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            resblock_down=True,
                        ) if self.resblock_updown
                        else Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                dropout,
                emb_channels=time_embed_dim,
                dims=dims,
                use_z=use_z,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels),
            ResBlock(
                ch,
                dropout,
                emb_channels=time_embed_dim,
                dims=dims,
                use_z=use_z,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        dropout,
                        emb_channels=time_embed_dim,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_z=use_z,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(
                        ResBlock(
                            ch,
                            self.dropout,
                            emb_channels=time_embed_dim,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            resblock_up=True,
                        ) if self.resblock_updown
                        else Upsample(ch, conv_resample, dims=dims)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, self.model_channels, self.out_channels, 3, padding=1)),
        )
        self.norm = torch.sigmoid if self.use_adaptive else nn.Identity()
    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, z=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # Incorporate latent code infomation (if any)
        z_emb = z
        # Incorporate class label information (if any)
        # if self.num_classes is not None:
        #     assert y.shape == (x.shape[0],)
        #     t_emb += self.label_emb(y)
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, t_emb, z_emb)
            hs.append(h)
        h = self.middle_block(h, t_emb, z_emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop() * 2 ** (-0.5)], dim=1)
            h = module(cat_in, t_emb, z_emb)
        h = self.out(h)
        if self.use_adaptive:
            other, adaptive_factor = torch.split(h, [6, 1], dim=1)
            adaptive_factor = self.norm(adaptive_factor)
            h = torch.cat((other, adaptive_factor), dim=1)
        h = h.type(x.dtype)
        return h


class AdaptiveUNetModel(UNetModel):
    """
    AdaptiveUNetModel for adaptive training and inference
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        ada_type="last",
        num_adaptive_blocks=3,
        emb_channels=512,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        z_dim=None,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        use_z=True,
        resblock_updown=True,
        use_mappingnet=False,
        use_x_hat=True,
        pred_mode="adaptive_form1",
    ):

        super().__init__(
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            emb_channels,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            z_dim,
            num_classes,
            use_checkpoint,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            use_z,
            resblock_updown,
            use_mappingnet,
            use_x_hat,
        )
        self.pred_mode = pred_mode
        self.use_adaptive = "adaptive" in self.pred_mode
        assert self.use_adaptive, "use_adaptive must be True"
        self.model_type = ada_type
        self.num_adaptive_blocks = num_adaptive_blocks
        self.encoder_num_adaptive_blocks = 0 if self.model_type == "last" else self.num_adaptive_blocks
        self.decoder_num_adaptive_blocks = self.num_adaptive_blocks
        self.num_blocks = len(self.input_blocks)

        assert self.num_adaptive_blocks < self.num_blocks, "num_adaptive_blocks must be less than num_blocks"
        assert self.num_adaptive_blocks % (num_res_blocks + 1) == 0, "num_adaptive_blocks must be divisible by 3"
        self.use_ada_model = True

        if self.model_type == "symmetric":
            self.adaptive_input_blocks1, self.adaptive_input_blocks2, self.adaptive_input_blocks3 = self.input_blocks[:self.num_adaptive_blocks], copy.deepcopy(self.adaptive_input_blocks1), \
                                                                                                    copy.deepcopy(self.adaptive_input_blocks1),
        elif self.model_type == "last":
            self.adaptive_input_blocks1, self.adaptive_input_blocks2, self.adaptive_input_blocks3 = [], [], []
        else:
            raise NotImplementedError("model_type must be symmetric or last")
        self.shared_input_blocks, self.shared_output_blocks = self.input_blocks[self.encoder_num_adaptive_blocks:], self.output_blocks[:-self.decoder_num_adaptive_blocks]
        self.shared_mid_blocks = self.middle_block
        self.adaptive_output_blocks1 = self.output_blocks[-self.num_adaptive_blocks:]
        self.adaptive_output_blocks2 = copy.deepcopy(self.adaptive_output_blocks1)
        self.adaptive_output_blocks3 = copy.deepcopy(self.adaptive_output_blocks1)

        self.adaptive_out1 = self.out
        self.adaptive_out2, self.adaptive_out3 = copy.deepcopy(self.adaptive_out1), copy.deepcopy(self.adaptive_out1)
        self.len_adaptive_blocks = len(self.adaptive_input_blocks1) + len(self.adaptive_output_blocks1) + 1

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, z=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param z:
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        adaptive_hs1, adaptive_hs2, adaptive_hs3, shared_hs = [], [], [], []
        t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # Incorporate latent code infomation (if any)
        z_emb = z
        # Incorporate class label information (if any)
        # if self.num_classes is not None:
        #     assert y.shape == (x.shape[0],)
        #     t_emb += self.label_emb(y)
        h = h1 = h2 = h3 = x.type(self.inner_dtype)
        if self.model_type == "symmetric":
            for module in self.adaptive_input_blocks1:
                h1 = module(h1, t_emb, z_emb)
                adaptive_hs1.append(h1)
            for module in self.adaptive_input_blocks2:
                h2 = module(h2, t_emb, z_emb)
                adaptive_hs2.append(h2)
            for module in self.adaptive_input_blocks3:
                h3 = module(h3, t_emb, z_emb)
                adaptive_hs3.append(h3)

            # todo change src of h
            h = (h1 + h2 + h3) / 3 # for more stable training
        for module in self.shared_input_blocks:
            h = module(h, t_emb, z_emb)
            shared_hs.append(h)
        h = self.shared_mid_blocks(h, t_emb, z_emb)
        for module in self.shared_output_blocks:
            cat_in = th.cat([h, shared_hs.pop()], dim=1)
            h = module(cat_in, t_emb, z_emb)

        # todo change src of h1, h2, h3
        h1 = h2 = h3 = h
        # 1 / sqrt(2) for more stable training, see more in "ScaleLong: Towards More Stable Training of Diffusion Model via Scaling Network Long Skip Connection"
        if self.model_type == "last":
            adaptive_hs1 = adaptive_hs2 = adaptive_hs3 = shared_hs
            for index, module in enumerate(self.adaptive_output_blocks1):
                cat_in = torch.cat([h1, adaptive_hs1[index] * 2 ** (-0.5)], dim=1)
                h1 = module(cat_in, t_emb, z_emb)

            for index, module in enumerate(self.adaptive_output_blocks2):
                cat_in = torch.cat([h2, adaptive_hs2[index] * 2 ** (-0.5)], dim=1)
                h2 = module(cat_in, t_emb, z_emb)

            for index, module in enumerate(self.adaptive_output_blocks3):
                cat_in = torch.cat([h3, adaptive_hs3[index] * 2 ** (-0.5)], dim=1)
                h3 = module(cat_in, t_emb, z_emb)
        elif self.model_type == "symmetric":
            for module in self.adaptive_output_blocks1:
                cat_in = torch.cat([h1, adaptive_hs1.pop() * 2 ** (-0.5)], dim=1)
                h1 = module(cat_in, t_emb, z_emb)

            for module in self.adaptive_output_blocks2:
                cat_in = torch.cat([h2, adaptive_hs2.pop() * 2 ** (-0.5)], dim=1)
                h2 = module(cat_in, t_emb, z_emb)

            for module in self.adaptive_output_blocks3:
                cat_in = torch.cat([h3, adaptive_hs3.pop() * 2 ** (-0.5)], dim=1)
                h3 = module(cat_in, t_emb, z_emb)
        else:
            raise NotImplementedError("model_type must be symmetric or last")
        # h1 for eps, h2 for x_start, h3 for ada_factor
        h1, h2, h3 = self.adaptive_out1(h1), self.adaptive_out2(h2), self.adaptive_out3(h3)
        h3 = torch.sigmoid(torch.sum(h3, dim=1, keepdim=True))
        h = torch.cat([h1, h2, h3], dim=1)
        return h


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels=in_channels, *args, **kwargs)
        self.in_channels = self.in_channels * 2 if self.use_x_hat else self.in_channels

    def forward(self, x, timesteps, z=None, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape

        # low_res or z
        if self.use_x_hat:
            assert low_res is not None, f'low_res should not be None'
            upsampled = F.interpolate(low_res, (new_height, new_width), mode="nearest")
            x = th.cat([x, upsampled], dim=1)
        else:
            if low_res is not None:
                warnings.warn(f'low_res should be None, but get {type(low_res)}')

        if not self.use_z:
            return super().forward(x, timesteps, **kwargs)
        else:
            assert z is not None, 'z should not be None'
            return super().forward(x, timesteps, z=z, **kwargs)


class DecoderWrapper(nn.Module):
    def __init__(
        self,
        original_decoder: nn.Module = None,
        betas: torch.Tensor = None,
        target_predict_type: str = "eps",
    ):
        super(DecoderWrapper, self).__init__()
        assert betas is not None and original_decoder is not None
        assert target_predict_type == "eps", f"only support eps for now, but got{target_predict_type}"
        self.original_deocder = original_decoder
        self.original_predict_type = self.original_deocder.pred_mode
        self.original_deocder_use_adaptive = self.original_deocder.use_adaptive
        self.target_predict_type = target_predict_type
        self.register_buffer("betas", betas)
        dev = self.betas.device
        self.alphas = 1.0 - self.betas
        alpha_bar = torch.cumprod(self.alphas, dim=0)
        alpha_bar_shifted = torch.cat([torch.tensor([1.0], device=dev), alpha_bar[:-1]])
        self.alpha_bar_next = torch.cat([alpha_bar[1:], torch.tensor([0.0], device=dev)])
        assert alpha_bar_shifted.shape == torch.Size(
            [
                self.T,
            ]
        )
        # Auxillary consts
        self.register_buffer("sqrt_alpha", torch.sqrt(self.alphas))
        self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / self.alphas))
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("minus_sqrt_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alpha_bar))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alpha_bar - 1)
        )

        # q(x_t-1 | x_t, x_0) mean coefficients
        self.register_buffer(
            "post_coeff_1",
            self.betas * torch.sqrt(alpha_bar_shifted) / (1.0 - alpha_bar),
        )
        self.register_buffer(
            "post_coeff_2",
            torch.sqrt(self.alphas) * (1 - alpha_bar_shifted) / (1 - alpha_bar),
        )
        # q(x_t-1 | x_t, eps) mean coefficients
        self.register_buffer(
            "post_coeff_3",
            self.betas / self.minus_sqrt_alpha_bar,
        )
        # p(eps| x_t, x_t-1) coefficients
        self.register_buffer(
            "post_coeff_4",
            self.minus_sqrt_alpha_bar / self.betas,
        )
        # q(eps | x_t, x_0) coefficients
        self.register_buffer(
            "post_coeff_5",
            self.sqrt_alpha_bar / self.minus_sqrt_alpha_bar,
        )
        self.register_buffer(
            "post_coeff_6",
            1. / self.minus_sqrt_alpha_bar,
        )

    def _get_posterior_mean_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, _eps_score=None):
        assert _eps_score is not None
        # re-derive posterior_mean to get mean from x_t and eps directly
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long) if isinstance(t, int) else t
        eps_score = _eps_score

        # Generate the reconstruction from x_t and eps
        post_mean = extract(self.sqrt_recip_alpha, t_, x_t.shape) * \
                    (x_t - extract(self.post_coeff_3, t_, x_t.shape) * eps_score)
        return post_mean

    def _get_posterior_mean_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, _x0_score=None, clip_denoised=True):
        assert _x0_score is not None
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long) if isinstance(t, int) else t
        x0_score = _x0_score

        # clip x0_score
        if clip_denoised:
            # todo add dynamic threshold
            x0_score.clamp_(-1.0, 1.0)

        # Generate the reconstruction from x_t and x_0
        post_mean = (
            extract(self.post_coeff_1, t_, x_t.shape) * x0_score
            + extract(self.post_coeff_2, t_, x_t.shape) * x_t
        )
        return post_mean

    def _get_posterior_eps_from_xt_xtminus1(self, x_t: torch.Tensor, x_t_minus1: torch.Tensor, t_: torch.Tensor):
        """
        get posterior eps from x_t and predicted x_t-1 from x_t
        :param x_t:
        :param x_t_minus1:
        :param t:
        :return:
        """
        return extract(self.post_coeff_4, t_, x_t.shape) * \
               (x_t - extract(self.sqrt_alpha, t_, x_t.shape) * x_t_minus1)

    def _get_posterior_eps_from_xt_x0(self, x_t, t_, x_0=None, clip_denoised=True, cond=None, z=None):
        t_ = t_
        if x_0 is None:
            x0_score = self.decoder(x_t, t_, low_res=cond, z=z)[..., 3:6, :, :] if self.use_adaptive else self.decoder(x_t, t_, low_res=cond, z=z)
        else:
            x0_score = x_0

        if clip_denoised:
            x0_score.clamp_(-1.0, 1.0)

        pred_eps = (
            extract(self.post_coeff_6, t_, x_t.shape) * x_t -
            extract(self.post_coeff_5, t_, x_t.shape) * x0_score
        )
        return pred_eps

    def forward(self, x_t, timesteps, cond=None, y=None, **kwargs):
        predict_score = self.decoder(x_t, timesteps, z=cond, y=y, **kwargs)
        t_ = timesteps if isinstance(timesteps, torch.Tensor) else torch.full((x_t.size(0),), timesteps, device=x_t.device, dtype=torch.long)
        if self.use_adaptive:
            eps_pred, x0_pred, adaptive_factor = predict_score[..., :3, :, :], predict_score[..., 3:6, :, :], predict_score[..., 6:7, :, :]
            if self.original_predict_type == "adaptive_form1":
                pred_xt_minus1_from_noise = self._get_posterior_mean_from_noise(
                    x_t,
                    t_,
                    _eps_score=eps_pred,
                )
                pred_xt_minus1_from_x0 = self._get_posterior_mean_from_x0(
                    x_t,
                    t_,
                    _x0_score=x0_pred,
                )
                pred_xt_minus1 = adaptive_factor * pred_xt_minus1_from_noise + \
                                 (1 - adaptive_factor) * pred_xt_minus1_from_x0
                eps_derived = self._get_posterior_eps_from_xt_xtminus1(
                    x_t,
                    pred_xt_minus1,
                    t_,
                )
                return eps_derived
            elif self.original_predict_type == "adaptive_form2":
                eps_from_x0_pred = self._get_posterior_eps_from_xt_x0(
                    x_t,
                    t_,
                    x0_pred,
                    clip_denoised=True,
                )
                pred_eps = (1 - adaptive_factor) * eps_from_x0_pred + adaptive_factor * eps_pred
                return pred_eps
            else:
                raise NotImplementedError()
        elif self.original_predict_type == "eps":
            return predict_score
        elif self.original_predict_type == "x0":
            raise NotImplementedError()
        else:
            raise NotImplementedError()


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model_ada_unet = AdaptiveUNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[16, ],
        channel_mult=[1, 1, 2, 3, 4],
        use_z=True,
        use_checkpoint=True,
    ).to(device)
    model_enc = Encoder(
        in_channels=3,
        model_channels=128,
        out_channels=512,
        num_res_blocks=2,
        attention_resolutions=[16, ],
        channel_mult=[1, 1, 2, 3, 4, 4],
    )
    # model_sup = SuperResModel(
    #     in_channels=3,
    #     model_channels=128,
    #     out_channels=3,
    #     num_res_blocks=2,
    #     attention_resolutions=[16, ],
    #     channel_mult=[1, 1, 2, 3, 4],
    #     use_z=True,
    #     use_x_hat=False,
    #     use_scale_shift_norm=False,
    # )

    # bs = 72
    # img_batch = torch.ones(bs, 3, 128, 128, device=device)
    # t = torch.ones(bs, device=img_batch.device)
    # z = torch.randn(bs, 512, device=img_batch.device)
    # eps = model_ada_unet(img_batch, t, z=z)

    # real_eps = torch.randn((bs, 9, 128, 128), device=img_batch.device)
    # loss_fn = torch.nn.MSELoss()
    # loss = loss_fn(eps, real_eps)
    # loss.backward()
