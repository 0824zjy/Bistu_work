#5.5
import torch
import torch.nn as nn

from ldm.modules.diffusionmodules.util import conv_nd


class ResidualBlock(nn.Module):
    def __init__(self, dims, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv_nd(
            dims,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.silu = nn.SiLU()
        self.conv2 = conv_nd(
            dims,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if in_channels != out_channels:
            self.residual = conv_nd(
                dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.silu(out)
        out = self.conv2(out)

        out = out + residual
        out = self.silu(out)

        return out


class PatchMerging(nn.Module):
    def __init__(self, patch_dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()

        self.patch_dim = patch_dim
        self.norm = norm_layer(4 * patch_dim)
        self.reduction = nn.Linear(4 * patch_dim, 2 * patch_dim)

    def forward(self, x):
        """
        Args:
            x: [B,H,W,C]

        Returns:
            x: [B,H/2,W/2,2C]
        """
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Height and Width must be even."

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        x = x.view(B, H // 2, W // 2, 2 * C)

        return x


class FeatureExtractor(nn.Module):
    """
    Dense Hint Input encoder.

    This structure can be used for:
        - region / mask branch
        - soft boundary prior branch

    In the new SBP-PG method, boundary should not be used as a hard binary
    condition. If this extractor is used for boundary, its input should be
    soft_boundary_prior rather than hard boundary.
    """

    def __init__(self, hint_channels, dim=2):
        super(FeatureExtractor, self).__init__()

        self.initial_conv = conv_nd(
            dim,
            hint_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.layer1 = ResidualBlock(dim, 16, 16)

        self.conv_before_res2 = conv_nd(
            dim,
            16,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.layer2 = ResidualBlock(dim, 32, 32)
        self.patch_merge1 = PatchMerging(patch_dim=32)

        self.conv_before_res3 = conv_nd(
            dim,
            64,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.layer3 = ResidualBlock(dim, 64, 64)
        self.patch_merge2 = PatchMerging(patch_dim=64)

        self.conv_before_res4 = conv_nd(
            dim,
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.layer4 = ResidualBlock(dim, 128, 128)
        self.patch_merge3 = PatchMerging(patch_dim=128)

        self.conv_before_res5 = conv_nd(
            dim,
            256,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.layer5 = ResidualBlock(dim, 256, 256)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)

        x = self.conv_before_res2(x)
        x = self.layer2(x)

        x = x.permute(0, 2, 3, 1)
        x = self.patch_merge1(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv_before_res3(x)
        x = self.layer3(x)

        x = x.permute(0, 2, 3, 1)
        x = self.patch_merge2(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv_before_res4(x)
        x = self.layer4(x)

        x = x.permute(0, 2, 3, 1)
        x = self.patch_merge3(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv_before_res5(x)
        x = self.layer5(x)

        return x


class BoundarySpatialModulator(nn.Module):
    """
    Weak soft-boundary-aware spatial modulation head.

    Input:
        soft boundary prior at current decoder scale, [B,1,H,W]

    Output:
        gamma, beta: both [B,1,H,W]

    Important:
        This module does NOT impose hard boundary control.

        In ControlledUnetModel.forward, gamma/beta should be applied only with
        small progressive scaling:

            h = h * (1 + alpha_t * boundary_mod_scale * gamma)
                + alpha_t * boundary_mod_scale * beta

        where:
            alpha_t is small at high-noise timesteps and larger at low-noise timesteps.
    """

    def __init__(self, in_channels=1, hidden_channels=32, dim=2):
        super().__init__()

        self.net = nn.Sequential(
            conv_nd(
                dim,
                in_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.SiLU(),
            conv_nd(
                dim,
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.SiLU(),
            conv_nd(
                dim,
                hidden_channels,
                2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, boundary_map):
        """
        Args:
            boundary_map: [B,1,H,W], soft prior in [0,1]

        Returns:
            gamma: [B,1,H,W], bounded to [-1,1]
            beta:  [B,1,H,W], bounded to [-1,1]
        """
        stats = self.net(boundary_map)

        gamma, beta = torch.chunk(stats, chunks=2, dim=1)

        # Keep modulation bounded. Final strength is further controlled by
        # alpha_t * boundary_mod_scale in ControlledUnetModel.
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)

        return gamma, beta
