import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# =========================================================
# Basic UNet
# =========================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_ch=64):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_ch)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch, base_ch * 2)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 2, base_ch * 4)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 4, base_ch * 8)
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 8, base_ch * 16)
        )

        self.up1 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_ch * 16, base_ch * 8)

        self.up2 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up4 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_ch * 2, base_ch)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def _cat(self, skip, x):
        if skip.shape[2:] != x.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, x], dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self._cat(x4, x)
        x = self.conv1(x)

        x = self.up2(x)
        x = self._cat(x3, x)
        x = self.conv2(x)

        x = self.up3(x)
        x = self._cat(x2, x)
        x = self.conv3(x)

        x = self.up4(x)
        x = self._cat(x1, x)
        x = self.conv4(x)

        return self.outc(x)


# =========================================================
# nnU-Net style 2D architecture
# =========================================================

class ConvINLeakyReLU(nn.Module):
    """
    nnU-Net 风格基础卷积块:
        Conv2d -> InstanceNorm2d -> LeakyReLU

    与普通 UNet 的主要区别:
        1. 使用 InstanceNorm
        2. 使用 LeakyReLU
        3. 使用 strided conv 下采样
        4. 通道数按 nnU-Net 风格逐层增加
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        negative_slope=1e-2,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StackedConvLayers(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_convs=2,
        first_stride=1,
    ):
        super().__init__()

        layers = []

        layers.append(
            ConvINLeakyReLU(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=first_stride,
            )
        )

        for _ in range(num_convs - 1):
            layers.append(
                ConvINLeakyReLU(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class NNUNet2D(nn.Module):
    """
    适配当前 dataloader 的 2D nnU-Net-like baseline。

    输入:
        x: [B, 3, H, W]

    输出:
        logits: [B, num_classes, H, W]

    默认输入 352x352 时:
        352 -> 176 -> 88 -> 44 -> 22
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        base_features=32,
        max_features=320,
        num_convs_per_stage=2,
    ):
        super().__init__()

        features = [
            base_features,
            base_features * 2,
            base_features * 4,
            base_features * 8,
            min(base_features * 10, max_features),
        ]

        # 例如 base_features=32:
        # [32, 64, 128, 256, 320]
        self.features = features

        # Encoder
        self.encoders = nn.ModuleList()

        self.encoders.append(
            StackedConvLayers(
                in_channels=in_channels,
                out_channels=features[0],
                num_convs=num_convs_per_stage,
                first_stride=1,
            )
        )

        for i in range(1, len(features)):
            self.encoders.append(
                StackedConvLayers(
                    in_channels=features[i - 1],
                    out_channels=features[i],
                    num_convs=num_convs_per_stage,
                    first_stride=2,
                )
            )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(len(features) - 1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose2d(
                    features[i],
                    features[i - 1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                )
            )

            self.decoders.append(
                StackedConvLayers(
                    in_channels=features[i - 1] * 2,
                    out_channels=features[i - 1],
                    num_convs=num_convs_per_stage,
                    first_stride=1,
                )
            )

        self.seg_head = nn.Conv2d(features[0], num_classes, kernel_size=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    a=1e-2,
                    mode="fan_out",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    a=1e-2,
                    mode="fan_out",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _cat(self, skip, x):
        if skip.shape[2:] != x.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat([skip, x], dim=1)

    def forward(self, x):
        skips = []

        out = x

        for encoder in self.encoders:
            out = encoder(out)
            skips.append(out)

        # 最后一层作为 bottleneck，不作为 skip 使用
        out = skips[-1]

        skips = skips[:-1][::-1]

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, skips):
            out = upconv(out)
            out = self._cat(skip, out)
            out = decoder(out)

        logits = self.seg_head(out)

        return logits


# =========================================================
# DeepLabV3
# =========================================================

def build_deeplabv3_resnet50(num_classes=1, pretrained_backbone=False):
    """
    torchvision DeepLabV3-ResNet50。

    输出格式:
        dict["out"]

    train/test 里面用 extract_logits 统一取出 logits。
    """

    try:
        from torchvision.models.segmentation import deeplabv3_resnet50
        from torchvision.models import ResNet50_Weights

        weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None

        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=weights_backbone,
            aux_loss=False,
        )

    except Exception:
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=False,
            pretrained_backbone=pretrained_backbone,
            aux_loss=False,
        )

    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model


# =========================================================
# Model factory
# =========================================================

def get_model(
    model_name: str,
    num_classes: int = 1,
    pretrained_backbone: bool = False,
):
    model_name = model_name.lower()

    if model_name in ["unet", "u-net"]:
        return UNet(
            in_channels=3,
            num_classes=num_classes,
            base_ch=64,
        )

    if model_name in ["nnunet", "nnunet2d", "nnunet_2d", "nn-unet"]:
        return NNUNet2D(
            in_channels=3,
            num_classes=num_classes,
            base_features=32,
            max_features=320,
            num_convs_per_stage=2,
        )

    if model_name in ["deeplabv3", "deeplabv3_resnet50", "deeplab"]:
        return build_deeplabv3_resnet50(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
        )

    raise ValueError(
        f"Unsupported model_name={model_name}. "
        f"Use one of: unet, nnunet, deeplabv3_resnet50"
    )


def extract_logits(output):
    """
    统一不同模型的输出格式。

    UNet / nnUNet2D:
        Tensor [B,1,H,W]

    torchvision DeepLabV3:
        dict["out"] [B,1,H,W]

    其他模型:
        tuple/list 默认取第一个输出
    """

    if isinstance(output, dict):
        return output["out"]

    if isinstance(output, (tuple, list)):
        return output[0]

    return output
