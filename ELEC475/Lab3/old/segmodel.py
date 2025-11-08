import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# ---------- ASPP MODULE ----------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=(1, 6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for r in dilation_rates
        ])

        # Global context branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Combine all parallel outputs
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilation_rates) + 1),
                      out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        features = [conv(x) for conv in self.convs]
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(
            global_feat, size=size, mode='bilinear', align_corners=False)
        x = torch.cat(features + [global_feat], dim=1)
        return self.project(x)


# ---------- MAIN MODEL ----------
class MobileNetV3_SegNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()

        # Load backbone
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT).features
        self.backbone = backbone

        # Define which layers correspond to which taps
        # indices determined from torchvision’s MobileNetV3-Small feature map sizes
        self.low_idx = 2   # stride ≈ 4, channels=24
        self.mid_idx = 3   # stride ≈ 8, channels=40
        self.high_idx = 12  # stride ≈16, channels=576 (final layer)

        # ASPP context module
        self.aspp = ASPP(in_channels=576, out_channels=256)

        # Low-level feature refinement before merging
        self.low_proj = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Final decoder after merging low + ASPP output
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        low, mid, high = None, None, None

        # Forward through encoder and collect feature taps
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == self.low_idx:
                low = x        # stride ≈ 4
            elif i == self.mid_idx:
                mid = x        # stride ≈ 8 (for KD)
            elif i == self.high_idx:
                high = x       # stride ≈16
                break

        # ASPP on high-level feature
        aspp_out = self.aspp(high)

        # Upsample ASPP output to low-level resolution
        aspp_up = F.interpolate(
            aspp_out, size=low.shape[2:], mode='bilinear', align_corners=False)

        # Fuse with low-level projection
        low_proj = self.low_proj(low)
        fused = torch.cat([aspp_up, low_proj], dim=1)

        # Decode and upsample to input resolution
        out = self.decoder(fused)
        out = F.interpolate(out, size=input_size,
                            mode='bilinear', align_corners=False)

        # Return predictions + taps (for KD)
        return {
            "out": out,
            "low": low,
            "mid": mid,
            "high": high
        }


# ---------- EXAMPLE ----------
if __name__ == "__main__":
    model = MobileNetV3_SegNet(num_classes=21)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out["out"].shape)  # → torch.Size([1, 21, 224, 224])
    print({k: v.shape for k, v in out.items() if k != "out"})
