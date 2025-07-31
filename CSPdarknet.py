import torch
import torch.nn as nn
from nets.Blocks import Focus, Conv, eca_block, se_block, cbam_block, MFEblock, C3, CA_Block, SPPF


class CSPDarknet(nn.Module):
    def __init__(self, base_channels=64, base_depth=1, phi='s', pretrained=False):
        super().__init__()
        
        self.stem = Focus(3, base_channels, k=3)

        self.dark2 = self._block(base_channels, base_channels * 2, base_depth, [2,4,8])
        self.dark3 = self._block(base_channels * 2, base_channels * 4, base_depth * 3, [2,4,8])
        self.dark4 = self._block(base_channels * 4, base_channels * 8, base_depth * 3, [1,2,3])
        self.dark5 = self._block(base_channels * 8, base_channels * 16, base_depth, [1,2,3], use_spp=True)

        # Train weights that are specific to your unique traits.
        if pretrained:
            print("✔ Advanced CSPDarknet can be trained from scratch to fit your dataset.")
            print("✔ If training is unstable, consider starting with a smaller configuration.")


    def _block(self, in_ch, out_ch, depth, scales, use_spp=False):
        layers = [
            Conv(in_ch, out_ch, k=3, s=2),
            eca_block(out_ch),
            se_block(out_ch),
            cbam_block(out_ch),
            MFEblock(out_ch, scales),
        ]
        if use_spp:
            layers.append(SPPF(out_ch, out_ch))
            layers.append(C3(out_ch, out_ch, depth, shortcut=False))
        else:
            layers.append(C3(out_ch, out_ch, depth))
        layers.append(CA_Block(out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x); feat1 = x
        x = self.dark4(x); feat2 = x
        x = self.dark5(x); feat3 = x
        return feat1, feat2, feat3

