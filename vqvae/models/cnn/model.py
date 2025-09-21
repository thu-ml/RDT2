import torch
import torch.nn as nn

from vqvae.models.cnn.blocks import *


class Encoder(nn.Module):
    """
    @func: 
    from [B,in_channels,act_horizon] 
    to [B,z_channels,act_horizon/4]
    @feature:
    conv | attn | linear

    """
    def __init__(
        self, 
        ch=64,
        ch_mult=(2, 4, 8),
        in_channels=1,
        z_channels=8, 
        act_horizon=24, 
        dropout=0.0,
    ):
        super().__init__()

        # init
        self.ch = ch
        self.ch_mult = ch_mult
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.act_horizon = act_horizon
        # self.padded_horizon = 2 ** (act_horizon - 1).bit_length() 

        # conv in
        self.conv_in = nn.Conv1d(self.in_channels, self.ch, kernel_size=3, stride=1, padding=1) # B,ch,16,1

        # down - block 1
        self.down_1 = nn.Module()
        self.down_1.block_1 = ConvBlock(in_channels=self.ch, num_groups=1, out_channels=self.ch*ch_mult[0], dropout=dropout) # B,2*ch,16,1
        self.down_1.downsample = Downsample2x(in_channels=self.ch*ch_mult[0]) # B,2*ch,8,1
        self.down_1.block_2 = ConvBlock(in_channels=self.ch*ch_mult[0], num_groups=1, out_channels=self.ch*ch_mult[0], dropout=dropout) # B,2*ch,8,1
        
        # down - block 2
        self.down_2 = nn.Module()
        self.down_2.block_1 = ConvBlock(in_channels=self.ch*ch_mult[0], num_groups=1, out_channels=self.ch*ch_mult[1], dropout=dropout) # B,4*ch,4,1
        self.down_2.downsample = Downsample2x(in_channels=self.ch*ch_mult[1]) # B,4*ch,4,1
        self.down_2.block_2 = ConvBlock(in_channels=self.ch*ch_mult[1], num_groups=1, out_channels=self.ch*ch_mult[1], dropout=dropout) # B,4*ch,4,1
        
        # down - block 3
        self.down_3 = nn.Module()
        self.down_3.block_1 = ConvBlock(in_channels=self.ch*ch_mult[1], num_groups=1, out_channels=self.ch*ch_mult[2], dropout=dropout) # B,4*ch,4,1
        self.down_3.downsample = Downsample2x(in_channels=self.ch*ch_mult[2]) # B,4*ch,4,1
        self.down_3.block_2 = ConvBlock(in_channels=self.ch*ch_mult[2], num_groups=1, out_channels=self.ch*ch_mult[2], dropout=dropout) # B,4*ch,4,1
        
        # # down - block 4
        # self.down_4 = nn.Module()
        # self.down_4.block_1 = ConvBlock(in_channels=self.ch*ch_mult[2], num_groups=1, out_channels=self.ch*ch_mult[3], dropout=dropout) # B,4*ch,4,1
        # self.down_4.downsample = Downsample2x(in_channels=self.ch*ch_mult[3]) # B,4*ch,4,1
        # self.down_4.block_2 = ConvBlock(in_channels=self.ch*ch_mult[3], num_groups=1, out_channels=self.ch*ch_mult[3], dropout=dropout) # B,4*ch,4,1
        
        # # down - block 5
        # self.down_5 = nn.Module()
        # self.down_5.block_1 = ConvBlock(in_channels=self.ch*ch_mult[3], num_groups=1, out_channels=self.ch*ch_mult[4], dropout=dropout) # B,4*ch,4,1
        # self.down_5.downsample = Downsample2x(in_channels=self.ch*ch_mult[4]) # B,4*ch,4,1
        # self.down_5.block_2 = ConvBlock(in_channels=self.ch*ch_mult[4], num_groups=1, out_channels=self.ch*ch_mult[4], dropout=dropout) # B,4*ch,4,1

        # Add a global block in the last block? (attention, non-local, etc; see Taming Transformers) 
        
        # conv out
        self.norm_out = Normalize(num_channels=self.ch*ch_mult[2], num_groups=1) # B,4*ch,4,1
        self.conv_out = torch.nn.Conv1d(self.ch*ch_mult[2], self.z_channels, kernel_size=3, stride=1, padding=1) # B,z_channels,4,1
    
    def forward(self, x):
        """
        @input:
        x has shape [B,in_channels,act_horizon]

        """
        # # Use interpolation to pad the input to the nearest power of 2
        # if x.shape[-1] != self.padded_horizon:
        #     x = F.interpolate(
        #         x, 
        #         size=(self.padded_horizon,), 
        #         mode='linear', 
        #         align_corners=False
        #     )

        # begin
        h = self.conv_in(x) # [B,ch,act_horizon]

        # down-1
        h = self.down_1.block_2(self.down_1.downsample(self.down_1.block_1(h))) # [B,2ch,act_horizon/2]
        
        # down-2
        h = self.down_2.block_2(self.down_2.downsample(self.down_2.block_1(h))) # [B,4ch,act_horizon/4]
        
        # down-3
        h = self.down_3.block_2(self.down_3.downsample(self.down_3.block_1(h))) # [B,4ch,act_horizon/4]
        
        # # down-4
        # h = self.down_4.block_2(self.down_4.downsample(self.down_4.block_1(h))) # [B,4ch,act_horizon/4]
        
        # # down-5
        # h = self.down_5.block_2(self.down_5.downsample(self.down_5.block_1(h))) # [B,4ch,act_horizon/4]

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True)) # [B,z_channels,act_horizon/4]
        
        # output
        return h # h has shape [B,z_channels,act_horizon/4]


class Decoder(nn.Module):
    """
    @func: 
    from [B,z_channels,act_horizon/4]
    to [B,in_channels,act_horizon] 
    @feature: 
    conv | attn | linear
    
    """
    def __init__(
        self, 
        ch=64,
        ch_mult=(2, 4, 8),
        in_channels=1,
        z_channels=8, 
        act_horizon=24, 
        dropout=0.0,
    ):
        super().__init__()
        # init
        self.ch = ch
        self.ch_mult = ch_mult
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.act_horizon = act_horizon

        # z to block_in
        self.norm_in = Normalize(num_channels=self.z_channels, num_groups=1) # B,z_channels,4
        self.conv_in = torch.nn.Conv1d(self.z_channels, self.ch*self.ch_mult[-1], kernel_size=3, stride=1, padding=1) # B,4ch,4

        # up - block 1
        self.up_1 = nn.Module()
        self.up_1.block_1 = ConvBlock(in_channels=self.ch*self.ch_mult[-1], num_groups=1, out_channels=self.ch*self.ch_mult[-1], dropout=dropout) # B,4ch,4
        self.up_1.upsample = Upsample2x_TF(in_channels=self.ch*self.ch_mult[-1]) # B,4ch,8
        self.up_1.block_2 = ConvBlock(in_channels=self.ch*self.ch_mult[-1], num_groups=1, out_channels=self.ch*self.ch_mult[-2], dropout=dropout) # B,2ch,8

        # up - block 2
        self.up_2 = nn.Module()
        self.up_2.block_1 = ConvBlock(in_channels=self.ch*self.ch_mult[-2], num_groups=1, out_channels=self.ch*self.ch_mult[-2], dropout=dropout) # B,2ch,8
        self.up_2.upsample = Upsample2x_TF(in_channels=self.ch*self.ch_mult[-2]) # B,2ch,16
        self.up_2.block_2 = ConvBlock(in_channels=self.ch*self.ch_mult[-2], num_groups=1, out_channels=self.ch*self.ch_mult[-3], dropout=dropout) # B,ch,16
        
        # up - block 3
        self.up_3 = nn.Module()
        self.up_3.block_1 = ConvBlock(in_channels=self.ch*self.ch_mult[-3], num_groups=1, out_channels=self.ch*self.ch_mult[-3], dropout=dropout) # B,2ch,8
        self.up_3.upsample = Upsample2x_TF(in_channels=self.ch*self.ch_mult[-3]) # B,2ch,16
        self.up_3.block_2 = ConvBlock(in_channels=self.ch*self.ch_mult[-3], num_groups=1, out_channels=self.ch, dropout=dropout) # B,ch,16
        
        # # up - block 4
        # self.up_4 = nn.Module()
        # self.up_4.block_1 = ConvBlock(in_channels=self.ch*self.ch_mult[-4], num_groups=1, out_channels=self.ch*self.ch_mult[-4], dropout=dropout) # B,2ch,8
        # self.up_4.upsample = Upsample2x_TF(in_channels=self.ch*self.ch_mult[-4]) # B,2ch,16
        # self.up_4.block_2 = ConvBlock(in_channels=self.ch*self.ch_mult[-4], num_groups=1, out_channels=self.ch, dropout=dropout) # B,ch,16
        
        # # up - block 5
        # self.up_5 = nn.Module()
        # self.up_5.block_1 = ConvBlock(in_channels=self.ch*self.ch_mult[-5], num_groups=1, out_channels=self.ch*self.ch_mult[-5], dropout=dropout) # B,2ch,8
        # self.up_5.upsample = Upsample2x_TF(in_channels=self.ch*self.ch_mult[-5]) # B,2ch,16
        # self.up_5.block_2 = ConvBlock(in_channels=self.ch*self.ch_mult[-5], num_groups=1, out_channels=self.ch, dropout=dropout) # B,ch,16

        # end
        self.conv_out = torch.nn.Conv1d(self.ch, self.in_channels, kernel_size=3, stride=1, padding=1) # B,1,16
    
    def forward(self, z):
        """
        @input:
        h has shape [batch_size, z_channels, act_horizon/4]
        """

        # begin
        h = self.conv_in(F.silu(self.norm_in(z), inplace=True)) # [B,4ch,4]

        # up-1
        h = self.up_1.block_2(self.up_1.upsample(self.up_1.block_1(h))) # [B,2ch,8]

        # up-2
        h = self.up_2.block_2(self.up_2.upsample(self.up_2.block_1(h))) # [B,ch,16]
        
        # up-3
        h = self.up_3.block_2(self.up_3.upsample(self.up_3.block_1(h))) # [B,ch,16]
        
        # # up-4
        # h = self.up_4.block_2(self.up_4.upsample(self.up_4.block_1(h))) # [B,ch,16]
        
        # # up-5
        # h = self.up_5.block_2(self.up_5.upsample(self.up_5.block_1(h))) # [B,ch,16]

        # end
        h = self.conv_out(h) # [B,in_channels,16]
        
        # # Use interpolation to map the input back to the original shape
        # if h.shape[-1] != self.act_horizon:
        #     h = F.interpolate(
        #         h, 
        #         size=(self.act_horizon,), 
        #         mode='linear', 
        #         align_corners=False
        #     )

        return h # h has shape [B,in_channels,16]


if __name__ == "__main__":
    encoder = Encoder(
        ch=64,
        in_channels=20,
        z_channels=64,
        act_horizon=24
    )
    
    decoder = Decoder(
        ch=64,
        in_channels=20,
        z_channels=64,
        act_horizon=24
    )
    
    x = torch.randn(4, 20, 24)
    print("x.shape", x.shape)
    z = encoder(x)
    print("z.shape", z.shape)
    x_recon = decoder(z)
    print("x_recon.shape", x_recon.shape)
