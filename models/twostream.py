"""
To be added
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,output_channels,
                               kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        input_channels = output_channels
        self.conv2 = nn.Conv2d(input_channels,output_channels,
                               kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv3 = nn.Conv2d(input_channels,output_channels,
                               kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)))
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)))
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)))
        return x3 


class DecoderBlock(nn.Module):
    def __init__(self, input_channels,output_channels):
        super(DecoderBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(input_channels,output_channels,
                                          kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        input_channels = output_channels
        self.deconv2 = nn.Conv2d(input_channels,output_channels,
                                          kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.deconv3 = nn.Conv2d(input_channels,output_channels,
                                          kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channels)
        #pass

    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.deconv1(x)))
        x2 = F.leaky_relu(self.bn2(self.deconv2(x1)))
        x3 = F.leaky_relu(self.bn3(self.deconv3(x2)))
        return x3 
        #pass    



class TwoStream(nn.Module):
    def __init__(self):
        super(TwoStream, self).__init__()
        self.block1A = EncoderBlock(3,16)
        self.block2A = EncoderBlock(16,32)
        self.block3A = EncoderBlock(32,64)
        self.block4A = EncoderBlock(64,128)
        self.block5A = EncoderBlock(128,256)
        self.block6A = EncoderBlock(256,512)
        self.block7A = EncoderBlock(512,1024)

        self.block1B = EncoderBlock(1,16)
        self.block2B = EncoderBlock(16,32)
        self.block3B = EncoderBlock(32,64)
        self.block4B = EncoderBlock(64,128)
        self.block5B = EncoderBlock(128,256)
        self.block6B = EncoderBlock(256,512)
        self.block7B = EncoderBlock(512,1024)
        
        self.block8_2 = EncoderBlock(1024,1024)
        self.block8_3 = EncoderBlock(1024,1024)

        self.block9 = DecoderBlock(1024,512)
        self.block10 = DecoderBlock(512,256)
        self.block11 = DecoderBlock(256,128)
        self.block12 = DecoderBlock(128,64)
        self.block13 = DecoderBlock(64,32)
        self.block14 = DecoderBlock(32,16)
        self.block15 = DecoderBlock(16,3)
        #pass

    def forward(self, x1, y1):
        outx1 = self.block1A(x1) 
        outx2 = self.block2A(outx1) 
        outx3 = self.block3A(outx2)
        outx4 = self.block4A(outx3)
        outx5 = self.block5A(outx4)
        outx6 = self.block6A(outx5)
        outx7 = self.block7A(outx6)

        outy1 = self.block1B(y1)  
        outy2 = self.block2B(outy1) 
        outy3 = self.block3B(outy2)
        outy4 = self.block4B(outy3)
        outy5 = self.block5B(outy4)
        outy6 = self.block6B(outy5)
        outy7 = self.block7B(outy6)

        out8_1 = outx7+outy7
        out8_2 = self.block8_2(out8_1)
        out8_3 = self.block8_3(out8_2)+outx7+outy7
        
        out9 = self.block9(out8_3)+outx6+outy6
        out10 = self.block10(out9)+outx5+outy5
        out11 = self.block11(out10)+outx4+outy4
        out12 = self.block12(out11)+outx3+outy3
        out13 = self.block13(out12)+outx2+outy2
        out14 = self.block14(out13)+outx1+outy1
        out15 = self.block15(out14)



        return out15