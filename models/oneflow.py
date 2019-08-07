class OneStream(nn.Module):
    def __init__(self):
        super(OneStream, self).__init__()
        self.conv1_1 = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.conv1_3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm2d(16)

        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.conv2_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(32)

        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.conv3_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(64)
        
        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.conv4_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_3 = nn.BatchNorm2d(128)
   
        self.conv5_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5_3 = nn.BatchNorm2d(256)

        self.conv6_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn6_1 = nn.BatchNorm2d(512)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6_2 = nn.BatchNorm2d(512)
        self.conv6_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6_3 = nn.BatchNorm2d(512)

        self.conv7_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn7_1 = nn.BatchNorm2d(1024)
        self.conv7_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.bn7_2 = nn.BatchNorm2d(1024)
        self.conv7_3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.bn7_3 = nn.BatchNorm2d(1024)
        
        self.conv8_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.bn8_1 = nn.BatchNorm2d(1024)

        self.conv9_1 = nn.ConvTranspose2d(1024, 512,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bn9_1 = nn.BatchNorm2d(512)
        self.conv9_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn9_2 = nn.BatchNorm2d(512)
     
        self.conv10_1 = nn.ConvTranspose2d(512, 256,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bn10_1 = nn.BatchNorm2d(256)
        self.conv10_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn10_2 = nn.BatchNorm2d(256)

        self.conv11_1 = nn.ConvTranspose2d(256, 128,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bn11_1 = nn.BatchNorm2d(128)
        self.conv11_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn11_2 = nn.BatchNorm2d(128)

        self.conv12_1 = nn.ConvTranspose2d(128, 64,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bn12_1 = nn.BatchNorm2d(64)
        self.conv12_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn12_2 = nn.BatchNorm2d(64)

        self.conv13_1 = nn.ConvTranspose2d(64, 32,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bn13_1 = nn.BatchNorm2d(32)
        self.conv13_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn13_2 = nn.BatchNorm2d(32)

        self.conv14_1 = nn.ConvTranspose2d(32, 16,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bn14_1 = nn.BatchNorm2d(16)
        self.conv14_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn14_2 = nn.BatchNorm2d(16)
        
        self.conv15_1 = nn.ConvTranspose2d(16, 3,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bn15_1 = nn.BatchNorm2d(3)
        self.conv15_2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.bn15_2 = nn.BatchNorm2d(3)
        
        
    
    def forward(self, x):
        out1 = F.leaky_relu(self.bn1_1(self.conv1_1(x)))
        out1 = F.leaky_relu(self.bn1_2(self.conv1_2(out1)))
        out1 = F.leaky_relu(self.bn1_3(self.conv1_3(out1)))

        out2 = F.leaky_relu(self.bn2_1(self.conv2_1(out1)))
        out2 = F.leaky_relu(self.bn2_2(self.conv2_2(out2)))
        out2 = F.leaky_relu(self.bn2_3(self.conv2_3(out2)))

        out3 = F.relu(self.bn3_1(self.conv3_1(out2)))
        out3 = F.relu(self.bn3_2(self.conv3_2(out3)))
        out3 = F.relu(self.bn3_3(self.conv3_3(out3)))

        out4 = F.relu(self.bn4_1(self.conv4_1(out3)))
        out4 = F.relu(self.bn4_2(self.conv4_2(out4)))
        out4 = F.relu(self.bn4_3(self.conv4_3(out4)))

        out5 = F.relu(self.bn5_1(self.conv5_1(out4)))
        out5 = F.relu(self.bn5_2(self.conv5_2(out5)))
        out5 = F.relu(self.bn5_3(self.conv5_3(out5)))

        out6 = F.relu(self.bn6_1(self.conv6_1(out5)))
        out6 = F.relu(self.bn6_2(self.conv6_2(out6)))
        out6 = F.relu(self.bn6_3(self.conv6_3(out6)))
        
        out7 = F.relu(self.bn7_1(self.conv7_1(out6)))
        out7 = F.relu(self.bn7_2(self.conv7_2(out7)))
        out7 = F.relu(self.bn7_3(self.conv7_3(out7)))

        out8 = out7
        out8 = F.relu(self.bn8_1(self.conv8_1(out8)))
        
        #???

        out9 = F.relu(self.bn9_1(self.conv9_1(out8)))
        out9 = F.relu(self.bn9_2(self.conv9_2(out9)))

        out10 = F.relu(self.bn10_1(self.conv10_1(out9)))
        out10 = F.relu(self.bn10_2(self.conv10_2(out10)))

        out11 = F.relu(self.bn11_1(self.conv11_1(out10)))
        out11 = F.relu(self.bn11_2(self.conv11_2(out11)))

        out12 = F.relu(self.bn12_1(self.conv12_1(out11)))
        out12 = F.relu(self.bn12_2(self.conv12_2(out12)))

        out13 = F.relu(self.bn13_1(self.conv13_1(out12)))
        out13 = F.relu(self.bn13_2(self.conv13_2(out13)))

        out14 = F.relu(self.bn14_1(self.conv14_1(out13)))
        out14 = F.relu(self.bn14_2(self.conv14_2(out14)))

        out15 = F.relu(self.bn15_1(self.conv15_1(out14)))
        out15 = F.relu(self.bn15_2(self.conv15_2(out15)))


        return output