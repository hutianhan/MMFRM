# MMFRM
### MMFRM
    A Multiscale and Multilevel Fusion Network Based on ResNet and MobileFaceNet for 
    Facial Expression  Recognition

#### Main software enviroment
    Torch 1.8.1
    Cuda11.1
    Python 3.7.10
    Numpy 1.20.3
    Matplotlib 3.4.2
    etc. 

#### Data Preparation
     Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset) dataset, 
              [FERPlus](dataset at https://www.worldlink.com.cn/osdir/ferplus.html)
              For [RAF-DB]and make sure it have a structure  like following:
                    ```
                    - data/raf-basic/
                         EmoLabel/
                             list_patition_label.txt
                         Image/aligned/
                             train_00001_aligned.jpg
                             test_0001_aligned.jpg
                             ...
                    ```
### Training
    Train on RAF-DB dataset，Run train_my.py directly, python train_my.py 

### Introduction to Key Code
    ### 1.The Triple CBAM Feature Fusion Module (TCFFM)
         Use CBAM to generate the attention weights of two different features separately, and then regenerate them into 
         the attention weights of two stacked features. TCFFM can effectively deal with the semantic difference between 
         facial features and landmark features, which can also represent the correlations between the channel and spatial 
         features.
                    class TCFFM(nn.Module):
                        def __init__(self, channel1,channel2):
                            super(TCFFM, self).__init__()
                    
                            self.conv1 = nn.Conv2d(channel1, channel1, kernel_size=3, stride=1, padding=1)
                            self.cbam1 = CBAM(channel1)
                            self.conv2 = nn.Conv2d(channel2, channel2, kernel_size=3, stride=1, padding=1)
                            self.cbam2 = CBAM(channel2)
                            self.conv3 = nn.Conv2d(channel1+channel2, channel1+channel2, kernel_size=3, stride=1, padding=1)
                            self.cbam3 = CBAM(channel1+channel2)
                    
                        def forward(self, x,y):
                            x=self.conv1(x)
                            x=self.cbam1(x)
                            y = self.conv2(y)
                            y = self.cbam2(y)
                            z=torch.cat([x, y], dim=1)
                            z=self.conv3(z)
                            z=self.cbam3(z)
                            return z

    ### 2.a Multiscale and Multilevel Fusion network based on ResNet and MobileFaceNet (MMFRM) for FER（MMFRM）
                    class MMFRM(nn.Module):
                        def __init__(self,  num_classes=7, drop_rate=0.0):
                            super(MMFRM, self).__init__()
                            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
                            self.bn1 = nn.BatchNorm2d(64)
                            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                            self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                                        RestNetBasicBlock(64, 64, 1))
                            self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                                        RestNetBasicBlock(128, 128, 1))
                            self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                                        RestNetBasicBlock(256, 256, 1))
                            self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                                        RestNetBasicBlock(512, 512, 1))
                            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                            self.drop_rate = drop_rate
                            self.fc = nn.Linear(512, num_classes)
                            self.alpha = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
                            self.cov1 =  Conv_Block(128 +64,128)
                            self.cov2 = Conv_Block(256 + 128, 256)
                            self.cov3 = Conv_Block(512 + 512, 512)
                            self.TCFMM1 = TCFFM(128, 64)
                            self.TCFMM2= TCFFM(256,128)
                            self.TCFMM3 = TCFFM(512, 512)
                    
                        def forward(self, x,feature1,feature2,feature3):
                            out = self.conv1(x)
                            out = self.layer1(out)
                            out = self.layer2(out)#[B, 128, 28, 28]
                            # out = torch.cat([out, feature1], dim=1)  #c:128 ,64
                            out=self.TCFMM1(out, feature1)
                            out= self.cov1(out)
                            out = self.layer3(out)#[B, 256, 14, 14]
                            # out = torch.cat([out, feature2], dim=1)  #c:256 ,128
                            out = self.TCFMM2(out, feature2)
                            out= self.cov2(out)
                            out = self.layer4(out)#[B, 512, 7, 7]
                            # out = torch.cat([out, feature3], dim=1) #c:512 ,512
                            out = self.TCFMM3(out, feature3)
                            out= self.cov3(out)
                            if self.drop_rate > 0:
                                out = nn.Dropout(self.drop_rate)(out)
                            out=self.avgpool(out)
                            out = out.view(out.size(0), -1)
                            attention_weights = self.alpha(out)
                            out = attention_weights * self.fc(out)
                            # out = self.fc(out)
                            return attention_weights, out

    ### 2.aThe Loss Function of Removing Facial Residual Features(RFLoss)
                    sm = torch.softmax(outputs, dim=1)
                    Pmax, predicted_labels = torch.max(sm, 1)
                    a0=a1=a2=a3=a4=a5=a6 = []
                    c0=c1=c2=c3=c4=c5=c6 = 0.0001
                    s0 =s1=s2=s3=s4=s5=s6= 0.0
                    sq0=sq1=sq2=sq3=sq4=sq5=sq6 = 0.0
                    for i  in range(batch_sz):
                        if targets[i]==0  :
                            s0=s0+sm[i][0]
                            c0=c0+1
                            a0.append(sm[i][0])
                        if targets[i]==1  :
                            s1=s1+sm[i][1]
                            c1=c1+1
                            a0.append(sm[i][1])
                        if targets[i] == 2:
                            s2 = s2 + sm[i][2]
                            c2 = c2 + 1
                            a0.append(sm[i][2])
                        if targets[i] == 3:
                            s3= s3 + sm[i][3]
                            c3 = c3 + 1
                            a0.append(sm[i][3])
                        if targets[i] == 4:
                            s4= s4 + sm[i][4]
                            c4 = c4 + 1
                            a0.append(sm[i][4])
                        if targets[i] == 5:
                            s5= s5 + sm[i][5]
                            c5 = c5 + 1
                            a0.append(sm[i][5])
                        if targets[i] == 6:
                            s6= s6 + sm[i][6]
                            c6 = c6 + 1
                            a0.append(sm[i][6])
        
                    aver0 = s0 / c0
                    aver1 = s1 / c1
                    aver2= s2 / c2
                    aver3 = s3 / c3
                    aver4= s4/ c4
                    aver5 = s5 / c5
                    aver6= s6/ c6
                    for j  in a0:
                        sq0=sq0+math.sqrt(pow(j-aver0,2))
                    for j  in a1:
                        sq1=sq1+math.sqrt(pow(j-aver1,2))
                    for j  in a2:
                        sq2=sq2+math.sqrt(pow(j-aver2,2))
                    for j in a3:
                        sq3 = sq3 + math.sqrt(pow(j - aver3, 2))
                    for j  in a4:
                        sq4=sq4+math.sqrt(pow(j-aver4,2))
                    for j  in a5:
                        sq5=sq5+math.sqrt(pow(j-aver5,2))
                    for j  in a6:
                        sq6=sq6+math.sqrt(pow(j-aver6,2))
                    RR_loss2=(sq0+sq1+sq2+sq3+sq4+sq5+sq6)/64
        
                    targets = targets.to(device)
                    if  t>50:
                        loss = 0.9*criterion(outputs, targets) + 0.2*RR_loss2
                    else:
                        loss = criterion(outputs, targets)

### Save weights in the models folder
