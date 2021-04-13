import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SymmetricalLayer(nn.Module):
    def __init__(self, input_features, num_classes ,scaler=16):
        super(SymmetricalLayer, self).__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        
        self.weight = nn.Parameter(torch.FloatTensor(2, input_features))
        nn.init.uniform_(self.weight)
        thetas = torch.arange(num_classes, dtype=torch.float32)
        self.thetas = (2*math.pi*thetas / thetas.shape[0]).to(device)
        self.I = torch.eye(input_features).to(device)
        self.input_features = input_features
        self.num_classes = num_classes
        self.s = scaler
    
    def rotateNd(self, v1, v2):

        n1 = v1 / torch.norm(v1)
        v2 = v2 - torch.dot(n1,v2) * n1
        n2 = v2 / torch.norm(v2)
        
        ger_sub = torch.ger(n2,n1) - torch.ger(n1,n2)
        ger_add = torch.ger(n1,n1) + torch.ger(n2,n2)
        sin_th = torch.unsqueeze(torch.unsqueeze(torch.sin(self.thetas),dim=-1),dim=-1)
        cos_th = torch.unsqueeze(torch.unsqueeze(torch.cos(self.thetas)-1,dim=-1),dim=-1)
        R = self.I + ger_sub*sin_th + ger_add*cos_th
        
        return torch.einsum('bij,j->bi',R,n1)        
        
    def forward(self, input_x):
        x = F.normalize(input_x)
        W = F.normalize(self.weight)
        Ws = self.rotateNd(W[0],W[1])
        cosine = F.linear(x, Ws)
        output = self.s*cosine
        
        return output

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.linear = SymmetricalLayer(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, labels):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])