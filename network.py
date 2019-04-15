import chainer
import chainer.links as L
import chainer.functions as F

# MLP for MNIST
class MLP(chainer.Chain):
    def __init__(self, n_units=500, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)
        
    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l1(h1))
        return self.l3(h2)

# CNN for CIFAR10
class NNet(chainer.Chain):
    def __init__(self, n_out=10):
        super(NNet, self).__init__()
        with self.init_scope():
            self.c1_0 = ConvBlock(3,128,3,1,1)
            self.c1_1 = ConvBlock(128,128,3,1,1)
            self.c1_2 = ConvBlock(128,128,3,1,1)
            
            self.c1_3 = ConvBlock(128,256,3,1,1)
            self.c1_4 = ConvBlock(256,256,3,1,1)
            self.c1_5 = ConvBlock(256,256,3,1,1)
            
            self.c1_6 = ConvBlock(256,512,3,1,0)
            self.c1_7 = ConvBlock(512,256,1,1,0)
            self.c1_8 = ConvBlock(256,128,1,1,0)
            
            self.l1 = L.Linear(128, n_out)
    
    def forward(self,x):
        h = self.c1_0(x)
        h = self.c1_1(h)
        h = self.c1_2(h)
        h = F.max_pooling_2d(h,2)
        
        h = self.c1_3(h)
        h = self.c1_4(h)
        h = self.c1_5(h)
        h = F.max_pooling_2d(h,2)
        
        h = self.c1_6(h)
        h = self.c1_7(h)
        h = self.c1_8(h)
        
        h = F.average_pooling_2d(h,6)
        h = self.l1(h)
        return h
        
## ConvBlock
class ConvBlock(chainer.Chain):
    def __init__(self,n_in,n_out,a,b,c):
        super(ConvBlock,self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(n_in,n_out,a,b,c)
            self.bn1 = L.BatchNormalization(n_out)
        
    def __call__(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.leaky_relu(h, slope=0.1)
        return h