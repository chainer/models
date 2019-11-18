import chainer
import chainer.functions as F
import chainer.links as L


class Net(chainer.Chain):
    
    def __init__(self):

        super(Net, self).__init__()

        with self.init_scope():

            self.down_conv1 = chainer.Sequential(
                L.Convolution2D(in_channels=1,
                                out_channels=48,
                                ksize=5,
                                stride=2,
                                pad=2),
                L.BatchNormalization(48),
                F.relu,

                L.Convolution2D(in_channels=48,
                                out_channels=128,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(128),
                F.relu,
                
                L.Convolution2D(in_channels=128,
                                out_channels=128,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(128),
                F.relu,
            )


            self.down_conv2 = chainer.Sequential(
                L.Convolution2D(in_channels=128,
                                out_channels=256,
                                ksize=3,
                                stride=2,
                                pad=1),
                L.BatchNormalization(256),
                F.relu,

                L.Convolution2D(in_channels=256,
                                out_channels=256,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(256),
                F.relu,

                L.Convolution2D(in_channels=256,
                                out_channels=256,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(256),
                F.relu
            )


            self.flat_conv = chainer.Sequential(    
                L.Convolution2D(in_channels=256,
                                out_channels=256,
                                ksize=3,
                                stride=2,
                                pad=1),
                L.BatchNormalization(256),
                F.relu,

                L.Convolution2D(in_channels=256,
                                out_channels=512,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(512),
                F.relu,
                
                L.Convolution2D(in_channels=512,
                                out_channels=1024,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(1024),
                F.relu,
                
                L.Convolution2D(in_channels=1024,
                                out_channels=1024,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(1024),
                F.relu,
                
                L.Convolution2D(in_channels=1024,
                                out_channels=1024,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(1024),
                F.relu,
                
                L.Convolution2D(in_channels=1024,
                                out_channels=1024,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(1024),
                F.relu,
                
                L.Convolution2D(in_channels=1024,
                                out_channels=512,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(512),
                F.relu,
                
                L.Convolution2D(in_channels=512,
                                out_channels=256,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(256),
                F.relu
            ) 


            self.up_conv1 = chainer.Sequential(
                L.Deconvolution2D(in_channels=256,
                                out_channels=256,
                                ksize=4,
                                stride=2,
                                pad=1),
                L.BatchNormalization(256),
                F.relu,
                
                L.Convolution2D(in_channels=256,
                                out_channels=256,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(256),
                F.relu,
            
                L.Convolution2D(in_channels=256,
                                out_channels=128,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(128),
                F.relu,
            )


            self.up_conv2 = chainer.Sequential(
                L.Deconvolution2D(in_channels=128,
                                out_channels=128,
                                ksize=4,
                                stride=2,
                                pad=1),
                L.BatchNormalization(128),
                F.relu,
                
                L.Convolution2D(in_channels=128,
                                out_channels=128,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(128),
                F.relu,
                
                L.Convolution2D(in_channels=128,
                                out_channels=48,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(48),
                F.relu,
            )


            self.up_conv3 = chainer.Sequential(
                L.Deconvolution2D(in_channels=48,
                                out_channels=48,
                                ksize=4,
                                stride=2,
                                pad=1),
                L.BatchNormalization(48),
                F.relu,
                
                L.Convolution2D(in_channels=48,
                                out_channels=24,
                                ksize=3,
                                stride=1,
                                pad=1),
                L.BatchNormalization(24),
                F.relu,
                
                L.Convolution2D(in_channels=24,
                                out_channels=1,
                                ksize=3,
                                stride=1,
                                pad=1),
                F.sigmoid
            )


    def forward(self, x):
        x = self.down_conv1(x)
        x = self.down_conv2(x)
        x = self.flat_conv(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        out = self.up_conv3(x)
        return out
