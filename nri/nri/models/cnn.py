import chainer
import chainer.links as L
import chainer.functions as F


class CNN(chainer.Chain):

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(CNN, self).__init__()
        w = chainer.initializers.LeCunNormal()

        self.conv1 = L.Convolution1D(n_in, n_hid, ksize=5, stride=1, pad=0, initialW=w)
        self.bn1 = L.BatchNormalization(n_hid)
        self.conv2 = L.Convolution1D(n_hid, n_hid, kernel_size=5, stride=1, padding=0, initialW=w)
        self.bn2 = L.BatchNormalization(n_hid)
        self.conv_predict = L.Convolution1D(n_hid, n_out, ksize=1, initialW=w)
        self.conv_attention = L.Convolution1D(n_hid, 1, ksize=1, initialW=w)
        self.dropout_prob = do_prob

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = self.bn1(F.relu(self.conv1(inputs)))
        x = F.dropout(x, self.dropout_prob)
        x = F.max_pooling_2d(x, 2, 2, 0)
        x = self.bn2(F.relu(self.conv2(x)))
        pred = self.conv_predict(x)

        attention = my_softmax(self.conv_attention(x), axis=2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob
