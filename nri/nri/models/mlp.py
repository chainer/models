import chainer
import chainer.links as L
import chainer.functions as F


class MLP(chainer.Chain):

    """ MLP

    Two-layer fully-connected ELU net with batch norm.

    """

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()

        # Xavier normal = Glorot normal
        w = chainer.initializers.GlorotNormal()

        with self.init_scope():
            self.fc1 = L.Linear(n_in, n_hid, initialW=w, initial_bias=0.1)
            self.fc2 = L.Linear(n_hid, n_out, initialW=w, initial_bias=0.1)
            self.bn = L.BatchNormalization(n_out)
        self.dropout_prob = do_prob

    def forward(self, inputs):
        # Input shape: [batch_size, num_nodes, feature_dims]
        batch_size, num_nodes = inputs.shape[:2]
        inputs = inputs.reshape(batch_size * num_nodes, -1)
        # New shape: [batch_size * num_nodes, feature_dims]

        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob)
        x = F.elu(self.fc2(x))
        x = self.bn(x)

        return x.reshape(batch_size, num_nodes, -1)
