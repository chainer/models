import chainer
from chainer import links as L
from chainer import functions as F


class RelationNetwork(chainer.Chain):

    '''Includes a CNN feature extractor, g, f and the relation network.'''

    def __init__(self, n_out):
        super(RelationNetwork, self).__init__()

        with self.init_scope():
            self.feat_extractor = ImageFeatureExtractor()
            self.g = GTheta()
            self.f = FPhi()
            self.fc = L.Linear(n_out)

    def __call__(self, imgs, questions):
        feat = self.feat_extractor(imgs)

        # Append relative coordinates to each location in the feature maps.
        n, c, h, w = feat.shape
        spatial_area = h * w

        xp = self.xp
        coords_h = xp.linspace(-1, 1, h, dtype=feat.dtype)
        coords_w = xp.linspace(-1, 1, w, dtype=feat.dtype)
        coords_hh, coords_ww = xp.meshgrid(coords_h, coords_w)
        coords_hh = coords_hh[None]
        coords_ww = coords_ww[None]
        coords = xp.concatenate((coords_hh, coords_ww), axis=0)
        coords = coords.reshape(2, -1)
        coords = coords[None]  # (1, 2, spatial_area * spatial_area)
        coords = xp.repeat(coords, n, axis=0)

        # Coordinates may be cached here but the performance gain is not
        # significant so it is skipped in favor of readability.

        feat = feat.reshape(n, c, spatial_area)
        h = F.concat((feat, coords), axis=1)  # (n, c + 2, spatial_area)

        # Create coordinate pairs (differentiable meshgrid).
        h_hh = F.expand_dims(h, 2)
        h_ww = F.expand_dims(h, 3)
        h_hh = F.repeat(h_hh, spatial_area, axis=2)
        h_ww = F.repeat(h_ww, spatial_area, axis=3)
        h = F.concat((h_hh, h_ww), axis=1)

        # Append questions to each coordinate pair.
        questions = questions.astype(imgs.dtype)
        questions = questions[:, :, None, None]
        questions = F.tile(questions, (1, 1, spatial_area, spatial_area))
        h = F.concat((h, questions), axis=1)
        # (n, (c + 2) * 2 + questions_length, spatial_area, spatial_area)

        # g.
        h = F.transpose(h, (0, 2, 3, 1))
        h = F.reshape(h, (n * spatial_area * spatial_area, -1))
        h = self.g(h)
        h = F.reshape(h, (n, spatial_area * spatial_area, -1))
        h = F.sum(h, axis=1)

        h = self.f(h)

        # Logits.
        h = self.fc(h)

        return h


class ImageFeatureExtractor(chainer.Chain):

    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()

        with self.init_scope():
            self.conv0 = L.Convolution2D(3, 32, ksize=3, stride=2, pad=1)
            self.conv1 = L.Convolution2D(32, 64, ksize=3, stride=2, pad=1)
            self.conv2 = L.Convolution2D(64, 128, ksize=3, stride=2, pad=1)
            self.conv3 = L.Convolution2D(128, 256, ksize=3, stride=2, pad=1)
            self.bn0 = L.BatchNormalization(32)
            self.bn1 = L.BatchNormalization(64)
            self.bn2 = L.BatchNormalization(128)

    def __call__(self, imgs):
        h = F.relu(self.bn0(self.conv0(imgs)))
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.conv3(h)
        return h


class GTheta(chainer.Chain):

    def __init__(self):
        super(GTheta, self).__init__()

        with self.init_scope():
            self.fc0 = L.Linear(2000)
            self.fc1 = L.Linear(2000)
            self.fc2 = L.Linear(2000)
            self.fc3 = L.Linear(2000)

    def __call__(self, h):
        h = F.relu(self.fc0(h))
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return h


class FPhi(chainer.Chain):

    def __init__(self):
        super(FPhi, self).__init__()

        with self.init_scope():
            self.fc0 = L.Linear(2000)
            self.fc1 = L.Linear(1000)
            self.fc2 = L.Linear(500)
            self.fc3 = L.Linear(100)

    def __call__(self, h):
        h = F.relu(self.fc0(h))
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return h
