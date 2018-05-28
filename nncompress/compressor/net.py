# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import numpy as np
from chainer import Chain
from chainer import reporter


class EmbeddingCompressor(Chain):
    def __init__(self, n_codebooks, n_centroids, n_vocab, embed_dim, tau, embed_mat):
        super(EmbeddingCompressor, self).__init__()
        """
        M: number of codebooks (subcodes)
        K: number of vectors in each codebook
        """
        self.M = n_codebooks
        self.K = n_centroids
        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.tau = tau

        M = self.M
        K = self.K
        u_init = I.Uniform(scale=0.01)
        with self.init_scope():
            self.embed_mat = L.EmbedID(n_vocab, embed_dim, initialW=embed_mat)
            self.l1 = L.Linear(embed_dim, M * K // 2, initialW=u_init, initial_bias=u_init)
            self.l2 = L.Linear(M * K // 2, M * K, initialW=u_init, initial_bias=u_init)
            self.codebook = chainer.Parameter(initializer=u_init, shape=(M * K, embed_dim))

    def _encode(self, xs):
        exs = self.embed_mat(xs)
        h = F.tanh(self.l1(exs))
        logits = F.softplus(self.l2(h))
        logits = F.log(logits + 1e-10).reshape(-1, self.M, self.K)
        return logits, exs

    def _decode(self, gumbel_output):
        return F.matmul(gumbel_output, self.codebook)

    def __call__(self, xs):
        y_hat, input_embeds = self.predict(xs)
        loss = 0.5 * F.sum((y_hat - input_embeds) ** 2, axis=1)
        loss = F.mean(loss)
        reporter.report({'loss': loss.data}, self)
        return loss

    def predict(self, xs):
        # Encoding
        logits, exs = self._encode(xs)

        # Discretization
        D = F.gumbel_softmax(logits, self.tau, axis=2)
        gumbel_output = D.reshape(-1, self.M * self.K)
        with chainer.no_backprop_mode():
            maxp = F.mean(F.max(D, axis=2))
            reporter.report({'maxp': maxp.data}, self)

        # Decoding
        y_hat = self._decode(gumbel_output)
        return y_hat, exs

    def export(self, path):
        np.save(path, chainer.cuda.to_cpu(self.codebook.data))

    def retrieve_codes(self, xs, vocabs):
        logits, _ = self._encode(xs)
        indices_list = logits.data.argmax(axis=2).tolist()
        for vocab, indices in zip(vocabs, indices_list):
            assert len(indices) == self.M
            indices = ' '.join(str(x) for x in indices)
            out = '{}\t{}'.format(vocab, indices)
            yield out
