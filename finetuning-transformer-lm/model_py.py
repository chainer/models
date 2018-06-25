import sys
import re
import math
import json
import copy
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

# nn.Dropout2d is replaced with F.dropout by ignoring differences


def gelu(x):
    return 0.5 * x * (1 + F.tanh(math.sqrt(2 / math.pi)
                                 * (x + 0.044715 * (x ** 3))))


def swish(x):
    return x * F.sigmoid(x)


ACT_FNS = {
    'relu': F.relu,
    'swish': swish,
    'gelu': gelu
}


class LayerNorm(chainer.Chain):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        with self.init_scope():
            self.g = chainer.Parameter(1., n_state)
            self.b = chainer.Parameter(0., n_state)
        self.e = e

    def __call__(self, x):
        # chainer requires explicit broadcast for avoiding latent bugs
        u = F.mean(x, -1, keepdims=True)
        u = F.broadcast_to(u, x.shape)
        s = F.mean((x - u) ** 2, -1, keepdims=True)
        s = F.broadcast_to(s, x.shape)
        x = (x - u) / F.sqrt(s + self.e)
        return F.bias(F.scale(x, self.g, axis=2), self.b, axis=2)


class Conv1D(chainer.Chain):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            with self.init_scope():
                self.w = chainer.Parameter(
                    initializers.Normal(scale=0.02), (nf, nx))  # transposed
                self.b = chainer.Parameter(0., nf)
        else:  # was used to train LM
            raise NotImplementedError

    def __call__(self, x):
        if self.rf == 1:
            size_out = x.shape[:-1] + (self.nf,)
            x = F.linear(x.reshape(-1, x.shape[-1]), self.w, self.b)
            x = x.reshape(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(chainer.Chain):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        #[switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        with self.init_scope():
            self.b = chainer.Parameter(
                np.tril(np.ones((n_ctx, n_ctx)), 0)[None, None])
            self.b._requires_grad = False  # `b` is just registered without param update
            self.c_attn = Conv1D(n_state * 3, 1, nx)
            self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = lambda x: F.dropout(x, cfg.attn_pdrop)
        self.resid_dropout = lambda x: F.dropout(x, cfg.resid_pdrop)

    def _attn(self, q, k, v):
        w = F.batch_matmul(q.reshape(-1, *q.shape[-2:]),
                           k.reshape(-1, *k.shape[-2:]))
        if self.scale:
            w = w / math.sqrt(v.shape[-1])
        # TF implem method: mask_attn_weights
        w = w * self.b.array[0] + -1e9 * (1 - self.b.array[0])
        w = F.softmax(w, axis=2)
        w = self.attn_dropout(w)
        return F.batch_matmul(w, v.reshape(-1, *v.shape[-2:]))\
                .reshape(v.shape[0], v.shape[1], v.shape[2], -1)

    def merge_heads(self, x):
        x = F.transpose(x, (0, 2, 1, 3))
        new_x_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1], )
        return x.reshape(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.shape[:-1] + (self.n_head, x.shape[-1] // self.n_head)
        x = x.reshape(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return F.transpose(x, (0, 2, 3, 1))
        else:
            return F.transpose(x, (0, 2, 1, 3))

    def __call__(self, x):
        x = self.c_attn(x)
        query, key, value = F.split_axis(x, 3, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(chainer.Chain):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg.n_embd
        with self.init_scope():
            self.c_fc = Conv1D(n_state, 1, nx)
            self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = lambda x: F.dropout(x, cfg.resid_pdrop)

    def __call__(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(chainer.Chain):
    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.n_embd
        with self.init_scope():
            self.attn = Attention(nx, n_ctx, cfg, scale)
            self.ln_1 = LayerNorm(nx)
            self.mlp = MLP(4 * nx, cfg)
            self.ln_2 = LayerNorm(nx)

    def __call__(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class Model(chainer.Chain):
    """ Transformer model """

    def __init__(self, cfg, vocab=40990, n_ctx=512):
        super(Model, self).__init__()
        self.vocab = vocab
        with self.init_scope():
            self.embed = L.EmbedID(vocab, cfg.n_embd,
                                   initializers.Normal(scale=0.02))
            self.drop = lambda x: F.dropout(x, cfg.embd_pdrop)
            block = Block(n_ctx, cfg, scale=True)
            self.h = chainer.ChainList(*[copy.deepcopy(block)
                                         for _ in range(cfg.n_layer)])
        self.decoder = lambda x: F.linear(x, self.embed.W)
        # To reproduce the noise_shape parameter of TF implementation
        self.clf_dropout = lambda x: F.dropout(x, cfg.clf_pdrop)

    def __call__(self, x):
        x = x.reshape(-1, x.shape[2], x.shape[3])
        e = self.embed(x)
        h = F.sum(e, axis=2)
        for block in self.h:
            h = block(h)
        return h


class LMHead(chainer.Chain):
    """ Language Model Head for the transformer """

    def __init__(self, model, cfg):
        super(LMHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.decoder = lambda x: F.linear(x, model.embed.W)

    def __call__(self, h):
        # Truncated Language modeling logits
        # Shape: 252, 768
        h_trunc = h[:, :-1].reshape(-1, self.n_embd)
        lm_logits = self.decoder(h_trunc)
        return lm_logits


class ClfHead(chainer.Chain):
    """ Classifier Head for the transformer """

    def __init__(self, clf_token, cfg, single_prediction=False):
        super(ClfHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        # To reproduce the noise_shape parameter of TF implementation
        self.dropout = lambda x: F.dropout(x, cfg.clf_pdrop)
        with self.init_scope():
            self.linear = L.Linear(cfg.n_embd, 1,
                                   initialW=initializers.Normal(scale=0.02))
        self.single_prediction = single_prediction

    def __call__(self, h, x):
        # Classification logits
        clf_h = h.reshape(-1, self.n_embd)
        flat = x[:, :, :, 0].reshape(-1)
        #pool_idx = torch.eq(x[:, :, 0].contiguous().view(-1), self.clf_token)
        clf_h = clf_h[flat == self.clf_token, :]  # .index_select(0, pool_idx)
        clf_h = self.dropout(clf_h)
        clf_h = clf_h.reshape(-1, self.n_embd)
        clf_logits = self.linear(clf_h)
        if self.single_prediction:
            return F.concat([clf_logits, 1 - clf_logits], axis=-1)
        else:
            return clf_logits.reshape(-1, 2)


def load_openai_pretrained_model(
        model,
        n_ctx=-1,
        n_special=-1,
        n_transfer=12,
        n_embd=768,
        path='./model/',
        path_names='./'):
    # Load weights from TF model
    names = json.load(open(path_names + 'parameters_names.json'))
    shapes = json.load(open(path + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n))
                   for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape)
                   for param, shape in zip(init_params, shapes)]
    if n_ctx > 0:
        init_params[0] = init_params[0][:n_ctx]
    if n_special > 0:
        init_params[0] = np.concatenate([init_params[1], (np.random.randn(
            n_special, n_embd) * 0.02).astype(np.float32), init_params[0]], 0)
    else:
        init_params[0] = np.concatenate([init_params[1],
                                         init_params[0]
                                         ], 0)
    del init_params[1]
    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1 + n_transfer * 12
    init_params = [arr.squeeze() for arr in init_params]
    try:
        assert model.embed.W.shape == init_params[0].shape
    except AssertionError as e:
        e.args += (model.embed.W.shape, init_params[0].shape)
        raise
    model.embed.W.array[:] = init_params[0]
    for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if len(pointer.shape) == 2 and pointer.shape[::-1] == ip.shape:
            ip = ip.T
            # transpose matrix in a linear layer from tf to chainer
        try:
            assert pointer.shape == ip.shape
        except AssertionError as e:
            e.args += (pointer.shape, ip.shape)
            raise
        pointer.array[:] = ip


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DEFAULT_CONFIG = dotdict({
    'n_embd': 768,
    'n_head': 12,
    'n_layer': 12,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'afn': 'gelu',
    'clf_pdrop': 0.1})
