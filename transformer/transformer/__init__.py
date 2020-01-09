import copy

import chainer

from .encoder import Encoder, EncoderLayer
from .encoder_decoder import EncoderDecoder
from .decoder import DecoderLayer, Decoder
from .embedding import Embedding
from .positional_encoding import PositionalEncoding
from .position_wise_feed_forward import PositionwiseFeedForward
from .attention import MultiHeadedAttention


def build_decoder(vocab_size, N=6, model_size=512, ff_size=2048, num_heads=8, dropout_ratio=0.1):
    """
        Convenience function that returns the decoder, together with the embedding layer.
        Code using this function is expected to embed the input to the decoder by itself, using
        the supplied encoder Chain.
    :param vocab_size: the number of classes
    :param N: stack size of the decoder
    :param model_size: the number of hidden units in the transformer
    :param ff_size: the number of hidden units in the PositionwiseFeedForward part of the decoder
    :param num_heads: number of attention heads in the attention parts of the model
    :param dropout_ratio: dropout ratio for regularization
    :return: a tuple of two Chains. The first Chain is used for embedding the input to the decoder, the second is
    the decoder itself.
    """
    attention = MultiHeadedAttention(num_heads, model_size, dropout_ratio=dropout_ratio)
    feed_forward = PositionwiseFeedForward(model_size, ff_size, dropout_ratio=dropout_ratio)
    positional_encoding = PositionalEncoding(model_size, dropout_ratio=dropout_ratio)

    decoder_layer = DecoderLayer(
        model_size,
        copy.deepcopy(attention),
        copy.deepcopy(attention),
        feed_forward,
        dropout_ratio=dropout_ratio
    )

    decoder = Decoder(decoder_layer, N)

    embeddings = Embedding(model_size, vocab_size)

    return chainer.Sequential(embeddings, positional_encoding), decoder


def get_encoder_decoder(src_vocab_size, tgt_vocab_size, N=6, model_size=512, ff_size=2048, num_heads=8, dropout_ratio=0.1):
    """
        Convenience function that returns the full transformer model including encoder and decoder.
    :param src_vocab_size: the number of classes for the encoder
    :param tgt_vocab_size: the number of classes for the decoder
    :param N: stack size of the decoder
    :param model_size: the number of hidden units in the transformer
    :param ff_size: the number of hidden units in the PositionwiseFeedForward part of the decoder
    :param num_heads: number of attention heads in the attention parts of the model
    :param dropout_ratio: dropout ratio for regularization
    :return: the transformer model
    """
    attention = MultiHeadedAttention(num_heads, model_size, dropout_ratio=dropout_ratio)
    feed_forward = PositionwiseFeedForward(model_size, ff_size, dropout_ratio=dropout_ratio)
    positional_encoding = PositionalEncoding(model_size, dropout_ratio=dropout_ratio)

    encoder_layer = EncoderLayer(
        model_size,
        copy.deepcopy(attention),
        copy.deepcopy(feed_forward),
        dropout_ratio=dropout_ratio
    )
    encoder = Encoder(encoder_layer, N)

    decoder_layer = DecoderLayer(
        model_size,
        copy.deepcopy(attention),
        copy.deepcopy(attention),
        feed_forward,
        dropout_ratio=dropout_ratio
    )
    decoder = Decoder(decoder_layer, N)

    src_embeddings = Embedding(model_size, src_vocab_size)
    tgt_embeddings = Embedding(model_size, tgt_vocab_size)

    src_embeddings = chainer.Sequential(src_embeddings, positional_encoding)
    tgt_embeddings = chainer.Sequential(tgt_embeddings, positional_encoding)

    model = EncoderDecoder(
        encoder,
        decoder,
        src_embeddings,
        tgt_embeddings
    )

    return model
