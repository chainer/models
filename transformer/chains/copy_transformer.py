import chainer.functions as F
import chainer.links as L
from chainer import Chain

from transformer import get_encoder_decoder
from transformer.utils import subsequent_mask


class CopyTransformer(Chain):
    """
        This class shows how the transformer could be used.
        The copy transformer used in our example consists of an encoder/decoder
        stack with a stack size of two. We use greedy decoding during test time,
        to get the predictions of the transformer.

        :param vocab_size: vocab_size determines the number of classes we want to distinguish.
        Since we only want to copy numbers, the vocab_size is the same for encoder and decoder.
        :param max_len: determines the maximum sequence length, since we have no end of sequence token.
        :param start_symbol: determines the begin of sequence token.
        :param transformer_size: determines the number of hidden units to be used in the transformer
    """

    def __init__(self, vocab_size, max_len, start_symbol, transformer_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.start_symbol = start_symbol
        self.transformer_size = transformer_size

        with self.init_scope():
            model = get_encoder_decoder(
                vocab_size,
                vocab_size,
                N=2,
                model_size=transformer_size,
            )
            self.model = model
            self.mask = subsequent_mask(self.transformer_size)
            self.classifier = L.Linear(transformer_size, vocab_size)

    def __call__(self, x, t):
        result = self.model(x, t, None, self.mask)

        return self.classifier(result, n_batch_axes=2)

    def decode_prediction(self, x):
        """
            helper function for greedy decoding
            :param x: the output of the classifier
            :return: the most probable class index
        """
        return F.argmax(F.softmax(x, axis=2), axis=2)

    def predict(self, x):
        """
            This method performs greedy decoding on the input vector x
            :param x: the input data that shall be copied.
            :return: the (hopefully) copied data.
        """
        # first, we use the encoder on the input data
        memory = self.model.encode(x, None)

        # second, we create the start input for the decoder
        target = self.xp.full((len(x), 1), self.start_symbol, x.dtype)

        # third, we decode our encoded data, using our target as input to the decoder
        for i in range(self.max_len - 1):
            prediction = self.model.decode(memory, None, target, self.mask)
            prediction = self.classifier(prediction, n_batch_axes=2)
            decoded = self.decode_prediction(prediction)

            target = F.concat([target, decoded[:, -1:]])

        return target.array
