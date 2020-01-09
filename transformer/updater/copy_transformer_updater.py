import chainer
import chainer.functions as F

from chainer.training import StandardUpdater


class CopyTransformerUpdater(StandardUpdater):

    def update_core(self):
        with chainer.using_device(self.device):
            self.update_net()

    def update_net(self):
        batch = next(self.get_iterator('main'))
        batch = self.converter(batch, self.device)

        optimizer = self.get_optimizer('main')
        net = optimizer.target

        # for training we need one label less, since we right shift the output of the network
        predictions = net(batch['data'], batch['label'][:, :-1])

        batch_size, num_steps, vocab_size = predictions.shape
        predictions = F.reshape(predictions, (-1, vocab_size))
        labels = batch['label'][:, 1:].ravel()

        loss = F.softmax_cross_entropy(predictions, labels)
        accuracy = F.accuracy(F.softmax(predictions), labels)

        net.cleargrads()
        loss.backward()
        optimizer.update()

        chainer.reporter.report({
            "loss": loss,
            "train/accuracy": accuracy
        })
