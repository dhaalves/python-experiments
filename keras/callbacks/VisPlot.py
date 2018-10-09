import visdom as visdom

from keras import callbacks
import numpy as np


class VisPlot(callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.vis = visdom.Visdom()
        self.win_loss = None
        self.win_acc = None

    def on_epoch_end(self, epoch, logs={}):

        x = np.array([epoch])
        y_loss = np.column_stack((np.array(logs.get('loss')), np.array(logs.get('val_loss'))))
        y_acc = np.column_stack((np.array(logs.get('categorical_accuracy')),
                                 np.array(logs.get('val_categorical_accuracy'))))
        legend_loss = dict(legend=['train_loss', 'val_loss'])
        legend_acc = dict(legend=['train_acc', 'val_acc'])

        if epoch == 0:
            self.win_loss = self.vis.line(Y=y_loss,
                                          X=x,
                                          opts=legend_loss)
            self.win_acc = self.vis.line(Y=y_acc,
                                         X=x,
                                         opts=legend_acc)
        else:
            self.vis.line(win=self.win_loss,
                          update='append',
                          Y=y_loss,
                          X=x,
                          opts=legend_loss)
            self.vis.line(win=self.win_acc,
                          update='append',
                          Y=y_acc,
                          X=x,
                          opts=legend_acc)
