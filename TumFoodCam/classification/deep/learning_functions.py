import numpy as np
from misc import utils
import data_io.settings as Settings


class AdjustVariable(object):
    """
    Simple implementation of a class that can be hooked into on_epoch_finished.
    It will adjust a variable with name name like learning_rate or momentum to speed up learning.
    Improved with https://github.com/StevenReitsma/kaggle-diabetic-retinopathy/blob/master/deep/learning_rate.py
    TODO: Test if this actually improved anything or made things worse.
    """
    def __init__(self, name, start=0.03):
        self.name = name
        self.start = start
        self.ls = None

    def __call__(self, nn, train_history):
        #if self.ls is None:
        #    self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        ## detect if NN_EPOCHS has changed
        #if Settings.NN_EPOCHS != nn.max_epochs:
        #    changeEpochs = utils.radio_question("[!]", "Change in NN_EPOCHS detected. Change net?", None, ["Yes", "No"], [True, False])
        #    if changeEpochs:
        #        #getattr(nn, "max_epochs").set_value(Settings.NN_EPOCHS)
        #        nn.max_epochs = Settings.NN_EPOCHS
        #        self.ls = np.linspace(self.start, self.stop, Settings.NN_EPOCHS)
        #    else:
        #        raise StopIteration()

        epoch = train_history[-1]['epoch']

        stop = self.start * 10e-2 * 2
        stop2 = stop * 10e-4 * 2

        ls = np.linspace(self.start, stop, 50)
        ls2 = np.linspace(stop, stop2, nn.max_epochs - 100)

        if epoch <= 50:
            new_value = utils.to_float32(ls[epoch - 1])
        elif epoch <= 100:
            new_value = utils.to_float32(ls[-1])
        else:
            new_value = utils.to_float32(ls2[epoch - 1 - 100])

        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    """
    Stops the training if the validation_loss did not improve for 'patience' epochs.
    """

    def __init__(self, patience=30):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']

        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()

        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


