from tensorflow.keras.callbacks import Callback


class AfterEpochLogger(Callback):
    def __init__(self, no_epochs):
        self.no_epochs = no_epochs
        self.epochs_passed = 0
        self.history_loss = []
        self.history_accuracy = []
        self.val_history_loss = []
        self.val_history_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epochs_passed += 1
        self.history_loss.append(logs['loss'])
        self.history_accuracy.append(logs['accuracy'])
        self.val_history_loss.append(logs['val_loss'])
        self.val_history_accuracy.append(logs['val_accuracy'])
        if self.epochs_passed % self.no_epochs == 0:
            print(
                'Epoch: %d\n\tTrain Accuracy:\t%.2f%%\tValidation Accuracy\t%.2f%%\n\tTrain Loss\t%.2f\tValidation Loss:\t%.2f' %
                (self.epochs_passed, logs['accuracy'], logs['val_accuracy'], logs['loss'], logs['val_loss']))