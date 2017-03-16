import json
from keras.applications import ResNet50, VGG16, VGG19
from keras.optimizers import SGD
from keras.utils import np_utils

class ConvNet:
    """
    Keras wrapper for Convolutional Neural Networks
    """
    def __init__(self):

	# Defining initial flags
        self.model = None
        self._trained = False
        self._history = None
        self._to_categorical = False

        return

    def build_resnet50(self, include_top=True, weights=None, classes=1000):

	# Creating ResNet50 Model
        model = ResNet50(include_top=include_top, weights=weights, classes=classes)
        if weights != None:
                self._trained = True
        
        # Print model summary
        model.summary()

        # Applying model to ConvNet class
        self.model = model

        return

    def build_vgg16(self, include_top=True, weights=None, classes=1000):

	# Creating VGG16 Model
        model = VGG16(include_top=include_top, weights=weights, classes=classes)
        if weights != None:
                self._trained = True
        
        # Print model summary
        model.summary()

        # Applying model to ConvNet class
        self.model = model

        return

    def build_vgg19(self, include_top=True, weights=None, classes=1000):

	# Creating VGG19 Model
        model = VGG19(include_top=include_top, weights=weights, classes=classes)
        if weights != None:
                self._trained = True
        
        # Print model summary
        model.summary()

        # Applying model to ConvNet class
        self.model = model

        return

    def train(self, data_train, labels_train, n_classes, learning_rate=0.001,
              batch_size=32, n_epochs=20, validation_size=0.05,
              metric='binary_accuracy', loss_func='binary_crossentropy',
              patience=10):

	# Check if model has been built
        if self.model is None:
            print("Error: the model has not been built.")
            return

	# Apply categorical format for labels
        if n_classes > 2 or loss_func == 'categorical_crossentropy':
            labels_train = np_utils.to_categorical(labels_train, n_classes)
            self._to_categorical = True

        # Initialise the optimizer
        opt = SGD(lr=learning_rate, momentum=0.9, decay=0.0005, nesterov=True)

	# Compiling model
        print("\n[INFO] compiling...")
        self.model.compile(loss=loss_func, optimizer=opt, metrics=[metric])

	# Training model
        print("\n[INFO] training...")
        history = self.model.fit(data_train, labels_train, batch_size=batch_size,
                                 validation_split=validation_size,
                                 epochs=n_epochs, verbose=1)

	# Applying post-train flags
        self._trained = True
        self._history = history.history

        return

    def evaluate(self, data_test, labels_test, n_classes, batch_size=32):

        if self._trained == False:
            print("Error: the model has not been trained.")
            return

        # Switch to categorical labels if this is not a binary classification task
        if self._to_categorical == True:
            labels_test = np_utils.to_categorical(labels_test, n_classes)

        # Evaluating model on test set
        print("\n[INFO] evaluating...")
        (loss, acc) = self.model.evaluate(data_test, labels_test, batch_size=batch_size, verbose=1)

	# Displaying final accuracy and loss
        print("\n[INFO] test_accuracy: {:.2f}% test_loss: {:.4f}".format(acc * 100, loss))

        return acc

    def predict(self, x, batch_size=32, verbose=0):

	# Check if the network is already pre-trained or not
        if self._trained == False:
            print("Error: the model has not been trained.")
            return

	# Predicting numpy input 'x'
        print("[INFO] predicting...")

        return self.model.predict(x, batch_size=batch_size, verbose=verbose)

    def decode_predictions(self, preds, path=None, top=None):

	# Decodes the predictions into readable inputs from path file
        CLASS_INDEX = json.load(open(path))
        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
            result.sort(key=lambda x: x[2], reverse=True)
            results.append(result)

        return results

    def save_model(self, fpath):

	# Converting model to json
        model_json = self.model.to_json()

	# Saving model in json file
        with open(fpath, "w") as json_file:
                json_file.write(model_json)
        return

    def save_weight(self, fpath):

	# Saving trained weights
        self.model.save_weights(fpath, overwrite=True)

        return

    def get_history(self):

	# Returning network history
        if self._history is None:
            print("Error: the model has no history.")
            return

        return self._history

# Test the ConvNet class
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    print("Main")
