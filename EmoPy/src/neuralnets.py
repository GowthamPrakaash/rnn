from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, ConvLSTM2D, Conv3D, MaxPooling2D, Dropout, \
    MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.utils import plot_model
import json

from .callback import PlotLosses


class _FERNeuralNet(object):
    """
    Interface for all FER deep neural net classes.
    """

    def __init__(self, emotion_map):
        self.emotion_map = emotion_map
        self._init_model()

    def _init_model(self):
        raise NotImplementedError("Class %s doesn't implement _init_model()" % self.__class__.__name__)

    def fit(self, x_train, y_train):
        raise NotImplementedError("Class %s doesn't implement fit()" % self.__class__.__name__)

    def fit_generator(self, generator, validation_data=None, epochs=50):
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
        self.model.fit_generator(generator=generator, validation_data=validation_data, epochs=epochs,
                                 callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3), PlotLosses()])

    def predict(self, images):
        self.model.predict(images)

    def save_model_graph(self):
        plot_model(self.model, to_file='output/model.png')

    def export_model(self, model_filepath, weights_filepath, emotion_map_filepath, emotion_map):
        self.model.save_weights(weights_filepath)

        model_json_string = self.model.to_json()
        model_json_file = open(model_filepath, 'w')
        model_json_file.write(model_json_string)
        model_json_file.close()

        with open(emotion_map_filepath, 'w') as fp:
            json.dump(emotion_map, fp)


class ConvolutionalLstmNN(_FERNeuralNet):
    """
    Convolutional Long Short Term Memory Neural Network.

    :param image_size: dimensions of input images
    :param channels: number of image channels
    :param emotion_map: dict of target emotion label keys with int values corresponding to the index of the emotion probability in the prediction output array
    :param time_delay: number of time steps for lookback
    :param filters: number of filters/nodes per layer in CNN
    :param kernel_size: size of sliding window for each layer of CNN
    :param activation: name of activation function for CNN
    :param verbose: if true, will print out extra process information

    **Example**::

        net = ConvolutionalLstmNN(target_dimensions=(64,64), channels=1, target_labels=[0,1,2,3,4,5,6], time_delay=3)
        net.fit(features, labels, validation_split=0.15)

    """

    def __init__(self, image_size, channels, emotion_map, time_delay=2, filters=10, kernel_size=(4, 4),
                 activation='sigmoid', verbose=False):
        self.time_delay = time_delay
        self.channels = channels
        self.image_size = image_size
        self.verbose = verbose

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        super().__init__(emotion_map)

    def _init_model(self):
        """
        Composes all layers of CNN.
        """
        model = Sequential()
        model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation,
                             input_shape=[self.time_delay] + list(self.image_size) + [self.channels],
                             data_format='channels_last', return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation,
                             input_shape=(self.time_delay, self.channels) + self.image_size,
                             data_format='channels_last', return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=1, kernel_size=self.kernel_size, activation="sigmoid", data_format="channels_last"))
        model.add(Flatten())
        model.add(Dense(units=len(self.emotion_map.keys()), activation="sigmoid"))
        if self.verbose:
            model.summary()
        self.model = model

    def fit(self, features, labels, validation_split, batch_size=10, epochs=50):
        """
        Trains the neural net on the data provided.

        :param features: Numpy array of training data.
        :param labels: Numpy array of target (label) data.
        :param validation_split: Float between 0 and 1. Percentage of training data to use for validation
        :param batch_size:
        :param epochs: number of times to train over input dataset.
        """
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
        self.model.fit(features, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                       callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])
