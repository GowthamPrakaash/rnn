from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, ConvLSTM2D, Conv3D, MaxPooling2D, Dropout, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.utils import plot_model
import json


class ConvolutionalLstmNN(_FERNeuralNet):

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
        """Composes all layers of CNN."""
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
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
        self.model.fit(features, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                       callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])

