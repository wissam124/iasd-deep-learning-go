from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from matplotlib import pyplot as plt


class GoModel():
    def __init__(self, regParam, learningRate, inputDim, outputDim):
        self.regParam = regParam
        self.learningRate = learningRate
        self.inputDim = inputDim
        self.outputDim = outputDim

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, X, y, epochs, verbose, validation_split, batch_size):
        checkpoint = ModelCheckpoint('best_model.h5',
                                     monitor='loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto',
                                     period=1)

        csv_logger = CSVLogger('training.log', separator=',', append=False)

        return self.model.fit(X,
                              y,
                              epochs=epochs,
                              verbose=verbose,
                              validation_split=validation_split,
                              batch_size=batch_size,
                              callbacks=[checkpoint, csv_logger])

    def save_model(self, version):
        self.model.save('./model_params_version_' + "{}".format(version) +
                        '.h5')

    def summary(self):
        return self.model.summary()

    def plot_model(self):
        plot_model(self.model)

    def display_layers():
        pass


class NeuralNet(GoModel):
    def __init__(self, regParam, learningRate, inputDim, outputDim,
                 hiddenLayers, momentum):
        GoModel.__init__(self, regParam, learningRate, inputDim, outputDim)
        self.hidden_layers = hiddenLayers
        self.momentum = momentum
        self.num_layers = len(hiddenLayers)
        self.model = self.buildModel()

    def convLayer(self, x, numFilters, kernelSize):

        x = Conv2D(filters=numFilters,
                   kernel_size=kernelSize,
                   data_format='channels_last',
                   padding='same',
                   use_bias=False,
                   activation='linear',
                   kernel_regularizer=regularizers.l2(self.regParam))(x)

        x = BatchNormalization(axis=-1)(x)

        x = LeakyReLU()(x)

        return x

    def residualLayer(self, inputLayer, numFilters, kernelSize):

        x = self.convLayer(inputLayer, numFilters, kernelSize)

        x = Conv2D(filters=numFilters,
                   kernel_size=kernelSize,
                   data_format='channels_last',
                   padding='same',
                   use_bias=False,
                   activation='linear',
                   kernel_regularizer=regularizers.l2(self.regParam))(x)

        x = BatchNormalization(axis=-1)(x)

        x = add([inputLayer, x])

        x = LeakyReLU()(x)

        return (x)

    def value_head(self, x):

        x = Conv2D(filters=1,
                   kernel_size=(1, 1),
                   data_format='channels_last',
                   padding='same',
                   use_bias=False,
                   activation='linear',
                   kernel_regularizer=regularizers.l2(self.regParam))(x)

        x = BatchNormalization(axis=-1)(x)

        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(10,
                  use_bias=False,
                  activation='linear',
                  kernel_regularizer=regularizers.l2(self.regParam))(x)

        x = LeakyReLU()(x)

        x = Dense(1,
                  use_bias=False,
                  activation='sigmoid',
                  kernel_regularizer=regularizers.l2(self.regParam),
                  name='value')(x)

        return (x)

    def policy_head(self, x):

        x = Conv2D(filters=2,
                   kernel_size=(1, 1),
                   data_format='channels_last',
                   padding='same',
                   use_bias=False,
                   activation='linear',
                   kernel_regularizer=regularizers.l2(self.regParam))(x)

        x = BatchNormalization(axis=-1)(x)

        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(self.outputDim, activation='softmax', name='policy')(x)

        return (x)

    def buildModel(self):

        mainInput = Input(shape=self.inputDim, name='board')

        x = self.convLayer(mainInput, self.hidden_layers[0]['numFilters'],
                           self.hidden_layers[0]['kernelSize'])

        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self.residualLayer(x, h['numFilters'], h['kernelSize'])

        value_head = self.value_head(x)
        policy_head = self.policy_head(x)

        model = Model(inputs=[mainInput], outputs=[policy_head, value_head])
        model.compile(optimizer=SGD(lr=self.learningRate),
                      loss={
                          'value': 'mse',
                          'policy': 'categorical_crossentropy'
                      },
                      loss_weights={
                          'value': 1,
                          'policy': 1
                      },
                      metrics=['accuracy'])

        return model
