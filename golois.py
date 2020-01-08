# coding: utf-8
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
import golois

import config
from model import NeuralNet

planes = 8
moves = 361
dynamicBatch = True  # pour tester réseau sans installer la bibli golois


def generateData(N=10000, dynamicBatch=False):
    planes = 8
    moves = 361
    if dynamicBatch:
        input_data = np.random.randint(2, size=(N, 19, 19, planes))
        input_data = input_data.astype('float32')

        policy = np.random.randint(moves, size=(N, ))
        policy = keras.utils.to_categorical(policy)

        value = np.random.randint(2, size=(N, ))
        value = value.astype('float32')

        end = np.random.randint(2, size=(N, 19, 19, 2))
        end = end.astype('float32')

        golois.getBatch(input_data, policy, value, end)
    else:
        input_data = np.load('./input_data.npy')
        policy = np.load('./policy.npy')
        value = np.load('./value.npy')
        # end = np.load('./end.npy')

    return input_data, policy, value


input_data, policy, value = generateData(100000, True)


# Initial model training
goNeuralNet = NeuralNet(config.REG_CONST, config.LEARNING_RATE,
                        (19, 19, planes), moves, config.HIDDEN_CNN_LAYERS,
                        config.MOMENTUM)

GoNeuralNet.summary()

GoNeuralNet.fit(input_data, {
    'policy': policy,
    'value': value
},
                epochs=config.EPOCHS,
                verbose=1,
                validation_split=0.1,
                batch_size=config.BATCH_SIZE)

GoNeuralNet.save_model()
