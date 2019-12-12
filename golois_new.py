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
if dynamicBatch:
    N = 100000
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
    end = np.load('./end.npy')

GoNeuralNet = NeuralNet(config.REG_CONST, config.LEARNING_RATE,
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

GoNeuralNet.save_model(0.1)

import pandas as pd
df = pd.read_csv('./training.log')
epochs = df['epoch']
plt.clf()
f, ax = plt.subplots(2, 3)
ax[0][0].plot(epochs, df['loss'])
ax[0][0].plot(epochs, df['val_loss'])
ax[0][0].legend(['loss', 'val_los'])
ax[0][0].set_title('Total loss')
ax[0][1].plot(epochs, df['policy_loss'])
ax[0][1].plot(epochs, df['val_policy_loss'])
ax[0][1].legend(['policy_loss', 'val_policy_loss'])
ax[0][1].set_title('Policy loss')
ax[0][2].plot(epochs, df['value_loss'])
ax[0][2].plot(epochs, df['val_value_loss'])
ax[0][2].legend(['value_loss', 'val_value_loss'])
ax[0][2].set_title('Value loss')
ax[1][1].plot(epochs, df['policy_accuracy'])
ax[1][1].plot(epochs, df['val_policy_accuracy'])
ax[1][1].legend(['policy_accuracy', 'val_policy_accuracy'])
ax[1][1].set_title('Policy accuracy')
ax[1][2].plot(epochs, df['value_accuracy'])
ax[1][2].plot(epochs, df['val_value_accuracy'])
ax[1][2].legend(['value_accuracy', 'val_value_accuracy'])
ax[1][2].set_title('Value accuarcy')

# input = keras.Input(shape=(19, 19, planes), name='board')
# x = layers.Conv2D(32, 3, activation='relu', padding='same')(input)
# x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
# x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
# x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
# x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
# x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
# policy_head = layers.Conv2D(1, 3, activation='relu', padding='same')(x)
# policy_head = layers.Flatten()(policy_head)
# policy_head = layers.Dense(moves, activation='softmax',
#                            name='policy')(policy_head)
# value_head = layers.Flatten()(x)
# value_head = layers.Dense(1, activation='sigmoid', name='value')(value_head)

# model = keras.Model(inputs=input, outputs=[policy_head, value_head])

# model.summary()

# model.compile(optimizer=keras.optimizers.SGD(lr=0.1),
#               loss={
#                   'value': 'mse',
#                   'policy': 'categorical_crossentropy'
#               },
#               metrics=['accuracy'])

# model.fit(input_data, {
#     'policy': policy,
#     'value': value
# },
#           epochs=10,
#           batch_size=128,
#           validation_split=0.1)

# model.save('test.h5')

# idee: jouer sur le fit
# jouer sur le batch_size
# on a le droit de toucher à la taille de l'input
# dataset table a l'etat t, le move que le joueur intelligent a fait, les etats d'avant, target qui gagne la partie
# move du suivant, value probabilite que le joueur gagne
