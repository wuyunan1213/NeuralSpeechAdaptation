import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras_lr_multiplier import LRMultiplier

########################## Ordinary feed forward NN ###########################
class feed_forward_NN():
    def __init__(self, name, activations, layers, loss, learn_rate):
        if len(activations) != len(layers): 
          raise ValueError('layer number and activation number do not agree')
        self.name = name 
        self.activations = activations   # list of activation functions 
        self.layers = layers    # list of layer-wise unit (include input layer)
        self.loss = loss    # binary crossentropy for now
        self.learn_rate = learn_rate 
        self.model = self.createModel()   
    
    def createModel(self):
        layers = self.layers
        acts = self.activations
        lossF = self.loss
        learn = self.learn_rate
    
        model = Sequential()
        model.add(Dense(units = layers[1],
                        input_dim = layers[0],
                        activation = acts[0]))
        for i in range(2,len(layers)):
            model.add(Dense(unit = layers[i], 
                            activation = acts[i]))
        model.compile(loss = lossF, 
                      optimizer = keras.optimizers.Adam(learning_rate = learn))
        return model
    
    def fit_model(self, train_input, train_label, test_input, test_label):
        # loss, accuracy of train and test is saved in history dictionary
        history = self.model.fit(train_input, 
                                   train_label, 
                                   validation_data = (test_input,test_label), 
                                   epochs = 10, 
                                   batch_size = 10)
        return history
    
    def evaluate(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title("FF Model Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(['Train','Test'], loc = 'upper left')
        plt.savefig("FF_model_accuracy_plot.png")
        plt.show()
        plt.close()

############################## Slow-Fast pathway NN ############################
# ref : https://pypi.org/project/keras-lr-multiplier/
# for learning rate multiplier : pip install keras-lr-multiplier
# ref : https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer
# for different weight decays (regularizer) for each layers

class slow_fast_NN():
    def __init_(self, name):
        self.name = name
        self.model = self.createModel()
        
    def createModel(self):
        inputs = Input(shape=(self.layers[0],))
        h1 = Dense(layers[1], 
                   activation='relu', 
                   name = "slow",
                   kernel_regularizer = tf.keras.regularizers.l1(0.01),
                   activity_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
        h2 = Dense(layers[2], 
                   activation='relu',
                   name = "fast",
                   kernel_regularizer = tf.keras.regularizers.l1(0.1),
                   activity_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
        comb = keras.layers.concatenate([h1, h2])
        predictions = Dense(layers[3], activation='sigmoid')(comb)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer = 
                      LRMultiplier('adam', {'slow':0.0005, 'fast':0.05}),
                      loss = 'binary_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def fit_model(self, train_input, train_label, test_input, test_label):
        # loss, accuracy of train and test is saved in history dictionary
        history = self.model.fit(train_input, 
                                   train_label, 
                                   validation_data = (test_input,test_label), 
                                   epochs = 10, 
                                   batch_size = 10)
        return history
    
    def evaluate(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title("Slow-Fast Model Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(['Train','Test'], loc = 'upper left')
        plt.savefig("Slow_Fast_model_accuracy_plot.png")
        plt.show()
        plt.close()