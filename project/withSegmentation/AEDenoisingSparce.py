from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
import os
import math

def ae(input_shape, X, x_train_noisy, X_test, x_test_noisy):
    # this is the size of our encoded representations
    encoding_dim = 28
    # this is our input placeholder
    input_img = Input(shape=(input_shape,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(input_img)
    encoded = Dense(encoding_dim, activation='linear',
                    activity_regularizer=regularizers.l2(0.00001))(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_shape, activation='sigmoid')(decoded)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
    
    autoencoder.fit(x_train_noisy, X, 
              batch_size=28, 
              epochs=100,
              shuffle=True,
              validation_data=(x_test_noisy, X_test))

    return autoencoder, encoder

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)

for segment in [1,2,4,7,8]:

    directory = "./resultsDenoisingSparse/segments" + str(segment)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(10):
        target_class_labels = np.where([y_train[:]==i])[1]
        target_class_data = X_train[:][target_class_labels] 

        segments_list = []
        
        segment_length = int(784/segment)
        
        for j in range(segment):
            segments_list.append(target_class_data[:, (j*segment_length):((j+1)*segment_length)])
        
        noise_factor = 0.5

        fused_vector = np.empty((segments_list[0].shape[0],0))

        for item in segments_list:
            cur_x_train, cur_x_test = train_test_split(item, test_size=0.35)
            cur_x_train_noisy = cur_x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=cur_x_train.shape) 
            cur_x_test_noisy = cur_x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=cur_x_test.shape) 
            
            cur_x_train_noisy = np.clip(cur_x_train_noisy, 0., 1.)
            cur_x_test_noisy = np.clip(cur_x_test_noisy, 0., 1.)

            cur_autoencoder, cur_encoder = ae(cur_x_train_noisy.shape[1], cur_x_train, cur_x_train_noisy, cur_x_test, cur_x_test_noisy)

            encoded = cur_encoder.predict(item)
            
            if (segment == 1):
                np.savetxt(directory + "/AEResult_#" + str(i) + ".csv", encoded, delimiter=',')
            else:
                fused_vector = np.concatenate((fused_vector, encoded), axis=1)

        if (segment != 1):
            fused_x_train, fused_x_test = train_test_split(fused_vector, test_size=0.35)
    
            fused_x_train_noisy = fused_x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=fused_x_train.shape) 
            fused_x_test_noisy = fused_x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=fused_x_test.shape) 
            
            fused_x_train_noisy = np.clip(fused_x_train_noisy, 0., 1.)
            fused_x_test_noisy = np.clip(fused_x_test_noisy, 0., 1.)
            
            autoencoder, encoder = ae(fused_x_train_noisy.shape[1], fused_x_train, fused_x_train_noisy, fused_x_test, fused_x_test_noisy);
            
            encoded = encoder.predict(fused_vector)
            
            np.savetxt(directory + "/AEResult_#" + str(i) + ".csv", encoded, delimiter=',')
            
        