from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist

def ae(input_shape):
    # this is the size of our encoded representations
    encoding_dim = 32
    # this is our input placeholder
    input_img = Input(shape=(input_shape,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='linear',
                    activity_regularizer=regularizers.l2(0.00001))(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_shape, activation='sigmoid')(decoded)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    # create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-3]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    return autoencoder, encoder, decoder

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
 
X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)

for i in range(10):
    target_class_labels = y_train[y_train[:]==i]
    target_class_data = X_train[:][target_class_labels] 
    
    x_train, x_test = train_test_split(target_class_data, test_size=0.2)
    
    autoencoder, encoder, decoder = ae(784);
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
    
    test_stat = autoencoder.fit(x_train, x_train,
                epochs=30,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, x_test))
    
    encoded = encoder.predict(target_class_data)
    
    np.savetxt("./resultsSparse/AEResult_#" + str(i) + ".csv", encoded, delimiter=',')
            
        