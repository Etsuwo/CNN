import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import TruncatedNormal, Constant
from matplotlib import pyplot as plt

num_class = 10
channel = 3

def conv2d(filters, kernel_size, strides=(1,1), padding='same', bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean = 0.0, stddev = 0.01)
    cnst = Constant(value = bias_init)
    return Conv2D(filters, kernel_size, strides = strides, padding = padding, 
                    activation = "relu", 
                    kernel_initializer = trunc, 
                    bias_initializer = cnst,
                    **kwargs)

def dense(units, activation='tanh'):
    trunc = TruncatedNormal(mean = 0.0, stddev = 0.01)
    cnst = Constant(value = 1)
    return Dense(units, activation = activation, kernel_initializer = trunc, bias_initializer = cnst)

def AlexNet(input_shape):
    model = Sequential()

    model.add(conv2d(96, 9, strides = (1, 1), padding = 'valid', bias_init=0, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(BatchNormalization())
    model.add(conv2d(256, 5))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(BatchNormalization())
    model.add(conv2d(384, 3, bias_init=0))
    model.add(conv2d(384, 3))
    model.add(conv2d(256, 3))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))
    model.add(dense(4096))
    model.add(Dropout(0.5))
    model.add(dense(num_class, activation='softmax'))
    model.compile(optimizer = SGD(lr = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

def ShowTrainData(history):
    
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.show()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

model = AlexNet(x_train[0].shape)
model.summary()

history = model.fit(x_train, y_train, batch_size = 128, epochs = 100, verbose = 1, validation_split = 0.1)

ShowTrainData(history)
