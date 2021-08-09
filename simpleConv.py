import tensorflow as tf
from tensorflow.keras import Sequential

print(tf.__version__)
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(type(x_train))
print(x_train.shape)
print(x_train.dtype)
print(x_train.min(), '-', x_train.max())

#画素の正規化
x_train = x_train / 255
x_test = x_test / 255

print(x_train.dtype)
print(x_train.min(), '-', x_train.max())

#構築
model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten_layer'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
], name = 'my_model')

model.summary()

#学習プロセスの設定
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#学習の実行
callbacks = [tf.keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
                tf.keras.callbacks.ModelCheckpoint('./data/mnist_sequential_{epoch:03d}_{val_loss:.04f}.h5', save_best_only = True)]
history = model.fit(x_train, y_train, batch_size = 128, epochs = 20, validation_split = 0.2, callbacks = callbacks)
