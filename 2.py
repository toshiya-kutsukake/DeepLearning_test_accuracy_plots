import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

#Keras???????????????????????????????????????????
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#MNIST??????
fig = plt.figure(figsize=(9, 9))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
for i in range(81):
    ax = fig.add_subplot(9, 9, i + 1, xticks=[], yticks=[])
    ax.imshow(x_train[i].reshape((28, 28)), cmap='gray')

plt.show()


num_classes = 10
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test =  keras.utils.np_utils.to_categorical(y_test, num_classes)


x_train.tofile("X_train data.png")
x_test.tofile("X_test data.png")

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])



batch_size = 128
epochs = 20
history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test))

#???

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

        # 1) Accracy Plt
plt.plot(epochs, acc, 'bo' ,label = 'training acc')
plt.plot(epochs, val_acc, 'b' , label= 'validation acc')
plt.title('Training and Validation acc')

plt.legend(['train', 'test'], loc='upper left')

plt.savefig("Model accuracy.jpeg")
plt.show()


#loss

     # 2) Loss Plt
plt.plot(epochs, loss, 'bo' ,label = 'training loss')
plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
plt.title('Training and Validation loss')
 

plt.legend(['train', 'test'], loc='upper left')

plt.savefig("Model loss.jpeg")
plt.show()

