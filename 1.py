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

fig.savefig("MNIST data.png")
plt.show()