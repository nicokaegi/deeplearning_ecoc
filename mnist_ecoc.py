from ecoc_classifier import ecocModel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

def mnist_model():

    input_dim = (28, 28, 1)

    model = Sequential([Conv2D(28, kernel_size=(3,3), input_shape=input_dim),
                         MaxPooling2D(pool_size=(2,2)),
                         Flatten(),
                         Dense(128, activation=tf.nn.relu),
                         Dropout(0.2),
                         Dense(1, activation=tf.nn.sigmoid)])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def main():
    #ecoc_matrix = np.loadtxt('/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/10x10',delimiter=',')
    #main_model = ecocModel(mnist_model ,ecoc_matrix)

    #np.set_printoptions(edgeitems=30)

    #with np.load('/home/sindri/NicksPlayGround/ml/data/mnist_ecoc_data.npz') as data:

    lables = np.load('./mnist_notecoc_lables.npy')
    images = np.load('./mnist_images.npy')

    losses = cross_val_score(tf.keras.wrappers.scikit_learn.KerasClassifier(mnist_model()), images, lables, cv=10)

    print('mnist {}'.format(losses))
    print(np.mean(losses))
    print(np.std(losses))



if __name__ == '__main__':
    main()
