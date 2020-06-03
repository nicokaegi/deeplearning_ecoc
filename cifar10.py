from ecoc_classifier import ecocModel
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, AveragePooling2D
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

def cifar10_model():

    input_dim = (32, 32, 3)

    model = Sequential([Conv2D(32, kernel_size=(5,5), strides=1, padding="same", input_shape=input_dim),
                         MaxPooling2D(pool_size=(3,3), strides=2),
                         Activation(tf.nn.relu),
                         Conv2D(32, kernel_size=(5,5), strides=1, padding="same"),
                         Activation(tf.nn.relu),
                         AveragePooling2D(pool_size=(3,3), strides=2),
                         Conv2D(64, kernel_size=(5,5), strides=1, padding="same"),
                         Activation(tf.nn.relu),
                         AveragePooling2D(pool_size=(3,3), strides=2),
                         Flatten(),
                         Dense(64),
                         Dense(1, activation=tf.nn.sigmoid)])


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def main():
    ecoc_matrix = np.loadtxt('/home/sindri/NicksPlayGround/ml/deeplearning_ecoc/10x10',delimiter=',')
    main_model = ecocModel(cifar10_model,ecoc_matrix)

    labels = np.load('./cifar_ecoc_labels.npy')
    images = np.load('./cifar_images.npy')

    #x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.30, random_state=42, shuffle=True)

    points = cross_val_score(main_model,images, labels, cv=5, fit_params={'batch_size' : 300, 'epochs' : 10} )

    '''
    main_model.fit(x_train, y_train, batch_size=300, epochs=40)

    main_model.evaluate(x_test, y_test)
    main_model.save('/content/drive/My Drive/ml traning data')
    '''

    print(points)
    print("average {}".format(sum(points,0)/len(points)) )
    print("std {}".format(np.std(points)))


if __name__ == '__main__':
    main()
