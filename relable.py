import MatrixGeneration as mg
import numpy as np
import tensorflow_datasets as tfds
from enum import Enum

#mg.GenerateMatrixCSV(10,5)
ecoc_matrix = np.loadtxt("/home/sindri/NicksPlayGround/ml/cnn/10x10", delimiter=',')

train_data = tfds.load('cifar10', split='train').shuffle(42)
test_data = tfds.load('cifar10', split='test').shuffle(42)


#minist dims

'''
training_images = np.empty((70000,28,28,1))
training_lables = np.empty((70000,10))
'''


#cifar10 dims

training_images = np.empty((60000,32,32,3))
training_lables = np.empty((60000,10))


place = 0
for pic in train_data:
    training_images[place] = pic['image']
    training_lables[place] = ecoc_matrix[pic['label'].numpy()]

    place += 1

for pic in test_data:
    training_images[place] = pic['image']
    training_lables[place] = ecoc_matrix[pic['label'].numpy()]

    place += 1

#np.savez('mnist_ecoc_data.npz', training_images=training_images, training_lables=training_lables)
np.save('cifar_traning_images.npy', training_images)
np.save('cifar_traning_lables.npy', training_lables)
