import numpy as np

class ecoc_tool(object):
    """a set of functions for operating on a ecoc matrix"""

    def __init__(self, ecoc_matrix=None):

        self.ecoc_matrix = ecoc_matrix
        self.code_word_length = len(ecoc_matrix[0])

    def convert_labels(self, labels):

        unique_labels = np.unique(labels)

        if(len(unique_labels) == self.code_word_length):
            maping = {str(unique_labels[0]) : self.ecoc_matrix[0]}
            pos = 0
            for label in unique_labels:
                maping[str(label)] = self.ecoc_matrix[pos]
                pos += 1

            ecoc_labels = np.empty((len(labels),self.code_word_length))

            pos = 0
            for label in labels:
                ecoc_labels[0] = maping[str(label[0])]
                pos += 1

            return ecoc_labels
