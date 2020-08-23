import numpy as np
import os

class ecoc_tool(object):
    """a set of functions for operating on a ecoc lables"""

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

    def split_columns(self, lables, path):

        if not os.path.exists(path):
            os.mkdir(path)

        pos = 0
        while(pos < self.code_word_length):
            column = lables[:, pos]
            np.savetxt("{}/{}.bc".format(path, pos), column, fmt='%i')
            pos += 1

    def deterministic_dropout(self, number_of_samples, number_of_learners, path):
	
        p = number_of_samples/number_of_learners
        whole_range = range(0,number_of_samples)
        pos = 0
        while(pos < number_of_learners):
		
            index_array = np.array([x for x in whole_range if (x < p*pos or x > p*(pos + 1)) ])
            np.savetxt("{}/{}.dat".format(path,pos), index_array,fmt='%d' )

            pos += 1
		 
    def random_dropout(self, number_of_samples, number_of_learners, path):	
	
        whole_array = np.arange(0,number_of_samples, dtype=np.int)
        chunk_size = int(number_of_samples - number_of_samples/number_of_learners)
        pos = 0
        while(pos < number_of_learners):
            
            np.savetxt("{}/{}.dat".format(path, pos), np.sort(np.random.choice(whole_array, size=chunk_size, replace=False)), fmt='%d' )
            pos += 1  	
