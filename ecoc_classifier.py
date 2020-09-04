from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from typing import List
import numpy as np

class ecoc_classifier(ClassifierMixin, BaseEstimator):

    def __init__(self, model_constructer=None, ecoc_matrix=None, model_list=None, code_word_length=0):
        '''
        the whole point of the is class is to train a ecoc model using sklearn, keras, or pytorch models (depending on if you implemented a fit function on your pytorch model)
        while also being a sklearn compatable estimator, meaning I can use sklearn functions like crossvaidation. (note this had not be tested with all of sklearns functions)

        there are two ways to use this class,
        one is when there is already a function for defining blank models, which you can then simply supply using the model consuctre paramater
        there other is when you want to supply a list of blank models your self which you can do with by using the model_list paramater

        make sure to supply one or the other not both.

        it is also nessary to supply a ecoc matrix, but unessary to supply a code length. (sklearn complains if any class variables don't have defualt values)

        '''
        self.model_constructer = model_constructer
        self.ecoc_matrix = ecoc_matrix
        self.model_list = model_list
        self.code_word_length = code_word_length

    def Hdistance(self, model_output : List , code_word : List ):# determins hamming distance

        '''
        counts the diffreance of bits between two code words
        '''

        distance = 0
        pos = 0
        while(pos < self.code_word_length):

            if( int(model_output[pos]) !=  code_word[pos] ):

                distance += 1

            pos += 1

        return distance

    def determinLable(self, results):

        '''
        when given an list of output codes from the models, this assigns a list
        of code words from the ecoc matrix which are the smallest hamming distance
        '''

        output = np.empty( ( results.shape[0], self.code_word_length ) )

        item = 0
        while(item < results.shape[0]):

            smallest_distance = -1

            for code_word in self.ecoc_matrix :

                distance = self.Hdistance(results[item], code_word)

                if( distance < smallest_distance or smallest_distance == -1):

                    smallest_distance = distance
                    output_code = code_word

            output[item] = np.array(output_code, copy=True)
            item += 1

        return output

    def fit(self, X, y, **kwargs):

        self.code_word_length = len(self.ecoc_matrix[0])

        '''
        a standerd implementation of fit used by all sklearn models

        in this case it initalzes a model for each column of the ecoc matrix,
        and then calls fit to train it on the bits of the column. after wards the
        model is append to a list for latter use
        '''

        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        if(self.model_list == None):

            self.model_list = []

            if(self.model_constructer != None):

                bit_pos = 0
                while(bit_pos < self.code_word_length):

                    self.model_list.append(self.model_constructer())
                    bit_pos += 1

            bit_pos = 0
            while(bit_pos < self.code_word_length):

                columnBits = y[:, bit_pos]
                self.model_list[bit_pos].fit(X, columnBits , **kwargs)
                bit_pos += 1

        # Return the classifier
        return self

    def predict(self, X, y=None):

        '''
        a standerd implementation of the predict function used by all sklearn models.

        here after checking if the data is vailid it is feed into each model of the list, and a new output code
        is made from the outputs which is then check against the ecoc matrix to see which row the new code word
        is closest to.
        '''

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        print('predicting')

        results = np.empty((self.code_word_length,) + (X.shape[0],) + (1,))

        pos = 0;
        for model in self.model_list:

            results[pos] = model.predict( X )
            pos += 1

        results = results.reshape((self.code_word_length,X.shape[0])).T.round()

        return self.determinLable(results)

    def score(self, X ,y):

        results = self.predict(X)
        right = 0

        pos = 0
        for sample in results:

            if (sample == y[pos]).all():

                right += 1


            pos += 1


        return right/X.shape[0]
