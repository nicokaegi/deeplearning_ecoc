# deeplearning_ecoc

This repo contains the implemtation of an ensamble classifier based off error correcting output codes. 

how it works it that one can take a classification dataset like say image net and replace each class label with a binary string. 

these binary strings are then placed into a matrix nick named the code book.

you then take each column of the code book and train a binary classfied using whatever method that you prefer.

for preidction you then feed you input into your classifiers to get back a new binary string.

this binary string is then compared to strings present in the code book. 

the class assoisted to the string closest to your output string is considered the final predcited output class for the intial input image.

the Pros for this method are : 

  - When given a large enough number of classes. this type of classifier can give you some bounds on how bad the error can be thanks to the automatic error correction correction. 

  - The column classifiers can be trained in parallel

the Cons  for this method are : 

- when only given a few classes like the 10 in cfiar 10 the classifier only performs at the same level of a stadard model.
- one needs to be very careful in picking their output codes. all the rows and all the columns of your code book need to as distant from eachother as possible.



Written durring my time as an undergraduate reacearch assistant for Dr  hieu ngyuen of the rowan univeristy math department 
