# naivebayesclassifier
A Naive Bayes classifier for categorizing hotel reviews as Postive/Negative and Truth/Deceptive. 

Language used: Python. 

No NLTK libraies have been used. The following feature selection techniques are used:

1. Stop word removal
2. Punctuation removal
3. Frequency counts - remove words having very low frequencies. Or words which have similar frequency in all classes.

Have used words as tokens for featurization. 

Files created:

nblearn.py will learn a naive Bayes model from the training data, and nbclassify.py will use the model to classify new (test) data. 

How to run the classifier:

> python nblearn.py /path/to/input

The argument is the directory of the training data; the program will learn a naive Bayes model, and write the model parameters to a file called nbmodel.txt. 

The classification program will be invoked in the following way:

> python nbclassify.py /path/to/input

The argument is the directory of the test data; the program will read the parameters of a naive Bayes model from the file nbmodel.txt, classify each file in the test data, and write the results to a text file called nboutput.txt in the following format:

label_a label_b path1
label_a label_b path2 
⋮

In the above format, label_a is either “truthful” or “deceptive”, label_b is either “positive” or “negative”, and pathn is the path of the text file being classified.

  
