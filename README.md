# Introduction

This is the Naive Bayes classiﬁer developed by python to classify the sentiment polarity of the tweets. The data are divided into `Train.tsv`, `Test.tsv` and `Valid.tsv` for train the model, testing and finally validation of the results respectively.

# Instruction

1. Clone the project and change the input path for tweet_train_data, tweet_test_data and tweet_valid_data should be changed for your owned

   `f = open(r'path')`

2. Output path for the Test_data sentiment label should also be changed. A total of 3217 lines named as "prediction.tsv" will be generated in print_labelled_data function.

   `f_out = open('C:'path', 'w')`

# Results

The the `Valid.tsv` are generated for classification accuracy which is a set of result using the training results model for the comapision.A total of 1377 lines file is also genereated with sentiment label for each tweets with the file named as Valid_Test.tsv.

The final results indicate that there are 515 correct label over total of 1377 labels. The overall accuracy therefore are 37.4%.
Positive_Prob_Train_Data : 0.37227278394215185
Negative_Prob_Train_Data : 0.14524373519511283
Neutral_Prob_Train_Data : 0.4824834808627353
Accuracy of model in valid data : 0.37400145243282495
