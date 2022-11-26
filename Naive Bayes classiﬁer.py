import string

# Function for extracting significant words from tweets
def tweet_words_extract(tweets):
    words = []
    lower_alpha = string.ascii_lowercase
    upper_alpha = string.ascii_uppercase
    numbers = [str(n) for n in range(10)]
    for word in tweets:
        cur_word = ''
        for c in word:
            if (c not in lower_alpha) and (c not in upper_alpha) and (c not in numbers):
                if len(cur_word) >= 2:
                    words.append(cur_word.lower())
                cur_word = ''
                continue
            cur_word += c
        if len(cur_word) >= 2:
            words.append(cur_word.lower())
    return words

# Function for getting Training Data from the input file and seperated
def tweet_train_data():
    f = open(r"\Train.tsv")    # Input File link for the Train Data
    Train_data = []
    for line in f.readlines():
        line = line.strip()
        tweet_details = line.split()
        tweet_Id = tweet_details[0]
        user_ID = tweet_details[1]
        tweets_label = tweet_details[2]
        tweets = tweet_words_extract(tweet_details[3:])
        Train_data.append([tweet_Id, user_ID, tweets_label, tweets])
    f.close()

    return Train_data

# Function for getting Test Data from the input file and seperated
def tweet_test_data():
    f = open(r"\Test.tsv")     # Input File link for the Test Data
    Test_data = []
    for line in f.readlines():
        line = line.strip()
        tweet_details = line.split(' ')
        tweet_Id = tweet_details[0]
        user_ID = tweet_details[1]
        tweets = tweet_words_extract(tweet_details[2:])
        Test_data.append([tweet_Id, user_ID, '', tweets])

    f.close()

    return Test_data

# Get Valid Data from the input file and seperated
def tweet_valid_data():
    f = open(r"\Valid.tsv")    # Input File link for the Train Data
    Valid_data = []
    for line in f.readlines():
        line = line.strip()
        tweet_details = line.split()
        tweet_Id = tweet_details[0]
        user_ID = tweet_details[1]
        tweets_label = tweet_details[2]
        tweets = tweet_words_extract(tweet_details[3:])
        Valid_data.append([tweet_Id, user_ID, tweets_label, tweets])
    f.close()

    return Valid_data


# Function of getting list of words from the train data
def get_words(Train_data):
    words = []
    for data in Train_data:
        words.extend(data[3])
    return list(set(words))


# Function for find the probability of each word in the Train_data
def get_tweet_word_prob(Train_Data, sentiment_label=None):
    words = get_words(Train_data)
    freq = {}

    for word in words:
        freq[word] = 1

    total_count = 0
    for data in Train_data:
        if data[2] == sentiment_label or sentiment_label == None:
            total_count += len(data[3])
            for word in data[3]:
                freq[word] += 1

    prob = {}
    for word in freq.keys():
        prob[word] = freq[word]*1.0/total_count

    return prob

# Function for counting the probability of each label
def get_tweet_label_count(Train_data, sentiment_label):
    count = 0
    total_count = 0
    for data in Train_data:
        total_count += 1
        if data[2] == sentiment_label:
            count += 1
    return count*1.0/total_count

# Function for labeling the test data using Naive Bayes Model
def label_data(Test_data, positive_word_prob, negative_word_prob, neutral_word_prob, positive_prob, negative_prob, neutral_prob):
    labels = []
    for data in Test_data:
        data_prob_positive = positive_prob
        data_prob_negative = negative_prob
        data_prob_neutral = neutral_prob

        for word in data[3]:
            if word in positive_word_prob:
                data_prob_positive *= positive_word_prob[word]
                data_prob_negative *= negative_word_prob[word]
                data_prob_neutral *= neutral_word_prob[word]
            else:
                continue

        if data_prob_positive > data_prob_negative and data_prob_positive > data_prob_neutral:
            labels.append([data[0], 'postive', data_prob_positive, data_prob_negative, data_prob_neutral])
        elif data_prob_negative > data_prob_positive and data_prob_positive > data_prob_neutral:
            labels.append([data[0], 'negative', data_prob_positive, data_prob_negative, data_prob_neutral])
        else:
            labels.append([data[0], 'neutral', data_prob_positive, data_prob_negative, data_prob_neutral])

    return labels


# Function for printing the trained model results on the test data in file prediction.tsv
def print_labelled_data(labels):
    f_out = open('\prediction.tsv', 'w')
    for [tweet_id, label, prob_positive, prob_negative, prob_neutral] in labels:
        f_out.write('%s\n' % (label))

    f_out.close()


if __name__ == '__main__':  
    # Get the train, test and valid data
    Train_data = tweet_train_data()
    Test_data = tweet_test_data()
    Valid_data = tweet_valid_data()

    # Get the probabilities of each word for different labels
    word_prob = get_tweet_word_prob(Train_data)
    positive_word_prob = get_tweet_word_prob(Train_data, 'positive')
    negative_word_prob = get_tweet_word_prob(Train_data, 'negative')
    neutral_word_prob = get_tweet_word_prob(Train_data, 'neutral')

    # Get the probability of each label
    positive_prob = get_tweet_label_count(Train_data, 'positive')
    negative_prob = get_tweet_label_count(Train_data, 'negative')
    neutral_prob = get_tweet_label_count(Train_data, 'neutral')
    print('Positive_Prob_Train_Data : {}'.format(positive_prob))
    print('Negative_Prob_Train_Data : {}'.format(negative_prob))
    print('Neutral_Prob_Train_Data : {}'.format(neutral_prob))

    # Normalization of the data
    for (word, prob) in word_prob.items():
        positive_word_prob[word] /= prob
        negative_word_prob[word] /= prob
        neutral_word_prob[word] /= prob

    # Label the test data and print it
    test_labels = label_data(Test_data, positive_word_prob, negative_word_prob, neutral_word_prob, positive_prob, negative_prob, neutral_prob)
    print_labelled_data(test_labels)
    

    # Finding the accuracy for the Validation Data
    Valid_labels = label_data(Valid_data, positive_word_prob, negative_word_prob, neutral_word_prob, positive_prob, negative_prob, neutral_prob)
    f_out = open('C:\\Users\\p445c\\Desktop\\Assignment Data\\Q2\\Valid_Test.tsv', 'w')
    for [tweet_id, label, prob_positive, prob_negative, prob_neutral] in Valid_labels:
        f_out.write('%s\n' % (label))
    f_out.close()
    

def tweet_valid_test_data():
    f = open(r"\Valid_Test.tsv")
    Valid_test = []
    for line in f.readlines():
        line = line.strip()
        tweet_details = line.split()
        tweets_label = tweet_details[0]
        Valid_test.append([tweets_label])
    f.close()

    return Valid_test

Valid_results = tweet_valid_test_data()

count = 0
total_count = 0
for data in Valid_data:
    labels = []
    labels.append([data[2]])

    total_count += 1

    if labels == Valid_results:
        count += 1

print('Accuracy of model in valid data : {}'.format(count*1.0/total_count))


