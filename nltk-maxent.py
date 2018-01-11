import nltk
import pickle
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Number of training documents
TR_DOC = None
# Number of testing documents
TE_DOC = None
# Number of words in Bag of Words
BAG = 100


def tokenize(text):
    # Tokenizing the document
    all_tokens = word_tokenize(text)
    # Lowercasing all the tokens
    all_tokens = [w.lower() for w in all_tokens]
    # Lemmatizing to root and meaningful words
    lemma = WordNetLemmatizer()
    all_tokens = map(lemma.lemmatize, all_tokens)
    # Preventing repitions by using set()
    all_tokens = set(all_tokens)
    return all_tokens

# Making a dictionary of the top_features for a document
def dict(tokens, document):
    dict = {}
    for feature in tokens:
        if feature in document[0]:
            dict[feature] = 1
        else:
            dict[feature] = 0
    return dict

# Organizing data for training and testing
def all_documents(format_data, format_labels):
    all_docs = [(word_tokenize(format_data[i]), format_labels[i]) for i in range(len(format_data))]
    return all_docs


def train_data(tokens, data, labels):
    all_docs = all_documents(data, labels)
    training_data = []
    for document in all_docs:
        # Getting the training data into correct format for nltk.MaxEntClassifier.train
        temp = tuple((dict(tokens, document), document[1]))
        training_data.append(temp)
    return training_data

def test_data(tokens, data, labels):
    all_docs = all_documents(data, labels)
    testing_data = []
    for document in all_docs:
        # Getting the training data into correct format for nltk.MaxEntClassifier.train.classify
        testing_data.append(dict(tokens, document))
    return testing_data

def test_maxent(algorithms, train, test, tlabels):
    print('a')
    classifier = nltk.MaxentClassifier.train(train, 'IIS', trace = 0, max_iter = 1000)
    # Loading a saved pickle
    # classifier_saved = open("maxent.pickle", "rb")
    # classifier = pickle.load(classifier_saved)
    # classifier_saved.close()
    # Saving a pickle
    save_classifier = open("maxent.pickle", "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()
    print('b')
    error = 0
    for featureset, tlabel in zip(test, tlabels):
        # Showing the probability for each label
        # pdist = classifier.prob_classify(featureset)
        # print('%8.2f%6.2f%6.2f%6.2f%6.2f ===> %6.2f' % (pdist.prob(0), pdist.prob(1), pdist.prob(2), pdist.prob(3), pdist.prob(4), tlabel),)
        # Counting errors
        if(classifier.classify(featureset)-tlabel !=0):
            error = error + 1
        # Predicted Label /\ Correct Label
        print('%8.2f /\ %6.2f'%(classifier.classify(featureset),tlabel))
    # Printing out accuracy
    print("Accuracy : %f" % (1-(error/float(len(tlabels))))*100)


def main():
    # Loading articles of particular categories and also remove header, footer
    cats = ['alt.atheism', 'comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.guns']
    # Loading the training documents
    newsgroup_train = fetch_20newsgroups(subset='train', categories=cats, remove=('headers', 'footers'), shuffle=True)
    # Loading the testing documents
    newsgroup_test = fetch_20newsgroups(subset='test', categories=cats, remove=('headers', 'footers'), shuffle=True)
    # tfidf Vectorizer
    tfidfs = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    print(-3)
    # print(newsgroup_train.target[:TR_DOC])
    # Fit the documents to the tfidf Vectorizer
    vector = tfidfs.fit_transform(newsgroup_train.data[:TR_DOC])
    print(-2)
    # Getting the sorted indices based on tfidf values
    indices = np.argsort(tfidfs.idf_)[::-1]
    print(-1)
    # Getting the top most 'BAG' number of features
    top_features = [tfidfs.get_feature_names()[i] for i in indices[:BAG]]
    print(0)
    # Getting the training data into usable format for nltk.MaxEntClassifier.train
    training_data = train_data(top_features, newsgroup_train.data[:TR_DOC], newsgroup_train.target[:TR_DOC])
    print(1)
    # Getting the testing data into usable format for nltk.MaxEntClassifier.train.classify
    testing_data = test_data(top_features, newsgroup_test.data[:TE_DOC], newsgroup_test.target[:TE_DOC])
    print(2)
    # Testing the values
    test_maxent(nltk.classify.MaxentClassifier.ALGORITHMS, training_data, testing_data, newsgroup_test.target[:TE_DOC])
    print(3)
    # print(len(testing_data))


if __name__ == '__main__':
    main()