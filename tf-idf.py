import math
import string
from sklearn.datasets import fetch_20newsgroups

N_DOC = 10
BAG = 1000

def tokenize(document):
    document = document.replace("\n", " ")
    return document.lower().split(" ")

def sublinear_term_frequency(term, tokenized_document):
    # Counts the number of times the term is repeated in the document
    count = tokenized_document.count(term)
    # If the term is not present, return zero
    if count == 0:
        return 0
    return 1 + math.log(count)

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    # Get all unique tokens in the list of documents
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        # Count the number of documents in which this word occurs. (Helps us in getting relieved of stop words, articles and other unimportant stuff)
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        # Calculating the idf values as per formula
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    # t = sorted(idf, key=lambda x: idf[x], reverse=True)[:]
    # for x in t:
    #     print(x, idf[x])
    # exit(12345)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in sorted(idf, key=lambda x: idf[x], reverse=True)[:BAG]:
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents



def main():
    # Loading articles of particular categories and also remove header, footer and quotes
    cats = ['alt.atheism', 'comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.guns']
    newsgroup_train = fetch_20newsgroups(subset='train', categories=cats, remove=('headers', 'footers', 'quotes'), shuffle=True)
    print(tfidf(newsgroup_train.data[:N_DOC]))
    print(len(tfidf(newsgroup_train.data[:N_DOC])[0]))

if __name__ == '__main__':
    main()