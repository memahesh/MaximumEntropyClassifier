from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
# Loads the 20newsgroup data
bunch = fetch_20newsgroups(subset='train', shuffle=True)
# Prints all category names
print(bunch.target_names)
# Get the data (content) of the first article
print(bunch.data[0])
# Count of the number of lines in the first article
print(len(bunch.data[0].split("\n")))
# Prints the first three lines of the first article
print(bunch.data[0].split("\n")[:9])
#
# Remember the above mentioned are lines (not sentences)
#
# The shape of the entire 20newsgroup dataset
print(bunch.filenames.shape)
#
#
# Loading articles of particular categories and also remove header, footer and quotes
cats = ['alt.atheism', 'comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.guns']
newsgroup_train = fetch_20newsgroups(subset='train', categories=cats, remove=('headers', 'footers', 'quotes'))
print(newsgroup_train.filenames.shape)
print(newsgroup_train.target_names)
article_1 = newsgroup_train.data[0]
print(article_1)
vectorizer = TfidfVectorizer()
vector_articles = vectorizer.fit_transform(newsgroup_train.data)
print(vector_articles)
print(vector_articles.shape)