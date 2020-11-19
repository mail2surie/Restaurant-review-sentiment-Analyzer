import pandas as pd
import pickle
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

data = pd.read_csv('Restaurant_Reviews.tsv',sep='\t')



#Creating corpus.
corpus = []
ps = PorterStemmer()
for i in range(0,len(data)):
    
    #removing non alphabets,converting to lower case and then creating a array.
    review = re.sub('[^a-zA-Z]',' ',data['Review'][i]).lower().split()
    
    #apply stemming for non stop words.
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    
    #join the list to re form review.
    review = ' '.join(review)
    
    #create the corpus
    corpus.append(review)
    
# Creating the Bag of Words model
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:,-1].values

# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv-transform.pkl', 'wb'))

# Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'restaurant-sentiment-mnb-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
