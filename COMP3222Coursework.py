import numpy as np
import pandas as pd
import codecs
import nltk
import re
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

#Random Forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#SVM classification
from sklearn.svm import SVC
from sklearn import svm

#Naive Bayes
from sklearn.naive_bayes import GaussianNB

trainingPath = #insert path to the training set
testPath = #insert path to the test set

trainingSet = pd.read_csv(trainingPath, sep="\t")
testSet = pd.read_csv(testPath, sep="\t")

trainingDocs = []
testDocs = []
 
#"_values" refers to the "tweetText" column, and "_labels" to the "label" column
trainingSet_values = trainingSet.iloc[:, 1].values
trainingSet_labels = trainingSet.iloc[:, 6].values

testSet_values = testSet.iloc[:, 1].values
testSet_labels = testSet.iloc[:, 6].values    

#We treat 'humor' labels as 'fake'
for x in range(0, len(trainingSet_labels)):
    if (trainingSet_labels[x]=='humor'):
        trainingSet_labels[x] = 'fake'

        
stemmer = WordNetLemmatizer()

#Pre-processing data in both sets
for i in range(0, len(trainingSet)):
    trainingSet_values = trainingSet.iloc[:, 1].values
    
    #Turning everything to lowercase
    trainingDoc = trainingSet_values[i].lower()
    #Removing special characters
    trainingDoc = re.sub(r'\W', ' ', str(trainingSet_values[i]))
    #Turning multiple spaces into a single space
    trainingDoc = re.sub(r'\s+', ' ', trainingDoc, flags=re.I)
    #Removing all non-Latin characters
    #trainingDoc = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', u'', trainingDoc) 
    
    #Performing lemmatisation
    trainingDoc = trainingDoc.split()
    trainingDoc = [stemmer.lemmatize(word) for word in trainingDoc]
    trainingDoc = ' '.join(trainingDoc)
    
    trainingDocs.append(trainingDoc)

for i in range(0, len(testSet)):
    testSet_values = testSet.iloc[:, 1].values
    
    #Turning everything to lowercase
    testDoc = testSet_values[i].lower()
    #Removing special characters
    testDoc = re.sub(r'\W', ' ', str(testSet_values[i]))
    #Turning multiple spaces into a single space
    testDoc = re.sub(r'\s+', ' ', testDoc, flags=re.I)
    #Removing all non-Latin characters
    testDoc = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', u'', testDoc) 

    #Performing lemmatisation
    testDoc = testDoc.split()
    testDoc = [stemmer.lemmatize(word) for word in testDoc]
    testDoc = ' '.join(testDoc)
    
    testDocs.append(testDoc)

    
#Vectoriser for the BoW model
vectorizer = CountVectorizer(max_features=1000, min_df=1, max_df=0.8, stop_words=stopwords.words())

training = vectorizer.fit_transform(trainingDocs).toarray()
test = vectorizer.fit_transform(testDocs).toarray()

#TF-IDF transformer 
tfidfconverter = TfidfTransformer()#use_idf=False

training = tfidfconverter.fit_transform(training).toarray()
test = tfidfconverter.fit_transform(test).toarray()

#Training and running the Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=500, random_state=0, max_features=32)
classifier.fit(training, trainingSet_labels) 
prediction = classifier.predict(test)

#Printing score metrics
print("Confusion matrix:")
print(confusion_matrix(testSet_labels, prediction))
print()
print("Classification report:")
print(classification_report(testSet_labels, prediction))
accuracy = accuracy_score(testSet_labels, prediction)
print("Accuracy:", accuracy)