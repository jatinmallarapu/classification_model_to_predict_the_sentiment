#Project Descripion
#Problem Statement: classification model to predict the sentiment either (1 or 0) based on Amazon Alexa reviews

#Context: This dataset consists of a nearly 3000 Amazon customer reviews (input text), star
#ratings, date of review, variant and feedback of various amazon Alexa products like Alexa
#Echo, Echo dots, Alexa Firesticks etc. for learning how to train Machine for sentiment analysis.

#Dataset:
#https://drive.google.com/file/d/1NL4bM3M1MR2WsvlyA3qlRPYFqLEN12a/view?usp=sharing

#1)Read the dataset
import numpy as np
import pandas as pd
data=pd.read_csv("amazon_alexa_data.csv")
data.head()
#2) Remove handle null values (if any).
data.isnull().sum()

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
#3) Preprocess the Amazon Alexa reviews based on the following parameter:
lemmatizer=WordNetLemmatizer()
corpus=[]
for i in range(0,3150):
 #c)Removing Punctuations
 review=re.sub('[^a-zA-Z]',' ',data['verified_reviews'][i])
 #b)Convert words to lower case
 review=review.lower()
 review=review.split()
 #e)Stemming or lemmatizing the words
 #d)Removing Stop words
 review=[lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
 review=' '.join(review)
 corpus.append(review)
#Creating a Bag of Words
#4) Transform the words into vectors using
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
X=tfidf.fit_transform(corpus).toarray()
y=data.iloc[:,4].values

data['corpus']=corpus
data

#Splitting the data into train and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random
_state=0)
#Fitting the model(using NaiveBayes)
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train,y_train)


##Predicting the values
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classificat
ion_report
cm=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test, y_pred)
cr=classification_report(y_test, y_pred)
print('Confusion Matrix:\t \n',cm)
print('Accuracy: ', accuracy)

#classification report
print(cr)

#Logistic Regressio
df=data
x=df[['rating']]
y=df.iloc[:,-1]
print(x.shape)
print(y.shape)

df=data
x=df[['rating']]
y=df.iloc[:,-1]
print(x.shape)
print(y.shape)

from sklearn.linear_model import LogisticRegression
m1=LogisticRegression()
m1.fit(x_train,y_train)
print('Model Training Score ',m1.score(x_train,y_train))
print('Model Testing Score ',m1.score(x_test,y_test))

ypred_m1=m1.predict(x_test)
print(ypred_m1)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,ypred_m1)
print("confusion matrix \n",cm)
print("Accuracy score ",accuracy_score(ypred_m1,y_test))

print("CLASSIFICATION REPORT:- \n",classification_report(y_test,ypred_m1))

import seaborn as sns
def eval_metrics(y_test, y_pred, plt_title):
print("Accuracy score ",accuracy_score(y_pred,y_test))
cm=confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title(plt_title)
plt.show()

eval_metrics(y_test,ypred_m1,"Logistic regression Confusion Matrix")

m=m1.coef_
c=m1.intercept_
print('Coefficient ',m)
print('Intercept ',c)

#KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3,leaf_size=25)
knn.fit(x_train, y_train)
y_pred_knn=knn.predict(x_test)
print('KNN Classifier Accuracy Score: ',accuracy_score(y_test,y_pred_kn
n))
cm_rfc=eval_metrics(y_test, y_pred_knn, 'KNN Confusion Matrix')

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions
def knn_comparison(data, k):
plt.figure(figsize=(5,5))
x = data[["rating"]].values
y = data["feedback"].astype(int).values
clf = neighbors.KNeighborsClassifier(n_neighbors=k)
clf.fit(x, y)
# Plotting decision region
plot_decision_regions(x, y, clf=clf, legend=2)
# Adding axes annotations
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Knn with K="+ str(k))
plt.show()
knn_comparison(df,100)

import numpy as np
def sig(x,m,c):
logit=1/(1+np.exp(1-(m*x+c)))
print(logit)
ypred_values=m1.predict([[6]])
print("The predicted value for the above features",ypred_values)

ypred_values1=m1.predict([[2]])
print("The prdicted value for the above features",ypred_values1)

#Accuracy scores for each model:-

print("Accuracy score for Linear Regression model is ",accuracy_score(ypred_m1,y_test)) #0.83756
print("Accuracy score for KNN Classifier is ",accuracy_score(y_pred_knn,y_test)) #1.0
print("Accuracy score for Multinomial Naive Bayes Classifier is ",accuracy) #0.03714

#Conclusion:
#Here we can see that the model which has best accuracy is the KNN Classifier, which can 
#predict the target value more accurately then the Linear regression model and Multinomial 
#Naïve Bayes Classifier.
#From this project we can learn about the data preprocessing which is very most important
#before we use the data for analysis.
