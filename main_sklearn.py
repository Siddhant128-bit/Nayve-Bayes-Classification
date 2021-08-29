import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection  import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords

enc = preprocessing.LabelEncoder()

dataset=pd.read_csv('spam.csv')
input_x=dataset['Message'].values
output_y=dataset['Label'].values

#enc.fit(input_x)
#input_x = enc.transform(input_x)

vectorizer=CountVectorizer()
input_x = vectorizer.fit_transform(input_x)

#input_x=input_x.reshape(-1,1)
#output_y=output_y.reshape(-1,1)
X_train,X_test,Y_train,Y_test=train_test_split(input_x,output_y)
nb=MultinomialNB()
nb.fit(X_train,Y_train)
pred=nb.predict(X_test)
print(accuracy_score(pred,Y_test))

def input_process(text):
    translator = str.maketrans('', '', string.punctuation)
    nopunc = text.translate(translator)
    words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(words)



Z=str(input('Enter the message to test: '))
new_x=vectorizer.transform([input_process(Z)])
#print(new_x)
print(nb.predict(new_x))
