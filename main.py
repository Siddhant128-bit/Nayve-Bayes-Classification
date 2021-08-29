import pandas as pd
import re

data_set=pd.read_csv('spam.csv')
data_set['Message']=data_set['Message'].str.replace('\W',' ')
data_set['Message']=data_set['Message'].str.lower()
data_set['Message']=data_set['Message'].str.split()
vocabulary=[]
for message in data_set['Message']: #go through each message in the list
    for word in message:
        vocabulary.append(word)
vocabulary=list(set(vocabulary)) #converting the list into set and converting it to list again so that repeated text will not be repeated

#this part is essentially creating a new dataframe which has words and their count respectively
words_count_per_message={unique_word: [0]*len(data_set['Message']) for unique_word in vocabulary }
for index,message in enumerate(data_set['Message']):
    for words in message:
        words_count_per_message[word][index]+=1
word_counts=pd.DataFrame(words_count_per_message)
#must rewatch the above code again its kind of confusing.

data_set_clean=pd.concat([data_set,word_counts],axis=1)
#print(data_set_clean)

#formula using  P(spam|w1,w2,w3......)=P(spam)*sum i from 1 to n (P(wi|spam)) and p(w/spam)=(Nw|spam +alpha)/(Nspam+alpha*Nvocabulary)

# Isolating spam and ham messages first
spam_messages = data_set_clean[data_set_clean['Label'] == 'spam']
ham_messages = data_set_clean[data_set_clean['Label'] == 'ham']
# P(spam) and P(ham)
p_spam = len(spam_messages) / len(data_set_clean)
p_ham = len(ham_messages) / len(data_set_clean)
# N_spam
n_words_per_spam_message = spam_messages['Message'].apply(len)
n_spam = n_words_per_spam_message.sum()
# N_ham
n_words_per_ham_message = ham_messages['Message'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1
# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
   n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
   parameters_spam[word] = p_word_given_spam

   n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
   p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
   parameters_ham[word] = p_word_given_ham
def test_model():
    while True:
        message=str(input('Enter the message in Roman Nepali: '))
        message=re.sub('\W',' ',message)
        if message=='exit':
            break
        else:
            message=message.lower().split()
            p_spam_given_message=p_spam
            p_ham_given_message=p_ham

            for word in message:
                if word in parameters_spam:
                    p_spam_given_message*=parameters_spam[word]
                if word in parameters_ham:
                    p_ham_given_message*=parameters_ham[word]
            print('P(spam|message): ',p_spam_given_message)
            print('P(ham|message): ',p_ham_given_message)

            if p_ham_given_message > p_spam_given_message:
              print('Label: ham')
            elif p_ham_given_message < p_spam_given_message:
              print('Label: spam')
            else:
              print('Equal proabilities, have a human classify this!')

test_model()
