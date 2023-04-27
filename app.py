import streamlit as st

###Model Code###
import joblib
import nltk #importing a stop words function from nltk
from nltk.corpus import stopwords
import string #importing a package of usefull strings. Will use string.punctuation for this

stemmer = nltk.stem.PorterStemmer()
stopwords = stopwords.words('english')


def my_tokenizer(sentence):

    #remove newline characters (Source:dylancastillo.co)
    sentence = " ".join(sentence.split())

    # remove punctuation and set to lower case
    for punctuation_mark in string.punctuation:
        sentence = sentence.replace(punctuation_mark,'').lower()

    # split sentence into words
    listofwords = sentence.split(' ')
    listofstemmed_words = []
    
    # remove stopwords and any tokens that are just empty strings
    for word in listofwords:
        if (not word in stopwords) and (word!=''):
            # Stem words
            stemmed_word = stemmer.stem(word)
            listofstemmed_words.append(stemmed_word)
    return listofstemmed_words


model = joblib.load('mlknn_streamlit.sav') #model file
TFIDF = joblib.load('TFIDF.sav') #tokenizer file
genrelist = joblib.load('genrelist.sav') #genre name file


#Header
st.title("This is Michael Renken's Book Description Genre Classifier!!")
###ask for input###
input_string = st.text_area("Input a book description here (feel free to write your own)",value='“Once upon a time, a very long time ago now, about last Friday, Winnie-the-Pooh lived in a forest all by himself under the name of Sanders.” Curl up with a true children’s classic by reading A.A.Milne’s Winnie-the-Pooh with iconic decorations by E.H.Shepard. Winnie-the-Pooh may be a bear of very little brain, but thanks to his friends Piglet, Eeyore and, of course, Christopher Robin, he’s never far from an adventure. In this much-loved classic story collection Pooh gets into a tight place, nearly catches a Woozle and heads off on an ‘expotition’ to the North Pole with the other animals.')
###run the model###
test = TFIDF.transform([input_string])

output_test = model.predict(test).nonzero()[1]
#translate these predictions from the genre indexes to the genre titles
answerlist = []
for i in output_test:
    answerlist.append(list(genrelist.keys())[i])

st.write('The model predicts that this book belongs to the below genres (ordered from most common genres in the training set):')
st.write(answerlist)
