import streamlit as st

###Model Code###
import joblib
import nltk #importing a stop words function from nltk
import string #importing a package of usefull strings. Will use string.punctuation for this
from google.oauth2 import service_account
from google.cloud import storage

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

# Retrieve file contents.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data()
def read_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    content = bucket.blob(file_path).download_as_string().decode("utf-8")
    return content

bucket_name = "mr_streamlit_app"
file_path = "BR_streamlit.sav"

stemmer = nltk.stem.PorterStemmer()
st_words = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]


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
        if (not word in st_words) and (word!=''):
            # Stem words
            stemmed_word = stemmer.stem(word)
            listofstemmed_words.append(stemmed_word)
    return listofstemmed_words


#model = joblib.load('mlknn_streamlit.sav') #model file
model = read_file(bucket_name, file_path) #pulls from google cloud instead
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
