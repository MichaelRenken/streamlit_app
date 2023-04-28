#imports
import streamlit as st
import joblib
import nltk #importing a stop words function from nltk
import string #importing a package of usefull strings. Will use string.punctuation for this
from google.oauth2 import service_account
from google.cloud import storage
import io
import pandas as pd

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

bucket_name = "mr_streamlit_app"
file_path = "BR_streamlit.sav"

# Retrieve BR model contents.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_resource()
def read_file(bucket_name, file_path):
    bucket = client.get_bucket(bucket_name) #find gcs bucket
    content = bucket.blob(file_path) #get model from bucket as blob
    #turn model object into bytes to joblib back into correct form
    model_bytes = io.BytesIO() #create bytes object
    content.download_to_file(model_bytes)  #transfer the sav to the bytes object
    model_bytes.seek(0) #changes the position of the pointer
    model = joblib.load(model_bytes) #then use joblib to get this file.
    return model
    
model = read_file(bucket_name,file_path)

#define the tokenizer for the TFIDF
def my_tokenizer(sentence):
    stemmer = nltk.stem.PorterStemmer()
    #manually added stopwords as nltk was not working with streamlit
    st_words = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself', 'they', 'them', 'their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"]
    
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

#pull the TFIDF and genrelist from github
TFIDF = joblib.load('TFIDF.sav') #tokenizer file
genrelist = joblib.load('genrelist.sav') #genre name file

#-------------------APP CODE---------------------------------------
#Header
st.title("This is Michael Renken's Book Description Genre Classifier!!")
###ask for input###
with st.form(key='my_form'): #connect text input to button
    input_string = st.text_area("Input a book description here (feel free to write your own)",value='“Once upon a time, a very long time ago now, about last Friday, Winnie-the-Pooh lived in a forest all by himself under the name of Sanders.” Curl up with a true children’s classic by reading A.A.Milne’s Winnie-the-Pooh with iconic decorations by E.H.Shepard. Winnie-the-Pooh may be a bear of very little brain, but thanks to his friends Piglet, Eeyore and, of course, Christopher Robin, he’s never far from an adventure. In this much-loved classic story collection Pooh gets into a tight place, nearly catches a Woozle and heads off on an ‘expotition’ to the North Pole with the other animals.')
    submit_button = st.form_submit_button(label='Submit') #submit button so that moble users can use (otherwise have to hit ctrl+enter)
###run the model###
test = TFIDF.transform([input_string]) #run TFIDF on input
output_test = model.predict_proba(test)[0] #Get model probabilities
#translate these predictions from the genre indexes to the genre titles
final_output = pd.DataFrame(columns=['Confidence']) #create empty dataframe to store outputs
for i,v in enumerate(output_test.tolist()): #input all probabilities into dataframe >50% probability
    if v >.5:
        final_output.loc[list(genrelist.keys())[i]] = [v] #replace the genre # with a value and enter into dataframe
final_output.sort_values(by='Confidence',ascending=False,inplace=True) #sort dataframe from most to least likely

st.text('The model predicts that this book belongs to the below genres:') #text above output
st.dataframe(final_output) #output final dataframe
