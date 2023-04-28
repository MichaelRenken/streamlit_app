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
    input_string = st.text_area("Input a book description here: /n Feel free to use an example below, find a description on your favorite book vendor's website, or make up one on your own!", value='“Once upon a time, a very long time ago now, about last Friday, Winnie-the-Pooh lived in a forest all by himself under the name of Sanders.” Curl up with a true children’s classic by reading A.A.Milne’s Winnie-the-Pooh with iconic decorations by E.H.Shepard. Winnie-the-Pooh may be a bear of very little brain, but thanks to his friends Piglet, Eeyore and, of course, Christopher Robin, he’s never far from an adventure. In this much-loved classic story collection Pooh gets into a tight place, nearly catches a Woozle and heads off on an ‘expotition’ to the North Pole with the other animals.')
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
st.subheader('Here are some example descriptions you can use. Paste the below text into the textbox above')
st.markdown("**Stephen King: Fairy Tale**")
st.text("A #1 New York Times Bestseller and New York Times Book Review Editors' Choice!\
    Legendary storyteller Stephen King goes into the deepest well of his imagination in this spellbinding novel about a seventeen-year-old boy who inherits the keys to a parallel world\
     where good and evil are at war, and the stakes could not be higher—for that world or ours. Charlie Reade looks like a regular high school kid, great at baseball and football, a decent student.\
    But he carries a heavy load. His mom was killed in a hit-and-run accident when he was seven, and grief drove his dad to drink. Charlie learned how to take care of himself—and his dad.\
    When Charlie is seventeen, he meets a dog named Radar and her aging master, Howard Bowditch, a recluse in a big house at the top of a big hill, with a locked shed in the backyard.\
    Sometimes strange sounds emerge from it. Charlie starts doing jobs for Mr. Bowditch and loses his heart to Radar. Then, when Bowditch dies, he leaves Charlie a cassette tape telling\
    a story no one would believe. What Bowditch knows, and has kept secret all his long life, is that inside the shed is a portal to another world.\
    King’s storytelling in Fairy Tale soars. This is a magnificent and terrifying tale in which good is pitted against overwhelming evil, and a heroic boy—and his dog—must lead the battle.\
    Early in the Pandemic, King asked himself: “What could you write that would make you happy?” “As if my imagination had been waiting for the question to be asked,\
    I saw a vast deserted city—deserted but alive. I saw the empty streets, the haunted buildings, a gargoyle head lying overturned in the street.\
    I saw smashed statues (of what I didn’t know, but I eventually found out). I saw a huge, sprawling palace with glass towers so high their tips pierced the clouds.\
    Those images released the story I wanted to tell.”")
st.markdown('**Douglas Adams: The Hitchhiker\'s Guide to the Galaxy**')
st.text('It’s an ordinary Thursday morning for Arthur Dent . . . until his house gets demolished. The Earth follows shortly after to make way for a new hyperspace express route, and Arthur’s best friend has just announced that he’s an alien. After that, things get much, much worse. With just a towel, a small yellow fish, and a book, Arthur has to navigate through a very hostile universe in the company of a gang of unreliable aliens. Luckily the fish is quite good at languages. And the book is The Hitchhiker’s Guide to the Galaxy . . . which helpfully has the words DON’T PANIC inscribed in large, friendly letters on its cover. Douglas Adams’s mega-selling pop-culture classic sends logic into orbit, plays havoc with both time and physics, offers up pithy commentary on such things as ballpoint pens, potted plants, and digital watches . . . and, most important, reveals the ultimate answer to life, the universe, and everything. Now, if you could only figure out the question. . . .')
st.markdown('**The Merriam-Webster Dictionary**')
st.text('A revised and updated edition of the best-selling dictionary covering core vocabulary with over a hundred new entries and senses. More than 75,000 definitions and 8,000 usage examples aid understanding―and cover the words you need today Includes pronunciations, word origins, and synonym lists Features useful tables and special sections on Foreign Words & Phrases and Geographical Names')
