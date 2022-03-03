import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#import nltk
#nltk.download('stopwords')


def process_text(s):
    s=s.strip().lower() # lower
    punctuation='!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    s=re.sub(r'[{}]+'.format(punctuation),' ',s) #remove puctuations
    stop_words = set(stopwords.words('english'))
    s=" ".join([word for word in str(s).split() if word not in stop_words]) #remove stop words
    stemmer = PorterStemmer()
    s=" ".join([stemmer.stem(word) for word in s.split()]) #Stemming
    s=re.sub(' +', ' ', s) #remove extra spaces
    return s