from joblib import dump, load
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# nltk.download('gutenberg')
# nltk.download('stopwords')
# nltk.download('wordnet')



clf = load('medical_relevence_classifier.joblib')

#cleaning
def clean_text(text):
  tokenizer = nltk.RegexpTokenizer(r"\w+")
  tokenized_words = tokenizer.tokenize(text)
  tokenized_words = [token.lower() for token in tokenized_words]
  stop_words=set(stopwords.words("english"))
  filtered_words=[]
  for w in tokenized_words:
      if w not in stop_words:
          filtered_words.append(w)
  ps = PorterStemmer()
  wl=WordNetLemmatizer()
  stemmed=[]
  for w in filtered_words:
    st=ps.stem(w)
    stemmed.append(wl.lemmatize(st))
  return ' '.join(stemmed)

def predict_text(msg):
  clean_msg = clean_text(msg)
  pred = clf.predict([clean_msg])
  return pred[0]

# input: a sting that contains the message
# output : 1 if it's relevent to medical context and 0 otherwise
print(predict_text('i am bored'))
print(predict_text('i have pain'))
