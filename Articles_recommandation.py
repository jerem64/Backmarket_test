import streamlit as st
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import string
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


#prepare the necessary data
DIMENSION = 768

if "df" not in st.session_state:
    st.session_state.df = pd.read_csv("./dataset.csv")

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

if "model" not in st.session_state:
    st.session_state.model = AutoModel.from_pretrained("bert-base-uncased")

if "index" not in st.session_state:
    st.session_state.index = faiss.read_index("./bert_embeddings.index")




def bert_encoder(input):
  # Set the model to eval mode
  st.session_state.model.eval()

  # Encode with tokenizer Bert
  encoded_data = st.session_state.tokenizer.batch_encode_plus([input], add_special_tokens=True, max_length=DIMENSION, truncation=True, padding=True, return_attention_mask=True, return_tensors='pt')

  # Get the encoded inputs
  input_ids = encoded_data["input_ids"]
  attention_masks = encoded_data["attention_mask"]

  with torch.no_grad():
    model_output = st.session_state.model(input_ids, attention_mask=attention_masks)

  embeddings = model_output.last_hidden_state.mean(dim=1).numpy()
  
  # Set the model back to training mode
  st.session_state.model.train()

  return embeddings

def find_nearest_texts(input_text, num_neighbors=3):
  input_embeddings = bert_encoder(input_text)

  # Normalize the input embeddings
  faiss.normalize_L2(input_embeddings)

  # Perform FAISS similarity search
  _, indices = st.session_state.index.search(input_embeddings.reshape(1,DIMENSION), k=num_neighbors)

  # Retrieve nearest texts based on indices
  nearest_texts = st.session_state.df.loc[indices[0], 'summary'].tolist()

  return nearest_texts

def Nettoyer_HTML(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    return text

def Nettoyer_Majuscules(text):
    text = text.lower()
    return text

def Nettoyer_Contractions(text):
    # On remplace les contraction par des mots complets
    contraction_map = {"ain't": "am not", "aren't": "are not", "can't": "cannot", "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                       "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is", "i'd": "I would", "i'll": "I will", "i'm": "I am", "i've": "I have", "isn't": "is not", "it's": "it is", "let's": "let us", "mustn't": "must not",
                       "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have", "shouldn't": "should not", "that's": "that is", "there's": "there is", "they'd": "they would",
                       "they'll": "they will", "they're": "they are", "they've": "they have", "we'd": "we would", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is",
                       "what've": "what have", "where's": "where is", "who'd": "who would", "who'll": "who will", "who're": "who are", "who's": "who is", "who've": "who have", "won't": "will not", "would've": "would have", "wouldn't": "would not",
                       "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have", "'tis": "it is", "'twas": "it was", "'s": "is"}

    words = text.split()
    new_words = []
    for word in words:
        if word in contraction_map:
            new_words.append(contraction_map[word])
        else:
            new_words.append(word)
    text = " ".join(new_words)
    return text

def Nettoyer_Ponctuation(text):
    punctuations = string.punctuation.replace('#', '').replace('+', '').replace(".", "")
    text = " ".join(word.strip(punctuations) for word in text.split())
    return text

def Nettoyer_Stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words]
    text = ' '.join(filtered_tokens)
    return text

def Nettoyer_text(text):
    text = Nettoyer_HTML(text)
    text = Nettoyer_Majuscules(text)
    text = Nettoyer_Contractions(text)
    text = Nettoyer_Ponctuation(text)
    text = Nettoyer_Stop_words(text)
    return text




#Create the form
st.title('Articles Recomendation Engine') Get "http://localhost:8501/script-health-check": EOF

st.header('Base Article')
article = st.text_area(label="Write or copy & paste your an article summary here.")
st.text("")

if st.button('Search similar'):
    cleaned_article = Nettoyer_text(article)
    nearest_texts = find_nearest_texts(cleaned_article)
    for i, text in enumerate(nearest_texts, 1):
        st.header(i)
        st.text(text)

st.image('./resources/1.png', caption='Goal')
st.text("")
st.image('./resources/2.png', caption='ML Framing')
st.text("")
st.image('./resources/3.png', caption='Baseline Solution')
st.text("")
st.image('./resources/4.png', caption='Advanced solution')
st.text("")
st.image('./resources/5.png', caption='Host in the cloud, GCP')
st.text("")
st.image('./resources/6.png', caption='Host in the cloud, AWS')
st.text("")
st.image('./resources/7.png', caption='Host in the cloud at scale')