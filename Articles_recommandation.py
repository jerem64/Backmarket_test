import streamlit as st
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel


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


#Create the form
st.title('Articles Recomendation Engine')

st.header('Base Article')
article = st.text_area(label="Write or copy & paste your an article summary here.")
st.text("")

if st.button('Search similar'):
    nearest_texts = find_nearest_texts(article)
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

       
            

