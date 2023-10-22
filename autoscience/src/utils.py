import logging
import os
import re

import mysql.connector
from dotenv import load_dotenv

import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
from lexrank import degree_centrality_scores
from langchain.llms import OpenAI

from torch.cuda import is_available as cuda_available

load_dotenv("autoscience/.env")

def encodeQuotes(text):
    quotes = re.findall('“[^“”]*”', text)
    for i, quote in enumerate(quotes):
        text = text.replace(quote, f"%QUOTE_{i}")
    return text, quotes

def decodeQuotes(text, quotes):
    for i, quote in enumerate(quotes):
        text = text.replace(f"%QUOTE_{i}", quote)
    return text

def init_logging():
    logging.basicConfig(
        filename=os.environ['LOG_FILENAME'],
        format='%(asctime)s %(message)s')

def database_connection():
    cnx = mysql.connector.connect(
        user=os.environ["MYSQL_USERNAME"], 
        password=os.environ["MYSQL_PASSWORD"],
        host=os.environ["MYSQL_HOST"], 
        database=os.environ["MYSQL_DB_NAME"]
    )
    assert cnx.is_connected()
    return cnx


def getTextOrNull(soup_obj):
    try:
        return soup_obj.get_text()
    except AttributeError:
        return None



def lexSummary(document, first_sents=2, max_length=300):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    openai = OpenAI(temperature=0.9, max_tokens=2000)
    if openai.get_num_tokens(document) < max_length:
        return document

    #Split the document into sentences
    sentences = nltk.sent_tokenize(document)

    #Compute the sentence embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    #Compute the pair-wise cosine similarities
    cos_scores = util.cos_sim(embeddings, embeddings)
    if cuda_available():
        cos_scores = cos_scores.cpu().numpy()
    else:
        cos_scores = cos_scores.numpy()

    #Compute the centrality for each sentence
    centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

    #We argsort so that the first element is the sentence with the highest score
    most_central_sent_idx = np.argsort(-centrality_scores)

    # Remove the first n sentences from most_central_sentence_indices
    # These will be included in the summary regardless,
    most_central_sent_idx = most_central_sent_idx[
        ~np.isin(most_central_sent_idx, range(first_sents))]

    # Starting from the n-th sentence...
    num_tokens = 0
    num_sents = None
    for i in range(first_sents):
        num_tokens += openai.get_num_tokens(sentences[i])
    # ...get the number of sentences needed to fill the max summary length
    for eidx in most_central_sent_idx.tolist():
        num_tokens += openai.get_num_tokens(sentences[eidx])
        if num_tokens >= max_length:
            num_sents = i
            break

    # Get the most relevent sentences (ordered by appearance in the text)
    if num_sents:
        summ_sents = [
            sentences[i] for i in sorted(
                most_central_sent_idx[:num_sents]
            )
        ]
        summ_sents = sentences[:first_sents] + summ_sents
    else:
        summ_sents = sentences
    
    # Return final summary
    return "\n".join(summ_sents)


if __name__ == "__main__":
    cnx = database_connection()
    print(cnx.is_connected())