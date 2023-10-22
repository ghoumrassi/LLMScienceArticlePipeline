import os
from pathlib import Path
from dotenv import load_dotenv 
import openai
from transformers import pipeline
import numpy as np
from torch import tensor

from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from transformers import AutoTokenizer, AutoModel

import utils

load_dotenv("autoscience\.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
prompts = Path("autoscience/prompts")

# def createSummary(text, title, chunk_size=300, num_words=100):
#     with open(prompts / "summarise.txt", "r") as f:
#         sum_prompt = f.read()
    
#     raise NotImplementedError()

def BARTSummary(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    text, quotes = utils.encodeQuotes(text)
    min_len = int(len(text.split()) * 0.8)  # Neet bette
    max_len = int(len(text.split()) * 1.2)
    summary = summarizer(
            text, 
            min_length=min_len, 
            max_length=max_len
        )[0]['summary_text']
    summary = utils.decodeQuotes(summary, quotes)
    return summary

def createEmbedding(text):
    raise NotImplementedError()

def createTopics(text):
    raise NotImplementedError()

def getReleventText(embedding, n=10):
    raise NotImplementedError()

def splitTextToSegments(doc_text, n_clusters=4):
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize document into paragraphs
    para_text = doc_text.split("\n")
    para_text = [p for p in para_text if len(p.strip()) > 0]

    # Encode paragraphs as embeddings
    para_tokens = [tokenizer.encode(p, add_special_tokens=True) for p in para_text]
    para_tensors = [tensor([p]) for p in para_tokens]
    para_embs = [model(p)[1].detach().numpy() for p in para_tensors]

    # Compute pairwise distances between paragraph embeddings
    dist_matrix = np.zeros((len(para_embs), len(para_embs)))
    for i in range(len(para_embs)):
        for j in range(i, len(para_embs)):
            dist = dtw(para_embs[i], para_embs[j], global_constraint="sakoe_chiba", sakoe_chiba_radius=1)
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # Cluster paragraphs using K-means clustering
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=100)
    labels = kmeans.fit_predict(dist_matrix)

    # Find section boundaries based on clusters
    boundaries = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            boundaries.append(i)
    boundaries.append(len(labels))

    # Return segmented document
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        segment_text = "\n".join(para_text[start:end])
        segments.append(segment_text)
    return segments


if __name__ == "__main__":
    with open("autoscience/data/raw/romanconcrete_1.txt", 'r', encoding='utf8') as f:
        text = f.read()
    # print(BARTSummary(text))
    [print(out + "\n\n") for out in splitTextToSegments(text)]