# Science News Article Generator

This project generates science news articles from inputs of reference texts. It utilizes natural language processing techniques and AI models to analyze reference texts, ask contextual questions, and output original science journalism.
## Features

- Summarizes input reference texts
- Creates an outline and plan for the article
- Asks contextual questions to find supplementary information
- Writes sections of the article
- Revises and edits drafts into a final article

## Frameworks / Models Used

- Langchain (LLM prompting)
- HuggingFace Transformers (summarization, embeddings)
- OpenAI API (GPT-3, ChatGPT)

## Contents

The project contains the following files:


- /autoscience
    - .env (secrets for API keys)  
    - prompts (text prompts for models)
    - sys_prompts (system prompts)
    - data
        - raw (input reference texts)
    - notebooks (Jupyter notebooks for experiments)
- /src
    - contentAgent.py (main orchestration)
    - utils.py (helper functions)
    - getArticles.py (web scraping)
    - parseArticles.py (extract text from HTML)
    - processText.py (preprocess and embed text)
    - generateArticle.py (initial GPT-3 implementation)

## Instructions

- Sign up for API keys (OpenAI, HuggingFace, etc)
- Add .env file with secrets
- Run pip install -r requirements.txt
- Run contentAgent.py with input text
