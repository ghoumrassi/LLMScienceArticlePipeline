import os
from dotenv import load_dotenv
import langchain
import openai
import re
from nltk import sent_tokenize

import trafilatura
from transformers import pipeline
import utils


def articleSummaryPrompt():
    with open(r"autoscience\prompts\summarize_text.txt", 'r') as f:
        template = f.read()

    prompt = langchain.PromptTemplate(
        input_variables=["reftext", "wordcount"],
        template=template
    )

    with open(r"autoscience\sys_prompts\create_summary.txt", 'r') as f:
        template = f.read()

    sysprompt = langchain.PromptTemplate(
        input_variables=[],
        template=template
    )
    summ_prompt = langchain.prompts.chat.ChatPromptTemplate.from_messages([
        langchain.prompts.chat.SystemMessagePromptTemplate(prompt=sysprompt),
        langchain.prompts.chat.HumanMessagePromptTemplate(prompt=prompt)
    ])
    summ_chain = langchain.chains.LLMChain(
        llm=langchain.chat_models.ChatOpenAI(temperature=0.9, max_tokens=1000),
        prompt=summ_prompt)
    return summ_chain


def articlePlanningPrompt():
    with open(r"autoscience\prompts\create_summary.txt", 'r') as f:
        template = f.read()

    prompt = langchain.PromptTemplate(
        input_variables=["reftext", "wordcount"],
        template=template
    )

    with open(r"autoscience\sys_prompts\create_summary.txt", 'r') as f:
        template = f.read()

    sysprompt = langchain.PromptTemplate(
        input_variables=[],
        template=template
    )
    plan_prompt = langchain.prompts.chat.ChatPromptTemplate.from_messages([
        langchain.prompts.chat.SystemMessagePromptTemplate(prompt=sysprompt),
        langchain.prompts.chat.HumanMessagePromptTemplate(prompt=prompt)
    ])
    plan_chain = langchain.chains.LLMChain(
        llm=langchain.chat_models.ChatOpenAI(temperature=0.9, max_tokens=1000),
        prompt=plan_prompt)
    return plan_chain


def finalDraftPrompt():
    with open(r"autoscience\prompts\final_draft_gpt4.txt", 'r') as f:
        template = f.read()

    prompt = langchain.PromptTemplate(
        input_variables=["reftext", "wordcount"],
        template=template
    )

    with open(r"autoscience\sys_prompts\create_summary.txt", 'r') as f:
        template = f.read()

    sysprompt = langchain.PromptTemplate(
        input_variables=[],
        template=template
    )
    plan_prompt = langchain.prompts.chat.ChatPromptTemplate.from_messages([
        langchain.prompts.chat.SystemMessagePromptTemplate(prompt=sysprompt),
        langchain.prompts.chat.HumanMessagePromptTemplate(prompt=prompt)
    ])
    plan_chain = langchain.chains.LLMChain(
        llm=langchain.chat_models.ChatOpenAI(
            model_name='gpt-4', temperature=0.9, max_tokens=1000),
        prompt=plan_prompt)
    return plan_chain


def askingQuestionsPrompt():
    with open(r"autoscience\prompts\asking_questions.txt", 'r') as f:
        template = f.read()

    hum_questions_prompt = langchain.PromptTemplate(
        input_variables=["reftext", "articleplan", "numquestions"],
        template=template
    )

    with open(r"autoscience\sys_prompts\create_summary.txt", 'r') as f:
        template = f.read()

    sys_questions_prompt = langchain.PromptTemplate(
        input_variables=[],
        template=template
    )
    questions_prompt = langchain.prompts.chat.ChatPromptTemplate.from_messages(
        [
            langchain.prompts.chat.SystemMessagePromptTemplate(
                prompt=sys_questions_prompt),
            langchain.prompts.chat.HumanMessagePromptTemplate(
                prompt=hum_questions_prompt)
        ]
    )
    questions_chain = langchain.chains.LLMChain(
        llm=langchain.chat_models.ChatOpenAI(temperature=0.9, max_tokens=1000),
        prompt=questions_prompt)
    return questions_chain


def sectionWritingPrompt():
    with open(r"autoscience\prompts\write_section.txt", 'r') as f:
        template = f.read()

    prompt = langchain.PromptTemplate(
        input_variables=[
            "reftext", "supplementary", "articleplan", "sectionstr"
        ],
        template=template
    )
    return prompt


def reviseDraftPrompt():
    with open(r"autoscience\prompts\revise_draft.txt", 'r') as f:
        template = f.read()

    prompt = langchain.PromptTemplate(
        input_variables=["articletext"],
        template=template
    )
    return prompt


def additionalInformationPrompt():
    pass


def getAdditionalInformationBing():
    pass


def getReleventParagraphs():
    pass


def getCleanedInputText(url):
    """
    - Clean the input (duplicate spaces etc.)
    - Separate first 250 words (closest section ending)
    - For [250: End], summarise to ~1500 words using HF.
    """
    downloaded = trafilatura.fetch_url(url)
    # outputs main content and comments as plain text
    result = trafilatura.extract(downloaded)

    return result


def lists_atmost_n(lists, n=1024):
    totlen = 0
    sublists = []
    sublist = []
    for ls in lists:
        totlen += len(ls)
        if totlen > n:
            sublists.append(sublist)
            sublist = []
            totlen = len(ls)
        sublist.append(ls)
    return sublists


def summarizeInput(text, lead=250, context_length=1024, preserve_quotes=True):
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0
    )
    summarizer.tokenizer.model_max_length = context_length
    summarizer.model.config.n_positions = context_length

    if preserve_quotes:
        len_quoted = len(summarizer.tokenizer.encode(text))
        text, quotes = utils.encodeQuotes(text)

    tokenized = summarizer.tokenizer.encode(text)
    tokens = len(tokenized)

    text = summarizer.tokenizer.decode(tokenized)

    if preserve_quotes:
        # TODO: Functionality for preserving quotations.
        maxlen = maxlen - (len_quoted - tokens)

    if tokens < context_length:
        return text

    sentences = sent_tokenize(text)
    token_sents = [summarizer.tokenizer.encode(sent) for sent in sentences]

    curr_token = 0
    for i, sent in enumerate(sentences):
        prev_token = curr_token
        curr_token += len(sent)
        if curr_token > lead:
            if (curr_token - lead) > (lead - prev_token):
                sent_lead_idx = i
            else:
                sent_lead_idx = i-1
            break

    lead_text = " ".join(
        [summarizer.tokenizer.decode(s) for s in token_sents[:sent_lead_idx]])

    to_summ_token_sents = token_sents[sent_lead_idx:]

    tokenized_lists = lists_atmost_n(to_summ_token_sents, n=context_length//2)

    summary = None
    for ls in tokenized_lists:
        print(summary)
        to_summarize = "".join([summarizer.tokenizer.decode(s) for s in ls])
        if summary:
            to_summarize = summary + "\n" + to_summarize
        num_tokens = len(summarizer.tokenizer.encode(to_summarize))
        # Min is min of:
        # 1. 80% of the input size
        # 2. 80% of the max size
        max_length = min(context_length//2, num_tokens)
        min_length = int(max_length*0.8)

        summary = summarizer(
            to_summarize,
            min_length=min_length,
            max_length=max_length
        )[0]['summary_text']

    final_summ = lead_text + summary

    if preserve_quotes:
        final_summ = utils.decodeQuotes(final_summ, quotes)

    return final_summ


if __name__ == "__main__":
    # Plan article
    plan_chain = articlePlanningPrompt()

    # Get additional sources
    # - Ask quesions
    questions_chain = askingQuestionsPrompt()

    # Summarise chain
    summ_chain = articleSummaryPrompt()

    # Writing sections prompt
    section_prompt = sectionWritingPrompt()
    section_chain = langchain.chains.LLMChain(
        llm=langchain.llms.OpenAI(temperature=0.9, max_tokens=1000),
        prompt=section_prompt)

    # # Writing draft prompt
    # draft_prompt = reviseDraftPrompt()
    # chain3 = langchain.chains.LLMChain(
    #     llm=langchain.llms.OpenAI(temperature=0.7, max_tokens=2000),
    #     prompt=draft_prompt)
    chain3 = finalDraftPrompt()

    text = getCleanedInputText(
        'https://news.mit.edu/2023/speeding-drug-discovery-with-diffusion-generative-models-diffdock-0331'
    )
    assert text

    summary = utils.lexSummary(text, first_sents=2, max_length=1500)
    # summary = summarizeInput(text, preserve_quotes=False)

    plan_output = plan_chain.predict(reftext=text, wordcount=1200)

    questions_output = questions_chain.predict(
        reftext=summary,
        articleplan=plan_output,
        numquestions=5)

    serp = langchain.SerpAPIWrapper()
    supplementary_text = []
    search_queries = re.findall(r"\d+\.\s(.*)", questions_output)
    for search in search_queries:
        # 1. Get search results for query
        organic_results = serp.results(search)['organic_results']
        for result in organic_results:
            search_url = result['link']
            if search_url.split('.')[-1] != 'pdf':
                break
        # 2. Get text of top page for search
        search_text = getCleanedInputText(search_url)
        if not search_text:
            continue
        # 3. Summarise with huggingface
        search_summary = utils.lexSummary(
            search_text, first_sents=2, max_length=1500)
        # search_summary = summarizeInput(search_text, lead=10)
        # 4. Summarise in regular language with Chat
        # TODO: make a custom prompt to ensure that the summary aligns with
        # the question asked in the search query
        search_summ = summ_chain.predict(reftext=search_summary, wordcount=100)
        supplementary_text.append(search_summ)

    bullets = re.findall(r"(\d)\..*\(\d* words\)", plan_output)
    section_strs = []
    for i in range(0, len(bullets), 2):
        if i == len(bullets)-2:
            section_strs.append(bullets[i])
        else:
            section_strs.append(
                " and ".join(bullets[i: i+2])
            )

    print(section_strs)
    sections = []
    for sectionstr in section_strs:
        sections.append(
            section_chain.predict(
                reftext=text,
                supplementary="\n".join(supplementary_text),
                articleplan=plan_output,
                sectionstr=sectionstr
            )
        )

    article_fulltext = "\n".join(sections)

    final_article = chain3.predict(reftext=article_fulltext, wordcount=800)

    print(final_article)
