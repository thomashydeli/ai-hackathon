import json
import spacy
from heapq import nlargest
import requests
import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

DB_FAISS_PATH = 'indexer'

def getResponse(
    prompt,
    model,
    max_tokens,
    temperature=0.5,
):
    _response=requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        },
        data=json.dumps(
            {
                "model":model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens":max_tokens,
                "n":1
            }
        ),
    )
    response=_response.json()['choices'][0]['message']['content'].lstrip().rstrip()
    print(f'GPT response acquired: {response}')
    return response


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings)
    return vector_store


def text_summarize(text, per, stopwords, punctuation): # function for summarization
    nlp = spacy.load('en_core_web_sm') # analyzer using spacy
    doc= nlp(text)
    tokens=[token.text for token in doc] # tokenization
    word_frequencies={}
    for word in doc: # stopwords removal and frequency calculation
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys(): # achieve ratio of term frequency
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens: # acquire document scores
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get) # ranking the content for summarization based on sentence scoring and token length
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary


memory_prompt="""{long_memory}{short_memory}"""

history_prompt="""
User: {question}
Bot: {response}
"""

summary_prompt="""
Please summarize the following content using less than {characters} words：
{dialogue}"""

letter_prompt="""
"""