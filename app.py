# Test question:
# I am a patient suffering from a mental health condition. What considerations am I entitled to from emergency providers during a psychotic episode?

import os
import re
import time
import json
import openai
import requests
import pandas as pd
from collections import deque
from datetime import datetime
from utils import getResponse, text_summarize, memory_prompt, prompt_template
from utils import history_prompt, summary_prompt, conv_template
from utils import get_vectorstore
from flask import Flask, request, jsonify, make_response, render_template

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from langchain.prompts import PromptTemplate


N_docs=10
PATTERN = r'\[(.*?)\]\((.*?)\)'
DB_FAISS_PATH = 'indexer'
vector_store = get_vectorstore()
os.environ['TOKENIZERS_PARALLELISM']='true'

URL_HASH={
    "data\\aca.pdf":"[Affordable Care Act (ACA)](https://www.congress.gov/111/plaws/publ148/PLAW-111publ148.pdf)",
    "data\\ada.pdf":"[Americans with Disabilities Act (ADA)](https://archive.ada.gov/pubs/adastatute08.pdf)",
    "data\\emtala.pdf":"[Emergency Medical Treatment and Labor Act (EMTALA)](https://www.cms.gov/Regulations-and-Guidance/Legislation/EMTALA/downloads/emtala_chtr.pdf)",
    "data\\hipaa.pdf":"[Health Insurance Portability and Accountability Act (HIPAA)](https://www.govinfo.gov/content/pkg/PLAW-104publ191/pdf/PLAW-104publ191.pdf)"
}

print("setting up credentials with OpenAI's GPT")
# loading secrets locally:
with open('secrets.json') as f:
    secrets=json.load(f)
openai.api_key=secrets['openai']
os.environ["OPENAI_API_KEY"] = secrets['openai']
model = OpenAI(model_name="gpt-3.5-turbo-16k")


# loading stopwords:
with open('stopwords','r') as f:
    stopwords=f.read()
stopwords=set(stopwords.split('\n')) # stopwords from Baidu
# loading punctuations:
punctuation=' '.join(['"', "'", '\\', '{', '}', '<', '>', '[', ']', '(', ')', '*', '_', '-', ':', ';', '.', '!'])


# global variables
MAX_TOKENS=1024
MAX_TOKEN_LONG_MEMORY=512
MAX_TOKEN_SHORT_MEMORY=512
MAX_TOKEN_OTHERS=256
RETRY=5
LONG_MEMORY=[]
SHORT_MEMORY=deque(maxlen=1)


def process_chat_message(message, long_memory, short_memory):
    print('entering message chat')
    if len(long_memory) > 0:
        long_mem='\n'.join(long_memory)
        long_mem=text_summarize(long_mem, MAX_TOKEN_LONG_MEMORY / len(long_mem), stopwords, punctuation)
        long_mem='Summary of previous conversation: '+long_mem
    else:
        long_mem=''
    print(f'long memory: {long_mem}')
    
    if len(short_memory) > 0:
        short_mem=short_memory[0]
        # if len(short_mem) > MAX_TOKEN_SHORT_MEMORY:
        #     short_mem=text_summarize(short_mem, MAX_TOKEN_SHORT_MEMORY / len(short_mem), stopwords, punctuation)
        short_mem='\n'+short_mem
        short_mem+='\n'
    else:
        short_mem=''
    print(f'short memory: {short_mem}')

    previous_context=memory_prompt.format(
        short_memory=short_mem,
        long_memory=long_mem,
    )
    print(previous_context)

    print('setting up the prompt')
    PROMPT = PromptTemplate(
        template=prompt_template+previous_context+conv_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    print('setting up the qa chain')
    qa_chain = RetrievalQA.from_chain_type(
        llm=model, 
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={'k': N_docs}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents = True,
    )

    # TO-DO: customized implementation of a response letter
    print('ready to do qa chain')
    print('retrieving responses from Open AI GPT ...')
    response=qa_chain(
        {"query": message}
    )
    print(f'response received: {response}')
    # pdf=qa_chain()
    result, sources=(
        response['result'], 
        response["source_documents"]
    )
    print()
    print(f'sources: {sources}')
    source_outputs=[]
    print('parsing returned sources into links can be used later...')
    for _source in sources:
        v=URL_HASH[_source.metadata['source']]
        page_no=_source.metadata['page']
        match = re.search(PATTERN, v)
        text_inside_brackets = match.group(1)
        text_inside_parentheses = match.group(2)

        source_outputs.append('<a href="{}#page={}">{}: Page {}</a>'.format(
            text_inside_parentheses, page_no, text_inside_brackets, page_no
        ))
    sources='<br><br>'.join(source_outputs)

    return result, sources


if __name__ == '__main__':
    app=Flask(__name__)

    @app.route('/chat',methods=['POST'])
    def chat():
        global LONG_MEMORY
        global SHORT_MEMORY

        # TO-DO: set up initial message, and process first message always as user information
        message=request.json.get('message')
        print(f'message acquired as: {message}')
        if not message:
            return jsonify(
                {"error": "No message provided"}
            ), 400

        response, sources=process_chat_message(message, LONG_MEMORY, SHORT_MEMORY)
        history=history_prompt.format(question=message, response=response)

        LONG_MEMORY.append(history)
        SHORT_MEMORY.append(history)

        print(f'[INFO] length of chat history: {len(LONG_MEMORY)}, latest content: {SHORT_MEMORY[0]}')
        return jsonify(
            {
                "response": response,
                "sources": sources,
            }
        )

    @app.route('/')
    def home():
        return render_template('index.html') # what powered the html

    app.run(port=8888) # start the flask app