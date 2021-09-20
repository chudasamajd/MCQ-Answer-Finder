# pip install cdqa

import os, io
import errno
import urllib
import urllib.request
import hashlib
import re
import requests
from time import sleep
#from google.cloud import vision
#from google.cloud.vision import types
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
from ast import literal_eval
from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline
from cdqa.utils.converters import pdf_converter

import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline

# Download data and models
download_bnpp_data(dir='./data/bnpp_newsroom_v1.1/')
download_model(model='bert-squad_1.1', dir='./models')

# Loading data and filtering / preprocessing the documents
df = pd.read_csv('data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv', converters={'paragraphs': literal_eval})                       
df = filter_paragraphs(df)

# Loading QAPipeline with CPU version of BERT Reader pretrained on SQuAD 1.1                   
cdqa_pipeline = QAPipeline(reader='models/bert_qa.joblib')

# Fitting the retriever to the list of documents in the dataframe
cdqa_pipeline.fit_retriever(df)

# Sending a question to the pipeline and getting prediction
query = 'who is the founder of google'
prediction = cdqa_pipeline.predict(query)

print('query: {}\n'.format(query))
print('answer: {}\n'.format(prediction[0]))
print('title: {}\n'.format(prediction[1]))
print('paragraph: {}\n'.format(prediction[2]))

query = 'what is the capital of india'
prediction = cdqa_pipeline.predict(query)

print('query: {}\n'.format(query))
print('answer: {}\n'.format(prediction[0]))
print('title: {}\n'.format(prediction[1]))
print('paragraph: {}\n'.format(prediction[2]))

# from google.colab import drive 
# drive.mount('/content/gdrive')



question = ''
texts = ''
slugify_keyword = ''
def question(i):
  global question
  global texts
  global slugify_keyword
  if i == 'question1.jpg':
    question = "Which planet is closest to the sun?"
    texts = "Which planet is closest to the sun?"
    slugify_keyword = 'Which+planet+is+closest+to+the+sun'


  if i == 'question2.jpg':
    question = "Who is the prime minister of india?"
    texts = "Who is the prime minister of india?"
    slugify_keyword = 'Who+is+the+prime+minister+of+india'

  if i == 'question3.jpg':
    question = "What is the capital of Germany?"
    texts = "What is the capital of Germany?"
    slugify_keyword = 'What+is+the+capital+of+Germany'
    


img = "question1.jpg"

question(img)


#print(slugify_keyword)

result_urls = []


if '?' in texts:
    question = re.search('([^?]+)', texts).group(1)
    
elif ':' in texts:
    question = re.search('([^:]+)', texts).group(1)
    
elif '\n' in texts:
    question = re.search('([^\n]+)', texts).group(1)

slugify_keyword = urllib.parse.quote_plus(question)

#print(slugify_keyword)

print('Question : {}\n'.format(question))
print("------------------------Please Wait For Answer------------------------------")


def crawl_result_urls():
    req = Request('https://google.com/search?q=' + slugify_keyword, headers={'User-Agent': 'Mozilla/5.0'})                                
    html = urlopen(req).read()
    bs = BeautifulSoup(html, 'html.parser')
    results = bs.find_all('div', class_='ZINbbc')
    #print(results)
    try:
        for result in results:
          #print(result.find_all(href=True))
          for r in result.find_all(href=True):
            link = r.get('href')
            #print(link)
            if 'url' in link:
                result_urls.append(re.search('q=(.*)&sa', link).group(1))
    except (AttributeError, IndexError) as e:
        pass

def get_result_details(url):
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req).read()
        bs = BeautifulSoup(html, 'html.parser')
        try:
            title =  bs.find(re.compile('^h[1-6]$')).get_text().strip().replace('?', '').lower()

            #print(title)
            # Set your path to pdf directory
            filename =  "/content/gdrive/My Drive/pdfs/" + title + ".pdf"
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
            with open(filename, 'w') as f:
                for line in bs.find_all('p')[:5]:
                    f.write(line.text + '\n')
        except AttributeError:
            pass
    except urllib.error.HTTPError:
        pass

def find_answer():
    # Set your path to pdf directory
    df = pdf_converter(directory_path="/content/gdrive/My Drive/pdfs")
    cdqa_pipeline = QAPipeline(reader='models/bert_qa.joblib')
    cdqa_pipeline.fit_retriever(df)
    query = question + '?'
    prediction = cdqa_pipeline.predict(query)
    #print(prediction)

    #print('Question : {}\n'.format(query))
    #print("------------------------Please Wait For Answer------------------------------")
    #print('answer: {}\n'.format(prediction[0]))
    # print('title: {}\n'.format(prediction[1]))
    # print('paragraph: {}\n'.format(prediction[2]))
    return prediction[0]

crawl_result_urls()
#print(result_urls)

for url in result_urls[:3]:
    get_result_details(url)
    sleep(5)


answer = find_answer()
print('Answer: ' + answer)
