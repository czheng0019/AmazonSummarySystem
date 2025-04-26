import json
import nltk
import html
import swifter
import re
import pandas as pd
from nltk.corpus import stopwords
import requests
import gzip
import io

# fetching data sets
appliances_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz"
response = requests.get(appliances_url)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

with gzip.open(io.BytesIO(response.content), 'rt') as f:
    reviews_data = []
    for line in f:
        row = json.loads(line)
        reviews_data.append([row['rating'], row['title'] + ' ' + row['text'], row['parent_asin']])
reviews_df = pd.DataFrame(reviews_data, columns=["rating", 'text', 'parent_asin'])
reviews_df['parent_asin'] = reviews_df['parent_asin'].astype(str)
print(reviews_df.head(), reviews_df.dtypes)

meta_data = []
meta_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Appliances.jsonl.gz"
response = requests.get(meta_url)
with gzip.open(io.BytesIO(response.content), 'rt') as f:
    meta_data = []
    for line in f:
        row = json.loads(line)
        meta_data.append([row['title'], row['parent_asin']])
meta_df = pd.DataFrame(meta_data, columns=['product name', 'parent_asin'], dtype="str")
print(meta_df.head(), meta_df.dtypes)
combined_df = reviews_df.join(meta_df.set_index('parent_asin'), on="parent_asin", how='inner')

punctuations = "'\"\,<>./?@#$%^&*_~/!()-[]{};:"
stop_words = set(stopwords.words('english'))

def process_text(line):
    line = html.unescape(line)
    line = re.sub(r"<.*?>", "", line) # remove html
    line = line.strip().lower() # remove excess whitespace and make lowercase
    line = line.translate(str.maketrans("", "", punctuations)) # remove punctuation
    line = re.sub(r'\d+', '', line) # remove numbers
    words = [word for word in line.split() if word not in stop_words] # filter out stop words
    return ' '.join(words)    

combined_df['text'] = combined_df['text'].swifter.apply(process_text)
combined_df['product name'] = combined_df['product name'].swifter.apply(process_text)

is_noun = lambda pos: pos[:2] == 'NN'

def extract_nouns(line):
    tokenized = nltk.word_tokenize(line)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(nouns)



combined_df['product name'] = combined_df['product name'].swifter.apply(extract_nouns)
print(combined_df.head())
combined_df.to_csv("./data/processed_data_nouns.csv", index=False)
