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

appliances_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz" # fetching review data set
response = requests.get(appliances_url) # download review data

# download resources for tokenization, pos tagging, and stopword removal
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

# read and parse the review data
with gzip.open(io.BytesIO(response.content), 'rt') as f:
    reviews_data = []
    for line in f:
        row = json.loads(line)
        reviews_data.append([row['rating'], row['title'] + ' ' + row['text'], row['parent_asin']])

# convert the review data into a dataframe
reviews_df = pd.DataFrame(reviews_data, columns=["rating", 'text', 'parent_asin'])
reviews_df['parent_asin'] = reviews_df['parent_asin'].astype(str)
#print(reviews_df.head(), reviews_df.dtypes)

meta_data = []
meta_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Appliances.jsonl.gz" # fetching meta data set
response = requests.get(meta_url) # download meta data

# read and parse the meta data
with gzip.open(io.BytesIO(response.content), 'rt') as f:
    meta_data = []
    for line in f:
        row = json.loads(line)
        meta_data.append([row['title'], row['parent_asin']])

# convert the meta data into a dataframe
meta_df = pd.DataFrame(meta_data, columns=['product name', 'parent_asin'], dtype="str")
#print(meta_df.head(), meta_df.dtypes)
combined_df = reviews_df.join(meta_df.set_index('parent_asin'), on="parent_asin", how='inner')

punctuations = """'"\\,<>./?@#$%^&*_~/!()-[]{};:""" # punctuation to be removed
stop_words = set(stopwords.words('english')) # load stopwords

# function to process all text
def process_text(line):
    line = html.unescape(line)
    line = re.sub(r"<.*?>", "", line) # remove html
    line = line.strip().lower() # remove excess whitespace and make lowercase
    line = line.translate(str.maketrans("", "", punctuations)) # remove punctuation
    words = [word for word in line.split() if word not in stop_words and word.isalpha()] # filter out stop words
    return ' '.join(words)    

# clean process text for reviews and product name columns
combined_df['text'] = combined_df['text'].swifter.apply(process_text)
combined_df['product name'] = combined_df['product name'].swifter.apply(process_text)

is_noun = lambda pos: pos[:2] == 'NN' # to identify nouns

# function to only keep nouns in a string
def extract_nouns(line):
    tokenized = nltk.word_tokenize(line)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(nouns)

combined_df['product name'] = combined_df['product name'].swifter.apply(extract_nouns) # apply noun extraction to product name column
#print(combined_df.head())
combined_df.to_csv("./data/processed_data_nouns.csv", index=False) # save to csv
