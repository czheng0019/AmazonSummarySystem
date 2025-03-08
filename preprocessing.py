import json
import nltk
import html
import swifter
import re
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')

reviews_filename = "./data/Appliances.jsonl"
reviews_data = []
with open(reviews_filename, "r") as ratings:
    for line in ratings:
        row = json.loads(line)
        reviews_data.append([row['rating'], row['title'] + ' ' + row['text'], row['parent_asin']])
reviews_df = pd.DataFrame(reviews_data, columns=["rating", 'text', 'parent_asin'])
reviews_df['parent_asin'] = reviews_df['parent_asin'].astype(str)
print(reviews_df.head(), reviews_df.dtypes)
meta_data = []
meta_filename = "./data/meta_Appliances.jsonl"
with open(meta_filename) as meta_info:
    for line in meta_info:
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

print(combined_df.head())
combined_df.to_csv("./data/processed_data.csv", index=False)
