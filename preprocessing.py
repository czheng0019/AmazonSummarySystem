import json
import pandas as pd
reviews_filename = "./data/Appliances.jsonl"
reviews_data = []
with open(reviews_filename, "r") as ratings:
    for line in ratings:
        row = json.loads(line)
        reviews_data.append([row['rating'], row['title'], row['text'], row['parent_asin']])
reviews_df = pd.DataFrame(reviews_data, columns=["rating", 'title', 'text', 'parent_asin'])
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
print(combined_df.head())