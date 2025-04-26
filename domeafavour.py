import os
import sys

filename = 'data/processed_data_nouns.csv'
filestream = open(filename, 'r')
product_names = {}

MIN_CATEGORY_SIZE = 0.001
MAX_CATEGORY_SIZE = 0.01

for line in filestream:
    product_id, product_name = line.split(',')[2], line.split(',')[3]
    product_id, product_name = product_id.strip(), product_name.strip()
    product_names[product_id] = product_name


product_name_category = {}

document_frequency = {}
for product_name in product_names.values():
    for word in product_name.split():
        if not word in document_frequency:
            document_frequency[word] = 0
        document_frequency[word] += 1

def get_sub_df(subset: list[str], name_list: list[str]):
    sub_df = {}
    for product_name in name_list:
        words = product_name.split()
        if not subset is None and len(list(set(subset).intersection(words))) == 0:
            continue
        for word in words:
            if not subset is None and word in subset:
                continue
            if not word in sub_df:
                sub_df[word] = 0
            sub_df[word] += 1
    return sub_df

def prune_df(document_frequency, threshold):
    pruned_document_frequency = {}
    for word in document_frequency.keys():
        if document_frequency[word] >= threshold:
            pruned_document_frequency[word] = document_frequency[word]
    return pruned_document_frequency
        
def show_df(document_frequency):
    document_frequency_sorted = sorted(document_frequency.items(), key=lambda item: item[1], reverse=True)
    for rank, entry in enumerate(document_frequency_sorted[:100]):
        name, count = entry[0], entry[1]
        if count >= MAX_CATEGORY_SIZE * PRODUCT_NAME_SIZE:
            print(f"{name} ===")
            sub_df = prune_df(get_sub_df([name], product_names.values()), PRODUCT_NAME_SIZE * MIN_CATEGORY_SIZE)
            sorted_sub_df = sorted(sub_df.items(), key=lambda item: item[1], reverse=True)
            for sub_name, sub_count in sorted_sub_df[:5]:
                if sub_count >= MIN_CATEGORY_SIZE * PRODUCT_NAME_SIZE:
                    print(f"{sub_name} {sub_count}")
            print("===")
        else:
            print(f"{rank} {name} {count}")



PRODUCT_NAME_SIZE = len(product_names.values())



# print(len(pruned_document_frequency))
# print(prune_df(get_sub_df(["replacement"], product_names.values()),PRODUCT_NAME_SIZE * MIN_CATEGORY_SIZE))

# sort by descending order of frequence
show_df(prune_df(get_sub_df(None, product_names.values()),PRODUCT_NAME_SIZE * MIN_CATEGORY_SIZE))
