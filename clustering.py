import pandas as pd 
import time
from nltk.stem import PorterStemmer

class ProductNames():
    def __init__(self, datafile):
        self.stemmer = PorterStemmer()
        data = pd.read_csv(datafile, header=0)
        unfiltered_product_names = pd.unique(data['product name']).astype(str)
        print(unfiltered_product_names.shape)
        word_to_freq = {}
        for product_name in unfiltered_product_names:
            for word in product_name.split():
                if len(word) > 2:
                    word_to_freq[word] = word_to_freq.get(word, 0)+1
        max_freq = max(word_to_freq.values())
        freq_to_words = [[] for _ in range(max_freq+1)]
        for word in word_to_freq.keys():
            freq_to_words[word_to_freq[word]].append(word)
        top_500 = []
        ind = max_freq
        while len(top_500) < 500 and ind > -1:
            top_500 = top_500 + freq_to_words[ind]
            ind-=1
        print("Stopped before this freq:",ind)
        top_500[:500]
        self.most_common_product_words = set(top_500)
        self.filtered_product_names = []
        unfiltered_to_filtered_mapping = {}
        for product_name in unfiltered_product_names:
            seen_words = set()
            filtered_words = []
            for word in product_name.split():
                if len(word) > 2 and word in self.most_common_product_words:
                    stem = self.stemmer.stem(word)
                    if stem not in seen_words:
                        filtered_words.append(word)
                        seen_words.add(stem)
            filtered_product_name = ' '.join(filtered_words)
            unfiltered_to_filtered_mapping[product_name] = filtered_product_name
            self.filtered_product_names.append(filtered_product_name)
        data["product name"] = data["product name"].map(unfiltered_to_filtered_mapping)
        self.filtered_dataset = data
if __name__ == "__main__":
    start_time = time.time()
    final_product_names = ProductNames("./data/processed_data_nouns.csv")
    end_time = time.time()
    print("time taken", end_time - start_time)
    print(final_product_names.filtered_product_names[:10])
    print(final_product_names.filtered_product_names[-10:])
    print(len(final_product_names.filtered_product_names), len(set(final_product_names.filtered_product_names)))
    print(final_product_names.filtered_dataset.head())
    # product_names = pd.unique(data['product_name'])
    # print(product_names.shape)
    # for product_name in product_names:
    #     relevant_reviews = data[data['product_name']==product_name]
    # print("done")
    # print(data.groupby(["product name"]).head())