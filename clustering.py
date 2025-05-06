import pandas as pd 
import time
from nltk.stem import PorterStemmer
import pickle

class Clusterer():
    def __init__(self, datafile):
        self.stemmer = PorterStemmer()  # initialize stemmer
        data = pd.read_csv(datafile, header=0) # read the dataset
        unfiltered_product_names = pd.unique(data['product name']).astype(str) # get unique product names as strings
        print(unfiltered_product_names.shape)

        # count word frequencies in product names for words while excluding words with 2 or less characters
        word_to_freq = {}
        for product_name in unfiltered_product_names:
            for word in product_name.split():
                if len(word) > 2:
                    word_to_freq[word] = word_to_freq.get(word, 0)+1

        max_freq = max(word_to_freq.values()) # find the maximum frequency

        # map the frequency to words
        freq_to_words = [[] for _ in range(max_freq+1)]
        for word in word_to_freq.keys():
            freq_to_words[word_to_freq[word]].append(word)

        # get the top 500 most frequent words and store them
        top_500 = []
        ind = max_freq
        while len(top_500) < 500 and ind > -1:
            top_500 = top_500 + freq_to_words[ind]
            ind-=1
        print("Stopped before this freq:",ind)
        top_500[:500]
        self.most_common_product_words = set(top_500)

        # filter the product names using the most common word
        self.filtered_product_names = []
        unfiltered_to_filtered_mapping = {}
        for product_name in unfiltered_product_names:
            seen_words = set()
            filtered_words = []
            for word in product_name.split():
                if len(word) > 2 and word in self.most_common_product_words:
                    stem = self.stemmer.stem(word)
                    if stem not in seen_words: # only add if the stemmed word of product name is not duplicated
                        filtered_words.append(word)
                        seen_words.add(stem)
            filtered_product_name = ' '.join(filtered_words)
            if filtered_product_name != "nan" and len(filtered_product_name) > 0:
                unfiltered_to_filtered_mapping[product_name] = filtered_product_name
                self.filtered_product_names.append(filtered_product_name)
            elif filtered_product_name == "nan":
                print(filtered_product_name)

        # replace product names in the dataset with the filtered ones
        data["product name"] = data["product name"].map(unfiltered_to_filtered_mapping)
        self.filtered_dataset = data

    def create_clusters(self):
        # identify unique product names after filtering
        unique_product_names = pd.unique(self.filtered_dataset['product name']).astype(str)
        data = self.filtered_dataset
        product_name_clusters = {}
        
        # iterate through each product
        for product_name in unique_product_names:
            word_to_scores = {} # map the total rating per word
            word_to_num_reviews = {} # map how many reviews a word appears in 

            # get all reviews given a product name
            reviews_for_product = data[data["product name"] == product_name]
            for index, row in reviews_for_product.iterrows():
                rating = row[0]
                text = row[1]
                if not isinstance(text, str): # skip invalid text entries
                    continue
                #Disallow words stars, five, one, star since these occur when reviews saying one star or five stars
                seen_words = set(["stars", "five", "one", "star"])
                #For each word, update running sum of ratings words has been a part of with new rating
                for word in text.split():
                    if word not in seen_words:
                        word_to_scores[word] = word_to_scores.get(word, 0)+rating
                        seen_words.add(word)
                        word_to_num_reviews[word] = word_to_num_reviews.get(word,0)+1

            # categorize words by average rating
            average_score_one = []
            average_score_five = []
            for word in word_to_scores:
                #Average score calculated by sum of ratings of all reviews word appeared in / number of reviews with word
                avg_score = word_to_scores[word]/word_to_num_reviews[word]
                #Words that appeared in at least 5 reviews and have an average score <1.5 indicate bad product
                if avg_score < 1.5 and word_to_num_reviews[word] > 4:
                    average_score_one.append((word, avg_score))
                #Words that appeared in at least 5 reviews and have an average score >4.8 indicate good product
                elif avg_score > 4.8 and word_to_num_reviews[word] > 4:
                    average_score_five.append((word,avg_score))
            # cluster each product name
            product_name_clusters[product_name] = [average_score_one, average_score_five]
        self.product_name_clusters = product_name_clusters
    
    # save the clusters
    def save_clusters(self, filename):
        with open(filename, 'w+b') as save_file:
            pickle.dump(self.product_name_clusters, save_file)
            pickle.dump(self.filtered_dataset, save_file)

if __name__ == "__main__":
    start_time = time.time()
    clusterer = Clusterer("./data/processed_data_nouns.csv") # instantiate and process the data set
    end_time = time.time()
    print("time taken to build vocab", end_time - start_time)
    print(clusterer.filtered_product_names[:10])
    print(clusterer.filtered_product_names[-10:])
    print(len(clusterer.filtered_product_names), len(set(clusterer.filtered_product_names)))
    print(clusterer.filtered_dataset.head())
    
    # start creating clusters and measure how long it will take
    start_time = time.time()
    clusterer.create_clusters()
    end_time = time.time()
    print("time taken to cluster", end_time - start_time)
    clusterer.save_clusters("clusterer_state.pkl") # save clustering result
