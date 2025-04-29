import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
import gensim.downloader
import html
import re
import pickle

class ImportantWords():
    def __init__(self, clusters_file):
        """
        Class will have following state:
            product_name_clusters: from Clusterer,
                map from product name to 2 lists: 1st list is list of words that appeared in ratings averaging <1.5 while
                2nd list is list of words that appeared in ratings averaging >4.5
            product_names: list of filtered/processed product names from clusterer. Note: each product name will be a document
            w2v_model: pre-trained Word2Vec model
            product_name_representation: map from product name to average of word2vec embeddings of embedding for each word vector in document/product name
        """
        nltk.download('stopwords')
        self.stop_words = stopwords.words('english')
        self.punctuations = """'"\\,<>./?@#$%^&*_~/!()-[]{};:"""
        with open(clusters_file, 'rb') as data_file:
            self.product_name_clusters = pickle.load(data_file)
            filtered_dataset = pickle.load(data_file)
            self.product_names = list(set(filtered_dataset['product name'].astype('str').to_list()))
        self.w2v_model = gensim.downloader.load('fasttext-wiki-news-subwords-300')
        self.product_name_representation = {}
        for product_name in self.product_names:
            embeddings = []
            for word in product_name.split():
                #Using bag of words pre-trained model, only use embeddings of words in the model
                #Since otherwise program will crash and cannot approximate words not in the pretrained model
                if word in self.w2v_model:
                    embeddings.append(self.w2v_model[word])
            if len(embeddings) > 0:
                all_embeddings = np.array(embeddings)
                self.product_name_representation[product_name] = all_embeddings
        with open("product_names.txt", "w") as file:
            for name in self.product_name_representation.keys():
                file.write(name+"\n")

    def process_text(self, line):
        line = html.unescape(line)
        line = re.sub(r"<.*?>", "", line) # remove html
        line = line.strip().lower() # remove excess whitespace and make lowercase
        line = line.translate(str.maketrans("", "", self.punctuations)) # remove punctuation
        #line = re.sub(r'\d+', '', line) # remove numbers
        words = [word for word in line.split() if word not in self.stop_words and word.isalpha()] # filter out stop words
        return ' '.join(words)

    def get_query_embeddings(self, query_text):
        embeddings = []
        for word in query_text.split():
            if word in self.w2v_model:
                embeddings.append(self.w2v_model[word])
        return np.array(embeddings)

    def search(self, query):
        processed_query = self.process_text(query)
        query_embedding = self.get_query_embeddings(processed_query)
        if len(query_embedding) == 0:
            return None, None
        relevances = np.zeros(len(self.product_names))
        for i in range(len(self.product_names)):
            product_name = self.product_names[i]
            if product_name in self.product_name_representation:
                product_name_emb = self.product_name_representation[product_name]
                query_word_doc_rel = query_embedding @ product_name_emb.T
                query_doc_rel = query_word_doc_rel.mean(axis=1)
                relevance = np.sum(query_doc_rel)
                relevances[i] = relevance
        indices_for_5_best_docs = np.argsort(relevances)[-5:]
        words_for_good_reviews = set()
        words_for_bad_reviews = set()
        for i in range(indices_for_5_best_docs.shape[0]):
            ind = indices_for_5_best_docs[i]
            product_name = self.product_names[ind]
            print("Relevant product name", product_name)
            bad_words, good_words = self.product_name_clusters[product_name]
            words_for_good_reviews.update(good_words)
            words_for_bad_reviews.update(bad_words)
        return words_for_good_reviews, words_for_bad_reviews
    
if __name__=="__main__":
    word_retriever = ImportantWords("clusterer_state.pkl")
    while True:
        query = input("Enter appliance product to get terms to look out for: ")
        good_terms, bad_terms = word_retriever.search(query)
        if good_terms is None:
            print("This query is invalid. The appliance product does not exist")
        else:
            input()
            print("Terms associated with a good product:")
            print(good_terms)
            print("Terms associated with a bad product:")
            print(bad_terms)