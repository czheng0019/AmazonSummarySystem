import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
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
        try:
            nltk.find("corpora/stopwords")
        except:
            nltk.download('stopwords')
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download('wordnet')
        self.stop_words = stopwords.words('english')
        self.punctuations = """'"\\,<>./?@#$%^&*_~/!()-[]{};:"""
        with open(clusters_file, 'rb') as data_file:
            self.product_name_clusters = pickle.load(data_file)
            self.product_name_clusters = {str(prod_name_int):clusters for prod_name_int, clusters in self.product_name_clusters.items()}
            filtered_dataset = pickle.load(data_file)
            self.product_names = list(set(filtered_dataset['product name'].astype('str').to_list()))
        self.w2v_model = gensim.downloader.load('glove-wiki-gigaword-300')#('fasttext-wiki-news-subwords-300')
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
        self.nouns = {synset.name().split('.',1)[0] for synset in wordnet.all_synsets('n')}
        with open("product_names.txt", "w") as file:
            for name in self.product_name_representation.keys():
                file.write(name+"\n")
        #Explicitly filter out common relation terms since many reviews are my __ loved/hated/liked/etc this product
        self.common_words = set(["parent", "mother", "father", "brother", "sister", "wife", "husband", "boyfriend", "girlfriend", "son", "daughter", "niece", "nephew", "cousin", "uncle", "aunt", "grandpa", "grandma", "grandmother", "grandfather"])
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
        words_for_good_reviews = {}
        words_for_bad_reviews = {}
        for i in range(indices_for_5_best_docs.shape[0]):
            ind = indices_for_5_best_docs[i]
            product_name = self.product_names[ind]
            print("Relevant product name", product_name)
            bad_words, good_words = self.product_name_clusters[product_name]
            for i in range(len(good_words)):
                word, score = good_words[i]
                if word in self.w2v_model and word in self.nouns:
                    if score in words_for_good_reviews:
                        words_for_good_reviews[score].append(word)
                    else:
                        words_for_good_reviews[score] = [word]
            for i in range(len(bad_words)):
                word, score = bad_words[i]
                if word in self.w2v_model and word in self.nouns:
                    if score in words_for_bad_reviews:
                        words_for_bad_reviews[score].append(word)
                    else:
                        words_for_bad_reviews[score] = [word]
        final_good_words = []
        final_bad_words = []
        good_words_seen = set()
        for score in sorted(list(words_for_good_reviews.keys()), reverse=True):
            for word in words_for_good_reviews[score]:
                if word not in good_words_seen and word not in self.common_words and len(final_good_words) < 10:
                    final_good_words.append(word)
                    good_words_seen.add(word)
                elif len(final_good_words) == 10:
                    break
            if len(final_good_words) == 10:
                break
        bad_words_seen = set()
        for score in sorted(list(words_for_bad_reviews.keys()), reverse=True):
            for word in words_for_bad_reviews[score]:
                if word not in bad_words_seen and word not in self.common_words and len(final_bad_words) < 10:
                    final_bad_words.append(word)
                    bad_words_seen.add(word)
                elif len(final_bad_words) == 10:
                    break
            if len(final_bad_words) == 10:
                break
            
        return final_good_words, final_bad_words
    
if __name__=="__main__":
    word_retriever = ImportantWords("clusterer_state_1_2_4_8_score.pkl")
    while True:
        query = input("Enter appliance product to get terms to look out for(CTRL+C to exit program): ")
        good_terms, bad_terms = word_retriever.search(query)
        if good_terms is None:
            print("This query is invalid. The appliance product does not exist")
        else:
            input()
            print("Terms associated with a good product:")
            print(good_terms)
            print("Terms associated with a bad product:")
            print(bad_terms)