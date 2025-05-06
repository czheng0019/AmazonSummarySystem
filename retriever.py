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
    # download resources for stopword removal and wordnet to check for nouns
        try:
            nltk.find("corpora/stopwords")
        except:
            nltk.download('stopwords')
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download('wordnet')

        self.stop_words = stopwords.words('english') # load stopwords
        self.punctuations = """'"\\,<>./?@#$%^&*_~/!()-[]{};:""" # punctuation to be removed

        # load precomputed product clusters for good and bad words from the pickle file
        with open(clusters_file, 'rb') as data_file:
            self.product_name_clusters = pickle.load(data_file)
            self.product_name_clusters = {str(prod_name_int):clusters for prod_name_int, clusters in self.product_name_clusters.items()} # convert key to string
            filtered_dataset = pickle.load(data_file)
            self.product_names = list(set(filtered_dataset['product name'].astype('str').to_list())) # get unique product names

        self.w2v_model = gensim.downloader.load('glove-wiki-gigaword-300') # load pre-trained model

        # compute word2vec for each product name
        self.product_name_representation = {}
        for product_name in self.product_names:
            embeddings = []
            for word in product_name.split():
                # using bag of words pre-trained model, only use embeddings of words in the model
                # since otherwise program will crash and cannot approximate words not in the pretrained model
                if word in self.w2v_model:
                    embeddings.append(self.w2v_model[word])
            if len(embeddings) > 0:
                all_embeddings = np.array(embeddings)
                self.product_name_representation[product_name] = all_embeddings
        
        # store nouns for filtering
        self.nouns = {synset.name().split('.',1)[0] for synset in wordnet.all_synsets('n')}

        # save product names to a file
        with open("product_names.txt", "w") as file:
            for name in self.product_name_representation.keys():
                file.write(name+"\n")

        # explicitly filter out common relation terms since many reviews are my __ loved/hated/liked/etc this product
        self.common_words = set(["parent", "mother", "father", "brother", "sister", "wife", "husband", "boyfriend", "girlfriend", "son", "daughter", "niece", "nephew", "cousin", "uncle", "aunt", "grandpa", "grandma", "grandmother", "grandfather"])
    
    # function to process all text
    def process_text(self, line):
        line = html.unescape(line)
        line = re.sub(r"<.*?>", "", line) # remove html
        line = line.strip().lower() # remove excess whitespace and make lowercase
        line = line.translate(str.maketrans("", "", self.punctuations)) # remove punctuation
        #line = re.sub(r'\d+', '', line) # remove numbers
        words = [word for word in line.split() if word not in self.stop_words and word.isalpha()] # filter out stop words
        return ' '.join(words)

    # function to return word2vec embeddings for each word in the query
    def get_query_embeddings(self, query_text):
        embeddings = []
        for word in query_text.split():
            if word in self.w2v_model:
                embeddings.append(self.w2v_model[word])
        return np.array(embeddings)

    # function to find product names most relevant to the query and return the top 10 words for good and bad product characteristics
    def search(self, query):
        processed_query = self.process_text(query)
        query_embedding = self.get_query_embeddings(processed_query)
        if len(query_embedding) == 0:
            return None, None
        
        # initialize relevance scores
        relevances = np.zeros(len(self.product_names))
        for i in range(len(self.product_names)):
            product_name = self.product_names[i]
            if product_name in self.product_name_representation:
                product_name_emb = self.product_name_representation[product_name]
                query_word_doc_rel = query_embedding @ product_name_emb.T # for dot product similarity
                #Average over all document words
                query_doc_rel = query_word_doc_rel.mean(axis=1)
                #Sum over similarity for each word in query
                relevance = np.sum(query_doc_rel)
                relevances[i] = relevance

        # get the indices of the top 5 relevant product names
        indices_for_5_best_docs = np.argsort(relevances)[-5:]
        words_for_good_reviews = {}
        words_for_bad_reviews = {}

        # get the good and bad review words from the top 5 products
        for i in range(indices_for_5_best_docs.shape[0]):
            ind = indices_for_5_best_docs[i]
            product_name = self.product_names[ind]
            print("Relevant product name", product_name)
            bad_words, good_words = self.product_name_clusters[product_name]

            # good words with scores
            for i in range(len(good_words)):
                word, score = good_words[i]
                if word in self.w2v_model and word in self.nouns:
                    if score in words_for_good_reviews:
                        words_for_good_reviews[score].append(word)
                    else:
                        words_for_good_reviews[score] = [word]

            # bad words with scores
            for i in range(len(bad_words)):
                word, score = bad_words[i]
                if word in self.w2v_model and word in self.nouns:
                    if score in words_for_bad_reviews:
                        words_for_bad_reviews[score].append(word)
                    else:
                        words_for_bad_reviews[score] = [word]

        # find the top 10 good words and break when 10 are found
        final_good_words = []
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

        # find the top 10 bad words and break when 10 are found
        final_bad_words = []
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
    
# prompt for users to enter product names
if __name__=="__main__":
    word_retriever = ImportantWords("clusterer_state.pkl")
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