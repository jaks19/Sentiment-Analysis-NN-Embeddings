import numpy as np
import torch
import random

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

# Methods for Data Organization

# 1. Organize the Embedding Matrix and Word2Index Dict
'''
In ../word_vectors.txt, we have a word, a space, then space-separated floats until a newline character
The word's embedding is the vector next to it and we need to separate those unique words from their vectors and build:
- A dict of word to index
- A matrix of each of these vectors in order so that row i is the ith word's embedding vector
'''
def get_words_and_embeddings():
    filepath = "../word_vectors.txt"
    list_embeddings = open(filepath).readlines()
    words = []
    embedding_vectors = []
    for embedding_pair in list_embeddings:
        list_of_data = embedding_pair.split(" ")
        # Extract word
        words.append(list_of_data[0])
        # Remove space after it and remove newline char at the end
        embedding_vectors.append(list(float(i) for i in list_of_data[1:]))
    return words, embedding_vectors

def get_word_to_index_dict(words):
    dicti = {}
    for index, word in enumerate(words):
        dicti[word] = index
    return dicti


# 2. Organize the Test, Dev and Test ratings and comments
'''
Need to extract a list of reviews and corresponding ratings in two lists
Vectorize the reviews a list of tokens i.e. a list of indices corresponding to words in the sentence
Vocabulary for the indices is provided and is extracted from embeddings matrix
'''
def get_ratings_and_reviews(filename):
    filepath = "../data/stsa.binary." + filename
    list_reviews = open(filepath).readlines()
    list_comments = []
    list_ratings = []
    for review in list_reviews:
        # Extract rating, remove space after it and remove newline char at the end
        list_ratings.append(int(review[0]))
        list_comments.append(review[2:-1])
    return list_ratings, list_comments


def tokenize_from_vocabulary(sentence_list, vocabulary):
    all_sentences_tokenized = []
    for sentence in sentence_list:
        this_sentence_tokenized = []
        for word in sentence.split(" "):
            try: this_sentence_tokenized.append(vocabulary[word])
            except Exception: continue
        all_sentences_tokenized.append(this_sentence_tokenized)
    return all_sentences_tokenized


# Get data in ready-to-access-and-use form
''' 
Putting it all together:
1. Extract the embeddings matrix
2. Build the vocabulary corresponding to the matrix order
3. Extract ratings and comments
4. Tokenize comments based on above vocabulary
5. Make a comment be the mean of embeddings of all words in it
'''

def pad_sentences(sentences):
    longest = len(max(sentences, key=lambda sentence: len(sentence)))
    # Will pad so that can convert a Tensor of embedding vectors but when averaging,
    # will need to get rid of the extra vectors, so keep real
    real_lengths = []
    for i in range(len(sentences)):
        length = len(sentences[i])
        if length < longest:
            sentences[i] = sentences[i] + [0] * (longest - length)
    return sentences


def express_with_embeddings(sentences, embeddings):
    length_embedded_sentence = len(embeddings[0])
    embedded_sentences = []
    for sentence in sentences:
        embedded = np.zeros(length_embedded_sentence)
        if len(sentence) == 0:
            embedded_sentences.append(embedded)
            continue
        for index in sentence:
            embedded = np.add(embedded, embeddings[index])
        mean = np.divide(embedded, len(sentence))
        embedded_sentences.append(mean)
    return embedded_sentences


def prepare_all_data():
    words, embedding_vectors = get_words_and_embeddings()
    vocabulary = get_word_to_index_dict(words)
    data_set = {}
    data_set['embeddings'] = np.asarray(embedding_vectors)
    for filename in ['train', 'dev', 'test']:
        list_ratings, list_comments = get_ratings_and_reviews('train')
        tokenized_comments = tokenize_from_vocabulary(list_comments, vocabulary)
        embeddings_rich_comments = express_with_embeddings(tokenized_comments, embedding_vectors)
        data_set[filename] = [embeddings_rich_comments, list_ratings]
    return data_set


'''
Helper to organize training data in batches
'''
def get_batches(training_X, training_Y):
    assert len(training_X) == len(training_Y)
    c = list(zip(training_X, training_Y))
    random.shuffle(c)
    shuffled_X, shuffled_Y = zip(*c)

    batches = []
    batch_X_Y = [[],[]]
    for i in range (len(training_X)):
        batch_X_Y[0].append(shuffled_X[i])
        batch_X_Y[1].append(shuffled_Y[i])
        if i % 173 == 0 or i == len(training_X) - 1:
            batches.append(list(batch_X_Y))
            batch_X_Y = [[],[]]
    return batches