import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime


vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


# Read the data and append SENTENCE_START and SENTENCE_END tokens
def read_data(data_path):
    print("Reading CSV file...")
    with open(data_path, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)

        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])

        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed {} sentences.".format(len(sentences)))
    print(sentences[:2])
    return sentences


def preprocessing(sentences):

    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found {} unique words tokens.".format(len(word_freq.items())))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print("Using vocabulary size {}.".format(vocabulary_size))
    print("The least frequent word in our vocabulary is {} and appeared {} times.".format(vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print("Example sentence: {}".format(sentences[0]))
    print("Example sentence after Pre-processing: {}".format(tokenized_sentences[0]))
    return tokenized_sentences, index_to_word, word_to_index


def generate_train_data(tokenized_sentences, word_to_index):
    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    return X_train, y_train


def main():

    data_path = "data/reddit-comments-2015-08.csv"
    sentences = read_data(data_path)
    tokenized_sentences, index_to_word, word_to_index = preprocessing(sentences)

    X_train, y_train = generate_train_data(tokenized_sentences, word_to_index)

    # Print an training data example
    x_example, y_example = X_train[17], y_train[17]
    print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
    print("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))

if __name__ == "__main__":
    main()
