import pandas as pd
import numpy as np
import re
import emoji
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gensim.models import KeyedVectors


word2vec_model = KeyedVectors.load(r'models\artifacts\word2vec.kv')
glove_model = KeyedVectors.load(r'models\artifacts\glove.kv')


class text_preprocess():
    def __init__(self) -> None:
        pass

    def preprocess(self, df: pd.DataFrame, return_df: bool = False):
        stem_bow_vectorizer = CountVectorizer()
        lemmatize_bow_vectorizer = CountVectorizer()
        stemmed_tf_idf_vectorizer = TfidfVectorizer()
        lemmatized_tf_idf_vectorizer = TfidfVectorizer()

        df['Cleaned_text'] = df['Comment'].str.lower()                              # convert to lowercase
        df['Cleaned_text'] = df['Cleaned_text'].apply(self.remove_whitespace)       # removes whitespace charecters
        df['Cleaned_text'] = df['Cleaned_text'].apply(self.remove_special_chars)    # removes all other special charecters
        df['Cleaned_text'] = df['Cleaned_text'].apply(self.remove_stopwords)        # removes stopwords
        df['Cleaned_text'] = df['Cleaned_text'].apply(self.remove_emoji)            # removes emojis

        df['Stemmed_text'] = df['Cleaned_text'].apply(self.tokenize_stem)           # applies tokenization & stemming on cleaned text
        df['Lemmatized_text'] = df['Cleaned_text'].apply(self.tokenize_lemmatize)   # applies tokenization & lemmatization on cleaned text

        stemmed_bow = pd.DataFrame(stem_bow_vectorizer.fit_transform(df['Stemmed_text']).toarray())             # Stemmed data - BOW
        lemmatized_bow = pd.DataFrame(lemmatize_bow_vectorizer.fit_transform(df['Lemmatized_text']).toarray())  # Lemmatized data - BOW

        stemmed_tf_idf = pd.DataFrame(stemmed_tf_idf_vectorizer.fit_transform(df['Stemmed_text']).toarray())    # Stemmed data - TF/IDF
        lemmatized_tf_idf = pd.DataFrame(lemmatized_tf_idf_vectorizer.fit_transform(df['Lemmatized_text']).toarray())  # Lemmatized data - TF/IDF

        # load the word2vec model & vectorize the sentence
        stemmed_word2vec = pd.DataFrame(np.vstack(df['Stemmed_text'].apply(lambda x: self.vec_converter(sentence=x,
                                                                                                        keyedvector=word2vec_model))))
        lemmatized_word2vec = pd.DataFrame(np.vstack(df['Lemmatized_text'].apply(lambda x: self.vec_converter(sentence=x,
                                                                                                              keyedvector=word2vec_model))))

        # load the glove model & vectorize the sentence
        stemmed_glove = pd.DataFrame(np.vstack(df['Stemmed_text'].apply(lambda x: self.vec_converter(sentence=x,
                                                                                                     keyedvector=glove_model))))
        lemmatized_glove = pd.DataFrame(np.vstack(df['Lemmatized_text'].apply(lambda x: self.vec_converter(sentence=x,
                                                                                                           keyedvector=glove_model))))

        if return_df is True:
            return (df)
        else:
            self.save(stem_bow_vectorizer, r'models\artifacts\stem_bow_vectorizer.pkl')
            self.save(lemmatize_bow_vectorizer, r'models\artifacts\lemmatize_bow_vectorizer.pkl')
            self.save(stemmed_tf_idf_vectorizer, r'models\artifacts\stem_tf_idf_vectorizer.pkl')
            self.save(lemmatized_tf_idf_vectorizer, r'models\artifacts\lemmatize_tf_idf_vectorizer.pkl')
            return ((stemmed_bow, lemmatized_bow, stemmed_tf_idf, lemmatized_tf_idf,
                     stemmed_word2vec, lemmatized_word2vec, stemmed_glove, lemmatized_glove))

    def remove_whitespace(self, text: str):
        pattern = r'\\[tnr\x0b\x0c]'
        text = re.sub(pattern, ' ', text)
        return text

    def remove_special_chars(self, text: str):
        text = re.sub(pattern="[^a-zA-Z0-9]",
                      repl=" ",
                      string=text)
        text = re.sub(pattern="\s+",
                      repl=" ",
                      string=text)
        return text

    def remove_stopwords(self, text: str):
        stopwords_ = stopwords.words('english')
        stopwords_.extend(["i'm", "im", "u"])
        return (" ".join([word for word in text.split() if word not in stopwords_]))

    def remove_emoji(self, text: str):
        return (emoji.replace_emoji(text, ""))

    def tokenize_stem(self, sentence: str):
        stemmer = PorterStemmer()
        tokenized_sentence = word_tokenize(sentence)
        stemmed_tokens = [stemmer.stem(word) for word in tokenized_sentence]
        stemmed_sentence = " ".join(stemmed_tokens)
        return stemmed_sentence

    def tokenize_lemmatize(self, sentence: str):
        lemmatizer = WordNetLemmatizer()
        tokenized_sentence = word_tokenize(sentence)
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokenized_sentence]
        lemmatized_sentence = " ".join(lemmatized_tokens)
        return lemmatized_sentence

    def vec_converter(self, sentence: str, keyedvector):
        tokenized_sentence = word_tokenize(sentence)
        vector_token = [keyedvector[word] for word in tokenized_sentence if word in keyedvector]  # this is where we convert the tokens into vectors
        vector = np.mean(vector_token, axis=0) if vector_token else np.zeros(keyedvector.vector_size)
        return vector

    def save(self, object, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(object, file)

    def load(self, filepath):
        with open(filepath, 'rb') as file:
            object = pickle.load(file)
        return (object)
