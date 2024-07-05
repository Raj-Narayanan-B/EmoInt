import statsmodels.api as sm
from sklearn.model_selection import train_test_split

import pickle

import pandas as pd
import numpy as np
from text_preprocessing import text_preprocess
from evaluation import evaluate
from gensim.models import KeyedVectors


word2vec_model = KeyedVectors.load(r'models\artifacts\word2vec.kv')
glove_model = KeyedVectors.load(r'models\artifacts\glove.kv')


# word2vec_model = downloader_api.load('word2vec-google-news-300')
# glove_model = downloader_api.load('glove-wiki-gigaword-100')
preprocessor_obj = text_preprocess()
eval_obj = evaluate()


class statmodels():
    def __init__(self) -> None:
        pass

    # common function for all the selected statsmodels
    def stats_models(self, X, y, model_name: str, data_name: str, df_data_name: str):
        X = sm.add_constant(X)
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

        # if model_name in 'OLS':
        #     model = sm.OLS(endog=y_train,
        #                 exog=x_train)
        # elif model_name in "GLS":
        #     model = sm.GLS(endog=y_train,
        #                 exog=x_train)
        # elif model_name == "WLS":
        #     model = sm.WLS(endog=y_train,
        #                 exog=x_train)
        # elif model_name == "GLM":
        #     model = sm.GLM(endog=y_train,
        #                    exog=x_train)

        model = sm.OLS(endog=y_train, exog=x_train)
        result = model.fit()
        y_pred = result.predict(x_test)
        result.save(f'models\statsmodels\{model_name}_{data_name}_{df_data_name}.pkl')  # saving the instance of model for later usage
        result = eval_obj.eval(y_test, y_pred)
        return result

    def stats_models_predict(self, X: pd.DataFrame, model,):
        X = sm.add_constant(X)
        y_pred = model.predict(X)
        return y_pred

    def model_report(self, df: pd.DataFrame, df_data_name_: str):
        processed_data = (stemmed_bow, lemmatized_bow, stemmed_tf_idf, lemmatized_tf_idf,
                          stemmed_word2vec, lemmatized_word2vec, stemmed_glove, lemmatized_glove) = preprocessor_obj.preprocess(df)
        data_names = ['Stem_BOW', 'Lemmatize_BOW', 'Stem_TF_IDF', 'Lemmatize_TF_IDF',
                      'Stem_Word2Vec', 'Lemmatize_Word2Vec', 'Stem_GloVe', 'Lemmatize_GloVe']
        stats_models_names = ['OLS']  # ,'GLS','WLS','GLM']
        stats_report = {}
        for model_name in stats_models_names:
            stats_report[model_name] = {}
            for i in range(len(processed_data)):
                stats_report[model_name][data_names[i]] = self.stats_models(processed_data[i], df['Intensity'],
                                                                            model_name=model_name, data_name=data_names[i],
                                                                            df_data_name=df_data_name_)
        return stats_report['OLS']

    def statsmodel_test_prediction(self, df: pd.DataFrame, model_name: str, stem_or_lemma: str,
                                   feature_extraction_name: str, data_name: str):
        with open(f'models\statsmodels\{model_name}_{stem_or_lemma}_{feature_extraction_name}_{data_name}.pkl', 'rb') as file:
            model_result = pickle.load(file)
        processed_df = preprocessor_obj.preprocess(df, return_df=True)
        if stem_or_lemma in 'Stem':
            if feature_extraction_name in 'BOW':
                vectorizer = preprocessor_obj.load(r'models\artifacts\stem_bow_vectorizer.pkl')
                stemmed_df = pd.DataFrame(vectorizer.transform(processed_df['Stemmed_text']).toarray())
            elif feature_extraction_name in 'TF_IDF':
                vectorizer = preprocessor_obj.load(r'models\artifacts\stem_tf_idf_vectorizer.pkl')
                stemmed_df = pd.DataFrame(vectorizer.transform(processed_df['Stemmed_text']).toarray())
            elif feature_extraction_name in 'Word2Vec':
                stemmed_df = pd.DataFrame(np.vstack(processed_df['Stemmed_text'].apply(lambda x: preprocessor_obj.vec_converter(sentence=x,
                                                                                                                                keyedvector=word2vec_model))))
            elif feature_extraction_name in 'GloVe':
                stemmed_df = pd.DataFrame(np.vstack(processed_df['Stemmed_text'].apply(lambda x: preprocessor_obj.vec_converter(sentence=x,
                                                                                                                                keyedvector=glove_model))))
            else:
                raise ValueError(f"Unknown feature extraction name: {feature_extraction_name}")
            y_pred = self.stats_models_predict(stemmed_df, model_result)

        elif stem_or_lemma in 'Lemmatize':
            if feature_extraction_name in 'BOW':
                vectorizer = preprocessor_obj.load(r'models\artifacts\lemmatize_bow_vectorizer.pkl')
                lemmatized_df = pd.DataFrame(vectorizer.transform(processed_df['Lemmatized_text']).toarray())
            elif feature_extraction_name in 'TF_IDF':
                vectorizer = preprocessor_obj.load(r'models\artifacts\lemmatize_tf_idf_vectorizer.pkl')
                lemmatized_df = pd.DataFrame(vectorizer.transform(processed_df['Lemmatized_text']).toarray())
            elif feature_extraction_name in 'Word2Vec':
                lemmatized_df = pd.DataFrame(np.vstack(processed_df['Lemmatized_text'].apply(lambda x: preprocessor_obj.vec_converter(sentence=x,
                                                                                                                                      keyedvector=word2vec_model))))
            elif feature_extraction_name in 'GloVe':
                lemmatized_df = pd.DataFrame(np.vstack(processed_df['Lemmatized_text'].apply(lambda x: preprocessor_obj.vec_converter(sentence=x,
                                                                                                                                      keyedvector=glove_model))))
            else:
                raise ValueError(f"Unknown feature extraction name: {feature_extraction_name}")
            y_pred = self.stats_models_predict(lemmatized_df, model_result)

        return (y_pred)
