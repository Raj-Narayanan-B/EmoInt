import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.backend import clear_session  # type:ignore
from tensorflow.keras.models import load_model  # type:ignore
from tensorflow.keras import Sequential, Input  # type:ignore
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, TextVectorization, Embedding  # type:ignore
from text_preprocessing import text_preprocess
from evaluation import evaluate
preprocessor_obj = text_preprocess()
eval_obj = evaluate()
# word2vec_model = downloader_api.load('word2vec-google-news-300')
# glove_model = downloader_api.load('glove-wiki-gigaword-100')


class nn_models():
    def __init__(self) -> None:
        # The maximum length of a sentence that can be accepted by the model:
        # <<<< max(len(data) for data in train_df['Lemmatized_text']) = 127 >>>>#
        # Running the above code on Lemmatized data tells us that the maximum length of a sentence in lemmatized text is 127.
        # Hence we will round it to 130.
        self.sequence_length = 130
        self.nn_models = [SimpleRNN, LSTM, GRU]
        self.nn_model_names = ['RNN', 'LSTM', 'GRU']
        self.data_type_names = ['Stemmed', 'Lemmatized']

    def data_splitter(self, X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)
        return x_train, x_test, y_train, y_test

    def stem_data(self, df: pd.DataFrame):
        # split the data into train and test to avoid data leakage
        x_train_stemmed, x_test_stemmed, y_train_stemmed, y_test_stemmed = self.data_splitter(df['Stemmed_text'], df['Intensity'])

        # perform tokenization using TextVectorization
        text_vectorizer_stemmed = TextVectorization(output_sequence_length=self.sequence_length)
        text_vectorizer_stemmed.adapt(df['Stemmed_text'])
        # vocab_size = text_vectorizer_stemmed.vocabulary_size()

        # #Get the data in tensor format
        # x_train_stemmed = text_vectorizer_stemmed(x_train)
        # x_test_stemmed = text_vectorizer_stemmed(x_test)
        # y_train_stemmed = tf.convert_to_tensor(y_train)
        # y_test_stemmed = tf.convert_to_tensor(y_test)

        # save(text_vectorizer_stemmed, r"models\artifacts\text_vectorizer_stemmed")

        return (x_train_stemmed, x_test_stemmed, y_train_stemmed, y_test_stemmed, text_vectorizer_stemmed)

    def lemmatize_data(self, df: pd.DataFrame):
        # split the data into train and test to avoid data leakage
        x_train_lemmatized, x_test_lemmatized, y_train_lemmatized, y_test_lemmatized = self.data_splitter(df['Lemmatized_text'], df['Intensity'])

        # perform tokenization using TextVectorization
        text_vectorizer_lemmatized = TextVectorization(output_sequence_length=self.sequence_length)
        text_vectorizer_lemmatized.adapt(df['Lemmatized_text'])
        # vocab_size = text_vectorizer_lemmatized.vocabulary_size()

        # #Get the data in tensor format
        # x_train_lemmatized = text_vectorizer_lemmatized(x_train)
        # x_test_lemmatized = text_vectorizer_lemmatized(x_test)
        # y_train_lemmatized = tf.convert_to_tensor(y_train)
        # y_test_lemmatized = tf.convert_to_tensor(y_test)

        # save(text_vectorizer_lemmatized, r"models\artifacts\text_vectorizer_lemmatized")

        return (x_train_lemmatized, x_test_lemmatized, y_train_lemmatized, y_test_lemmatized, text_vectorizer_lemmatized)

    def model_builder(self, model_, text_vectorizer):
        model = Sequential()
        model.add(Input(shape=(1,), dtype=tf.string))
        model.add(text_vectorizer)
        model.add(Embedding(input_dim=text_vectorizer.vocabulary_size(),
                            output_dim=64,
                            input_length=self.sequence_length))
        model.add(model_(64, return_sequences=True))
        model.add(model_(32))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        return (model)

    def load_nn_model(self, file_path: str):
        return load_model(file_path)

    def nn_model_report(self, df: pd.DataFrame, df_data_name_: str):
        df = preprocessor_obj.preprocess(df, return_df=True)
        (x_train_stemmed, x_test_stemmed, y_train_stemmed, y_test_stemmed, text_vectorizer_stemmed) = self.stem_data(df)
        (x_train_lemmatized, x_test_lemmatized, y_train_lemmatized, y_test_lemmatized, text_vectorizer_lemmatized) = self.lemmatize_data(df)
        text_vectorizers = [text_vectorizer_stemmed, text_vectorizer_lemmatized]
        train_data = [[x_train_stemmed, y_train_stemmed], [x_train_lemmatized, y_train_lemmatized]]
        test_data = [[x_test_stemmed, y_test_stemmed], [x_test_lemmatized, y_test_lemmatized]]

        stats_report = {}
        for x in range(len(self.nn_models)):
            clear_session()
            model = self.model_builder(self.nn_models[x], text_vectorizers[0])
            stats_report[self.nn_model_names[x]] = {}
            for k, (i, j) in enumerate(zip(train_data, test_data)):
                print(model.summary(), '\n')
                model.fit(x=tf.convert_to_tensor(i[0]),
                          y=tf.convert_to_tensor(i[1]),
                          epochs=10,
                          validation_data=(tf.convert_to_tensor(j[0]), tf.convert_to_tensor(j[1])))
                model.save(f'models\deep_learning_models\{self.nn_model_names[x]}_{self.data_type_names[k]}_{df_data_name_}.tf')
                # rmse = pow(model.get_metrics_result()['loss'].numpy(),0.5)
                y_pred = model.predict(tf.convert_to_tensor(j[0]))
                result = eval_obj.eval(j[1], y_pred.flatten())
                stats_report[self.nn_model_names[x]][self.data_type_names[k]] = result
                clear_session()
                model = self.model_builder(self.nn_models[x], text_vectorizers[1])
        return stats_report

    def nn_models_test_prediction(self, model_filepath: str, df: pd.DataFrame, stem_or_lemma: str):
        df = preprocessor_obj.preprocess(df, return_df=True)
        model = self.load_nn_model(model_filepath)
        if stem_or_lemma == 'Stem':
            y_pred = model.predict(tf.convert_to_tensor(df['Stemmed_text']))
        elif stem_or_lemma == 'Lemmatize':
            y_pred = model.predict(tf.convert_to_tensor(df['Lemmatized_text']))
        else:
            raise ValueError(f"Unknown data name: {stem_or_lemma}")
        return y_pred


nn_models_obj = nn_models()
