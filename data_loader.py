from glob import glob
import pandas as pd

anger_path = glob('Emotion Intensity Data\Anger\*')
fear_path = glob('Emotion Intensity Data\Fear\*')
joy_path = glob('Emotion Intensity Data\Joy\*')
sadness_path = glob('Emotion Intensity Data\Sadness\*')


def data_loader_func(file_paths: list):
    train_dataframe_0 = pd.read_csv(file_paths[0], sep='\t', header=None, names=['ID', 'Comment', 'Emotion', 'Intensity'])
    train_dataframe_1 = pd.read_csv(file_paths[2], sep='\t', header=None, names=['ID', 'Comment', 'Emotion', 'Intensity'])
    train_df = pd.concat([train_dataframe_1, train_dataframe_0]).reset_index(drop=True)
    test_df = pd.read_csv(file_paths[1], sep='\t', header=None, names=['ID', 'Comment', 'Emotion', 'Intensity'])
    return (train_df, test_df)


def angry_data_loader():
    angry_train_df, angry_test_df = data_loader_func(anger_path)
    return (angry_train_df, angry_test_df)


def fear_data_loader():
    fear_train_df, fear_test_df = data_loader_func(fear_path)
    return (fear_train_df, fear_test_df)


def joy_data_loader():
    joy_train_df, joy_test_df = data_loader_func(joy_path)
    return (joy_train_df, joy_test_df)


def sadness_data_loader():
    sadness_train_df, sadness_test_df = data_loader_func(sadness_path)
    return (sadness_train_df, sadness_test_df)
