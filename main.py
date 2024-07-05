from data_loader import angry_data_loader, fear_data_loader, joy_data_loader, sadness_data_loader
from statmodel import statmodels
from evaluation import evaluate
from nn_model import nn_models
import pandas as pd

angry_train_df, angry_test_df = angry_data_loader()
fear_train_df, fear_test_df = fear_data_loader()
joy_train_df, joy_test_df = joy_data_loader()
sadness_train_df, sadness_test_df = sadness_data_loader()

statmodels_obj = statmodels()
nn_models_obj = nn_models()
eval_obj = evaluate()
############################################################## STATSMODELS ##############################################################
print("Training OLS on Anger, Fear, Joy & Sadness training data", '\n')
angry_report = pd.DataFrame(statmodels_obj.model_report(angry_train_df.copy(), 'Anger'))
print("Angry Report\n", angry_report.T, "\n")

fear_report = pd.DataFrame(statmodels_obj.model_report(fear_train_df.copy(), 'Fear'))
print("Fear Report\n", fear_report.T, "\n")

joy_report = pd.DataFrame(statmodels_obj.model_report(joy_train_df.copy(), 'Joy'))
print("Joy Report\n", joy_report.T, "\n")

sadness_report = pd.DataFrame(statmodels_obj.model_report(sadness_train_df.copy(), 'Sadness'))
print("Sadness Report\n", sadness_report.T, "\n")

print("Training OLS Complete!")

print("\nTesting on Fear test data using OLS_Lemmatized_test_data_Glove model \n")

y_pred_ = statmodels_obj.statsmodel_test_prediction(fear_test_df.copy(),
                                                    'OLS',
                                                    'Lemmatize',
                                                    'GloVe',
                                                    'Fear')
result = eval_obj.eval(fear_test_df['Intensity'], y_pred_)

print(f"\nThe Pearson and Spearman correlations are: {result}\n")

############################################################## NN Models ##############################################################

print("\nBeginning Neural Network Training")
print("\nSimple RNN, LSTM, GRU models are used")

anger_nn_report = nn_models_obj.nn_model_report(angry_train_df.copy(), 'Anger')
print("RNN report for Anger Data\n", pd.DataFrame(anger_nn_report['RNN']))
print("LSTM report for Anger Data\n", pd.DataFrame(anger_nn_report['LSTM']))
print("GRU report for Anger Data\n", pd.DataFrame(anger_nn_report['GRU']))

fear_nn_report = nn_models_obj.nn_model_report(fear_train_df.copy(), 'Fear')
print("RNN report for Fear Data\n", pd.DataFrame(fear_nn_report['RNN']))
print("LSTM report for Fear Data\n", pd.DataFrame(fear_nn_report['LSTM']))
print("GRU report for Fear Data\n", pd.DataFrame(fear_nn_report['GRU']))

joy_nn_report = nn_models_obj.nn_model_report(joy_train_df.copy(), 'Joy')
print("RNN report for Joy Data\n", pd.DataFrame(joy_nn_report['RNN']))
print("LSTM report for Joy Data\n", pd.DataFrame(joy_nn_report['LSTM']))
print("GRU report for Joy Data\n", pd.DataFrame(joy_nn_report['GRU']))

sadness_nn_report = nn_models_obj.nn_model_report(sadness_train_df.copy(), 'Sadness')
print("RNN report for Sadness Data\n", pd.DataFrame(sadness_nn_report['RNN']))
print("LSTM report for Sadness Data\n", pd.DataFrame(sadness_nn_report['LSTM']))
print("GRU report for Sadness Data\n", pd.DataFrame(sadness_nn_report['GRU']))

print("\nTraining NN models complete!")

print("\n Commencing Test Prediction")
print("\nPlease enter the following options (Case-Sensitive):")
model_name = input("Which model would you like to use? Enter (RMM/LSTM/GRU): ")
data_type = input("Which data type would like to use? Enter (Stem/Lemmatize): ")
dataset_name = input("Which dataset would you like to test? Enter (Anger/Fear/Joy/Sadness): ")

if dataset_name == 'Anger':
    test_df = angry_test_df
elif dataset_name == "Fear":
    test_df = fear_test_df
elif dataset_name == "Joy":
    test_df = joy_test_df
elif dataset_name == "Sadness":
    test_df = sadness_test_df
else:
    raise (ValueError("Incorrect value!"))

y_pred = nn_models_obj.nn_models_test_prediction(f'models\deep_learning_models\{model_name}_{"Stemmed" if data_type == "Stem" else "Lemmatized"}_{dataset_name}.tf',
                                                 test_df.copy(),
                                                 data_type)
result = eval_obj.eval(test_df['Intensity'], y_pred.flatten())
print(f"The Pearson and Spearman correlations are: {result}")
