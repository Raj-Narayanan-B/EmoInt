- To install the required libraries
    - python install -r requirements.txt

- To run the program
    - python main.py

- Once the program is running, the statistical model will execute first and then the neural network models will execute.

- The program trains the statistical model first on all the datas (Anger, Fear, Joy, Sadness) and then tests on Fear data

- Once the statitical model is done, the program proceeds to train the Neural Network on all the data again by stemming and lemmatizing them.

- The overall training of model (~55 models) takes around 15 to 20 minutes to complete depending on the hardware.

- Once the training of NN models are complete, the user is prompted to enter details about the test data - their choice will be taken into consideration
  for testing of NN nets.

- We are unable to check the pearson and spearman coefficients for Anger dataset because the test data has been not been given the golden values.

- As the models and artifacts size exceed 1GB, they are being added to gitignore.