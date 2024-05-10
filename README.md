# KeyTitle
This is a repository with python code for the paper "KeyTitle: Towards Better Bug Report Title Generation by Keywords Planning".

## Required python packages
* pytorch
* transformers
* rouge
* spacy
* ...

## To train models
1. Modify the config.py file to suit your running environment. (path of pre-trained models, path of datasets, epoches, learning rate, batch size...)
2. You may need to use the python scripts in preprocess folder to create the suitable data files in csv.
3. Run the train_keywords.py file to train the model. Wait for the training process to finish.

## To evaluate the performance
1. Modify the config.py file. Change to your settings. (model saved path, data for evaluating...)
2. Run the eval.py file to evaluate your model. Wait for the evaluation process to finish.
