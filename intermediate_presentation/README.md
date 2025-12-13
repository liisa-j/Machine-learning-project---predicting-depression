**#Intermediate presentation code and data retrieval**

In this folder you will find code and data we used for intermediate presentation. 

You will find: 
1) Indermediate presentation code from Liisa - This is a colab notebook that has all the code and data retrieval done for the intermediate presentation. 
2) create_parquets.py - this is a script that gets and transforms the data. Was necessary for comaparing models and cross-testing along datasets. 
In specific, the code does this: handles two CSV inputs (depression + fitness), creates two parquet files saved in data/:; full_dataset.parquet has all features after preprocessing (same as the intermediate presentation): it adds absolutist word features, shuffles dataset and drops irrelevant columns (tf-idfs); text_classification_dataset.parquet has user_id, text, label. 
3) README.md - explain the contents of this folder


**Getting the data**
In order to use the create_parquets.py script you need to download these two files into the \data: 
https://zenodo.org/records/3941387/files/depression_pre_features_tfidf_256.csv?download=1
https://zenodo.org/records/3941387/files/fitness_pre_features_tfidf_256.csv?download=1

Then run create_parquets.py and you will have the datasets ready in your data folder (NB! this task expects the same data structure detailed under this projects main README.me)
