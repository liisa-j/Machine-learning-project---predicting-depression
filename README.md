# Machine-learning-project---predicting-depression
This is a repository for 'Machine learning' project, autumn 2025


# Structure of this repository
Documents in main/root folder:
- **README.md** - contains all the instructions and explanations on how to use this repo. You are reading this now. 
- **requirements.txt** - contains list of required depencies for running the code
- **LICENCE** - an MIT licence
- **Background, methods and goals.pdf** - to give insight to the problem at hand
- **Results.pdf** - a pdf with a results report from running the models

Folders in main/root folder: 
- **src** - contains all the code necessary for the project (getting, cleaning, preprocessing the main Twitter dataset, feature extraction scripts and running models)
- **intermediate_presentation** - contains code for intermediate presentation dealing with the original Reddit dataset and extracting parquet files from this dataset (for more specific overview see the intermediate_presentation/README.md)


---

# **I Datasets**

We are using 2 different datasets: one from Twitter and one from Reddit. 

**The Twitter dataset** used in this project is from: https://zenodo.org/records/5854911
    The larger dataset created based on these .json files is 24 million rows and a shorter one used primarily in modeling is 1 million rows (equal classes). From the available original features we use author, tweet text and time (label is created based on anchor text and /or dataset class). See below for feature extraction and original data citation. 

**The Reddit Mental Health dataset** used in this project is from: https://zenodo.org/records/3941387 
    This dataset contains Reddit posts collected between 2018 and 2020 from 28 subreddits, including 15 mental health support subreddits and 13 control subreddits. For this project datasets from r/depression and r/fitness are used. The Reddit dataset comes with already preprocessed features (metadata, LIWC, TF-IDF, text and readability metrics, sentiment features, custom dictionaries) and weak labels (r/depression as a weak positive case label). The dataset derived from this source has 43373 rows (both classes are equal).


## **1. Getting the data**

### a) Getting Twitter data
The datasets are **not included** in this repository due to size (3+ GB). Please download datasets manually:

- **Positive class (depressed users):** [depression.zip](https://zenodo.org/records/5854911/files/depression.zip?download=1)  
- **Negative class (control users):** [neg.zip](https://zenodo.org/records/5854911/files/neg.zip?download=1)  


In your repo root, create a data/ folder:
```bash
mkdir data
```

Unzip these depression.zip and neg.zip files into your project folder:

## Positive class
```bash
unzip depression.zip -d data/depression
````

## Negative class
```bash
unzip neg.zip -d data/neg
```

The file structure should look like this: 

```swift
data/
├─ depression/
│  └─ media/data_dump/udit/submission/dataset/depression/cleaned_data_depression/
├─ neg/
│  └─ media/data_dump/udit/submission/dataset/neg/cleaned_data_neg/
```

### b) Getting Reddit data

The Reddit datasets also needs to be manually downloaded and saved in the /data folder:
https://zenodo.org/records/3941387/files/depression_pre_features_tfidf_256.csv?download=1 
https://zenodo.org/records/3941387/files/fitness_pre_features_tfidf_256.csv?download=1



## **2. Scripts for data preprocessing**
*NB - If you for any reason want to rerun these scripts, please understand that running these scripts takes times, as does downloading the zipped datasets you need for running them. Thanks!*
All scripts for data preprocessing begin with P and a number (shows the order in which the scripts
should be run; e.g. P1_, P2_). 
All scripts that extract features start with F_. 

### a) Twitter data
**P1_process_posclass.py** — Processes the positive (depressed) class.

Reads JSON files in depression_data/...
Flattens tweets, filters by ±90 days around the anchor tweet, and labels them
Outputs: data/tweets.parquet

**P2_process_negclass.py** — processes the negative/control class.

Reads JSON files in neg_data/...
Flattens tweets in batches and labels them as control
Outputs: data/neg_combined.parquet


**P3_combined_df.py** - creates a final dataframe of 20 million lines (combing previously produced negative and positive class datasets)

Loads datasets using pandas.read_parquet.
Converts user_id to string to ensure consistent data types.
Combines datasets with pd.concat.
Shuffles the combined df randomly to remove ordering effects.
Saves the final combined df to Parquet. 

**P4_short_df.py** - creates a smaller (1 million rows), randomly sampled version of the previous combined_tweets.parquet. 

Loads the combined df using pandas.read_parquet.
Randomly samples rows.
Shuffles df to remove any residual ordering.
Saves the sampled dataset to Parquet. 

**P5_cleaning.py** - cleans tweets
Reads the raw parquet dataset: data/shorty.parquet. Cleans the data with preprocess_dataframe (Converts to lowercase, replaces URLs with <URL>, replaces mentions with @, removes standalone RT, emojis to text, encodes label col w LabelEncoder,  creates label_encoded, drops is_anchor, original text column, original label, removes extra whitespace).
Writes the cleaned dataframe to: data/shorty_clean.parquet

### b) Reddit data 

**P6_intermediate_presentation/create_parquets.py** - run this script that preprocesses the Reddit csv-s in your /data and saves 2 ready-to-use Reddit datasets into your /data folder. 


## **3. Scripts for feature extraction**

### a) Twitter data

**F_features_from_Twitter.py** - code for extracting text derived linguistic features. 

**Basic text metrics:**
char_count (total number of characters in the text), 
word_count (total number of words),
avg_word_len (average word length),
unique_word_ratio (ratio of unique words to total words), 
sentence_count (structure and length at sentence level)

**Punctuation & stylistic:**
uppercase_ratio (proportion of words in all caps), 
elongation_count (count of repeated letters (e.g., “soooo”)),
ellipsis_count (count of “...”),
count_exclamation (number of !),
count_question (number of ?), 
repeated_exclamation, repeated_question (for expressive punctuation)

**Emoji features:**
emoji_count (total number of emojis),
emoji_unique_count (number of unique emojis)

**Sentiment (TextBlob):**
polarity (sentiment polarity (-1 to 1)),
subjectivity (sentiment subjectivity (0 to 1)),
vader_neg, vader_neu, vader_pos, vader_compound (for more nuanced sentiment breakdown)

**Empath categories:**
sadness,
fear,
anger,
negative_emotion,
positive_emotion,
social,
family,
friendship,
swearing,
health,
money,
work,
leisure,
time,
past,
future,
cognitive_mechanisms,
perceptual,
motion,
body,
sexual,
religion,
affection,
achievement,
reward,
dominance,
power,
anxious,
tentative,
certainty

**Grammar / lexical patterns:**
first_person_count (count of first-person pronouns (I, me, etc.)),
negation_count (count of negations (not, never, none, etc.)),
max_word_repeat (maximum number of times a single word is repeated), 
function_word_count (common function words)

**Readability (textstat):**
flesch_reading_ease (Flesch reading ease score),
flesch_kincaid_grade (Flesch-Kincaid grade level)


### b) Reddit data

**F_features_from_reddit_dataset.py** - this script takes the short Reddit preprocessed dataset and creates a new dataframe for Reddit data with the same features as the abovementioned *features_shorter2.py*. 


# **II Requirements**

Install the required Python packages:

```` bash
pip install -r requirements.txt
````

Also download the SpaCy English model: 

````bash
python -m spacy download en_core_web_sm
````

# **III Models**

**RF.py and RF2.py** - code for training random forest on the data

**linear_regression.py** - code for training linear regression on the data

**svm.py** - code for training svm on the data

.....

# **IV**

...


# **V Citations**

**Twitter dataset:** 

Suhavi, Singh, A., Arora, U., Shrivastava, S., Singh, A., Shah, R. R., & Kumaraguru, P. (2022). Twitter Self-reported Temporally-contextual Mental Health Diagnosis (Twitter-STMHD) Dataset (Version 1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5854911 


**Reddit dataset:** 

Low, D. M., Rumker, L., Talker, T., Torous, J., Cecchi, G., & Ghosh, S. S. (2020). Reddit Mental Health Dataset (Version 01) [Data set]. Zenodo. https://doi.org/10.17605/OSF.IO/7PEYQ