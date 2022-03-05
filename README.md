![image](https://user-images.githubusercontent.com/7256682/154767526-b3135f7f-5da9-481c-8f22-d1f504f16d3f.png)

# Data Science Nanodegree - Project 2: Disaster Response Pipeline

This project is related to the Data Science Nanodegree on Udacity (https://classroom.udacity.com/nanodegrees/nd025/dashboard/overview)

## Goal:

The goal of this project is to extract meaningful information of unstructured textual data (messages such as tweets) to be able to categorize new unseen messages as being related to one or more out of various categories about disasters.
This could be used by emergency workers to help in their jobs in prioritizing assistances during disasters.


## Pre-requisites:

The following instructions will only work if one has, in advance these files:
- `disaster_messages.csv`: file with the dataset containing real messages that were sent during disaster events;
- `disaster_categories.csv`: file with the list of different categories for the messages.


## Details:

The outputs of this project are:
- a database with the original dataset processed, cleaned and ready to be used to train a supersived learning model;
- a model already trained on the dataset previously processed;
- a web app where an emergency worker can input a new message and get classification results in several categories.


This repository consists the following scripts and files:
- process_data.py: 
- train_classifier.py:
- run.py:
- ....
- ....


## Instructions:

To be able to reproduce my work, one can follow these steps:

1. Save the files `disaster_messages.csv` and `disaster_categories.csv` on the same folder than the `process_data.py` script;

2. Run the following commands in the project's root directory to set up your database and model:

  2.1. Run the line of code below to run an ETL pipeline that cleans data and stores in a database:
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

  2.2. Run the line of code below to run a ML pipeline that trains a classifier and saves it on disk:
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the line of code below in the app's directory to run your web app:
    `python run.py`

4. After the app is up and running, one can open this url and try the web app: http://0.0.0.0:3001/

