# Disaster Response Pipeline Project

## The author
Welcome to my Project! I'm Carlos. This project began July 2021. 

## Installations

The code requires Python versions of 3.* and general libraries available through the Anaconda package. In addition, the nltk package needs to be installed for the program to run successfully. For more details on the required packages, please refer to the files train_classifier.py and process_data.py.

**Download** the latest GitHub repository.

## Project Motivation 
The goal of the project is to classify the disaster messages into categories. In this project, I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages. Through a web app, the user can input a new message and get classification results in several categories. The web app also display visualizations of the data.

## Project Descriptions

The project has three componants which are:
ETL Pipeline: process_data.py file contain the script to create ETL pipline which:

    Loads the messages and categories datasets.
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

ML Pipeline: train_classifier.py file contain the script to create ML pipline which:

    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file

Flask Web App: the web app enables the user to enter a disaster message, and then view the categories of the message.

    The web app also contains some visualizations that describe the data.

Files Descriptions
The files structure is arranged as below:

    README.md: read me file
    Project Components.txt: Project requirements to complete
    workspace

        \app
            run.py: flask file to run the app

        \templates
            master.html: main page of the web application
            go.html: result web page

        \data
            disaster_categories.csv: categories dataset
            disaster_messages.csv: messages dataset
            DisasterResponse.db: disaster response database
            process_data.py: ETL process

        \models
            train_classifier.py: classification code
            MLclassifier.pkl: model pickle file

        \Pipeline Preparation
            ETL Pipeline Preparation.py
            ML Pipeline Preparation.py


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/carsimoes/)
