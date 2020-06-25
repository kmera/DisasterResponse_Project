### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)

## Installation <a name="installation"></a>

The libaries needed to run the code, using Python version 3.* are:

* Pandas
* NumPy
* Sys
* Sqlalchemy
* Nltk
* String
* re
* word_tokenize
* WordNetLemmatizer
* Stopwords
* RandomForestClassifier
* KNeighborsClassifier
* train_test_split
* GridSearchCV
* MultiOutputClassifier
* Pipeline
* FeatureUnion
* BaseEstimator
* TransformerMixin
* CountVectorizer
* TfidfTransformer
* classification_report

## Project Motivation<a name="motivation"></a>

When there is a disaster there are plenty of messages, and the disaster response organizations have to handle and filter those messages according to the importance, and send then to the corresponding organization in oder to take care of the people that have reported a problem. So, to improve and speed up the process to attend each case, the messages need to be classified according to a category and to do that a Machine Learning model will support this request.

Basen on the data provided by Figure Eight which includes real life disaster messages and categories, this project is aimed to develop a web app that classifies that messages.  

disaster_categories.csv and disaster_messages.csv were provided by Figure Eight. Those files were cleaned in an ETL Pipeline. Next, a Machine Learning Pipeline was developed to build a Supervised ML model. Finally, a web app will be the interface to enter a message and classify it in some categories.

## File Descriptions <a name="files"></a>

There are two Python scripts. The first one is process_data.py, which includes all the tasks of the ETL process such as load datasets, merge them and and store them in a SQL database. To run this script: 'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'. The second script is train_classifier.py, which is in charge of load the data from SQL database, split the data into training and test datasets, build a pipeline model which use GridSearchCV and then train, predict, and measure the performance of the model. To run ML pipeline:'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'. Finally, there is a web app that first shows the Distribution of Message Genres in a plot, a text box to enter the message and it will be classified according to the Machine Learning model, and the web app will be shown the categories of the message.


## Results<a name="results"></a>

The results of this project is the categories' message which are highlighted in green. To access the wew app you have to go to: http://0.0.0.0:3001/


