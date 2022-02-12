# Udacity_DSProject2_DisasterResponse

# Description:
Following a natural disaster there are usually large volumes of disaster-related communication. Disaster response organisations need to efficiently sift through and action the messages that matter whilst being overwhelmed with reacting to the given disaster. This project involves analysing disaster data in the form of messages to build a supervised machine learning model for an API that classifies disaster messages. FigureEight provided over 25k tagged disaster messages which were used to train or test the model. 
1.	An ETL pipeline was built to prepare messages and category data, and then load the cleaned data into a SQLite database.
2.	A ML pipeline was then used to build a multi-output supervised learning model using the data from the SQLite database.
3.	A web app was built to extract data from the database, to display data visualisations and to use the model to classify new messages. 

# Files in repository: 
1.	disaster_messages.csv: Messages dataset containing 4 columns including the message id, translated message, original message, and the type of message (e.g. news, social media). 
2.	disaster_cateogires.csv: Categories dataset containing 2 columns including the message id and the category(ies) of the message.
3.	DisasterResponse.db: SQLite database where the wrangled data is loaded to by the ETL script and from which the training and test data is loaded from in the ML pipeline script.
4.	process.py: ETL script
•	Loads the messages and categories datasets
•	Merges the two datasets
•	Cleans the data 
•	Stores it in a SQLite database
5.	train.py: ML pipeline script
•	Loads data from the SQLite database
•	Splits data into training and test sets
•	Text processing and machine learning pipeline
•	Produce classification report which include F1 score, precision and recall for each category on the test set 
•	Exports the final model as a pickle file
6.	run.py: Flask webapp script provided by Udacity:
•	Loads data from the SQLite database
•	Creates visualisation to appear on the webapp
•	Uses model to classify new messages

# Note on running python scripts:
1. process.py: To run this script, and three arguments need to be provided - two dataset csvs (disaster_messages.csv and disaster_categories.csv) and the database filepath (DisasterResponse.db).
2. train.py: To run this script, the two arguments need to be given the filepath for the database (DisasterResponse.db) and a pickle file name (classifier_LSVC.pkl) to which the final trained model will be exported.
3. run.py: No arguments need to be provided. 

# Libraries used: 
1.	pandas
2.	nltk
3.	sqlalchemy
4.	sqlite3
5.	re
6.	sklearn
7.	pickle
8.	joblib
9.	flask
10.	plotly

# Notes: 
-	RandomForestClassifier and LinearSVC were both used and LinearSVC gave higher precision, recall and F1-scores.
-	Precision and recall were appropriate evaluation metrics since the dataset was quite imbalanced. Messages were labelled under multiple categories if appropriate but the the majority of the messages were labelled as (in addition to ‘related’), ‘aid-related,’ ‘weather-related’. Some categories (e.g ‘shop’) has as little as 27 corresponding messages which was likely insufficient for the model to be trained well. 
-	There were a total of 36 categories in the dataset, however there were no messages tagged as 'child-alone' so this category was dropped from the dataset. It was thought that removing it from the training (and test) data would have no impact on the model.  
