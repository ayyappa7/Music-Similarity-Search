# Music-Similarity-Search
Cloud based music similarity search using Machine Learning

This project aims at developing a Cloud-based Web Application to retrieve the music tracks that are similar to the user's input track by using Machine Learning on the server side.

All the audio tracks in the Audio datasets, which are GTZAN Dataset and MagnaTagATune Dataset are preprocessed to get Mel Frequency Cepstral Coefficients (MFCC). Using these obtained audio features, an ensemble of classifiers consisting of Neural Network, Logistic Regression, and a Support Vector Machine is trained to detect the genre of a given audio track. This trained ensemble of classifiers is placed on the server side. The web application is implemented using Flask framework. This framework handles the requests, responses and routing. On the front end of the web application, the user will be able to drag and drop an audio track file on the UI. This audio file is uploaded to a location on the server side. The same audio features of the audio track are calculated and is passed as input to the classifier on the server side which detects the genre of that track. Most similar audio tracks belonging to the same genre in the datasets are identified and their details are sent back to the front end and are displayed on the UI. 

This Web Application to be deployed on AWS Cloud's Linux instance and will be accessible from anywhere over the internet. 
