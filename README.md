### Description
* This is a proposed medical chatbot deployed into DialogFlow that takes user's input, analyzes it and diagnoses the case into a disease from a pre-defined [dataset](https://www.kaggle.com/itachi9604/disease-symptom-description-dataset?select=symptom_precaution.csv).
* The bot analyses the user's sentence by checking whether it's medically relevant or not, then if it finds the input relevant, it goes to the next step.
* The bot then check for the number of the symptoms found -if any- in the user's inquiry, if they match the threshold then the model is called to send the predicted diagnosis based on these symtoms,\ otherwise the bot starts to ask follow-up questions for the user comprising the most relevant symptom to what he has mentioned before.

### Dependencies

numpy==1.18.1.\
scikit-learn==0.24.2.\
pandas==1.0.1.\
nltk==3.4.5.\
joblib==0.14.1.\
pickle version==4.0.

### Files Description

##### chatbot/disease_classifier.py 
Python file that contains the classification code (using SVM only) for the disease which the input symptoms belong to.

##### chatbot/get_next_symptom.py 
Python file that contains a function that takes some symptoms and predicts the next most important symptom to be asked as a follow-up question for the user.
##### chatbot/medical_relevence.py 
Python file that contains a function that takes user's message and decide if it's relevent to the medical context or not.

##### chatbot/clean_data.csv
Csv file that contains a subset of the clean data used for training the model.

##### chatbot/disease_classifier.pkl
The SVM classification model used in `disease_classifier.py` serialized using pickle.

##### chatbot/medical_relevence_classifier.joblib
The model used in `medical_relevence.py`

##### medical_relevence_classification.ipynb
Python notebook for the medical relevance function and its needed steps.

##### Disease_Prediction_Clustering_NextSymptom.ipynb
Python notebook for different clustering techniques applied (K-means and DBSCAN) and get_next_symptom function applied.

##### Disease_Prediction_Classification,ipynb
Python notebook for different classification techniques applied (Random forest, XGBoost and SVM)


