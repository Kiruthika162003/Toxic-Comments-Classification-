# Social Media Toxic Comments Classification

A machine learning project on NLP to detect different types of toxicity like threats, obscenity, insults, and identity-based hate in the comments given in the dataset.

## Description

The dataset used in this project consists of three files present in the 'data' folder: `train.csv`, `test.csv`, and `test_labels.csv`. The data in the training set is in the form of comments which have been labeled by human raters for toxic behavior. These comments are classified into six types of toxicity: toxic, severe_toxic, obscene, threat, insult, and identity_hate.

## Implementation

### Explorative Data Analysis

- The train data has 159,571 observations with 8 columns and the test data has 153,164 observations with 2 columns.
- A plot showing the count of each of the six labels was plotted and it was observed that the label 'toxic' has the most observations in the training dataset while 'threat' label has the least observations.
- A cross-correlation matrix for each label was plotted to see which labels are likely to appear together with a comment. It was observed that the 'obscene' label had a higher chance to be 'insulting' at the same time.
- To visualize the most common words contributing to different labels, separate word clouds were generated for each label.

### Feature Engineering

- Tokenization was used to remove punctuations, special characters, and non-ascii characters from the comments.
- Lemmatization was applied and all the comments with length less than three were filtered out.
- TFIDF vectorizer was used to scale down the impact of tokens that occur very frequently in a given corpus which are empirically less informative than features that occur in a small fraction of the training corpus.

### Model Selection

- Three models known to perform well in text classification were compared against each other: Linear SVM, Multinomial Naive Bayes, and Logistic Regression.
- The evaluation metrics used to check the performance were F1-score, Recall, and Hamming Loss.
- Initially, the cross-validation F1-score and Recall were compared using the training dataset. It was observed that Linear SVM and Logistic Regression models performed much better than Multinomial Naive Bayes.
- On the test dataset, the Multinomial Naive Bayes model did not perform well compared to others. It was observed that the Linear SVM model performed slightly better than the Logistic Regression model.
- Confusion matrices were plotted for the most common label 'toxic'. It was observed that all three models predicted non-toxic labels fairly well, probably because most of the data was non-toxic.
- Aggregate Hamming Loss was calculated for each model. It was found that Logistic Regression had the least percentage of labels incorrectly classified.
- Pipelines were constructed to compare Linear SVM and Logistic Regression models. The 'class_weight' hyperparameter was manually chosen to aim for better results than the basic models themselves.
- The result showed that Linear SVC performed better than the Logistic Regression model.

### Hyperparameter Tuning

- The optimal hyperparameters for the basic models were found using Grid Search, considering only the label 'toxic' since it was the most common label, to tune the hyperparameters.

### Ensembling

- For ensembling different models, three popular tree-based boosting models: Adaptive Boosting, Gradient Boosting, and XGBoosting were compared against each other on the evaluation metrics described earlier.
- The result showed that the XGBoost Classifier performed the best out of all three classifiers. A voting classifier was used to ensemble the XGBoost model with our Logistic Regression and Linear SVM models.

## Results

- Linear SVM performed the best overall, followed by Logistic Regression and Multinomial Naive Bayes.
- The confusion matrices and evaluation metrics indicated that the models were able to predict non-toxic labels more accurately due to the imbalance in the dataset.
- The ensembling approach with XGBoost, Logistic Regression, and Linear SVM provided the best results in terms of F1-score and Recall.

Feel free to reach out if you have any questions or need further assistance!
