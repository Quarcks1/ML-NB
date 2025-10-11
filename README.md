# ML-NB
Machine Learning model that uses Bayesian Inference tecniques (Naive-Bayes) to improve his predictions through probabilities

# Naive Bayes
Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable. Bayes’ theorem states the following relationship, given class variable "y" and dependent feature vector X1 through Xn .....
https://scikit-learn.org/stable/modules/naive_bayes.html

For more information consult the reference: Chapter 4 "Algorithms: the basics methods"

# Scikit-learn
Is an open-source Python library for machine learning. It provides a wide range of algorithms and tools for various machine learning tasks
<img width="1528" height="990" alt="image" src="https://github.com/user-attachments/assets/09fc755b-aec6-45df-8f12-bd8671498ce6" />

# Steps to follow 
Do it for each dataset

1.- Choose a dataset (e.g., 10,000 records). 

2.- Load the data into Python using pandas.

3.- Explore and clean the data:
    * Check for missing values.
    * Ensure classes are balanced (same number of examples per class).

4.- Split the data:
  70-90% for training.
  10-30% for evaluation.
  Save a separate set as a test set (new data the model hasn't seen).

5.- Select the problem type:
  *Is it classification (categories)? Use accuracy as a metric.

6.- Choose a Bayesian model:
We will use the **GaussianNB** from scikit-learn for classification.

7.- Train the model on the training data.

8.- Validate the model using k-fold cross-validation (e.g., k=5).

9.- Evaluate the model on the evaluation set.

10.- Test the model with the test set (new data).

11.- Measure performance with the appropriate metrics.
