import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

#load datasets
df_main = pd.read_csv('./Data/dataset1000.csv')
df_test = pd.read_csv('./Data/dataset100.csv')

#Define categorical columns 
features_columns = ['Director', 'Production', 'User', 'Genre']

# split features
x = df_main[features_columns]  #features
y = df_main['Rank']                 #target variable

#print(df_test.head())


#Transform categorical features to numerical
def preprocess_data(df, label_encoders=None):
    df_encoded = df.copy()
    if label_encoders is None:
        label_encoders = {}
        for column in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column])
            label_encoders[column] = le
    else:
        for column in df_encoded.columns:
            df_encoded[column] = label_encoders[column].transform(df_encoded[column])
    return df_encoded, label_encoders


def train_model(x_train, y_train):
    model = MultinomialNB(alpha=0.5)
    model.fit(x_train_encoded, y_train)
    return model
    
    
def evaluate_model(model, x_evaluation, y_evaluation):
    y_pred = model.predict(x_evaluation)
    accuracy = accuracy_score(y_evaluation, y_pred)
    return accuracy
    

def encode_test_data(df_test, label_encoders):
    df_test_encoded = df_test.copy()

    for column in label_encoders:
        df_test_encoded[column] = label_encoders[column].transform(df_test_encoded[column])
    return df_test_encoded

#divide training (70%) and evaluation (30%) sets
x_train, x_evaluation, y_train, y_evaluation = train_test_split(
    x, y, test_size=0.3, stratify=y, random_state=42
    )

x_train_encoded, label_encoders = preprocess_data(x_train)
x_evaluation_encoded, _ = preprocess_data(x_evaluation, label_encoders)



#train model
model = train_model(x_train_encoded, y_train)

# k-fold cv = folds
# Within the training set (70%), the model is trained and evaluated 5 times:
scores = cross_val_score(model, x_train_encoded, y_train, cv=5)
print("Cross-validation mean accuracy:", np.mean(scores))

#evaluate model
y_pred = model.predict(x_evaluation_encoded)
print("Accuracy:", accuracy_score(y_evaluation, y_pred))
print(classification_report(y_evaluation, y_pred))
print(confusion_matrix(y_evaluation, y_pred)) 

print(f"\nCross-validation mean accuracy: {np.mean(scores):.3f}")
print(f"Evaluation accuracy: {accuracy_score(y_evaluation, y_pred):.3f}")


def show_stats(y_train):
    print("Class distribution in training set:")
    y_train.value_counts().plot(kind='bar')
    plt.title("Distribución de clases en Rank")
    plt.xlabel("Rank")
    plt.ylabel("Frecuencia")
    plt.show()
show_stats(y_train)
