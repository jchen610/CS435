import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

def main():

    #Headers for the dataset
    cols = ["age", "workclass", "fnlwgt", "education", "education-num", "martial-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", ">50K, <=50K"]

    #Cleaning data by dropping all tuples with missing data
    train_df = pd.read_csv("adult.data", names = cols)
    train_df = train_df.replace(r'\?', np.nan, regex=True)
    train_df.dropna()

    test_df = pd.read_csv("adult.test", names = cols)
    test_df = test_df.replace(r'\?', np.nan, regex=True)
    test_df.dropna()
    
    #Save clean data
    train_df.to_csv('clean_adult.data', index=False, header = False) 
    test_df.to_csv('clean_adult.test', index=False, header = False) 

    #convert categorical data attributes to contionus data for both datasets
    label_encoders = {}

    for column in train_df.select_dtypes(include=['object']):
        label_encoders[column] = LabelEncoder()
        train_df[column] = label_encoders[column].fit_transform(train_df[column])
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    for column in test_df.select_dtypes(include=['object']):
        label_encoders[column] = LabelEncoder()
        test_df[column] = label_encoders[column].fit_transform(test_df[column])
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    #create our model
    bayes_model = GaussianNB()
    bayes_model = bayes_model.fit(X_train, y_train)

    #test our model
    y_pred = bayes_model.predict(X_test)
    error_rate = 1 - accuracy_score(y_test, y_pred)
    #pipe output into a file
    with open('output.txt', 'w') as f:
        #f.write(classification_report(y_test, y_pred)+ "\n")
        f.write("Classification Error Rate For Naive Bayes: " + str(error_rate) + "\n")


    #fit and scale attributes of both datasets and oversample training dataset to get equal label occurances
    scaler = StandardScaler()
    improved_X_train = scaler.fit_transform(X_train)
    improved_X_test = scaler.fit_transform(X_test)
    ros = RandomOverSampler()
    improved_X_train, improved_y_train = ros.fit_resample(improved_X_train, y_train)

    #create our model
    improved_bayes_model = GaussianNB()
    improved_bayes_model = improved_bayes_model.fit(improved_X_train, improved_y_train)

    #test our model
    improved_y_pred = improved_bayes_model.predict(improved_X_test)
    improved_error_rate = 1 - accuracy_score(y_test, improved_y_pred)
    #pipe output into a file
    with open('output.txt', 'a') as f:
        #f.write(classification_report(y_test, y_pred)+ "\n")
        f.write("Classification Error Rate For Improved Naive Bayes: " + str(improved_error_rate) + "\n")

        
    #Sampling for 50% 60% 70% 80% and 90%

    sampling_arr = [0.5, 0.6, 0.7, 0.8, 0.9]
    mean_err = []
    std_err = []

    for percentage in sampling_arr:
        error_rate_for_sample = []
        for i in range(5):
            #count of classes
            class_counts = train_df[">50K, <=50K"].value_counts()

            #calculate the ratio of the classes
            class_ratio = class_counts / class_counts.sum()

            #assign weight based on the ratio of classes
            weights = train_df[">50K, <=50K"].map(class_ratio)

            #sample data using weight to maintain ratio of classes
            sampled_train_df = train_df.sample(frac=percentage, weights=weights)

            #split data frame into X and Y
            X_sampled_train = sampled_train_df.iloc[:, :-1]
            y_sampled_train = sampled_train_df.iloc[:, -1]
            
            #create our model
            sampled_bayes_model = GaussianNB()
            sampled_bayes_model = sampled_bayes_model.fit(X_sampled_train, y_sampled_train)

            #test our model
            sampled_y_pred = sampled_bayes_model.predict(X_test)
            sampled_error_rate = 1 - accuracy_score(y_test, sampled_y_pred)
            error_rate_for_sample.append(sampled_error_rate)

        mean_err = np.mean(error_rate_for_sample)
        std_err = np.std(error_rate_for_sample)
        with open('output.txt', 'a') as f:
            f.write(f"Mean Error Rate For {percentage} Sampling: " + str(mean_err) + "\n")
            f.write(f"Standard Deviation Error Rate For {percentage} Sampling: " + str(std_err) + "\n")
if __name__ == "__main__":
    main()