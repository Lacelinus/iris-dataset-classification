# Iris Dataset Classification

This Python script performs classification of the Iris dataset using six different machine learning algorithms. Each algorithm is explained and evaluated using confusion matrices.

## Overview

The script begins by importing the necessary libraries, including Pandas for working with data frames. It reads the 'iris.csv' file into a data frame, separates independent variables (X) and the dependent variable (y), and removes the first 5 rows of data. It then splits the dataset into training and testing subsets using the `train_test_split` function from Scikit-Learn.

The data is standardized using the `StandardScaler` class. Then, six different classification algorithms are applied:

1. **Logistic Regression**: A logistic regression model is created and trained. Predictions are made, and a confusion matrix is printed.

2. **K-Nearest Neighbors Classifier**: A K-Nearest Neighbors model is created and trained. Predictions are made, and a confusion matrix is printed.

3. **Support Vector Classifier**: A Support Vector Classifier (SVC) model with an RBF kernel is created and trained. Predictions are made, and a confusion matrix is printed.

4. **Gaussian Naive Bayes Classifier**: A Gaussian Naive Bayes model is created and trained. Predictions are made, and a confusion matrix is printed.

5. **Decision Tree Classifier**: A Decision Tree model with entropy as the criterion is created and trained. Predictions are made, and a confusion matrix is printed.

6. **Random Forest Classifier**: A Random Forest model with entropy as the criterion and 10 estimators is created and trained. Predictions are made, and a confusion matrix is printed.

## Usage

1. Clone or download this project to your local machine.
2. Ensure you have Python installed.
3. Install the necessary libraries by running `pip install pandas scikit-learn`.
4. Make sure you have the 'iris.csv' file in the specified path ('D:/herşey/python/csvs/iris/iris.csv') or provide the correct file path to ensure the script can read the data.
5. Run the Python script.
6. The script will preprocess the Iris dataset, apply the chosen classification algorithms, and display confusion matrices for evaluation.

Feel free to customize the code by adjusting hyperparameters, trying different algorithms, or exploring additional evaluation metrics.

## Dependencies

- [Pandas](https://pandas.pydata.org/): Used for working with data frames.
- [Scikit-Learn](https://scikit-learn.org/stable/): Provides machine learning algorithms and tools.

## Important Notes

- This script is intended for educational purposes and as a demonstration of various machine learning algorithms on the Iris dataset.
- Make sure you have the necessary Python libraries (`pandas`, `scikit-learn`) installed to run the code successfully.
- Feel free to reach out if you have any questions or need assistance with the code.

**Author**

This script was created by Lacelinus. You can reach out to me at ekremkprn2@gmail.com for any questions or suggestions.

Enjoy exploring machine learning with the Iris dataset!
