# Import the Pandas library, which is used to work with data frames.
import pandas as pd 

# Read the 'iris.csv' file to create a data frame (DataFrame).
df = pd.read_csv('D:/herşey/python/csvs/iris/iris.csv')

# Separate independent variables (X) and the dependent variable (y) from the data frame.

# Independent variables (Features)
x = df.iloc[:, 1:5].values  

# Dependent variable (Target class)
y = df.iloc[:, 5].values   

# Remove the first 5 rows from the data frame.
df = df.drop(range(0, 5), axis=0)

# Use the train_test_split function from the sklearn library to split the dataset into training and testing subsets.
from sklearn.model_selection import train_test_split

# When splitting the data into training and testing sets, specify the size of the test set and the seed for random data splitting.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Use the StandardScaler class from the sklearn library to scale the data.

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object.
sc = StandardScaler()  

# Scale (standardize) the training data.
X_train = sc.fit_transform(x_train)  

# Transform the test data using the same scaling as the training data.
X_test = sc.transform(x_test)       

#%%

# Import the Logistic Regression class.
from sklearn.linear_model import LogisticRegression  

# Create a Logistic Regression model and specify the random seed for randomness.
log_reg = LogisticRegression(random_state=0) 

# Train the model using X_train and y_train.
log_reg.fit(X_train, y_train)  

# Use the trained model to predict X_test data.
y_pred_logr = log_reg.predict(X_test)  

# Import the confusion_matrix library to calculate the confusion matrix.
from sklearn.metrics import confusion_matrix  

# Calculate a confusion matrix between predictions and actual values.
cm_logr = confusion_matrix(y_test, y_pred_logr)  

# Print the confusion matrix.
print('Logr\n', cm_logr)  


#%%

# Import the K-Nearest Neighbors class.
from sklearn.neighbors import KNeighborsClassifier 

# Create a K-Nearest Neighbors model with a specified number of neighbors (n_neighbors) and using the Minkowski distance metric.
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')  

# Train the model using x_train and y_train.
knn.fit(x_train, y_train)  

# Use the trained model to predict x_test data.
y_pred_knn = knn.predict(x_test)  
print('\n') # Print an empty line to separate output sections.

# Calculate a confusion matrix between predictions and actual values.
cm_knn = confusion_matrix(y_test, y_pred_knn) 

# Print the confusion matrix.
print('Knn\n', cm_knn)  


#%%

# Import the Support Vector Classifier (SVC) class.
from sklearn.svm import SVC  

# Create an SVC model with an RBF (Radial Basis Function) kernel.
svc = SVC(kernel='rbf')

# Train the model using X_train and y_train.
svc.fit(X_train, y_train)

# Use the trained model to predict X_test data.
y_pred_svc = svc.predict(X_test)

# Calculate a confusion matrix between predictions and actual values.
cm_svc = confusion_matrix(y_test, y_pred_svc)

# Print the confusion matrix.
print('\n\nSvc\n', cm_svc)  

#%%

# Import the Gaussian Naive Bayes class.
from sklearn.naive_bayes import GaussianNB  

# Create a Gaussian Naive Bayes model.
gnb = GaussianNB()

# Train the model using X_train and y_train.
gnb.fit(X_train, y_train)

# Use the trained model to predict X_test data.
y_pred_gnb = gnb.predict(X_test)

# Calculate a confusion matrix between predictions and actual values.
cm_gnb = confusion_matrix(y_test, y_pred_gnb)

# Print the confusion matrix.
print('\nNaive Bayes\n', cm_gnb)  

#%%

# Import the Decision Tree class.
from sklearn.tree import DecisionTreeClassifier  

# Create a Decision Tree model with entropy as the criterion.
dtc = DecisionTreeClassifier(criterion='entropy')

# Train the model using X_train and y_train.
dtc.fit(X_train, y_train)

# Use the trained model to predict X_test data.
y_pred_dtc = dtc.predict(X_test)

# Calculate a confusion matrix between predictions and actual values.
cm_dtc = confusion_matrix(y_test, y_pred_dtc)

# Print the confusion matrix.
print('\nDtc\n', cm_dtc)  

#%%

# Import the Random Forest class.
from sklearn.ensemble import RandomForestClassifier  

# Create a Random Forest model with entropy as the criterion and 10 estimators.
rfc = RandomForestClassifier(criterion='entropy', n_estimators=10)

# Train the model using X_train and y_train.
rfc.fit(X_train, y_train)

# Use the trained model to predict X_test data.
y_pred_rfc = rfc.predict(X_test)

# Calculate a confusion matrix between predictions and actual values.
cm_rfc = confusion_matrix(y_test, y_pred_rfc)

# Print the confusion matrix.
print('\nRfc\n', cm_rfc)  
