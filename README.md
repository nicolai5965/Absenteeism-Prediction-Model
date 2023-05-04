# Absenteeism-Prediction-Model
## TensorFlow_Absenteeism_Udemy_Course

This project aims to predict employee absenteeism by analyzing the given dataset from an Udemy course. The dataset contains various features like reason for absence, date, transportation expense, distance to work, age, daily work load average, body mass index, education, children, and pets. The main goal is to create a machine learning model that can predict whether an employee will have excessive absenteeism based on these features.

## Data Preprocessing

The preprocessing steps include:

1. Dropping the 'ID' and 'Age' columns as they are not useful for our analysis.
2. Creating dummy variables for the 'Reason for Absence' column and grouping them into four different groups.
3. Concatenating the dummy variables into a new dataframe and dropping the original reason columns.
4. Formatting the 'Date' column to pandas datetime format and extracting the month and day of the week.
5. Mapping the values in the 'Education' column to new values.
6. Creating a target variable based on the median of the 'Absenteeism Time in Hours' column.
7. Standardizing the non-dummy feature columns using the StandardScaler from sklearn.
8. Combining the dummy and standardized non-dummy columns into a single dataframe.
9. Splitting the standardized data into training and testing sets using the train_test_split function from sklearn.

## Machine Learning Model

The machine learning model used in this project is a logistic regression model implemented using TensorFlow. The model architecture includes:

1. Input layer with a size equal to the number of features.
2. Two hidden layers with 128 neurons each, ReLU activation, and L2 regularization.
3. Batch normalization and dropout layers to reduce overfitting.
4. An output layer with a single neuron and sigmoid activation function for binary classification.

The model is compiled using the binary cross-entropy loss function, Adam optimizer, and accuracy as a metric. It is trained with a batch size of 16, validation split of 0.2, and early stopping to prevent overfitting. The model is then evaluated on the test set and its accuracy is reported.

## Results

The model achieved a test accuracy of 0.7714, indicating its ability to predict employee absenteeism based on the given features. Efforts were made to prevent overfitting, such as using L2 regularization, batch normalization, and dropout layers in the model architecture. An early stopping callback was also employed to stop training when no further improvement in validation accuracy was observed.

Despite these efforts to prevent overfitting and improve the model's performance, the small size of the dataset made it challenging to achieve better results. With a larger dataset, it would be possible to train the model more effectively and potentially achieve higher accuracy.
