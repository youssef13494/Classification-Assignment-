import numpy as np

import tkinter as tk
from tkinter import filedialog
from tkinter import Entry, Button, Label, Frame, PhotoImage
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from tkinter.scrolledtext import ScrolledText
from sklearn.model_selection import train_test_split


# Create a Tkinter window
root = tk.Tk()
root.title("Diabetes Prediction")

# Set the background color
root.configure(bg="#FFDE76")#
root.iconbitmap("C:\\Users\\DELL\\Downloads\\Assignment 3\\Assignment3_20201224\\rubber-duck_icon-icons.com_55299.ico")

# Set the size of the root window
root.geometry('1470x820')

# Disable resizing
root.resizable(False, False)

Percentage_data = tk.DoubleVar()
Training_size = tk.DoubleVar()
# Standardize Data
scaler = StandardScaler()

# Function to browse for CSV file
def browse_file():
    file_path = filedialog.askopenfilename()
    entry_path.delete(0, tk.END)
    entry_path.insert(tk.END, file_path)

# Load CSV file, preprocess data, and evaluate models
def analyze_data():
    global Percentage_data
    global Training_size
    global scaler
    file_path = entry_path.get()
    if not file_path:
        result_text1.insert(tk.END, "Please select a CSV file.")
        return

    # Load CSV file
    df = pd.read_csv(file_path)
    # Prompt user for the percentage of data and the number of clusters (K)
    percentage = Percentage_data.get()
    train_size = Training_size.get()

    # Calculate the number of records based on the percentage provided
    num_records = int(len(df) * percentage / 100)
    # Randomly select the calculated number of records from the dataset
    df = df.sample(n=num_records, random_state=42)

    # Preprocess data
    df = pd.get_dummies(df, columns=['smoking_history'], prefix='smoking')
    df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1})
    df = df[df['gender'] != 'Other']
    df = df.astype(int)

    # Select features and target variable
    x = df[['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
            'smoking_No Info', 'smoking_current', 'smoking_ever', 'smoking_former', 'smoking_never',
            'smoking_not current']]
    y = df['diabetes']

    x_scaled = scaler.fit_transform(x)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=(1-(train_size/100)), random_state=42)

    # Naive Bayes
    result_text1.insert(tk.END, "Naive Bayes Results:\n")
    result_text1.insert(tk.END, "---------------------\n")
    Result1=NaiveBayes(X_train, y_train, X_test, y_test)
    result_text1.insert(tk.END, Result1)

    # Decision Tree Classifier
    result_text1.insert(tk.END, "\nDecision Tree Classifier Results:\n")
    result_text1.insert(tk.END, "---------------------\n")
    Result2=evaluate_DecisionTreeClassifier(X_train, y_train, X_test, y_test)
    result_text1.insert(tk.END, Result2)



#Models
#NaiveBayes Model

class GaussianNaiveBayes:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.parameters = []
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.parameters.append({
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0)
            })

    def predict(self, X_test):
        posteriors = []
        for parameters in self.parameters:
            mean, var = parameters['mean'], parameters['var']
            posterior = np.sum(np.log(self.normal_pdf(X_test, mean, var)), axis=1)
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors, axis=0)]

    def normal_pdf(self, X, mean, var):
        return 1 / np.sqrt(2 * np.pi * var) * np.exp(-(X - mean)**2 / (2 * var))



#DecisionTreeClassifier 

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold          # Threshold value for the feature
        self.left = left                    # Left child (subtree)
        self.right = right                  # Right child (subtree)
        self.value = value                  # Value if the node is a leaf

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        predicted_class = np.argmax(num_samples_per_class)

        # Stopping criteria
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return Node(value=predicted_class)

        # Find the best split
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None
        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]  # Fix indexing here
                right_indices = np.where(X[:, feature_index] > threshold)[0]  # Fix indexing here
                gini = self._gini_impurity(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        # Split the dataset
        left_indices = np.where(X[:, best_feature_index] <= best_threshold)[0]  # Fix indexing here
        right_indices = np.where(X[:, best_feature_index] > best_threshold)[0]  # Fix indexing here
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=best_feature_index, threshold=best_threshold,
                    left=left_subtree, right=right_subtree)

    def _gini_impurity(self, *groups):
        total_samples = sum(len(group) for group in groups)
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in range(self.num_classes):
                p = sum(group == class_val) / total_samples
                score += p ** 2
            gini += (1.0 - score) * (size / total_samples)
        return gini

    def predict(self, X):
        return [self._predict_tree(x, self.tree) for x in X]

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        if node.threshold is not None:  # Check if threshold is defined
            if x[node.feature_index] <= node.threshold:
                return self._predict_tree(x, node.left)
            else:
                return self._predict_tree(x, node.right)
        else:  # Leaf node, return the most common class label
            class_labels = [self._predict_tree(x, leaf_node) for leaf_node in [node.left, node.right] if leaf_node]
            return max(set(class_labels), key=class_labels.count)

#Load Data
import pandas as pd
import numpy as np
import pandas as pd

def NaiveBayes(X_train, y_train, X_test, y_test):
    global scaler
    # Instantiate the Gaussian Naive Bayes model.
    model = GaussianNaiveBayes()
    # Fit the model to the training data.
    model.fit(X_train, y_train)
    # Predict the labels for the test data.
    predictions = model.predict(X_test)

    # Create a DataFrame for predictions
    df_predictions = pd.DataFrame(predictions, columns=['predictions'])
    
    # Inverse transform the scaled data to obtain the original scale
    X_test_original = scaler.inverse_transform(X_test)
    
    # Create DataFrame for original scale features with column names
    df_result = pd.DataFrame(X_test_original, columns=[
        'gender', 'age', 'hypertension', 'heart_disease', 'bmi',
        'HbA1c_level', 'blood_glucose_level', 'smoking_No Info',
        'smoking_current', 'smoking_ever', 'smoking_former',
        'smoking_never', 'smoking_not current'
    ])
    
    # Merge smoking history columns into one column
    smoking_history = df_result[['smoking_No Info', 'smoking_current', 'smoking_ever', 'smoking_former', 'smoking_never', 'smoking_not current']]
    df_result['smoking history'] = smoking_history.apply(lambda row: row.idxmax().split('_')[1], axis=1)
    df_result.drop(['smoking_No Info', 'smoking_current', 'smoking_ever', 'smoking_former', 'smoking_never', 'smoking_not current'], axis=1, inplace=True)
    # Concatenate predictions DataFrame with X_test DataFrame
    df_result = pd.concat([df_result, df_predictions], axis=1)
    # Select the first 20 rows
    df_result_first_20 = df_result.head(20)
    text_result=df_result_first_20.to_string()
    Result=""
    # Calculate the accuracy of the model.
    accuracy = np.mean(predictions == y_test)
    Result=Result+"Accuracy:  "+str(accuracy)+"\n \n"
    Result=Result+text_result+" \n"
    return Result



from sklearn.metrics import accuracy_score

def evaluate_DecisionTreeClassifier(X_train, y_train, X_test, y_test, max_depth=2):
    global scaler
    y_train=np.array(y_train)
    X_train=np.array(X_train)
    # Create an instance of the DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=max_depth)

    # Fit the model on the training data
    clf.fit(X_train, y_train)

    # Predict the labels for the test data
    predictions = clf.predict(X_test)

    # Create a DataFrame for predictions
    df_predictions = pd.DataFrame(predictions, columns=['predictions'])
    
    # Inverse transform the scaled data to obtain the original scale
    X_test_original = scaler.inverse_transform(X_test)
    
    # Create DataFrame for original scale features with column names
    df_result = pd.DataFrame(X_test_original, columns=[
        'gender', 'age', 'hypertension', 'heart_disease', 'bmi',
        'HbA1c_level', 'blood_glucose_level', 'smoking_No Info',
        'smoking_current', 'smoking_ever', 'smoking_former',
        'smoking_never', 'smoking_not current'
    ])
    
    # Merge smoking history columns into one column
    smoking_history = df_result[['smoking_No Info', 'smoking_current', 'smoking_ever', 'smoking_former', 'smoking_never', 'smoking_not current']]
    df_result['smoking history'] = smoking_history.apply(lambda row: row.idxmax().split('_')[1], axis=1)
    df_result.drop(['smoking_No Info', 'smoking_current', 'smoking_ever', 'smoking_former', 'smoking_never', 'smoking_not current'], axis=1, inplace=True)
    # Concatenate predictions DataFrame with X_test DataFrame
    df_result = pd.concat([df_result, df_predictions], axis=1)
    #Select First 20 Rows 
    df_result_first_20 = df_result.head(20)
    text_result=df_result_first_20.to_string()
    # Calculate the accuracy of the model.
    accuracy = np.mean(predictions == y_test)
    Result=''
    Result=Result+'Accuracy: '+str(accuracy)+'\n \n'
    Result=Result+text_result+" \n"
    return Result


# Frame for file selection
frame_file = Frame(root)
frame_file.grid(row=0, column=0, padx=10, pady=10)

label_path = Label(frame_file, text="Select CSV file:")
label_path.grid(row=0, column=0)

entry_path = Entry(frame_file, width=50)
entry_path.grid(row=0, column=1, padx=10)

button_browse = Button(frame_file, text="Browse", command=browse_file)
button_browse.grid(row=0, column=2)

# Frame for minimum support input
frame_support = Frame(root)
frame_support.grid(row=1, column=0, padx=10, pady=10)

label_Num_Cluster = Label(frame_support, text="Enter Percentage of Training data :")
label_Num_Cluster.grid(row=1, column=0)

entry_Num_Cluster = Entry(frame_support, textvariable=Training_size, width=10)
entry_Num_Cluster.grid(row=1, column=1, padx=10)

label_Percentage_record = Label(frame_support, text="Percentage of Record")
label_Percentage_record.grid(row=2, column=0)

entry_Percentage_record = Entry(frame_support, textvariable=Percentage_data, width=10)
entry_Percentage_record.grid(row=2, column=1, padx=10)


# Analyze button
button_analyze = Button(root, text="Analyze Data", command=analyze_data)
button_analyze.grid(row=3, column=0, pady=10)

result_frame = tk.Frame(root)
result_frame.grid(row=4, column=0, padx=10, pady=10)

result_label = tk.Label(result_frame, text="Results")
result_label.grid(row=0, column=0, padx=10, pady=10)

result_text1 = ScrolledText(result_frame, height=42, width=163)
result_text1.grid(row=0, column=1, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()