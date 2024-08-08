from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import numpy as np
from IPython.display import clear_output
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, send_file
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.circuit.library import ZZFeatureMap
from matplotlib import pyplot as plt
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from flask_cors import CORS
from sklearn.svm import SVC
import json
from qiskit_machine_learning.algorithms import QSVC
from qiskit.circuit.library import RealAmplitudes,ZZFeatureMap, TwoLocal
# from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import SPSA, COBYLA, L_BFGS_B, POWELL
import os
from qiskit_machine_learning.algorithms import VQC
import time
from flask import Flask, jsonify, request, render_template
import io
import math

app = Flask(__name__)
CORS(app) 

@app.route('/')
def index():
    return render_template('index.html')

def generate_data(training_size, test_size, adhoc_dimension):
    # Generate ad-hoc data
    train_features, train_labels, test_features, test_labels, _ = ad_hoc_data(
        training_size=training_size,
        test_size=test_size,
        n=adhoc_dimension,
        gap=0.3,  # Note: Gap parameter is still required by the ad_hoc_data function
        plot_data=False,
        one_hot=False,
        include_sample_total=True,
    )
    
    res = {
        'train_features': train_features.tolist(),
        'train_labels': train_labels.tolist(),
        'test_features': test_features.tolist(),
        'test_labels': test_labels.tolist()
    }

    filename = 'model_results.csv'
    try:
        # Check if the file exists
        if os.path.exists(filename):
            # Remove the file
            os.remove(filename)
            print(f"File '{filename}' has been deleted.")
        else:
            print(f"The file '{filename}' does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting the file: {e}")
    with open('adhoc_data.json', 'w') as file:
        json.dump(res, file, indent=2)

    # Convert to DataFrame for easier display
    train_df = pd.DataFrame(train_features, columns=[f'Feature_{i}' for i in range(adhoc_dimension)])
    train_df['Label'] = train_labels
    test_df = pd.DataFrame(test_features, columns=[f'Feature_{i}' for i in range(adhoc_dimension)])
    test_df['Label'] = test_labels
    
    # Calculate class imbalance ratio
    def calculate_imbalance_ratio(labels):
        counts = pd.Series(labels).value_counts()
        total = len(labels)
        ratio = {f"Label {label}": f"{(count / total * 100):.1f}%" for label, count in counts.items()}
        return ratio



    train_imbalance_ratio = calculate_imbalance_ratio(train_labels)
    test_imbalance_ratio = calculate_imbalance_ratio(test_labels)

    return train_df, test_df, train_imbalance_ratio, test_imbalance_ratio


@app.route('/generate-data', methods=['POST'])
def generate_data_endpoint():
    file_path = 'adhoc_data.json'

# Check if the file exists
    if os.path.isfile(file_path):
        try:
            # Remove the file
            os.remove(file_path)
            print(f"File '{file_path}' has been removed.")
        except Exception as e:
            print(f"Error occurred: {e}")
    else:
        print(f"File '{file_path}' does not exist.")
    data = request.json
    # print(data)
    try:
        training_size = data['training_size']
        test_size = data['test_size']
        adhoc_dimension = data['adhoc_dimension']
    except KeyError:
        return jsonify({'error': 'Missing required fields'}), 400
    
    if not all(isinstance(i, int) for i in [training_size, test_size, adhoc_dimension]):
        return jsonify({'error': 'Invalid data types'}), 400

    # Validate adhoc_dimension
    if adhoc_dimension not in [2, 3]:
        return jsonify({'error': 'Supported values of adhoc_dimension are 2 and 3 only.'}), 400

    # Generate the data
    try:
        train_df, test_df, train_imbalance_ratio, test_imbalance_ratio = generate_data(training_size, test_size, adhoc_dimension)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    # Filter data to include at least one record of each class
    def filter_top_records(df):
        # Retrieve records from each class
        df = df.drop_duplicates()
        target_column = 'Label'
    
         # Filter out rows where any feature value is 0 (excluding the target column)
        feature_columns = [col for col in df.columns if col != target_column]
        df = df[df[feature_columns].ne(0).all(axis=1)]
        class_0 = df[df['Label'] == 0]
        class_1 = df[df['Label'] == 1]
        combined = pd.concat([class_0[:2],class_1[:3]])
        combined = combined.drop_duplicates().sample(frac=1).head(5)
        return combined

    # Apply filtering to both train and test data
    train_df_filtered = filter_top_records(train_df)
    test_df_filtered = filter_top_records(test_df)
    

    # Prepare data for response
    response = {
        'train_data': train_df_filtered.to_dict(orient='records'),
        'test_data': test_df_filtered.to_dict(orient='records'),
        'train_shape': train_df.shape,
        'test_shape': test_df.shape,
        'train_imbalance_ratio': train_imbalance_ratio,
        'test_imbalance_ratio': test_imbalance_ratio
    }
    # with open('data.json', 'w') as file:
    #     json.dump(response, file)
    # print(response)
    return jsonify(response), 200

def generate_data_from_csv(training_size, test_size, feature_count, imbalance_ratio):
    print(training_size, test_size, feature_count, imbalance_ratio)
    # Load the dataset from CSV
    df = pd.read_csv('./creditcard_filtered.csv')
    # Remove duplicate rows
    df = df.drop_duplicates()
    # Remove rows with any null values
    df = df.dropna()
    
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    
    # Check if 'Class' column exists
    if 'Class' not in df.columns:
        raise ValueError("The CSV file must contain a column named 'Class'.")
    
    # Ensure feature_count is within the available features
    if feature_count > len(df.columns) - 1:
        raise ValueError("Feature count exceeds the number of available features in the dataset.")
    
    selected_features = df.iloc[:, :feature_count]
    selected_features['Class'] = df['Class']
    
    df = selected_features
    
    # Separate features and labels
    features = df.drop(columns=['Class'])
    labels = df['Class']
    
    # Calculate the number of samples required for each class in train and test sets
    def calculate_sample_counts(total_size, imbalance_ratio):
        if imbalance_ratio >= 1:
            raise ValueError("Imbalance ratio must be less than 1.")
        num_class1 = math.ceil(total_size * imbalance_ratio)
        num_class0 = math.ceil(total_size - num_class1)
        # print(num_class0,num_class1)
        return num_class0, num_class1

    def sample_classes(df, class0_count, class1_count):
        # Separate the majority and minority class samples
        class0_samples = df[df['Class'] == 0]
        class1_samples = df[df['Class'] == 1]
        # print("class1_count",class1_count)
        # print("len(class1_samples):",len(class1_samples))
        # print(len(class0_samples),class0_count)
        # Handle cases where the requested number of samples is more than available
        if class0_count > len(class0_samples):
            class0_count = len(class0_samples)
        if class1_count > len(class1_samples):
            class1_count = len(class1_samples)
        # print("class1_count",class1_count)
        # # Sample the classes
        if class0_count > 0:
            class0_samples = class0_samples.sample(n=class0_count, random_state=42)
            # print(class0_count)
            # print(class0_samples)
        if class1_count > 0:
            # print(class1_count)
            # print(class1_samples)
            class1_samples = class1_samples.sample(n=class1_count, random_state=42)

        # Combine the undersampled classes
        sampled_df = pd.concat([class0_samples, class1_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
        # print(sampled_df)
        return sampled_df

    # Calculate sample counts for training and testing
    train_class0_count, train_class1_count = calculate_sample_counts(training_size, imbalance_ratio)
    test_class0_count, test_class1_count = calculate_sample_counts(test_size, imbalance_ratio)

    # Sample data for training and testing sets
    df_train = sample_classes(df, train_class0_count, train_class1_count)
    df_test = sample_classes(df, test_class0_count, test_class1_count)
    
    # Separate features and labels from the sampled data
    train_features = df_train.drop(columns=['Class'])
    train_labels = df_train['Class']
    test_features = df_test.drop(columns=['Class'])
    test_labels = df_test['Class']
    
    # Prepare results
    def convert_df_to_list_of_lists(df):
        return df.values.tolist()

# Assuming train_features, train_labels, test_features, and test_labels are your DataFrames
# train_features_list = convert_df_to_list_of_lists(train_features)
# test_features_list = convert_df_to_list_of_lists(test_features)

# Create the result dictionary
    res = {
        'train_features': convert_df_to_list_of_lists(train_features),
        'train_labels': train_labels.tolist(),
        'test_features': convert_df_to_list_of_lists(test_features),
        'test_labels': test_labels.tolist()
    }
    # res = {
    #     'train_features': train_features.to_dict(orient='records'),
    #     'train_labels': train_labels.tolist(),
    #     'test_features': test_features.to_dict(orient='records'),
    #     'test_labels': test_labels.tolist()
    # }

    # Save the results to JSON
    filename = 'model_results.csv'
    try:
        # Check if the file exists
        if os.path.exists(filename):
            # Remove the file
            os.remove(filename)
            print(f"File '{filename}' has been deleted.")
        else:
            print(f"The file '{filename}' does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting the file: {e}")
    with open('adhoc_data.json', 'w') as file:
        json.dump(res, file, indent=2)

    # Convert to DataFrame for easier display
    train_df = pd.DataFrame(train_features)
    train_df['Label'] = train_labels
    test_df = pd.DataFrame(test_features)
    test_df['Label'] = test_labels
    
    # Calculate class imbalance ratio
    def calculate_imbalance_ratio(labels):
        counts = pd.Series(labels).value_counts()
        total = len(labels)
        ratio = {f"Label {label}": f"{(count / total * 100):.1f}%" for label, count in counts.items()}
        return ratio

    train_imbalance_ratio = calculate_imbalance_ratio(train_labels)
    test_imbalance_ratio = calculate_imbalance_ratio(test_labels)

    return train_df, test_df, train_imbalance_ratio, test_imbalance_ratio, train_class0_count, train_class1_count, test_class0_count, test_class1_count

@app.route('/generate-fraud-data', methods=['POST'])
def generate_fraud_data_endpoint():
    file_path = 'adhoc_data.json'
    
    data = request.json
    # print(data)
    try:
        training_size = data['training_size']
        test_size = data['test_size']
        adhoc_dimension = data['adhoc_dimension']
        imbalance_ratio =data['imbalance_ratio']
    except KeyError:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        train_df, test_df, train_imbalance_ratio, test_imbalance_ratio, train_class0_count, train_class1_count, test_class0_count, test_class1_count = generate_data_from_csv(
            training_size, test_size, adhoc_dimension, imbalance_ratio
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Filter data to include at least one record of each class
    def filter_top_records(df):
        target_column = 'Label'
    
        # Filter out rows where any feature value is 0 (excluding the target column)
        feature_columns = [col for col in df.columns if col != target_column]
        df = df[df[feature_columns].ne(0).all(axis=1)]
        class_0 = df[df['Label'] == 0]
        class_1 = df[df['Label'] == 1]
        combined = pd.concat([class_0[:2], class_1[:3]])
        combined = combined.drop_duplicates().sample(frac=1).head(5)
        return combined

    # Apply filtering to both train and test data
    train_df_filtered = filter_top_records(train_df)
    test_df_filtered = filter_top_records(test_df)
    
    # Prepare data for response
    response = {
        'train_data': train_df_filtered.to_dict(orient='records'),
        'test_data': test_df_filtered.to_dict(orient='records'),
        'train_shape': train_df.shape,
        'test_shape': test_df.shape,
        'train_imbalance_ratio': train_imbalance_ratio,
        'test_imbalance_ratio': test_imbalance_ratio, 
        'train_class0_count':train_class0_count,
        'train_class1_count':train_class1_count, 
        'test_class0_count':test_class0_count, 
        'test_class1_count':test_class1_count
    }

    return jsonify(response), 200

def load_data():
    with open('adhoc_data.json') as file:
        data = json.load(file)
    
    train_features = np.array(data['train_features'])
    train_labels = np.array(data['train_labels'])
    test_features = np.array(data['test_features'])
    test_labels = np.array(data['test_labels'])
    
    return train_features, train_labels, test_features, test_labels


def save_results(results, filename='model_results.csv'):
    # Ensure results is a dictionary
    if not isinstance(results, dict):
        raise ValueError("Results should be a dictionary.")
    
    # Define the columns for the CSV
    columns = ['classifier_type', 'model_type', 'train_accuracy', 'train_precision', 'train_recall', 'test_accuracy', 'test_precision', 'test_recall']

    # Convert results to a DataFrame
    df_new = pd.DataFrame([results], columns=columns)
    
    # Check if the file exists
    if os.path.exists(filename):
        # Append the new results to the existing CSV
        df_new.to_csv(filename, mode='a', header=False, index=False)
    else:
        # Create a new CSV with the column headers and new results
        df_new.to_csv(filename, mode='w', header=True, index=False)

@app.route('/logistic-regression', methods=['GET'])
def run_logistic_regression():
    train_features, train_labels, test_features, test_labels = load_data()
    # print(train_features, train_labels, test_features, test_labels)
    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    start_time = time.time()
    # print(start_time)
    model.fit(train_features, train_labels)
    training_time = time.time() - start_time
    # Predict and calculate metrics
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    # print("Length of test_labels:", len(test_labels))
    # print("Length of test_predictions:", len(test_predictions))

    metrics = {
        'classifier_type': 'Logistic Regression',
        'model_type': 'Classical ML',
        'train_accuracy': round(accuracy_score(train_labels, train_predictions), 2),
        'train_precision': round(precision_score(train_labels, train_predictions, average='weighted'), 2),
        'train_recall': round(recall_score(train_labels, train_predictions, average='weighted'), 2),
        'test_accuracy': round(accuracy_score(test_labels, test_predictions), 2),
        'test_precision': round(precision_score(test_labels, test_predictions, average='weighted'), 2),
        'test_recall': round(recall_score(test_labels, test_predictions, average='weighted'), 2),
        'training_time': round(training_time, 2)
    }
    # print(metrics)
    # Save results to a JSON file
    save_results(metrics)
    
    return jsonify(metrics), 200

@app.route('/decision-tree', methods=['GET'])
def run_decision_tree():
    train_features, train_labels, test_features, test_labels = load_data()
    
    # Train Decision Tree model
    model = DecisionTreeClassifier()
    start_time = time.time()
    model.fit(train_features, train_labels)
    training_time = time.time() - start_time
    # Predict and calculate metrics
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    metrics = {
        'classifier_type': 'Decision Tree',
        'model_type': 'Classical ML',
        'train_accuracy': round(accuracy_score(train_labels, train_predictions), 2),
        'train_precision': round(precision_score(train_labels, train_predictions, average='weighted'), 2),
        'train_recall': round(recall_score(train_labels, train_predictions, average='weighted'), 2),
        'test_accuracy': round(accuracy_score(test_labels, test_predictions), 2),
        'test_precision': round(precision_score(test_labels, test_predictions, average='weighted'), 2),
        'test_recall': round(recall_score(test_labels, test_predictions, average='weighted'), 2),
        'training_time': round(training_time, 2)
    }
        
    # Save results to a JSON file
    save_results(metrics)
    
    return jsonify(metrics), 200

@app.route('/random-forest', methods=['GET'])
def run_random_forest():
    train_features, train_labels, test_features, test_labels = load_data()
    
    # Train Random Forest model
    model = RandomForestClassifier()
    start_time = time.time()
    model.fit(train_features, train_labels)
    training_time = time.time() - start_time
    # Predict and calculate metrics
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    metrics = {
        'classifier_type': 'Random Forest',
        'model_type': 'Classical ML',
        'train_accuracy': round(accuracy_score(train_labels, train_predictions), 2),
        'train_precision': round(precision_score(train_labels, train_predictions, average='weighted'), 2),
        'train_recall': round(recall_score(train_labels, train_predictions, average='weighted'), 2),
        'test_accuracy': round(accuracy_score(test_labels, test_predictions), 2),
        'test_precision': round(precision_score(test_labels, test_predictions, average='weighted'), 2),
        'test_recall': round(recall_score(test_labels, test_predictions, average='weighted'), 2),
        'training_time': round(training_time, 2)
    }
        
    # Save results to a JSON file
    save_results(metrics)
    
    return jsonify(metrics), 200

@app.route('/naive-bayes', methods=['GET'])
def run_naive_bayes():
    train_features, train_labels, test_features, test_labels = load_data()
    
    # Train Naive Bayes model
    model = GaussianNB()
    start_time = time.time()
    model.fit(train_features, train_labels)
    training_time = time.time() - start_time
    # Predict and calculate metrics
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    metrics = {
        'classifier_type': 'Naive Bayes',
        'model_type': 'Classical ML',
        'train_accuracy': round(accuracy_score(train_labels, train_predictions), 2),
        'train_precision': round(precision_score(train_labels, train_predictions, average='weighted'), 2),
        'train_recall': round(recall_score(train_labels, train_predictions, average='weighted'), 2),
        'test_accuracy': round(accuracy_score(test_labels, test_predictions), 2),
        'test_precision': round(precision_score(test_labels, test_predictions, average='weighted'), 2),
        'test_recall': round(recall_score(test_labels, test_predictions, average='weighted'), 2),
         'training_time': round(training_time, 2)
    }
        
    save_results(metrics)
    
    return jsonify(metrics), 200

@app.route('/gradient-boosting', methods=['GET'])
def run_gradient_boosting():
    train_features, train_labels, test_features, test_labels = load_data()
    
    # Train Gradient Boosting model
    model = GradientBoostingClassifier()
    start_time = time.time()
    model.fit(train_features, train_labels)
    training_time = time.time() - start_time
    # Predict and calculate metrics
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    metrics = {
        'classifier_type': 'Gradient Boosting',
        'model_type': 'Classical ML',
        'train_accuracy': round(accuracy_score(train_labels, train_predictions), 2),
        'train_precision': round(precision_score(train_labels, train_predictions, average='weighted'), 2),
        'train_recall': round(recall_score(train_labels, train_predictions, average='weighted'), 2),
        'test_accuracy': round(accuracy_score(test_labels, test_predictions), 2),
        'test_precision': round(precision_score(test_labels, test_predictions, average='weighted'), 2),
        'test_recall': round(recall_score(test_labels, test_predictions, average='weighted'), 2),
        'training_time': round(training_time, 2)
    }
        
    # Save results to a JSON file
    save_results(metrics)
    
    return jsonify(metrics), 200

@app.route('/classical-qsvm', methods=['GET'])
def run_qsvm():
    train_features, train_labels, test_features, test_labels = load_data()

    # Prepare QSVM
    adhoc_dimension = train_features.shape[1]
    adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)
    
    # Train QSVM model and measure training time
    adhoc_svc = SVC(kernel=adhoc_kernel.evaluate)
    start_time = time.time()
    adhoc_svc.fit(train_features, train_labels)
    training_time = time.time() - start_time
    train_predictions = adhoc_svc.predict(train_features)
    test_predictions = adhoc_svc.predict(test_features)
    
    # Calculate metrics
    metrics = {
        'classifier_type': 'Classical SVM with Quantum Kernel',
        'model_type': 'Quantum ML',
        'train_accuracy': round(adhoc_svc.score(train_features, train_labels), 2),
        'train_precision': round(precision_score(train_labels, train_predictions, average='weighted'), 2),
        'train_recall': round(recall_score(train_labels, train_predictions, average='weighted'), 2),
        'test_accuracy': round(adhoc_svc.score(test_features, test_labels), 2),
        'test_precision': round(precision_score(test_labels, test_predictions, average='weighted'), 2),
        'test_recall': round(recall_score(test_labels, test_predictions, average='weighted'), 2),
        'training_time': round(training_time, 3)
    }
        
    # Save results to a JSON file
    save_results(metrics)
    
    return jsonify(metrics), 200

@app.route('/quantum-qsvm', methods=['GET'])
def run_classical_qsvm():
    train_features, train_labels, test_features, test_labels = load_data()

    # Prepare QSVM
    adhoc_dimension = train_features.shape[1]
    adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)
    
    # Train QSVM model and measure training time
    adhoc_svc = QSVC(kernel=adhoc_kernel.evaluate)
    start_time = time.time()
    adhoc_svc.fit(train_features, train_labels)
    training_time = time.time() - start_time
    train_predictions = adhoc_svc.predict(train_features)
    test_predictions = adhoc_svc.predict(test_features)
    
    # Calculate metrics
    metrics = {
        'classifier_type': 'Quantum SVM',
        'model_type': 'Quantum ML',
        'train_accuracy': round(adhoc_svc.score(train_features, train_labels), 2),
        'train_precision': round(precision_score(train_labels, train_predictions, average='weighted'), 2),
        'train_recall': round(recall_score(train_labels, train_predictions, average='weighted'), 2),
        'test_accuracy': round(adhoc_svc.score(test_features, test_labels), 2),
        'test_precision': round(precision_score(test_labels, test_predictions, average='weighted'), 2),
        'test_recall': round(recall_score(test_labels, test_predictions, average='weighted'), 2),
        'training_time': round(training_time, 3)
    }
        
    # Save results to a JSON file
    save_results(metrics)
    
    return jsonify(metrics), 200

@app.route('/vqc', methods=['GET'])
def run_vqc():
    train_features, train_labels, test_features, test_labels = load_data()

    adhoc_dimension = train_features.shape[1]
    adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")
    sampler = Sampler()
    
    # Prepare VQC
    ansatz = RealAmplitudes(adhoc_dimension, entanglement='linear', reps=2, insert_barriers=True)

    objective_func_vals = []

    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)

    optimizer = COBYLA(maxiter=15)
    vqc = VQC(
        sampler=sampler,
        feature_map=adhoc_feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback_graph,
    )

    # Train VQC model and measure training time
    objective_func_vals = []
    start_time = time.time()
    vqc.fit(train_features, train_labels)
    training_time = time.time() - start_time

    # Predict on training and test data
    train_predictions = vqc.predict(train_features)
    test_predictions = vqc.predict(test_features)

    # Calculate metrics
    metrics = {
        'classifier_type': 'Variational Quantum Classifier',
        'model_type': 'Quantum ML',
        'train_accuracy': round(vqc.score(train_features, train_labels), 3),
        'train_precision': round(precision_score(train_labels, train_predictions, average='weighted'), 3),
        'train_recall': round(recall_score(train_labels, train_predictions, average='weighted'), 3),
        'test_accuracy': round(vqc.score(test_features, test_labels), 3),
        'test_precision': round(precision_score(test_labels, test_predictions, average='weighted'), 3),
        'test_recall': round(recall_score(test_labels, test_predictions, average='weighted'), 3),
        'training_time': round(training_time, 3)
    }

    # Save results to a JSON file
    save_results(metrics)

    # Plot objective function value against iteration
    # plt.figure(figsize=(12, 6))
    # plt.title("Objective function value against iteration")
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective function value")
    # plt.plot(range(len(objective_func_vals)), objective_func_vals)
    # plt.show()

    return jsonify(metrics), 200

@app.route('/comparison-plot')
def plot():
    # Set Matplotlib to use the 'Agg' backend
    plt.switch_backend('Agg')

    # Load the data from the CSV file
    filename = 'model_results.csv'
    df = pd.read_csv(filename)
    # print(df)
    # Group by classifier_type and calculate the mean test_recall for each
    df_grouped = df.groupby('classifier_type')['test_recall'].mean().reset_index()
    # print(df_grouped)
    # Create a BytesIO buffer to save the plot image
    img = io.BytesIO()
    plt.clf()
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(df_grouped['classifier_type'], df_grouped['test_recall'], color='skyblue')
    plt.xlabel('Test Recall')
    plt.title('Comparison of Test Recall for Different Classifiers')
    plt.tight_layout()
    plt.xlim(0, 1)
    # Save the plot to the BytesIO buffer
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close('all')
    # Return the image as a response
    return send_file(img, mimetype='image/png')
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
