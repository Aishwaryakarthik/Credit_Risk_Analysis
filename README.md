# Credit_Risk_Analysis
# Overview of Project
the goal is to build up your skills in data preparation, statistical reasoning, and machine learning. You are now ready to apply machine learning to solve a real-world challenge: credit card risk.

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, you’ll need to employ different techniques to train and evaluate models with unbalanced classes. Jill asks you to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, you’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, you’ll use a combinatorial approach of over and undersampling using the SMOTEENN algorithm. Next, you’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once you’re done, you’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

# Deliverables:

This new assignment consists of three technical analysis deliverables and a written report.

1.Deliverable 1: Use Resampling Models to Predict Credit Risk
2.Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk
3.Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
4.Deliverable 4: A Written Report on the Credit Risk Analysis README.md

# Deliverables:

This new assignment consists of three technical analysis deliverables and a proposal for further statistical study:

Data Source: Module-17-Challenge-Resources.zip and LoanStats_2019Q1.csv
Data Tools: credit_risk_resampling_starter_code.ipynb and credit_risk_ensemble_starter_code.ipynb.
Software: Python 3.9, Visual Studio Code 1.50.0, Anaconda 4.8.5, Jupyter Notebook 6.1.4 and Pandas.

# Supervised Machine Learning and Credit Risk

## Predicting Credit Risk

### Create a Machine Learning Environment

Your new virtual environment will use Python 3.7 and accompanying Anaconda packages. After creating the new virtual environment, you'll install the imbalanced-learn library in that environment.

NOTE Consult the imbalanced-learn documentation for additional information about the imbalanced-learn library.

Check out the macOS instructions below, or go down to the Windows instructions.

macOS Setup Before we create a new environment in macOS, we'll need to update the global conda environment:

If your PythonData environment is activated when you launch the command line, deactivate the environment.

Update the global conda environment by typing conda update conda and press Enter.

After all the packages are collected, you'll see the prompt Proceed ([y]/n)?. Press the "Y" key (for "yes") and press Enter.

In the command line, type conda create -n mlenv python=3.7 anaconda. The name of your new environment is mlenv.

After all the packages are collected, you'll see the prompt Proceed ([y]/n)?. Press the "Y" key (for "yes") and press Enter.

Activate your mlenv environment by typing conda activate mlenv and press Enter.

Check Dependencies for the imbalanced-learn Package
Before we install the imbalanced-learn package, we need to confirm that all of the package dependencies are satisfied in our mlenv environment:

NumPy, version 1.11 or later
SciPy, version 0.17 or later
Scikit-learn, version 0.21 or later
On the command line, you can check all packages that begin with numpy, scipy, and scikit-learn when you type conda list | grep and press Enter. The grep command will search for patterns of the text numpy in our conda list. For example, when we type conda list | grep numpy and press Enter, the output should be as follows:

![image](https://user-images.githubusercontent.com/99555513/179023176-36e371eb-9291-4dc0-bdc8-052253333e7d.png)

### Check Dependencies for the imbalanced-learn Package

Before we install the imbalanced-learn package, we need to confirm that all of the package dependencies are satisfied in our mlenv environment:

NumPy, version 1.11 or later
SciPy, version 0.17 or later
Scikit-learn, version 0.21 or later
In the Anaconda Prompt, you can check all packages that begin with numpy, scipy, and scikit-learn when you type conda list | findstr and press Enter. The findstr command will search for patterns of the text in our conda list. For example, when we type conda list | findstr numpy and press Enter, the output should be as follows:

![image](https://user-images.githubusercontent.com/99555513/179023391-845de56a-5ada-43d6-9db8-b24ae3a23780.png)

From the output, we can see that our numpy dependency meets the installation requirements for the imbalanced-learn package.

Additionally, you can type python followed by the command argument -c, and then "import package_name;print(package_name.version)" to verify which version of a package is installed in an environment, where package_name is the name of the package you want to verify:

Type python -c "import numpy;print(numpy.version)" and press Enter to see the version of numpy in your mlenv environment.

### Install the imbalanced-learn Package
Now that our dependencies have been met, we can install the imbalanced-learn package in our mlenv environment.

With the mlenv environment activated, either in the Terminal in macOS or in the Anaconda Prompt (mlenv) in Windows, type the following:

conda install -c conda-forge imbalanced-learn

Then press Enter.

After all the packages are collected, you'll see the prompt Proceed ([y]/n)?. Press the "Y" key (for "yes") and press Enter.

### Add the Machine Learning Environment to Jupyter Notebook

To use the mlenv environment we just created in the Jupyter Notebook, we need to add it to the kernels. In the command line, type python -m ipykernel install --user --name mlenv and press Enter.

To check if the mlenv is installed, launch the Jupyter Notebook and click the "New" dropdown menu:

![image](https://user-images.githubusercontent.com/99555513/179023641-3321e59f-3f93-4de0-a21e-1d6fff34e4f1.png)



# Deliverable 1:

## Use Resampling Models to Predict Credit Risk

### Deliverable Requirements:

Using your knowledge of the imbalanced-learn and scikit-learn libraries, you’ll evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, you’ll use the oversampling RandomOverSampler and SMOTE algorithms, and then you’ll use the undersampling ClusterCentroids algorithm. Using these algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

To Deliver.

#### Follow the instructions below:

Follow the instructions below and use the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 1.

Open the credit_risk_resampling_starter_code.ipynb file, rename it credit_risk_resampling.ipynb, and save it to your Credit_Risk_Analysis folder.

Using the information we’ve provided in the starter code, create your training and target variables by completing the following steps:

Create the training variables by converting the string values into numerical ones using the get_dummies() method.
Create the target variables.
Check the balance of the target variables.
Next, begin resampling the training data. First, use the oversampling RandomOverSampler and SMOTE algorithms to resample the data, then use the undersampling ClusterCentroids algorithm to resample the data. For each resampling algorithm, do the following:

1.Use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
2.Calculate the accuracy score of the model.
3.Generate a confusion matrix.
4.Print out the imbalanced classification report.
5.Save your credit_risk_resampling.ipynb file to your Credit_Risk_Analysis folder.

### Deliverable 1 Requirements

For all three algorithms, the following have been completed:

1.An accuracy score for the model is calculated
2.A confusion matrix has been generated
3.An imbalanced classification report has been generated

# Deliverable 2:

## Use the SMOTEENN algorithm to Predict Credit Risk

### Deliverable Requirements:

Using your knowledge of the imbalanced-learn and scikit-learn libraries, you’ll use a combinatorial approach of over and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the SMOTEENN algorithm, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

To Deliver.

### Follow the instructions below:

Follow the instructions below and use the information in the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 2.

Continue using your credit_risk_resampling.ipynb file where you have already created your training and target variables.
Using the information we have provided in the starter code, resample the training data using the SMOTEENN algorithm.
After the data is resampled, use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
Save your credit_risk_resampling.ipynb file to your Credit_Risk_Analysis folder.

### Deliverable 2 Requirements
The combinatorial SMOTEENN algorithm does the following:

1.An accuracy score for the model is calculated
2.A confusion matrix has been generated
3.An imbalanced classification report has been generated


# Deliverable 3:

## Use Ensemble Classifiers to Predict Credit Risk

### Deliverable Requirements:

Using your knowledge of the imblearn.ensemble library, you’ll train and compare two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model. Using both algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

To Deliver.

### Follow the instructions below:

#### Follow the instructions below and use the information in the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 3.

1.Open the credit_risk_ensemble_starter_code.ipynb file, rename it credit_risk_ensemble.ipynb, and save it to your Credit_Risk_Analysis folder.

2.Using the information we have provided in the starter code, create your training and target variables by completing the following:

  Create the training variables by converting the string values into numerical ones using the get_dummies() method.
  Create the target variables.
  Check the balance of the target variables.
  
3.Resample the training data using the BalancedRandomForestClassifier algorithm with 100 estimators.
  Consult the following Random Forest documentation for an example.
  
5.Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.

6.Print the feature importance sorted in descending order (from most to least important feature), along with the feature score.

7.Next, resample the training data using the EasyEnsembleClassifier algorithm with 100 estimators.
  Consult the following Easy Ensemble documentation for an example.
  
8.After the data is resampled, use the LogisticRegression classifier to make predictions and evaluate the model’s performance.

9.Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
Save your credit_risk_ensemble.ipynb file to your Credit_Risk_Analysis folder.

#### Deliverable 3 Requirements

##### The BalancedRandomForestClassifier algorithm does teh following

1.An accuracy score for the model is calculated

2.A confusion matrix has been generated

3.An imbalanced classification report has been generated

4.The features are sorted in descending order by feature importance

##### The EasyEnsembleClassifier algorithm does the following:

1.An accuracy score of the model is calculated

2.A confusion matrix has been generated

3.An imbalanced classification report has been generated

### DELIVERABLE RESULTS:

#### Below are the results from the various techniques used to predictive model for High-Risk loans.

<img width="527" alt="Imbalanced classification report" src="https://user-images.githubusercontent.com/99555513/179028532-f60f55b5-b75e-4f19-add0-948beae3d87e.png">

<img width="356" alt="ACCURACY SCORE" src="https://user-images.githubusercontent.com/99555513/179028581-6ec3fe30-c4ce-4619-aac9-2d1733c5e79f.png">

<img width="467" alt="smoteen" src="https://user-images.githubusercontent.com/99555513/179028695-e40410f4-7662-4f91-809b-600e7d3fbe9e.png">

<img width="359" alt="balanced accuracy score" src="https://user-images.githubusercontent.com/99555513/179029287-360063d7-ff91-45b3-a75b-494c812a8ef5.png">

<img width="480" alt="confusion matrix" src="https://user-images.githubusercontent.com/99555513/179029329-a8461b48-6d00-44ee-8dd7-fd543ddb245e.png">

<img width="628" alt="imbalanced classification report2" src="https://user-images.githubusercontent.com/99555513/179029387-1ed62593-719f-47f2-9db0-01322c58d7da.png">


