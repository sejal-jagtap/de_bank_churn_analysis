# BANK CHURN ANALYSIS
Predict which customers might leave your bank before it happens! 🏃

This project analyzes customer churn in a banking dataset. The goal is to predict whether a customer will leave the bank based on their demographic and account information. Using machine learning models such as Random Forest and Logistic Regression, we evaluate the performance and identify key features that influence churn.

This project uses Python, Pandas, Scikit-learn, and Seaborn.
It trains Random Forrest and Logistic Regression models to predict bank churn and includes exploratory data analysis visualizations.
### Jupyter notebook attached as a reference

### 1. ABOUT DATA
   
  Dataset: Customer-Churn-Records.csv (file added in repo)
  The dataset contains information on bank customers, including:

  | Column             | Description                        | 
  | ------------------ | ---------------------------------- | 
  | RowNumber          | Row index                          | 
  | CustomerId         | Unique ID                          | 
  | Surname            | Customer surname                   | 
  | CreditScore        | Credit score                       | 
  | Geography          | Country                            | 
  | Gender             | Male/Female                        | 
  | Age                | Age in years                       | 
  | Tenure             | Number of years with the bank      | 
  | Balance            | Account balance                    | 
  | NumOfProducts      | Number of bank products            | 
  | HasCrCard          | Has credit card (0/1)              | 
  | IsActiveMember     | Active member (0/1)                | 
  | EstimatedSalary    | Annual salary                      | 
  | Exited             | Churn (0/1)                        | 
  | Complain           | Customer complaint (0/1 or yes/no) | 
  | Satisfaction Score | Customer satisfaction              | 
  | Card Type          | Type of card (GOLD/PLATINUM/etc.)  | 
  | Point Earned       | Reward points                      | 

LIBRARIES USED

  -> pandas for data manipulation
  
  -> numpy for numerical operations
  
  -> seaborn & matplotlib for visualization
  
  -> scikit-learn for preprocessing, train/test split, and modeling

### 2. DATA CLEANING
   
  -> Check if data has duplicates.
  
  -> Check if data has NULL/NA values.
  
  -> Drop unnecessary columns that do not influence churn (eg, RowNumber, CustomerId, Surname).
  
  -> Explore data to observe correlation patterns and check if certain columns need encoding.

<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/6720d692-58d7-4319-8836-7c79727ad8ea" />
<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/fa84755a-4bdf-4562-9010-d1192c767a8d" />
<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/f7976700-6bfd-4701-ba15-32a5a33acc2c" />

  
### 3. DATA PREPROCESSING
   
  -> Dropped categorical columns - Geography, Gender, Card Type, and encoded them using one-hot encoding.
  
  -> Concatenated numeric features and encoded categorical variables into a single dataset.
  
  -> Split the dataset into training (80%) and testing (20%) sets.

### 4. MODEL TRAINING

  ##### Divide Train and Test:

  # Training set: To train the model (ML algorithm)
  # Test Set: Model will be evaluated on test set.

  -> Test will consist of 20% of the total dataset.
  -> Use train_test_split from package - sklearn.model_selection (import).
  -> train_test_split shuffles the data and outputs 4 arrays/dataframes.
  -> input args - features: churn_data excluding exited,

                  label: target column, 

                  test_size: 20% for test,

  -> random_state= seed to ensure reproducibilty

  ##### Approach:
  The most commonly used metrics are precision and recall, F1 measure, accuracy and confusion matrix.
  The Scikit Learn library contains classes that can be used to calculate these metrics.

  ##### ML Models:
   
   a. Random Forest Classifier
   
   b. Logistic Regression

   -> Calculate precision, recall, F1 measure, and support using the below scikit learn classes:
   
     classification_report()
   
     confusion_matrix()
   
     accuracy_score

## TESTING

Tests are written with pytest in test_bank_churn.py

Tests cover:

  -> Data loading

  -> Data preprocessing

  -> Feature/Label split
  
  -> Random Forest Training

  -> Logistic Regression Training

  -> Predictions and evaluation

Test Result:

<img width="681" height="257" alt="image" src="https://github.com/user-attachments/assets/ca2f6651-3d26-4511-b1c6-599d6cdf65f6" />

## HOW TO USE THE CODE 💻

Repository structure: de_bank_churn_analysis

<img width="250" height="152" alt="image" src="https://github.com/user-attachments/assets/656ae696-b262-458d-8b52-887ce8e01d9e" />

Note: Dockerfile can be used to build a standalone Docker image if you don’t want to use the dev container.

### 1. Clone repository: (cmd)
   
  git clone https://github.com/sejal-jagtap/de_bank_churn_analysis.git

### 2. Navigate to the folder: (cmd)

  cd de_bank_churn_analysis

## Environment Setup
You can set up your environment in two ways: using VS Code dev container or using Docker directly.

### A] Using Dev Container (Recommended)
  1. Install VSCode (https://code.visualstudio.com/) and the Dev Containers, Remote - Containers extension.
  2. VS Code detects the .devcontainer folder automatically.
  3. Open the cloned project in VS Code.
  4. Open Command Palette:
     Press Ctrl+Shift+P (Windows/Linux), Cmd+Shift+P (Mac)
  5. Reopen in Container:
     Type 'Dev Containers: Reopen in Container' and select.
     Note:
     VS Code uses the .devcontainer/devcontainer.json configuration and the Dockerfile at the repo 
     root to build the container
  6. Dependencies are installed automatically (make install is run via postCreateCommand)
  7. Once the container is ready, you can run the project using the following commands:
     make run    #runs the main python script
     make test   #runs the test file
     make cleans #cleans cache

### OR

### B] Using Docker

Note: Docker Desktop should be installed on your device.
      Check Docker version to confirm (bash): docker --version
      Docker Desktop app should be running

  1. Build Docker Image (bash)
     
     docker build -t bank_churn_analysis  .

  2. Run a container from the image  (bash)
    
     docker run -it --name bank_churn_container -v ${PWD}:/workspaces/de_bank_churn_analysis bank_churn_analysis
  
    Note: after running this you'll be inside the container:
    root@<container_id>:/workspaces/de_bank_churn_analysis#

  3. Run commands inside the container (bash)
     make install #install dependencies
     make run     #runs the main Python script
     make test    #runs the test file
     make cleans  #cleans cache

  4. To exit container (bash)

     exit

  5. To see all containers
    
     docker ps -a

## REFACTORING 
     


