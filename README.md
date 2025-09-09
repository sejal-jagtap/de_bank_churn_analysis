# BANK CHURN ANALYSIS
Predict which customers might leave your bank before it happens! 🏃

This project analyzes customer churn in a banking dataset. The goal is to predict whether a customer will leave the bank based on their demographic and account information. Using machine learning models such as Random Forest and Logistic Regression, we evaluate the performance and identify key features that influence churn.

### 1. DATA
   
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

  <img width="471" height="316" alt="image" src="https://github.com/user-attachments/assets/b3208a41-cf94-4e34-a21f-2e51f85acbdc" />
  <img width="471" height="316" alt="image" src="https://github.com/user-attachments/assets/d1874a9f-e64f-406d-9ddc-cf0b5eac400c" />
  <img width="471" height="316" alt="image" src="https://github.com/user-attachments/assets/dab61408-c567-48d7-a0a5-f55489901d0b" />
  
### 3. DATA PREPROCESSING
   
  -> Dropped categorical columns - Geography, Gender, Card Type, and encoded them using one-hot encoding.
  
  -> Concatenated numeric features and encoded categorical variables into a single dataset.
  
  -> Split the dataset into training (80%) and testing (20%) sets.

### 4. ML MODEL
   
   a. Random Forest Classifier
   
   b. Logistic Regression

   -> Calculate precision, recall, F1 measure, and support using the below scikit learn classes:
   
     classification_report()
   
     confusion_matrix()
   
     accuracy_score

   
## HOW TO USE THE CODE 💻

Repository structure:

de_bank_churn_analysis/
└── Customer-Churn-Records.csv             # Dataset used for analysis
├── Makefile                               # Automation for Docker commands
└── README.md                              # Project documentation
├── bank_churn_analysis.py                 # Main Python Flask application
├── bank_churn_analysis_interactive.ipynb  # Jupyter nbk with results
├── requirements.txt                       # Python dependencies

### 1. Clone repository: (cmd)
   
  git clone https://github.com/sejal-jagtap/de_bank_churn_analysis.git

### 2. Create and activate the virtual environment: (cmd)
   
  .de_bank_churn_analysis\Scripts\activate.bat

### 3. Install Dependencies (cmd)
   
   pip install -r requirements.txt
   
### 4. Make file commands (cmd)
   
   make install #Installs dependencies
   
   make lint #Runs flake8

   make format #Formats code with black
   
   make clean #Cleans cache and coverage files

### 6. Run the Python script (cmd)
   
   python bank_churn_analysis.py

### 7. Jupyter notebook attached as a reference





