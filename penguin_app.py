import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values  
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

@st.cache()
def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm,body_mass_g,sex):
  specie = model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm,body_mass_g,sex]])
  specie = specie[0]
  if specie == 0:
  	specie_pred = "Adelie"
  elif specie == 1:
   specie_pred = "Chinstrap"
  elif specie == 2:
   specie_pred = "Gentoo"
  return specie_pred

st.title("Penguin Specie Predictor")
bill_length_mm = st.slider("Bill Length (mm)",float(df['bill_length_mm'].min()),float(df['bill_length_mm'].max()))
bill_depth_mm = st.slider("Bill Depth (mm)",float(df['bill_depth_mm'].min()),float(df['bill_depth_mm'].max()))
flipper_length_mm = st.slider("Flipper Length (mm)",float(df['flipper_length_mm'].min()),float(df['flipper_length_mm'].max()))
body_mass_g = st.slider("Body Mass (g)",float(df['body_mass_g'].min()),float(df['body_mass_g'].max())) 

sex_select = st.sidebar.selectbox("Select Sex of Penguin",("Male","Female"))
if sex_select == "Male":
  sex_select = 0
else:
  sex_select = 1

island_select= st.sidebar.selectbox("Select Island",("Biscoe","Dream","Torgersen"))
if island_select == "Biscoe":
  island_select = 0
elif island_select == "Dream":
  island_select = 1
else:
  island_select = 2
model = st.sidebar.selectbox("Select Classifier",("SVM","Random Forest Classifier","Logistic Regression")) 

if st.sidebar.button("Predict"):
  if model == "SVM":
    specie = prediction(svc_model, island_select, bill_length_mm, bill_depth_mm, flipper_length_mm,body_mass_g,sex_select)
    score = svc_score
  if model == "Random Forest Classifier":
    specie = prediction(rf_clf, island_select, bill_length_mm, bill_depth_mm, flipper_length_mm,body_mass_g,sex_select)
    score = rf_clf_score
  if model == "Logistic Regression":
    specie = prediction(log_reg, island_select, bill_length_mm, bill_depth_mm, flipper_length_mm,body_mass_g,sex_select)
    score = log_reg_score
  
  st.write("The predicted Penguin specie is",specie)
  st.write("Accuracy Score of the model is",score)
