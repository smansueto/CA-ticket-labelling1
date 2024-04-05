# Install all dependencies required below.
# !pip install pandas numpy seaborn matplotlib scikit-learn neattext

# Load EDA Pkgs
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# Load ML Pkgs
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Transformers
from sklearn.feature_extraction.text import CountVectorizer

# Others
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Load Pkgs
from sklearn.multioutput import MultiOutputClassifier
import neattext.functions as nfx

# Initialize Flask app
app = Flask(__name__)

 # Load Dataset
df = pd.read_csv("/Users/stephen/Desktop/SPROUT/[SPROUT] CA Project 2.0/database/[SPROUT] 03 Sample External Dataset - Sheet1.csv")
df = df.rename(columns=lambda x: x.strip())

# Concatenate Ticket Subject and Body
df['Complete Ticket'] = df['Client Complaint'].str.cat(df['Ticket Body'], sep='; ')
df.insert(loc=2, column='Complete Ticket', value=df.pop('Complete Ticket'))

# Turn int into a str
df['Support Level'] = df['Support Level'].astype(str)

# Features & Labels
Xfeatures = df['Complete Ticket']
ylabels = df[['Type of Product','Priority','Type of Complaint','Support Level']]

# Split Data
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=7)

# Build a pieline for the model
pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),
                        ('lr_multi',MultiOutputClassifier(LogisticRegression()))])

# Route to display the HTML form
@app.route('/')
def index():
    return render_template('submit_page.html')

# Route to handle the form submission; initial HTML
@app.route('/submit', methods=['POST'])
def submit():
    # Retrieving data from form
    subject_type = request.form['subjectType']
    ticket_body = request.form['ticketBody']
    
    global input_text
    input_text = subject_type + "; " + ticket_body

    # Fit on Dataset
    pipe_lr.fit(x_train,y_train)

    # Accuracy Score
    pipe_lr.score(x_test,y_test)

    # ML estimators for multi-output classification
    pipe = Pipeline(steps=[('cv',CountVectorizer()),('rf',KNeighborsClassifier(n_neighbors=4))])
    pipe.fit(x_train,y_train)

    # Print results
    arr = pipe.predict([input_text])
    global value_1, value_2, value_3, value_4
    value_1 = arr[0][0]  
    value_2 = arr[0][1] 
    value_3 = arr[0][2]  
    value_4 = arr[0][3]

    return redirect(url_for('table'))

# Final HTML
@app.route('/table')
def table():
    return render_template('results_page.html', type_product=value_1, priority=value_2, type_complaint=value_3, support=value_4)

if __name__ == '__main__':
    app.run(debug=True)