import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

df_1=pd.read_csv("csv/first_telc.csv")


@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    
    '''
    SeniorCitizen
    MonthlyCharges
    TotalCharges
    gender
    Partner
    Dependents
    PhoneService
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaperlessBilling
    PaymentMethod
    tenure
    '''
    

    inputQuery1 = request.form['query1'] if request.form['query1'] != "" else 0
    inputQuery2 = request.form['query2'] if request.form['query2'] != "" else 1.00
    inputQuery3 = request.form['query3'] if request.form['query3'] != "" else 2.00
    inputQuery4 = request.form['query4'] if request.form['query4'] != "" else "Female"
    inputQuery5 = request.form['query5'] if request.form['query5'] != "" else "Yes"
    inputQuery6 = request.form['query6'] if request.form['query6'] != "" else "No"
    inputQuery7 = request.form['query7'] if request.form['query7'] != "" else "No"
    inputQuery8 = request.form['query8'] if request.form['query8'] != "" else "No phone service"
    inputQuery9 = request.form['query9'] if request.form['query9'] != "" else "DSL"
    inputQuery10 = request.form['query10'] if request.form['query10'] != "" else "Yes"
    inputQuery11 = request.form['query11'] if request.form['query11'] != "" else "Yes"
    inputQuery12 = request.form['query12'] if request.form['query12'] != "" else "Yes"
    inputQuery13 = request.form['query13'] if request.form['query13'] != "" else "Yes"
    inputQuery14 = request.form['query14'] if request.form['query14'] != "" else "Yes"
    inputQuery15 = request.form['query15'] if request.form['query15'] != "" else "Yes"
    inputQuery16 = request.form['query16'] if request.form['query16'] != "" else "One year"
    inputQuery17 = request.form['query17'] if request.form['query17'] != "" else "Yes"
    inputQuery18 = request.form['query18'] if request.form['query18'] != "" else "Electronic check"
    inputQuery19 = request.form['query19'] if request.form['query19'] != "" else 1

    model = pickle.load(open("model/model.sav", "rb"))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]

    
    new_df = pd.DataFrame(data, columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                           'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                           'PaymentMethod', 'tenure'])
    
    df_2 = pd.concat([df_1, new_df], ignore_index = True) 
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    #drop column customerID and tenure
    df_2.drop(columns= ['tenure'], axis=1, inplace=True)   
    
    
    
    
    new_df__dummies = pd.get_dummies(df_2, columns=['gender', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod','tenure_group'])
    
    new_df__dummies = new_df__dummies.drop('Unnamed: 0',axis=1)
        
    
    single = model.predict(new_df__dummies.tail(1))
    probablity = model.predict_proba(new_df__dummies.tail(1))[:,1]
    
    if single==1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('home.html', output1=o1, output2=o2, 
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'], 
                           query7 = request.form['query7'], 
                           query8 = request.form['query8'], 
                           query9 = request.form['query9'], 
                           query10 = request.form['query10'], 
                           query11 = request.form['query11'], 
                           query12 = request.form['query12'], 
                           query13 = request.form['query13'], 
                           query14 = request.form['query14'], 
                           query15 = request.form['query15'], 
                           query16 = request.form['query16'], 
                           query17 = request.form['query17'],
                           query18 = request.form['query18'], 
                           query19 = request.form['query19'])


if __name__ == "__main__":
    app.run(debug=False)