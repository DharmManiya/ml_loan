import streamlit as st 
import pandas as pd 
import pickle as pk 

st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="🏦",
    layout="centered"
)

n_model=pk.load(open('n_model.pkl','rb')) 
regressor=pk.load(open('regressor.pkl','rb')) 
d_model=pk.load(open('d_model.pkl','rb')) 
model=pk.load(open('model.pkl','rb')) 
scaler=pk.load(open('scaler.pkl','rb')) 
 
st.header('Loan Prediction App')

no_of_dep = st.slider('Choose No of dependents', 0, 10) 
grad = st.selectbox('Choose Education',['Graduated','Not Graduated']) 
self_emp = st.selectbox('Self Emoployed ?',['Yes','No']) 
Annual_Income = st.slider('Choose Annual Income', 0, 10000000) 
Loan_Amount = st.slider('Choose Loan Amount', 0, 10000000) 
Loan_Dur = st.slider('Choose Loan Duration', 0, 20) 
Cibil = st.slider('Choose Cibil Score', 0, 1000) 
Assert = st.slider('Choose Assert', 0, 10000000)
Select_model = st.selectbox('Select model',['Logistic Regression','Decision Tree Regressor','Random Forest','Naive Bayes','All']) 

# import streamlit as st

# # Use 'value' for sliders
# no_of_dep = st.slider('Choose No of dependents', 0, 10, value=1) 

# # Use 'index' for selectboxes (0 is 'Graduated', 1 is 'Not Graduated')
# grad = st.selectbox('Choose Education', ['Graduated', 'Not Graduated'], index=1) 

# self_emp = st.selectbox('Self Employed ?', ['Yes', 'No'], index=0) 

# Annual_Income = st.slider('Choose Annual Income', 0, 100000000, value=5000) 

# Loan_Amount = st.slider('Choose Loan Amount', 0, 100000000, value=100000) 

# Loan_Dur = st.slider('Choose Loan Duration', 0, 20, value=1) 

# Cibil = st.slider('Choose Cibil Score', 0, 1000, value=200) 

# Assert = st.slider('Choose Assert', 0, 100000000, value=100000)

# Select_model = st.selectbox('Select model',['Logistic Regression','Decision Tree Regressor','Random Forest','Naive Bayes']) 


if grad =='Graduated': 
    grad_s = 1
else: 
    grad_s = 0
 
if self_emp =='Yes': 
    emp_s =1
else: 
    emp_s = 0
 
if st.button("Predict"): 
    pred_data =pd.DataFrame([[no_of_dep,grad_s,emp_s,Annual_Income,Loan_Amount,Loan_Dur,Cibil,Assert]],columns=['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score','Assert']) 
    pred_data = scaler.transform(pred_data)
    if Select_model == 'Logistic Regression':
        predict = model.predict(pred_data)
        if predict[0] == 1: 
            st.success('Logistic Regression : Loan Approved ✅') 
        else: 
            st.error('Logistic Regression : Loan Rejected ❌')
        
    elif Select_model == 'Decision Tree Regressor':
        predict = d_model.predict(pred_data)
        if predict[0] == 1: 
            st.success('Decision Tree Regressor : Loan Approved ✅') 
        else: 
            st.error('Decision Tree Regressor : Loan Rejected ❌')
    
    elif Select_model == 'Random Forest':
        predict = regressor.predict(pred_data)
        if predict[0] == 1: 
            st.success('Random Forest : Loan Approved ✅') 
        else: 
            st.error('Random Forest : Loan Rejected ❌')
            
    elif Select_model == 'Naive Bayes':
        predict = n_model.predict(pred_data)
        if predict[0] == 1: 
            st.success('Naive Bayes : Loan Approved ✅') 
        else: 
            st.error('Naive Bayes : Loan Rejected ❌')
    
    elif Select_model == 'All':
        predict = model.predict(pred_data)
        if predict[0] == 1: 
            st.success('Logistic Regression : Loan Approved ✅') 
        else: 
            st.error('Logistic Regression : Loan Rejected ❌')
         
            
        predict = d_model.predict(pred_data)
        if predict[0] == 1: 
            st.success('Decision Tree Regressor : Loan Approved ✅') 
        else: 
            st.error('Decision Tree Regressor : Loan Rejected ❌') 
            
        predict = regressor.predict(pred_data)
        if predict[0] == 1: 
            st.success('Random Forest : Loan Approved ✅') 
        else: 
            st.error('Random Forest : Loan Rejected ❌') 
            
        predict = n_model.predict(pred_data)
        if predict[0] == 1: 
            st.success('Naive Bayes : Loan Approved ✅') 
        else: 
            st.error('Naive Bayes : Loan Rejected ❌')
              
    else:
        st.error('=====Error=====')
        
