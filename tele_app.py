#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.metrics import ConfusionMatrixDisplay
import plotly.express as px
from imblearn.over_sampling import RandomOverSampler
# from category_encoders import OneHotEncoder
from sklearn.pipeline import make_pipeline
from category_encoders import OrdinalEncoder
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# import tensorflow as tf
import streamlit as st

# In[13]:


st.markdown("<h1 style='text-align: center;'>Project: Telecom Churn Prediction</h1> <hr style='border-top: 8px "
            "solid;border-radius: 5px;' class='rounded''>", unsafe_allow_html=True)
###

###
st.markdown("<h3 style='text-align: left;'>Table of Contents :</h3>", unsafe_allow_html=True)
st.markdown(""" <ul>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#predctive_model">Predictive Model</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>""", unsafe_allow_html=True)


# <a id="wrangling"></a>
# # Data Wrangling

# In[2]:


# merge all data files into  DataFrame
def wrangle(customer_data="Data/customer_data.csv", inter_data="Data/internet_data.csv",
            churn_data="Data/churn_data.csv"):
    df_churn = pd.read_csv(churn_data)
    df_data = pd.read_csv(customer_data)
    df_internet = pd.read_csv(inter_data)

    df = pd.merge(pd.merge(df_data, df_internet, on='customerID'), df_churn, on='customerID')

    return df


# In[3]:


df = wrangle()

# In[4]:


df.info()


# <a id="data_cleaning"></a>
# # Data Cleaning

# In[5]:


def cleaning(df):
    # Drop unnecessary columns
    df.drop(columns=["customerID"], inplace=True)

    # filter TotalCharges from null values
    df = df[df["TotalCharges"] != ' ']

    # convert TotalCharges into float
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # Remove null values
    df.dropna(inplace=True)

    # Remove Duplicates Row
    df.drop_duplicates(inplace=True)

    return df


# In[6]:


df = cleaning(df)

# In[7]:


df.head()


# In[8]:


def visual_hist(df, x_axis, title=False, color="Churn"):
    fig = px.histogram(df, x=x_axis, color=color, barmode='group', text_auto=".2s", title=title, nbins=15)

    return fig


# In[9]:


def visual_bar(df, title=False):
    fig = px.bar(df, text_auto=".2%", title=title, labels={"value": "percent %", "index": "Churn"})

    return fig


#
st.markdown("<a id='eda'></a>", unsafe_allow_html=True)
st.subheader("Exploratory Data Analysis")
# In[10]:


st.plotly_chart(visual_bar(df.Churn.value_counts(normalize=True), title="Churn percent %"), use_container_width=True)

st.text("* The plot shows a class imbalance of the data between churners and non-churners.")

# In[11]:


st.plotly_chart(visual_hist(df, "InternetService", title="Internet Service Types"), use_container_width=True)

st.text("Customers with InternetService fiber optic as part of their contract have much higher churn rate.")

# In[12]:


st.plotly_chart(visual_hist(df, "Dependents", title="Dependents"), use_container_width=True)

# ### Much higher churn rate for customers without children.

# In[13]:


st.plotly_chart(visual_hist(df, "tenure"), use_container_width=True)

# ### High tenure ranks as the strongest factor for not churning

# In[14]:


st.plotly_chart(visual_hist(df, "PaymentMethod"), use_container_width=True)

# ### Payment method electronic check shows much higher churn rate than other payment methods.

# In[15]:


st.plotly_chart(visual_hist(df, "PaymentMethod", color="PaperlessBilling"), use_container_width=True)

# ### Payment method electronic check shows much higher churn rate than other payment methods because it's Paperless
# Billing

# In[16]:


st.plotly_chart(visual_hist(df, "Partner"), use_container_width=True)

# ### Moderately higher churn rate for customers without partners.

# In[17]:


st.plotly_chart(visual_hist(df, "Contract", title="Contracts"), use_container_width=True)

# ### Churn rate for month-to-month contracts much higher that for other contract durations.

# In[22]:


st.plotly_chart(visual_hist(df, "SeniorCitizen", title="Senior Citizen"), use_container_width=True)

# ### Senior citizens churn rate is much higher than non-senior churn rate.

# In[19]:


target = "Churn"
X = df.drop(columns=[target])
y = df[target]

# In[20]:


ros = RandomOverSampler(random_state=42)

# In[21]:


X_over, y_over = ros.fit_resample(X, y)

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.1, random_state=42)

# In[23]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# <a id="predctive_model"></a>
# # Predictive  Model


model_forest = make_pipeline(
    OrdinalEncoder(),
    RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
)

# In[37]:


model_forest.fit(X_train, y_train)

# In[38]:


model_forest.score(X_train, y_train)

# In[39]:


model_forest.score(X_val, y_val)

# In[40]:


model_forest.score(X_test, y_test)

feature_imp = pd.Series(model_forest.named_steps["randomforestclassifier"].feature_importances_,
                        index=model_forest.named_steps["randomforestclassifier"].feature_names_in_)
feature_imp.sort_values(ascending=False, inplace=True)

# In[43]:


st.plotly_chart(px.bar(feature_imp, title="Feature Importance", color=feature_imp.index, text_auto=".1%"),
                use_container_width=True)


def user_input_features():
    Gender = st.sidebar.selectbox("Gender", X_test["gender"].value_counts().index)
    Senior_Citizen = st.sidebar.selectbox('Senior Citizen', X_test["SeniorCitizen"].value_counts().index)
    Partner = st.sidebar.selectbox("partner", X_test["Partner"].value_counts().index)
    Dependents = st.sidebar.selectbox("Dependents", X_test["Dependents"].value_counts().index)
    MultipleLines = st.sidebar.selectbox("MultipleLines", X_test["MultipleLines"].value_counts().index)
    InternetService = st.sidebar.selectbox("InternetService", X_test["InternetService"].value_counts().index)
    OnlineSecurity = st.sidebar.selectbox("OnlineSecurity", X_test["OnlineSecurity"].value_counts().index)
    OnlineBackup = st.sidebar.selectbox("OnlineBackup", X_test["OnlineBackup"].value_counts().index)
    DeviceProtection = st.sidebar.selectbox("DeviceProtection", X_test["DeviceProtection"].value_counts().index)
    TechSupport = st.sidebar.selectbox("TechSupport", X_test["TechSupport"].value_counts().index)
    StreamingTV = st.sidebar.selectbox("StreamingTV", X_test["StreamingTV"].value_counts().index)
    StreamingMovies = st.sidebar.selectbox("StreamingMovies", X_test["StreamingMovies"].value_counts().index)
    tenure = st.sidebar.slider("tenure", 1, 1000, 50)
    PhoneService = st.sidebar.selectbox("PhoneService", X_test["PhoneService"].value_counts().index)
    Contract = st.sidebar.selectbox("Contract", X_test["Contract"].value_counts().index)
    PaperlessBilling = st.sidebar.selectbox("PaperlessBilling", X_test["PaperlessBilling"].value_counts().index)
    PaymentMethod = st.sidebar.selectbox("PaymentMethod", X_test["PaymentMethod"].value_counts().index)
    MonthlyCharges = st.sidebar.slider("MonthlyCharges", 0, 5000, 2000)
    TotalCharges = st.sidebar.slider("TotalCharges", 0, 100000, 5000)

    data = {"gender": Gender,
            "SeniorCitizen": Senior_Citizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges

            }

    features = pd.DataFrame(data, index=[0])
    return features


# In[44]:
input_feature = user_input_features()
st.subheader('User Input')

st.write(input_feature)
# model = open("model.pkl", 'r')
# pickle.dump(model_forest,file)
predict = model_forest.predict(input_feature)
if predict == "No":
    mesg = "  User is not Churn"

else:
    mesg = "  User is  Churn"

st.markdown("<h1 style='text-align: center;'>Prediction</h1> <hr style='border-top: 8px "
            "solid;border-radius: 5px;' class='rounded''>", unsafe_allow_html=True)
st.title(mesg)
st.markdown(f"<h1 style='text-align: center;'></h1><hr style='border-top: 8px "
            "solid;border-radius: 5px;' class='rounded''>", unsafe_allow_html=True)

st.subheader('Prediction Probability')
st.write(model_forest.predict_proba(input_feature))

# model_forest.predict()
# In[16]:

st.markdown("<a id='conclusions'></a>", unsafe_allow_html=True)
#
st.title('Conclusions')

# <!-- ##   Contract duration: Contract duration month-to-month is the second-biggest driver of churn → supported
#     
# ##     Number of additional services: This feature does not rank among the top features → refused
#     
# ##     Partners and children: Having children ranks as the fourth feature that drives not churning, but strength is
# relatively low → partially supported
#     
# ##     Tenure: High tenure ranks as the strongest factor for not churning and the strongest feature overall. This
# is also supported by the boxplot in the EDA step. → supported
#     
# ##     Monthly payment: Total payments, which is the product of tenure and monthly payment ranks as the strongest
# factor for churn. Indirectly, high monthly payments lead to churn. However, tenure is the highest driver of not
# churning → refused
#     
# ##     Senior citizens: Senior citizens does not have high feature weights. Also, the ratio of senior citizens who
# churn is much higher than that of non-churners → refused -->

# In[17]:


st.subheader('1- Contract duration: Contract duration month-to-month is the second biggest driver of churn.')
st.divider()
st.subheader(
    '2- Partners and children: Having children ranks as the fourth feature that drives not churning, but strength is '
    'relatively low.')
st.divider()
st.subheader('3- Tenure: High tenure ranks as the strongest factor for not churning and the strongest feature overall.')
st.divider()
st.subheader(
    '4- Monthly payment: Total payments, which is the product of tenure and monthly payment ranks as the strongest '
    'factor for churn. Indirectly, high monthly payments lead to churn. However, tenure is the highest driver of not '
    'churning.')
st.divider()
st.subheader(
    '5- Senior citizens: Senior citizens does not have high feature weights. Also the ratio of senior citizens who '
    'churn is much higher than that of non-churners.')
st.divider()

# In[ ]:


# %%
