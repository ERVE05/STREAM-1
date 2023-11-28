import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import sklearn
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import plotly_express as px
import pickle
import joblib
from attr.filters import exclude
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recal

st.title('PROJET DE VISUALISATION DES CHURNS SUR LE RESEAU TELEPHONIQUE')
st.subheader('Churn Express')
def main():
    @st._cache_data(persist=True)
    def load_data():
        data = pd.read_csv("Expresso_churn_dataset .csv",skiprows=0, skipfooter=2001500)
        return data
    #affichage de la table
    df= load_data()
    df_sample= df.sample(300)
    if st.sidebar.checkbox("affiche les données brutes",False):
        st.subheader('jeu de données reduit de 2001500 lignes')
        st.write(df_sample)
    seed=123
    #suppression de variables cat non utilisé et remplissage de variables numeriques
    df.drop('user_id', axis=1, inplace=True)
    df.drop('REGION', axis=1, inplace=True)
    df.drop('TENURE', axis=1, inplace=True)
    df.drop('MRG', axis=1, inplace=True)
    df.drop('TOP_PACK', axis=1, inplace=True)
    # transformons 'REGULARITY' en float
    df['REGULARITY'] = df['REGULARITY'].astype(float)
    #cherchons les variable categorielles et numeriques
    var_list = df.columns.tolist()
    var_cat = []
    var_num = []
    for i in var_list:
        if df[i].dtype == 'object':
            var_cat.append(i)
        else:
            var_num.append(i)
    #renseignons les variables numeriques
    for var in var_num:
        df[var].fillna(df[var].median(), inplace=True)

    # Train/test split
    @st._cache_data(persist=True)
    def split (df):
        y = df['CHURN']
        x = df.drop('CHURN',axis=1)
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=seed)

    #x_train, x_test, y_train, y_test = split(df)
    classifier = st.sidebar.selectbox("Classificateur", ("Decision Tree Classifier",
                                                         "Random Forest Classifier")
    )
    # Random Forest
    if classifier == "Random Forest Classifier":
        st.sidebar.subheader("les hyperparametres du model ")
        n_arbres = st.sidebar.number_input("choisir le nombre d'arbre dans la foret",1,20000,
                                           step=1
    #0.000000e+00, 1.000000e+00, 1.875474e-01,step=1

        )

        profondeur_arbre= st.sidebar.number_input("profondeur d'un arbre",1,20,step=1)
        bootstrap =st.sidebar.radio("echantiloons boostrap lors de la creation d'arbres",
            ("True","False")
        )
        if st.sidebar.button('execution'):   #faisait partie de (,key=classify)
            st.subheader("Random Forest Result")
            model = RandomForestClassifier(
                n_estimators = n_arbres,
                max_depth = profondeur_arbre,
                bootstrap= bootstrap
            )
            #entrainement algo
            model.fit(x_train,y_train)
            #prediction
            y_pred= model.predict(x_test)
            #metrics de performance
            accuracy= model.score(x_test,y_test)
            precision=precision_score(y_test, y_pred)
            recall=recall_score(y_test,y_pred)
            #affichons les resultats
            st.write(accuracy.round(3))
            st.write(precision.round(3))
            st.write(recall.round(3))

if __name__=='__main__':
    main()
