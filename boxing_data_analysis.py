import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Streamlit Title
st.title('Boxing Data Analysis and Match Prediction')

# Sidebar for navigation
page = st.sidebar.radio("Choose a page", ["Home", "Data Exploration & Predictive Modeling"])

# Load the data outside the conditional blocks so it's accessible everywhere
data_fighters = pd.read_csv('fighters.csv', delimiter=',')
data_pop = pd.read_csv('popular_matches.csv', delimiter=',')

# Home Page
if page == "Home":
    st.write("Welcome to the Boxing Data Analysis and Match Prediction app!")
    st.markdown("""
    ### Project Overview:
    This project is designed to analyze data from historical boxing matches and use machine learning techniques to predict match outcomes. The app consists of two main sections: 

    - **Data Exploration & Visualization**: This section provides insights into the boxers' data, including their wins, losses, and other statistics, as well as the number of boxers by country and their knockout rates. Various visualizations help in exploring trends in the data.

    - **Predictive Modeling**: Using machine learning models such as **XGBoost** and **Random Forest**, we predict the outcomes of boxing matches based on historical data. These models are trained on features like boxers' wins, losses, height, and reach.

    The primary goal of this project is to demonstrate how data science can be applied to sports analytics and to build models that help predict future match outcomes.
    """)
# Data Exploration Page
if page == "Data Exploration & Predictive Modeling":
    
    # Read the data
    st.subheader("Boxers Data")
    st.markdown("""
    This section loads and displays the dataset of boxers. It provides basic information about the boxers, such as their number of wins, losses, knockout rates, and more. Data cleaning steps are also applied here, such as replacing unknown values and converting certain columns to numeric types.
    """)
    data_fighters = pd.read_csv('fighters.csv', delimiter=',')
    st.write(data_fighters.head())
    st.write(data_fighters.info())

    # Data preparation for EDA & Visualization
    data_fighters[data_fighters.columns.drop('stance')] = data_fighters[data_fighters.columns.drop('stance')].replace('Unknown', '0.0')
    data_fighters['ko_rate'] = data_fighters['ko_rate'].apply(lambda value: str(value).replace('%', ' '))
    data_fighters[['wins', 'looses', 'draws', 'ko_rate', 'age']] = data_fighters[['wins', 'looses', 'draws', 'ko_rate', 'age']].astype(float)
    
    st.markdown("""
    We then proceed to visualize the data. The following visualizations provide insights into the distribution of boxers based on their countries, their win rates, and knockout rates, which help us understand general trends and distributions in the data.
    """)

    # Count plot of boxers per country
    st.subheader("Number of Boxers by Country")
    fig1, ax1 = plt.subplots()
    sns.countplot(x=data_fighters['country'], order=data_fighters['country'].value_counts().head(30).index, ax=ax1)
    plt.xticks(rotation=90)
    st.pyplot(fig1)

    # Wins per country
    st.subheader("Mean Wins per Country")
    fig2, ax2 = plt.subplots()
    sns.barplot(x=data_fighters['country'], y=data_fighters['wins'], order=data_fighters['country'].value_counts().head(30).index, ax=ax2)
    plt.xticks(rotation=90)
    st.pyplot(fig2)

    # Wins by stance
    st.markdown("""
    The plots above show the distribution of boxers by country and their win rates. This helps identify the countries that have the highest number of professional boxers and the average number of wins for boxers from each country.
    """)
    st.subheader("Wins per Stance")
    fig3, ax3 = plt.subplots()
    sns.barplot(x=data_fighters['stance'], y=data_fighters['wins'], ax=ax3)
    plt.xticks(rotation=90)
    st.pyplot(fig3)

    # KO rate per country
    st.subheader("KO Rate per Country")
    fig4, ax4 = plt.subplots()
    sns.barplot(x=data_fighters['country'], y=data_fighters['ko_rate'], order=data_fighters['country'].value_counts().head(30).index, ax=ax4)
    plt.xticks(rotation=90)
    st.pyplot(fig4)

    st.markdown("""
    These additional visualizations give us insights into the relationship between the boxers' stance (e.g., Orthodox, Southpaw) and their win rates, as well as the knockout rates by country. This helps explore how certain attributes of boxers might relate to their success in the ring.
    """)

    # Normalize the name of the fighters
    def normalize_the_name(name):
        name = name.upper()
        name = str(name).replace(' ', '')
        return name

    data_fighters['name'] = data_fighters['name'].apply(lambda name: normalize_the_name(name))
    data_fighters['reach'] = data_fighters['reach'].apply(lambda row: row[14:-4] if row != '0.0' else '0.0')
    data_fighters['height'] = data_fighters['height'].apply(lambda row: row[9:-3] if row != '0.0' else '0.0')

    # One-hot encoding for stance
    data_fighters = pd.get_dummies(data_fighters, columns=['stance'])

    # Load Popular Matches Dataset
    st.subheader("Popular Matches Data")
    data_pop = pd.read_csv('popular_matches.csv', delimiter=',')
    st.write(data_pop.head())
    st.write(data_pop.info())

    # Data preparation for Popular Matches
    data_pop = data_pop.drop('place', axis=1).fillna(0.0)
    data_pop[['opponent_1_estimated_punch_power', 'opponent_2_estimated_punch_power', 'opponent_1_rounds_boxed']] = data_pop[['opponent_1_estimated_punch_power', 'opponent_2_estimated_punch_power', 'opponent_1_rounds_boxed']].astype(float)

    # Create new columns for last names of boxers
    def take_the_name(name, position):
        latters = str(name.upper()).split()
        if latters[position] == 'JR':
            del latters[-1]
        return latters[position]

    data_pop['op_1'] = data_pop['opponent_1'].apply(lambda name: take_the_name(name, -1))
    data_pop['op_2'] = data_pop['opponent_2'].apply(lambda name: take_the_name(name, -1))
    data_pop.insert(1, 'last_name_1', data_pop['op_1'])
    data_pop.insert(2, 'last_name_2', data_pop['op_2'])

    # Normalize names
    data_pop['opponent_1'] = data_pop['opponent_1'].apply(lambda name: normalize_the_name(name))
    data_pop['opponent_2'] = data_pop['opponent_2'].apply(lambda name: normalize_the_name(name))

    # Take the reason for the winner
    def take_the_reason_to_winner(row):
        latters = str(row).split()
        if len(latters) == 7:
            return latters[3]
        if len(latters) > 7:
            return latters[4]
        else:
            return 0.0
        
    data_pop['reason_winner'] = data_pop['verdict'].apply(lambda row: take_the_reason_to_winner(row))
    data_pop['verdict'] = data_pop['verdict'].apply(lambda row: take_the_name(row, 0))

    # Function to take the names of the opponents from data_pop
    def take_the_opponents():
        list_name_1 = data_pop['opponent_1'].tolist()
        list_name_2 = data_pop['opponent_2'].tolist()
        return list_name_1, list_name_2

    # Create full_data based on data_fighters
    def take_the_parameters(name):
        row = data_fighters.loc[data_fighters['name'] == str(name)] 
        row = np.array(row[row.columns].values)
        
        if row.shape[0] == 1.0:
            row = np.delete(row, [0])
        else:
            row = np.zeros(11, dtype=float)

        return row

    # Create full_data from data_pop and data_fighters
    def full_DATA():
        list_name_opponent_1, list_name_opponent_2 = take_the_opponents()

        wins_1, looses_1, draws_1, ko_rate_1, age_1, height_1, reach_1, country_1, stance_Orthodox_1, stance_Southpaw_1, stance_Unknown_1 = [], [], [], [], [], [], [], [], [], [], [] 
        wins_2, looses_2, draws_2, ko_rate_2, age_2, height_2, reach_2, country_2, stance_Orthodox_2, stance_Southpaw_2, stance_Unknown_2 = [], [], [], [], [], [], [], [], [], [], [] 
        
        list_columns_1 = [wins_1, looses_1, draws_1, ko_rate_1, age_1, height_1, reach_1, country_1, stance_Orthodox_1, stance_Southpaw_1, stance_Unknown_1]
        list_columns_2 = [wins_2, looses_2, draws_2, ko_rate_2, age_2, height_2, reach_2, country_2, stance_Orthodox_2, stance_Southpaw_2, stance_Unknown_2]

        for name in list_name_opponent_1:
            row = take_the_parameters(name)
            for i in range(len(list_columns_1)):
                list_columns_1[i].append(row[i])

        for name in list_name_opponent_2:
            row = take_the_parameters(name)
            for i in range(len(list_columns_2)):
                list_columns_2[i].append(row[i])

        # Assign the columns back to data_pop, including country_1 and country_2
        data_pop['country_1'] = country_1
        data_pop['country_2'] = country_2

        return data_pop

    # Generate full_data
    full_data = full_DATA()

    # Function to normalize the verdict (the match outcome)
    def normalize_verdict():
        list_last_name_1 = np.array(data_pop['last_name_1'])
        list_last_name_2 = np.array(data_pop['last_name_2'])
        list_verdict = np.array(data_pop['verdict'])

        normalize_verdict = []
        for i in range(data_pop.shape[0]):
            if str(list_verdict[i]) == 'DRAW':
                normalize_verdict.append(3.0)
            elif str(list_verdict[i]) == str(list_last_name_1[i]):
                normalize_verdict.append(1.0)
            elif str(list_verdict[i]) == str(list_last_name_2[i]):
                normalize_verdict.append(2.0)
            else:
                normalize_verdict.append(0.0)

        y_winner = normalize_verdict
        return y_winner

    # Function to prepare full data for predictive modeling
    def first_type_data():
        data_firts_type = data_pop
        data_firts_type = data_firts_type.drop(['date', 'last_name_1', 'last_name_2', 'opponent_1', 'opponent_2', 'verdict', 'reason_winner'], axis=1)

        data_1_p1 = data_firts_type.iloc[:, ::2]
        data_2_p1 = data_firts_type.iloc[:, 1::2]

        data_full = full_data
        data_1_p2 = data_full.iloc[:, 21:32]
        data_2_p2 = data_full.iloc[:, 32:]

        data_total_1 = pd.concat([data_1_p1, data_1_p2], axis=1, join='inner')
        data_total_2 = pd.concat([data_2_p1, data_2_p2], axis=1, join='inner')

        data_total = pd.concat([data_total_1, data_total_2], axis=1, join='inner')

        return data_total_1, data_total_2, data_total

    # Preprocess data for model input
    data_total_1, data_total_2, data_total = first_type_data()
    y_winner = normalize_verdict()

    # Drop non-numeric columns before scaling
    non_numeric_columns = ['last_name_1', 'last_name_2', 'opponent_1', 'opponent_2', 'verdict', 'reason_winner', 'op_1', 'op_2', 'country_1', 'country_2']
    data_total = data_total.drop(non_numeric_columns, axis=1, errors='ignore')

    # Standardize the numeric data
    scaler = StandardScaler()
    data_total = scaler.fit_transform(data_total)


    # Predictive modeling
    st.header("Predictive Modeling")

    st.markdown("""
    In this section, we build two machine learning models to predict the outcome of boxing matches based on historical data. We use two models:

    - **XGBoost (Extreme Gradient Boosting)**: A powerful machine learning algorithm known for its performance in structured/tabular data. It uses decision trees and gradient boosting to predict outcomes.
    - **Random Forest Classifier**: A tree-based ensemble model that combines the predictions of multiple decision trees to improve accuracy and avoid overfitting.

    Both models are trained on features like wins, losses, age, height, and reach of the boxers, and they predict the winner of the match.
    """)


    # XGBoost Classifier Explanation
    st.subheader("XGBClassifier Model")

    st.markdown("""
    We use the XGBoost classifier with the following hyperparameters:
    - Max depth: 2
    - Learning rate: 0.001
    - Number of estimators: 100

    The model is trained on 70% of the data, and we evaluate its performance on the remaining 30%. Below, we display the training and test accuracy, as well as the overall accuracy.
    """)
    model = XGBClassifier(max_depth=2, learning_rate=0.001, n_estimators=100, gamma=0, 
    min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)
    x_train, x_test, y_train, y_test = train_test_split(data_total, y_winner, test_size=0.3, random_state=15)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    y_pred_winner = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_winner)

    st.write(f"Train score: {train_score}")
    st.write(f"Test score: {test_score}")
    st.write(f"Accuracy: {accuracy}")

    # RandomForestClassifier model
    st.subheader("RandomForestClassifier Model")
    st.markdown("""
    The Random Forest model is trained similarly to the XGBoost model. It creates multiple decision trees and averages their predictions to produce a final prediction. We use the following hyperparameters for the Random Forest:
    - Number of estimators: 1000
    - Max depth: 2
    - Criterion: Entropy

    Below, we display the model’s performance on the training and test sets, as well as the accuracy.
    """)
    model = RandomForestClassifier(n_estimators=1000, max_depth=2, criterion='entropy', random_state=50)
    x_train, x_test, y_train, y_test = train_test_split(data_total, y_winner, test_size=0.3, random_state=40)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Train score: {train_score}")
    st.write(f"Test score: {test_score}")
    st.write(f"Accuracy: {accuracy}")