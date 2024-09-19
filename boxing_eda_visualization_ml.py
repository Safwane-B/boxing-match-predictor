

# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# read the data
data_fighters = pd.read_csv('../input/boxing-matches-dataset-predict-winner/fighters.csv', delimiter = ',')

data_fighters.head()

data_fighters.info()

# prepare the data for EDA & Visualization
data_fighters[data_fighters.columns.drop('stance')] = data_fighters[data_fighters.columns.drop('stance')].replace('Unknown', '0.0')
data_fighters['ko_rate'] = data_fighters['ko_rate'].apply(lambda value: str(value).replace('%', ' '))
data_fighters[['wins', 'looses', 'draws', 'ko_rate', 'age']] = data_fighters[['wins', 'looses', 'draws', 'ko_rate', 'age']].astype(float)

# count the boxers of a country
sns.countplot(x = data_fighters['country'], order = data_fighters['country'].value_counts().head(30).index)
plt.xticks(rotation = 90)
plt.show()

# the mean of list_ per country 
sns.barplot(x = data_fighters['country'], y = data_fighters['wins'], order = data_fighters['country'].value_counts().head(30).index)
plt.xticks(rotation = 90)
plt.show()

# the mean of list_ per prefer stance
sns.barplot(x = data_fighters['stance'], y = data_fighters['wins'])
plt.xticks(rotation = 90)
plt.show()


# the mean of list_ per country 
sns.barplot(x = data_fighters['country'], y = data_fighters['ko_rate'], order = data_fighters['country'].value_counts().head(30).index)
plt.xticks(rotation = 90)
plt.show()

# normalize the name 
def normalize_the_name(name):
    name = name.upper()
    name = str(name).replace(' ', '')
    return name

data_fighters['name'] = data_fighters['name'].apply(lambda name: normalize_the_name(name))
data_fighters['name']

# take just the value(cm) in height and reach column
def take_the_cm(string, a, b):
    if string == '0.0':
        return '0.0'
    else: 
        return string[a:-b]
    
data_fighters['reach'] = data_fighters['reach'].apply(lambda row: take_the_cm(row, 14, 4))
data_fighters['height'] = data_fighters['height'].apply(lambda row: take_the_cm(row, 9, 3))
data_fighters['reach'], data_fighters['height']

# for input at the prediction model
data_fighters = pd.get_dummies(data_fighters, columns = ['stance'])
data_fighters

# read the DATA
data_pop = pd.read_csv('../input/boxing-matches-dataset-predict-winner/popular_matches.csv', delimiter = ',')

data_pop.head()

data_pop.info()

## prepare the DATA
data_pop = data_pop.drop('place', axis = 1)
data_pop = data_pop.fillna(0.0)
data_pop[['opponent_1_estimated_punch_power', 'opponent_2_estimated_punch_power', 'opponent_1_rounds_boxed']] = data_pop[['opponent_1_estimated_punch_power', 'opponent_2_estimated_punch_power', 'opponent_1_rounds_boxed']].astype(float)

# create new columns with the last name of the boxers
def take_the_name(name, position):
    latters = str(name.upper()).split()
    if latters[position] == 'JR':
        del latters[-1]
    return latters[position]

data_pop['op_1'] = data_pop['opponent_1'].apply(lambda name: take_the_name(name, -1))
data_pop['op_2'] = data_pop['opponent_2'].apply(lambda name: take_the_name(name, -1))

data_pop.insert(1, 'last_name_1', data_pop['op_1'])
data_pop.insert(2, 'last_name_2', data_pop['op_2'])

data_pop = data_pop.drop('op_1', axis = 1)
data_pop = data_pop.drop('op_2', axis = 1)

# normalize the name of boxers
def normalize_the_name(name):
    name = name.upper()
    name = str(name).replace(' ', '')
    return name

data_pop['opponent_1'] = data_pop['opponent_1'].apply(lambda name: normalize_the_name(name))
data_pop['opponent_2'] = data_pop['opponent_2'].apply(lambda name: normalize_the_name(name))

# clean the verdict column and create the column with the reason of the winner
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

data_pop

# take the names of the boxers in data_popular_matches
def take_the_opponents():
    list_name_1, list_name_2 = [], []
    for i in range(data_pop.shape[0]):
        name_opponent_1 = data_pop['opponent_1'][i]
        name_opponent_2 = data_pop['opponent_2'][i]

        list_name_1.append(name_opponent_1)
        list_name_2.append(name_opponent_2)
    
    return list_name_1, list_name_2

list_name_opponent_1, list_name_opponent_2 = take_the_opponents()

# create the list of the new columns
wins_1, looses_1, draws_1, ko_rate_1, age_1, height_1, reach_1, country_1, stance_Orthodox_1, stance_Southpaw_1, stance_Unknown_1 = [], [], [], [], [], [], [], [], [], [], [] 

list_columns_1_str = ['wins_1', 'looses_1', 'draws_1', 'ko_rate_1', 'age_1',
'height_1', 'reach_1', 'country_1', 'stance_Orthodox_1', 'stance_Southpaw_1',
'stance_Unknown_1']

list_columns_1 = [wins_1, looses_1, draws_1, ko_rate_1, age_1,
height_1, reach_1, country_1, stance_Orthodox_1, stance_Southpaw_1,
stance_Unknown_1]

wins_2, looses_2, draws_2, ko_rate_2, age_2, height_2, reach_2, country_2, stance_Orthodox_2, stance_Southpaw_2, stance_Unknown_2 = [], [], [], [], [], [], [], [], [], [], [] 

list_columns_2_str = ['wins_2', 'looses_2', 'draws_2', 'ko_rate_2', 'age_2',
'height_2', 'reach_2', 'country_2', 'stance_Orthodox_2', 'stance_Southpaw_2',
'stance_Unknown_2']

list_columns_2 = [wins_2, looses_2, draws_2, ko_rate_2, age_2,
height_2, reach_2, country_2, stance_Orthodox_2, stance_Southpaw_2,
stance_Unknown_2]

# function that take all parameters/columns per name in data_fighters
''' 
    The condition if/else is there because some boxers ins`t
    at data_fighters. From the future, if necessary, take the parameters 
    with webscrapping. 220 boxers are found.
'''
def take_the_parameters(name):
    row = data_fighters.loc[data_fighters['name'] == str(name)] 
    row = np.array(row[row.columns].values)
    
    if row.shape[0] == 1.0:
        row = np.delete(row, [0])
    else:
        row = np.zeros(11 , dtype = float)

    return row


# create the DATA with all parameters
def full_DATA():
    # count = 0.0
    for name in list_name_opponent_1:
        row = take_the_parameters(name)
        # if row[0] > 0.0:
        #     count += 1.0
        for i in range(len(list_columns_1)):
            list_columns_1[i].append(row[i])

    for name in list_name_opponent_2:
        row = take_the_parameters(name)
        # if row[0] > 0.0:
        #     count += 1.0
        for i in range(len(list_columns_2)):
            list_columns_2[i].append(row[i])
    # print(count)

    for count, values in enumerate(list_columns_1_str): 
        data_pop[list_columns_1_str[count]] = list_columns_1[count]

    for count, values in enumerate(list_columns_2_str): 
        data_pop[list_columns_2_str[count]] = list_columns_2[count]

    return data_pop

full_data = full_DATA()
full_data

# first type 
'''
    -version 1.0
    2 separated datasets --> 1 with all atributtes of the firts oponnent and
    another with all atributtes of the second oponnent.
    (data_total_1, data_total_2)
    &
    1 dataset --> the first block of atributtes is from the firts oponnent and
    the second block of atributtes is from the second oponnent. 
    (data_total)
'''
def first_type_data():
    data_firts_type = data_pop
    data_firts_type = data_firts_type.drop(['date', 'last_name_1', 'last_name_2', 'opponent_1', 'opponent_2', 'verdict', 'reason_winner'], axis = 1)

    data_1_p1 = data_firts_type.iloc[:, ::2]
    data_2_p1 = data_firts_type.iloc[:, 1::2]

    data_full = full_data
    data_1_p2 = data_full.iloc[:, 21:32]
    data_2_p2 = data_full.iloc[:, 32:]

    data_total_1 = pd.concat([data_1_p1, data_1_p2], axis = 1, join = 'inner')
    data_total_2 = pd.concat([data_2_p1, data_2_p2], axis = 1, join = 'inner')

    data_total = pd.concat([data_total_1, data_total_2], axis = 1, join = 'inner')

    return data_total_1, data_total_2, data_total

# the predict
'''
    -version 1.0
    Have 3 type of verdict in the column : WINNER, DRAW AND UNKNOWN.
    Normalize the verdict:
    WINNER: 1.0 or 2.0
    DRAW: 3.0
    UNKNOWN: 0.0
'''
def normalize_verdict():
    data_analyse = data_pop
    data_analyse = data_analyse[['last_name_1', 'last_name_2', 'verdict']]

    list_last_name_1 = np.array(data_analyse['last_name_1'])
    list_last_name_2 = np.array(data_analyse['last_name_2'])
    list_verdict = np.array(data_analyse['verdict'])

    normalize_verdict = []
    for i in range(data_analyse.shape[0]):
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

# the reason_winner
'''
    -version 1.0 
    Have 7 type of reason winner in the column: 
    ['UD', 'KO', 'RTD', 'TKO', 'SD', 'MD', 0.0(UNKNOWN), 'PTS', 'DQ']

    To predict the KO:
    KO: 0.0
    the rest: 1.0
'''
def normalize_reason_winner():
    data_analyse = data_pop
    data_analyse = data_analyse['reason_winner']

    def normalize_KO(row):
        if str(row) == 'KO':
            return 0.0
        else:
            return 1.0

    y_reason_winner = data_analyse.apply(lambda row: normalize_KO(row))

    return y_reason_winner

data_total_1, data_total_2, data_total = first_type_data()
data_total

data_total.columns

# data
y_winner = normalize_verdict()
y_reason_winner = normalize_reason_winner()

# preprocessing data
scaler = StandardScaler()

data_total = data_total.drop(['country_1', 'country_2'], axis = 1)
data_total = scaler.fit_transform(data_total)

# Linear model
model = RidgeCV()
x_train, x_test, y_train, y_test = train_test_split(data_total, y_winner, test_size = 0.3, random_state = 10)
model.fit(x_train, y_train)
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
error = mean_squared_error(y_pred, y_test)

print("////////////////////////////////////////")
print("Resuls: RidgeCV(winner)")
print("Train score: %s" % train_score)
print("Test score: %s" % test_score)
print("Error(MSE): %s" % error)
print("////////////////////////////////////////")

# XGBClassifier
model = XGBClassifier(max_depth=2, learning_rate=0.001, n_estimators=100, gamma=0, 
min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)

x_train, x_test, y_train, y_test = train_test_split(data_total, y_winner, test_size = 0.3, random_state = 15)

model.fit(x_train, y_train)
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
y_pred_winner = model.predict(x_test)
accuracy = accuracy_score(y_test,  y_pred_winner)

print("////////////////////////////////////////")
print("Resuls: XGBClassifier(winner)")
print("Train score: %s" % train_score)
print("Test score: %s" % test_score)
print("Accuracy: %s" % accuracy)
print("////////////////////////////////////////")

# RandomForestClassifier
model = RandomForestClassifier(n_estimators = 1000, max_depth = 2, criterion = 'entropy', random_state = 50)
x_train, x_test, y_train, y_test = train_test_split(data_total, y_winner, test_size = 0.3, random_state = 40)
model.fit(x_train, y_train)
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
errors = abs(y_pred - y_test)
accuracy = accuracy_score(y_test, y_pred)

print("////////////////////////////////////////")
print("Resuls: RandomForestClassifier(winner)")
print("Train score: %s" % train_score)
print("Test score: %s" % test_score)
print("Accuracy: %s" % accuracy)
print("////////////////////////////////////////")

