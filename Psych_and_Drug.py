# -*- coding: utf-8 -*-
'''
Course: Machine Learning Lab given by Prof. Brefeld
Project: Influence of psychological factors on drug consumption
Authors: Johanna Regenthal and Sofija Engelson
Due date: 
'''

#-----------------------0.Preliminaries-----------------------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from matplotlib import patches as mpatches
from pomegranate import *

#----------------------1.Reading and cleaning data-------------------------------

#----------------------1.1 Reading--------------------------------------------------

colnames = ['ID','Age','Gender','Education','Country','Ethnicity','Nscore','Escore','Oscore','Ascore','Cscore','Impulsiv','SS','Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc','Coke','Crack','Ecstasy','Heroin','Ketamine','Legalh','LSD','Meth','Mushrooms','Nicotine','Semer','VSA']

PsychDrug = pd.read_csv('drug_consumption.data', names = colnames, header = None)
PsychDrug.head()

#----------------------1.2 Cleaning---------------------------------------------------

#----------1.2.1 Classification into Users and Non-users for all drugs------------------
# Imported from https://github.com/deepak525/Drug-Consumption/blob/master/drug.ipynb

DrugUse = ['Alcohol','Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack',
           'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms','Nicotine', 'Semer', 'VSA']
ClassificationDrugUse = ['User_Alcohol','User_Amphet', 'User_Amyl', 'User_Benzos', 'User_Caff', 'User_Cannabis', 'User_Choc', 'User_Coke', 'User_Crack',
           'User_Ecstasy', 'User_Heroin', 'User_Ketamine', 'User_Legalh', 'User_LSD', 'User_Meth', 'User_Mushrooms','User_Nicotine', 'User_Semer', 'User_VSA']

for column in DrugUse:
    le = LabelEncoder()
    PsychDrug[column] = le.fit_transform(PsychDrug[column])

for i in range(len(DrugUse)):
    PsychDrug.loc[((PsychDrug[DrugUse[i]]==0) | (PsychDrug[DrugUse[i]]==1)), ClassificationDrugUse [i]] = False
    PsychDrug.loc[((PsychDrug[DrugUse[i]]==2) | (PsychDrug[DrugUse[i]]==3) | (PsychDrug[DrugUse[i]]==4) | (PsychDrug[DrugUse[i]]==5) | (PsychDrug[DrugUse[i]]==6)), ClassificationDrugUse [i]] = True

# PsychDrug.drop(columns = DrugUse)

#------------------1.2.2 Dequantification of explanatory variables------------------------

# Building dictionary which has the columns as key and the mapping of quantified number and 
# translation saved as a tuple in value
MappingDict = {}

MappingDict['Age'] = (-0.95197,'18-24'),\
    (-0.07854,'25-34'),\
    (0.49788,'35-44'),\
    (1.09449,'45-54'),\
    (1.82213,'55-64'),\
    (2.59171,'65+')
    
MappingDict['Gender'] = (0.48246,'Female'),(-0.48246,'Male')

MappingDict['Education'] =(-2.43591,'Left School Before 16 years'),\
    (-1.73790,'Left School at 16 years'),\
    (-1.43719,'Left School at 17 years'),\
    (-1.22751,'Left School at 18 years'),\
    (-0.61113,'Some College,No Certificate Or Degree'),\
    (-0.05921,'Professional Certificate/ Diploma'),\
    (0.45468,'University Degree'),\
    (1.16365,'Masters Degree'),\
    (1.98437,'Doctorate Degree')

MappingDict['Country'] =(-0.09765,'Australia'),\
    (0.24923,'Canada'),\
    (-0.46841,'New Zealand'),\
    (-0.28519,'Other'),\
    (0.21128,'Republic of Ireland'),\
    (0.96082,'UK'),\
    (-0.57009,'USA')

MappingDict['Ethnicity'] = (-0.50212,'Asian'),\
    (-1.10702,'Black'),\
    (1.90725,'Mixed-Black/Asian'),\
    (0.12600,'Mixed-White/Asian'),\
    (-0.22166,'Mixed-White/Black'),\
    (0.11440,'Other'),\
    (-0.31685,'White')
    
# MappingDict is missing for the following columns: Nscore','Escore','Oscore','Ascore','Cscore'

# Rounding all floats to 5 places after comma for further replacement
PsychDrug = round(PsychDrug,5)

# Function to replace the 
def mapping(data, col):
    rep = data[col]
    for value in MappingDict[col]:
        rep = rep.replace(value[0], value[1])
    return rep

PsychDrug['Age'] = mapping(PsychDrug,'Age')
PsychDrug['Gender'] = mapping(PsychDrug,'Gender')
PsychDrug['Education'] = mapping(PsychDrug,'Education')
PsychDrug['Country'] = mapping(PsychDrug,'Country')
PsychDrug['Ethnicity'] = mapping(PsychDrug,'Ethnicity')

#---------------------2. Exploratory analysis------------------------------------
# Imported from https://github.com/deepak525/Drug-Consumption/blob/master/drug.ipynb

# Visualization of frequency of usage for each drug
fig, axes = plt.subplots(5,3,figsize = (16,16))
fig.suptitle("Count of Different Classes Vs Drug",fontsize=14)
k=0
for i in range(5):
    for j in range(3):
        sns.countplot(x=DrugUse[k], data=PsychDrug,ax=axes[i][j])
        k+=1

plt.tight_layout()
plt.show()

# Visualization of Users vs. Non-users for each drug
count_of_users = []
count_of_non_users = []

for i in range(len(DrugUse)):
    s = PsychDrug.groupby([ClassificationDrugUse[i]])[DrugUse[i]].count()
    count_of_users.append(s[1])
    count_of_non_users.append(s[0])

bins = np.arange(1,20,1)
plt.figure(figsize=(16,6))
plt.bar(bins+0,count_of_users,width=0.4,label ='User')
plt.bar(bins+.30,count_of_non_users,width=0.4,label ='Non-User')
plt.xticks(bins,DrugUse,rotation=50,fontsize=13)
plt.ylabel("Count",fontsize=13)
plt.title("Drug Vs User Or Non-user",fontsize=15)
plt.legend()
plt.show()

#----------------------------------------------------------------------------
# Visualization of the correlations and covariances between all continuous variables
# grouped by drug use frequencies and psychological variables

PsychVar = ['Nscore','Escore','Oscore','Ascore','Cscore','Impulsiv','SS']

corMatDrugUse = PsychDrug[DrugUse].corr()
corMatPsychVar = PsychDrug[PsychVar].corr()

covMatDrugUse = PsychDrug[DrugUse].cov()
covMatPsychVar = PsychDrug[PsychVar].cov()

ax = plt.axes()
sns.heatmap(corMatDrugUse, ax = ax)
ax.set_title('Correlation heatmap for drug use frequencies')
plt.show()

ax = plt.axes()
sns.heatmap(corMatPsychVar, ax = ax)
ax.set_title('Correlation heatmap for psychological factors')
plt.show()

ax = plt.axes()
sns.heatmap(covMatDrugUse, ax = ax)
ax.set_title('Covariance heatmap for drug use frequencies')
plt.show()

ax = plt.axes()
sns.heatmap(covMatPsychVar, ax = ax)
ax.set_title('Covariance heatmap for psychological factors')
plt.show()

# Chi-Square test for correlation between two categorial variables (demographic variable --> usage classification)
# For example: Education and usage of nicotine

def perform_chi_test(v1, v2):
    table = pd.crosstab(PsychDrug[v1], PsychDrug[v2], margins = False)
    stat, p, dof, expected = chi2_contingency(table)
    alpha = 0.05
    #print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print(v1 + ' and ' + v2 + ' are dependent because H0 can be rejected with a p-value of p=%.3f.' % p)
    else:
        print(v1 + ' and ' + v2 + ' are independent because H0 can not be rejected with a p-value of p=%.3f.' % p)

perform_chi_test('Education', 'User_Nicotine')

# Definition of a cumulative variable which indicates users (dacade-based) of legal and illegal drugs
LegalDrugs = ['User_Alcohol', 'User_Caff', 'User_Choc', 'User_Coke', 'User_Nicotine']
IllegalDrugs = ['User_Amphet', 'User_Amyl', 'User_Benzos','User_Cannabis', 'User_Crack',\
                'User_Ecstasy', 'User_Heroin', 'User_Ketamine', 'User_Legalh', 'User_LSD',\
                'User_Meth', 'User_Mushrooms','User_Semer', 'User_VSA']

def make_cumvar (list):
    User_cum = ['Non-user'] * len(PsychDrug)
#    User_cum = [False] * len(PsychDrug)
    for i in range(len(PsychDrug)):
        for drug in list:
            if PsychDrug[drug][i]==True:
                User_cum[i] = 'User'
#                User_cum[i] = True
    return User_cum

PsychDrug['User_LegalDrugs'] = make_cumvar(LegalDrugs)
PsychDrug['User_IllegalDrugs'] = make_cumvar(IllegalDrugs)

# Visualization of a continous variable for certain groups
# For example: Big Five scores for users and non-users of illegal drugs in comparison

NonUser = mpatches.Patch(color='blue')
User = mpatches.Patch(color='red')
mypal = {'Non-user': 'b', 'User': 'r'}

f, axes = plt.subplots(5, 1, sharex=True, sharey=True)
f.subplots_adjust(hspace=.75)

sns.boxplot(x = 'Nscore', y = 'User_IllegalDrugs', data = PsychDrug, ax = axes[0], palette = mypal)
sns.boxplot(x = 'Escore', y = 'User_IllegalDrugs', data = PsychDrug, ax = axes[1], palette = mypal)
sns.boxplot(x = 'Oscore', y = 'User_IllegalDrugs', data = PsychDrug, ax = axes[2], palette = mypal)
sns.boxplot(x = 'Ascore', y = 'User_IllegalDrugs', data = PsychDrug, ax = axes[3], palette = mypal)
sns.boxplot(x = 'Cscore', y = 'User_IllegalDrugs', data = PsychDrug, ax = axes[4], palette = mypal)
f.legend(handles = [NonUser,User], labels = ['Non-user','User'], loc = 'lower right')
f.suptitle('Big Five for users and non-users of illegal drugs (decade-based)')

for boxplot in range(5):
    axes[boxplot].yaxis.set_visible(False)
    

# Visualization of Big Five scores for users and non-users of illegal drugs
# Separate plots

sns.catplot(y = 'Nscore', x = 'User_IllegalDrugs', data = PsychDrug, kind="box") 
sns.catplot(y = 'Escore', x = 'User_IllegalDrugs', data = PsychDrug, kind="box") 
sns.catplot(y = 'Oscore', x = 'User_IllegalDrugs', data = PsychDrug, kind="box") 
sns.catplot(y = 'Ascore', x = 'User_IllegalDrugs', data = PsychDrug, kind="box") 
sns.catplot(y = 'Cscore', x = 'User_IllegalDrugs', data = PsychDrug, kind="box") 

# Visualization of a contious variable for a certain subset
# For example: Big Five for users of illegal drugs
    
subset = PsychDrug[PsychDrug['User_IllegalDrugs']==True]

f, axes = plt.subplots(5, 1, sharex=True, sharey=True)
f.subplots_adjust(hspace=.75)

sns.boxplot(x = 'Nscore', data = subset, ax = axes[0])
sns.boxplot(x = 'Escore', data = subset, ax = axes[1])
sns.boxplot(x = 'Oscore', data = subset, ax = axes[2])
sns.boxplot(x = 'Ascore', data = subset, ax = axes[3])
sns.boxplot(x = 'Cscore', data = subset, ax = axes[4])
    
#----------------------------------3. Bayesian Network--------------------

#----------------------------------3.1 Conditional Probabilities--------------------

# TESTING

DemoVar = ['Education','Gender', 'Age','Country']
#PredVar = PsychVar + DemoVar

# Producing a table of conditional probabilities
list_outer = []
while DemoVar != []:
    print(DemoVar)
    CondProbTable = (PsychDrug.groupby(DemoVar+['User_LSD']).count() / PsychDrug.groupby(DemoVar).count())
    for index, row in CondProbTable.iterrows():
        list_inner = []
        for ind in index:
            list_inner.append(ind)
        list_inner.append(row.min())
        list_outer.append(list_inner)
    DemoVar.pop()

# The guests initial door selection is completely random
Education = DiscreteDistribution({'Education': 1/5, 'Gender': 1/5, 'Age': 1/5, 'Country': 1/5, 'User_LSD': 1/5})
Gender = DiscreteDistribution({'Education': 1/5, 'Gender': 1/5, 'Age': 1/5, 'Country': 1/5, 'User_LSD': 1/5})
Age = DiscreteDistribution({'Education': 1/5, 'Gender': 1/5, 'Age': 1/5, 'Country': 1/5, 'User_LSD': 1/5})
Country = DiscreteDistribution({'Education': 1/5, 'Gender': 1/5, 'Age': 1/5, 'Country': 1/5, 'User_LSD': 1/5})
User_LSD = DiscreteDistribution({'Education': 1/5, 'Gender': 1/5, 'Age': 1/5, 'Country': 1/5, 'User_LSD': 1/5})

# Monty is dependent on both the guest and the prize. 
monty = ConditionalProbabilityTable(list_outer, [Education, Gender, Age, Country, User_LSD])

# State objects hold both the distribution, and a high level name.
s1 = State(Education, name="Education")
s2 = State(Gender, name="Gender")
s3 = State(Age, name="Age")
s4 = State(Country, name="Country")
s5 = State(User_LSD, name="User_LSD")

# Create the Bayesian network object with a useful name
model = BayesianNetwork("Monty Hall Problem")

# Add the three states to the network 
model.add_states(s1, s2, s3, s4, s5)


# Add edges which represent conditional dependencies, where the second node is 
# conditionally dependent on the first node (Monty is dependent on both guest and prize)
model.add_edge(s1, s5)
model.add_edge(s2, s5)
model.add_edge(s3, s5)
model.add_edge(s4, s5)

model.bake()

model.probability([['Doctorate Degree', 'Female', '25-34','UK', False]])

model.structure
