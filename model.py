import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy
df = pd.read_csv('Brain_Stroke_Data.csv')
print(df)
age_group=[]
for i in df['age']:
    if i<13.0:
        age_group.append('Toddler')
    elif i>12.0 and i<=19.0:
        age_group.append('Teen')
    elif i>19.0 and i<60.0:
        age_group.append('Adult')
    else:
        age_group.append('Senior')
df['age_group']=age_group
df.head()
df.drop('age',axis=1,inplace=True)

df_cod=pd.get_dummies(df,drop_first=True)
df_cod.head()

df1 = df_cod.copy()

for i in [i for i in df1.columns]:
    if df1[i].nunique()>=12:
        Q1 = df1[i].quantile(0.20)
        Q3 = df1[i].quantile(0.80)
        IQR = Q3 - Q1
        df1 = df1[df1[i] <= (Q3+(1.5*IQR))]
        df1 = df1[df1[i] >= (Q1-(1.5*IQR))]
df1 = df1.reset_index(drop=True)
display(df1.head())
print('\n\033[1mInference:\033[0m Before removal of outliers, The dataset had {} samples.'.format(df.shape[0]))
print('\033[1mInference:\033[0m After removal of outliers, The dataset now has {} samples.'.format(df1.shape[0]))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dffit= pd.DataFrame(scaler.fit_transform(df1),columns=df1.columns)
dffit.head()

from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
X=dffit.drop(['stroke'],axis=1)
y=dffit['stroke']
X_over, y_over = oversample.fit_resample(X, y)

df_final=pd.concat([X_over, y_over ],axis=1)
print(df_final)

df_final.drop('age_group_Toddler',axis=1,inplace=True)



#X= df_final.iloc[:,0:10]  # all features
#Y= df_final.iloc[:,-1]  # target output (stroke)
#%%
X = df1[['hypertension', 'work_type_children', 'heart_disease','ever_married_Yes','avg_glucose_level','work_type_Self-employed','bmi','Residence_type_Urban']]  #the top 3 features
Y = df1[['stroke']] # the target output
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=100)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

logreg_pred = logreg.predict(X_test)

print ("Probabilty of having brainstroke is:", accuracy_score(y_test,logreg_pred)*100 )
import pickle
pickle.dump(logreg, open("model.pkl", "wb"))

