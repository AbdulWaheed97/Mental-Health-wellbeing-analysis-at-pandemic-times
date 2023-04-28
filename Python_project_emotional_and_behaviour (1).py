                              #########################  PROJECT ############################

#Project : Analyze to identify which factors are influenced learing of child during covid period.
          # Because of we are conducting survey again not need to ask all questions and more focus on influced factors.
          
#Objective: Identify the emotional and behavioral difficulties of children which influencing child wellbeing   
#Constraints : We have more categarical data    

#Using Tools : Python and Excel    
           
#Load required Packages


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine.outliers import Winsorizer

#dataset

df = pd.read_excel(r'F:\Project\drive-download-20220222T153244Z-001\data.xlsx')

#we are also following Crisp ML(Q) Frame work :
    
#DATA UNDERSTANDING  
#===================  

df.shape #This dataset have 986 rows and 59 columns
#And this dataset have more columns and maybe we are facing more difficulties to handled this much cloumns.

df.info()

features = df.columns
#We want to Know how dataset was preseted.then we are sepearte the numerical and categarical data.
#Because of categarical data not useful prediction.
#We are now Convert categarical into Numerical Value.

numeric_data = df.select_dtypes(include=[np.number])
categorical_data = df.select_dtypes(exclude=[np.number])

categorical_data.shape #(986, 48)
numeric_data.shape #(986, 11)


#======================================================================================

#DATA PREPROCESSING
#=================== 

#Duplicate Value
#----------------


#Checking any duplicate Value PRESENTED OR NOT

df.duplicated().sum() #No duplicates


#Checking Null values (Missing Values)
#---------------------------------------

null_values =  pd.DataFrame (data = df.isna().sum() )

#More columns have null values , we can use imputer for fill nul values

#most of the null values are categarical we are use mode imputer

from sklearn.impute import SimpleImputer


mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df= pd.DataFrame(mode_imputer.fit_transform(df))
null_values = df.isna().sum()

df.columns = features
#Now there no missing values 

#Drop unnessory Columns
#------------------------------

df.columns

df.drop(['ID'],axis=1,inplace = True) #Not useful prediction

#Below are Mentioning children's age but same value in another column

df.drop(['Year'],axis=1,inplace = True)
df.drop(['Month'],axis=1,inplace = True)
df.drop(['Day'],axis=1,inplace = True)

col = df.columns

#Change Column Names for easy to access

df = df.rename(columns= {df.columns[0]:'still_going_to_school', df.columns[1]:'child_inhouse', df.columns[2]:'family_members',df.columns[3]:'age',df.columns[4]:'gender',df.columns[5]:'breakfast',df.columns[6]:'eat_fruits_or_not',df.columns[7]:'brush_count',df.columns[8]:'sleeping_start_time',df.columns[9]:'wakeup_time',df.columns[10]:'sports_activity_in_a_week',df.columns[11]:'watching_tv_in_a_week',df.columns[12]:'felling_tired_in_a_week',df.columns[13]:'concentratepay_attention_well_on_your_school_work',df.columns[14]:'cool_drink_in_a_week',df.columns[15]:'sugary_snack',df.columns[16]:'chinese_food',df.columns[17]:'safe_scale_in_your_area',df.columns[18]:'Easily_go_to_park',df.columns[19]:'Easily_go_to_outside',df.columns[20]:'do_u_have_garden',df.columns[21]:'how_many_times_u_going_outside_to_play',df.columns[22]:'enough_time_of_play',df.columns[23]:'playing_places',df.columns[24]:'playing_place_like_or_not',df.columns[25]:'relax_place_in_home',df.columns[26]:'i_am_doing_well_with_my_school_work',df.columns[27]:'i_feel_part_of_my_school_community',df.columns[28]:'i_have_lots_of_choice_over_things_that_are_important_to_me',df.columns[29]:"there_are_lots_of_things_I_am_good_at",df.columns[30]:'health',df.columns[31]:'School',df.columns[32]:'Family',df.columns[33]:'Friends',df.columns[34]:'appearance',df.columns[35]:'Life',df.columns[36]:'i_feel_lonely',df.columns[37]:'i_cry_a_lot',df.columns[38]:'i_am_unhappy',df.columns[39]:'i_feel_nobody_likes_me',df.columns[40]:'i_worry_a_lot',df.columns[41]:'i_have_problems_sleeping',df.columns[42]:'i_wake_up_in_the_night',df.columns[43]:'i_am_shy',df.columns[44]:'I feel scared',df.columns[45]:'i_worry_when_I_am_at_school',df.columns[46]:'i_get_very_angry',df.columns[47]:'i_lose_my_temper',df.columns[48]:'i_hit_out_when_i_am_angry',df.columns[49]:'i_do_things_to_hurt_people',df.columns[50]:'i_am_calm',df.columns[51]:'I_break_things_on_purpose',df.columns[52]:'keep_touchong_ur_family',df.columns[53]:'keep_touchong_ur_friends',df.columns[54]:'how_to_touch_famly_friends'})
#now changing column names for our comfort



df.columns

#--------------------------------------------------------------------------------------------

#import os
#os.getcwd()

#df.to_excel('new_column_names.xlsx',encoding='utf-8')#save into localsystem

#--------------------------------------------------------------------------------------------

df.keep_touchong_ur_family.value_counts()
df['keep_touchong_ur_family'].mode()
df['keep_touchong_ur_family']=df['keep_touchong_ur_family'].replace({9:'yes'})#Replace value 0f 9

df.family_members.value_counts()
df['family_members']=df['family_members'].replace({'6+': 6 })#Replace value 0f 6+
df.family_members.value_counts()


#age

df.age.value_counts()
df['age']=df['age'].replace({'Year 4':4,'Year 6':6,'Year 5':5,'Year 3':3})#Replace value 0f 6+

#gender

df.gender.value_counts()
df['gender']=df['gender'].replace({'Prefer not to say':np.nan})#prefer_not_to_say this rows are not use to prediction and replace the values into nan vales becayse of easy to drop
df.gender.isna().sum()
df ['gender'] = df.gender.dropna()

df['gender']=df['gender'].replace({'Girl':2,'Boy':1})
df.gender.value_counts()



#eat_fruits_or_not

df.eat_fruits_or_not.value_counts()
df['eat_fruits_or_not']=df['eat_fruits_or_not'].replace({'2 Or More Fruit and Veg':2,'1 Piece':1,'No':0})#Replace meaning full values

#Child_in_house


# Label Encoder
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()

#Child_in_house

df['child_inhouse']= labelencoder.fit_transform(df['child_inhouse'])

#still_going_to_school

df['still_going_to_school']= labelencoder.fit_transform(df['still_going_to_school'])

#gender

df['gender']= labelencoder.fit_transform(df['gender'])

#Easily_go_to_park

df.Easily_go_to_park.value_counts()
df['Easily_go_to_park']= labelencoder.fit_transform(df['Easily_go_to_park'])

#Easily_go_to_outside

df.Easily_go_to_outside.value_counts()
df['Easily_go_to_outside']= labelencoder.fit_transform(df['Easily_go_to_outside'])

#do_u_have_garden

df.do_u_have_garden.value_counts()
df['do_u_have_garden']= labelencoder.fit_transform(df['do_u_have_garden'])

#sleeping_time and wakeup_time

df.info()
df.sleeping_start_time.head()

df['sleeping_start_time'] = pd.to_datetime(df.sleeping_start_time)
df['sleeping_start_time']

df['wakeup_time'] = pd.to_datetime(df.wakeup_time)
df['wakeup_time']


sleeping_hours = ((df['wakeup_time']) - (df['sleeping_start_time']))


sleeping_hours.dt.components

sleeping_hour = pd.DataFrame(sleeping_hours.dt.components.hours)
sleeping_minutes = pd.DataFrame(sleeping_hours.dt.components.minutes)
sleeping_minutes = sleeping_minutes/60
sleeping_minutes

sleeping_minutes.value_counts()

sleeping_hour['sleeping_minutes'] = sleeping_minutes

sleeping_hour['sleeping_hour'] = sleeping_hour.sum(axis=1)
sleeping_hour.info()

#sleeping_hour = sleeping_hour.astype('str')
#sleeping_hour = sleeping_hour['hours'].str.cat(sleeping_hour['sleeping_minutes'], sep =".")

#sleeping_hour.astype(float)

print(sleeping_hour)

sleeping_hour

df['sleeping_hours'] = sleeping_hour.sleeping_hour

df = df.drop(['sleeping_start_time','wakeup_time'],axis=1)

#Now,we can calculate sleeping hours


#sports_activity_in_a_week,
#watching_tv_in_a_week	,
#felling_tired_in_a_week
#concentratepay_attention_well_on_your_school_work	
#cool_drink_in_a_week	
#sugary_snack	
#chinese_food

col = df.columns

df.sports_activity_in_a_week.value_counts()
df['sports_activity_in_a_week'].mode()
df['sports_activity_in_a_week']=df['sports_activity_in_a_week'].replace({'1-2 days':1,'3-4 days':3,'5-6 days':5,'7 days':7,'0 days':0})
#Replace value 0f 1-2 days ,3-4 days,5-6 days

df.watching_tv_in_a_week.value_counts()
df['watching_tv_in_a_week'].mode()
df['watching_tv_in_a_week']=df['watching_tv_in_a_week'].replace({'1-2 days':1,'3-4 days':3,'5-6 days':5,'7 days':7,'0 days':0})
#Replace value 0f 1-2 days ,3-4 days,5-6 days

df.felling_tired_in_a_week.value_counts()
df['felling_tired_in_a_week'].mode()
df['felling_tired_in_a_week']=df['felling_tired_in_a_week'].replace({'1-2 days':1,'3-4 days':3,'5-6 days':5,'7 days':7,'0 days':0})
#Replace value 0f 1-2 days ,3-4 days,5-6 days


#df = df.rename(columns= {df.columns[11]:'concentratepay_attention_well_on_your_school_work'})


df.concentratepay_attention_well_on_your_school_work.value_counts()
df['concentratepay_attention_well_on_your_school_work'].mode()
df['concentratepay_attention_well_on_your_school_work']=df['concentratepay_attention_well_on_your_school_work'].replace({'1-2 days':1,'3-4 days':3,'5-6 days':5,'7 days':7,'0 days':0})
#Replace value 0f 1-2 days ,3-4 days,5-6 days

df.sports_activity_in_a_week.value_counts()
df['cool_drink_in_a_week'].mode()
df['cool_drink_in_a_week']=df['cool_drink_in_a_week'].replace({'1-2 days':1,'3-4 days':3,'5-6 days':5,'7 days':7,'0 days':0})
#Replace value 0f 1-2 days ,3-4 days,5-6 days

df.sports_activity_in_a_week.value_counts()
df['sugary_snack'].mode()
df['sugary_snack']=df['sugary_snack'].replace({'1-2 days':1,'3-4 days':3,'5-6 days':5,'7 days':7,'0 days':0})
#Replace value 0f 1-2 days ,3-4 days,5-6 days

df.sports_activity_in_a_week.value_counts()
df['chinese_food'].mode()
df['chinese_food']=df['chinese_food'].replace({'1-2 days':1,'3-4 days':3,'5-6 days':5,'7 days':7,'0 days':0})
#Replace value 0f 1-2 days ,3-4 days,5-6 days


#playing_place_like_or_not
#chinese_food
#cool_drink_in_a_week



#df = df.drop(['playing_place_like_or_not'],axis=1)
#df = df.drop(['chinese_food'],axis=1)
#df = df.drop(['cool_drink_in_a_week'],axis=1)

#relax_place_in_home
#i_am_doing_well_with_my_school_work
#i_feel_part_of_my_school_community	
#i_have_lots_of_choice_over_things_that_are_important_to_me
#there_are_lots_of_things_I_am_good_at
#keep_touchong_ur_family
#keep_touchong_ur_friends

#Label_Encoder
col = df.columns


df['relax_place_in_home']= labelencoder.fit_transform(df['relax_place_in_home'])

df['i_am_doing_well_with_my_school_work']= labelencoder.fit_transform(df['i_am_doing_well_with_my_school_work'])

df['i_feel_part_of_my_school_community']= labelencoder.fit_transform(df['i_feel_part_of_my_school_community'])

df['i_have_lots_of_choice_over_things_that_are_important_to_me']= labelencoder.fit_transform(df['i_have_lots_of_choice_over_things_that_are_important_to_me'])

#df = df.rename(columns= {df.columns[24]:'there_are_lots_of_things_I_am_good_at'})

df['there_are_lots_of_things_I_am_good_at']= labelencoder.fit_transform(df['there_are_lots_of_things_I_am_good_at'])

df['keep_touchong_ur_family']= labelencoder.fit_transform(df['keep_touchong_ur_family'])

df['keep_touchong_ur_friends']= labelencoder.fit_transform(df['keep_touchong_ur_friends'])


#how_many_times_u_going_outside_to_play
#enough_time_of_play
#playing_place_like_or_not


df['how_many_times_u_going_outside_to_play']= labelencoder.fit_transform(df['how_many_times_u_going_outside_to_play'])

df['enough_time_of_play']= labelencoder.fit_transform(df['enough_time_of_play'])

df['playing_place_like_or_not']= labelencoder.fit_transform(df['playing_place_like_or_not'])


#breakfast

df.breakfast.value_counts()

import re

## Healthy Food
## Sugary Food
## Toast Food
## Cooked Food
## Other Food
## Nothing

df['breakfast']=df['breakfast'].str.lower()

for i in range(len(df['breakfast'])):
    if re.search('healthy', df['breakfast'][i]):
        df['breakfast'][i] = 'Healthy Food'
    elif re.search('sugary', df['breakfast'][i]):
        df['breakfast'][i] = 'Sugary Food'
    elif re.search('toast', df['breakfast'][i]):
        df['breakfast'][i] = 'Toast Food'
    elif re.search('cooked', df['breakfast'][i]):
        df['breakfast'][i] = 'Cooked Food'
    elif re.search('nothing', df['breakfast'][i]):
        df['breakfast'][i] = 'Nothing'
    else:
        df['breakfast'][i] = 'Other Food'
        
#        
        
df.breakfast.value_counts()   

#Apply label encoding

df['breakfast']= labelencoder.fit_transform(df['breakfast'])
  


#playing_places


df.playing_places = df.playing_places.astype(str)


for i in range(len(df['playing_places'])):

    if'In my house' in df['playing_places'][i]:
            df['playing_places'][i]=1

    elif 'my garden' in df['playing_places'][i]:
            df['playing_places'][i]=2

    elif 'a local grassy area' in df['playing_places'][i]:
            df['playing_places'][i]=3

    elif 'with bushes, trees and flowers' in df['playing_places'][i]:
            df['playing_places'][i]=4
            
    elif'bike or skate park' in df['playing_places'][i]:
            df['playing_places'][i]=5

    else:
            df['playing_places'][i] =0

col = df.columns


#how_to_touch_famly_friends

#One Way

a = df.how_to_touch_famly_friends.value_counts()
a

df['how_to_touch_famly_friends'] = df['how_to_touch_famly_friends'].str.lower()

for i in range(len(df['how_to_touch_famly_friends'])):

    if'by phone' in df['how_to_touch_famly_friends'][i]:
            df['how_to_touch_famly_friends'][i]=1

    elif 'i live near' in df['how_to_touch_famly_friends'][i]:
            df['how_to_touch_famly_friends'][i]=2
          
    else:
            df['how_to_touch_famly_friends'][i] =0


#Another Way

#a = df.how_to_touch_famly_friends
#a = pd.DataFrame(a)
#a = labelencoder.fit_transform(a)
#a.value_counts()





#Work on output_Column

col = df.columns
output = df.columns[34:44]
out = ' , '.join(output)
out


out_emotional = ['i_feel_lonely' ,' i_cry_a_lot' , 'i_am_unhappy' ,' i_feel_nobody_likes_me ',' i_worry_a_lot' ,' i_have_problems_sleeping ',' i_wake_up_in_the_night' ,' i_am_shy ',' I feel scared ',' i_worry_when_I_am_at_school']

df.i_feel_lonely.value_counts()


df['i_feel_lonely']=df['i_feel_lonely'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_cry_a_lot']=df['i_cry_a_lot'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_am_unhappy']=df['i_am_unhappy'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_feel_nobody_likes_me']=df['i_feel_nobody_likes_me'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_worry_a_lot']=df['i_worry_a_lot'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_have_problems_sleeping']=df['i_have_problems_sleeping'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_wake_up_in_the_night']=df['i_wake_up_in_the_night'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_am_shy']=df['i_am_shy'].replace({'Never':0,'Sometimes':1,'Always':2})
df['I feel scared']=df['I feel scared'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_worry_when_I_am_at_school']=df['i_worry_when_I_am_at_school'].replace({'Never':0,'Sometimes':1,'Always':2})


out_behaviour =  [ 'i_get_very_angry ',' i_lose_my_temper ',' i_hit_out_when_i_am_angry' ,' i_do_things_to_hurt_people ',' i_am_calm ',' I_break_things_on_purpose']
out

    

df.i_get_very_angry.value_counts()


df['i_get_very_angry']=df['i_get_very_angry'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_lose_my_temper']=df['i_lose_my_temper'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_hit_out_when_i_am_angry']=df['i_hit_out_when_i_am_angry'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_do_things_to_hurt_people']=df['i_do_things_to_hurt_people'].replace({'Never':0,'Sometimes':1,'Always':2})
df['i_am_calm']=df['i_am_calm'].replace({'Never':2,'Sometimes':1,'Always':0})
df['I_break_things_on_purpose']=df['I_break_things_on_purpose'].replace({'Never':0,'Sometimes':1,'Always':2})

col = df.columns

emo = df.iloc[:,34:44]
beh = df.iloc[:,44:49]


emo['add_e'] = emo.sum(axis = 1)


# Creating bins

bins = [-1, 9, 11, 20]
emo['ED_Subscale'] = pd.cut(emo['add_e'], bins, labels = ["Expected", "Borderline_difficulties", "Clinically_Significant_difficulties"])
emo['ED_Subscale'].value_counts()

beh['add_b'] = beh.sum(axis = 1)

# Creating bins
bins = [-1, 5, 6, 12]
beh['BD_Subscale'] = pd.cut(beh['add_b'], bins, labels = ["Expected", "Borderline_difficulties", "Clinically_Significant_difficulties"])
beh['BD_Subscale'].value_counts()

#df.info()
#df = df.astype(int64)

# now all inputs are converted into numerical

#add input and output

df["emotional"] = emo['ED_Subscale']
df["Behavioural"] = beh['BD_Subscale']

df.shape #(986,56)

df

#============

df_sample = df
col = df_sample.columns


#df_sample.drop(['i_feel_lonely' ,' i_cry_a_lot' , 'i_am_unhappy' ,' i_feel_nobody_likes_me ',' i_worry_a_lot' ,' i_have_problems_sleeping ',' i_wake_up_in_the_night' ,' i_am_shy ',' I feel scared ',' i_worry_when_I_am_at_school'],axis = 1,inplace=True) 
#df_sample.drop([],axis=1)

df_sample.drop(df_sample.columns[[range(34,50,1)]], axis = 1, inplace = True)

df_sample.shape#986,40

info = df_sample.info()

input_data = df_sample.iloc[:,0:38].astype(int)
#input_data = df_sample.iloc[:,0:38]
df_sample.info()
#input_data['sleeping_hours'] = df_sample.sleeping_hours.astype(float)


output_data_emo = df_sample.iloc[:,38:39]
output_data_emo = pd.DataFrame(output_data_emo)
output_data_emo.value_counts().plot(kind = 'barh')

import seaborn as sns



#emotional                          
#Expected                               0.888438
#Borderline_difficulties                0.077079
#Clinically_Significant_difficulties    0.034483

#Dataset was imbalanced 

output_data_beh = df_sample.iloc[:,39:]

output_data_beh.value_counts()/output_data_beh.value_counts().sum()

output_data_beh.value_counts().plot(kind = 'barh')

#(Behavioural                        
#Expected                               0.943205
#Clinically_Significant_difficulties    0.035497
#Borderline_difficulties                0.021298)

#Dataset was imbalanced 

predictors = input_data
predictors.info()

target_01 = output_data_emo
target_02 = output_data_beh



#First Model

################################# HANDLING IMBLANCE #######################################

#K-FOLD
#Emoional


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import GridSearchCV


10.0 **np.arange(-2,3)

log_class=LogisticRegression()
grid={'C':10.0 **np.arange(-2,3),'penalty':['l1','l2']}
cv=KFold(n_splits=5,random_state=None,shuffle=False)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(predictors,target_01,train_size=0.7)
clf=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred)) #0.88

print(classification_report(y_test,y_pred))

y_pred=clf.predict(X_train)
print(accuracy_score(y_train,y_pred)) #0.89
print(classification_report(y_train,y_pred))


############### Behaviral ###############################


10.0 **np.arange(-2,3)

log_class=LogisticRegression()
grid={'C':10.0 **np.arange(-2,3),'penalty':['l1','l2']}
cv=KFold(n_splits=5,random_state=None,shuffle=False)
#from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(predictors,target_02,train_size=0.7)
clf=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred)) #0.93

print(classification_report(y_test,y_pred))

y_pred=clf.predict(X_train)
print(accuracy_score(y_train,y_pred)) #0.94
print(classification_report(y_train,y_pred))


######################################## FEATURE SELECTION ##################################



#For emotional


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

### Apply SelectKBest Algorithm

ordered_rank_features=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_features.fit(predictors,target_01)
dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(predictors.columns)
features_rank=pd.concat([dfcolumns,dfscores],axis=1)
features_rank.columns=['Features','Score']
features_rank

features_rank.nlargest(10,'Score')



#############################  Choose only important column for emotional ##########################

good_feature_emotional = predictors.iloc[:,[10,15,28,11,32,29,8,33,31,19]]

good_feature_emotional['target_01'] = target_01

g_x = good_feature_emotional.iloc[:,0:10]
g_y = good_feature_emotional.iloc[:,10]

good_feature_emotional.info()
col = good_feature_emotional.columns


#for Emotional


log_class=LogisticRegression()
grid={'C':10.0 **np.arange(-2,3),'penalty':['l1','l2']}
cv=KFold(n_splits=5,random_state=None,shuffle=False)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(g_x , g_y , train_size=0.7)
clf=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred)) #0.90

print(classification_report(y_test,y_pred))

y_pred=clf.predict(X_train)
print(accuracy_score(y_train,y_pred)) #0.88
print(classification_report(y_train,y_pred))



#For behavioural


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

### Apply SelectKBest Algorithm

ordered_rank_features=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_features.fit(predictors,target_02)
dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(predictors.columns)
features_rank=pd.concat([dfcolumns,dfscores],axis=1)
features_rank.columns=['Features','Score']
features_rank

features_rank.nlargest(10,'Score')



#############################  Choose only important column  ##########################

good_feature_behaviour = predictors.iloc[:,[10,11,29,32,8,12,28,33,31,19]]

good_feature_behaviour['target_02'] = target_02

g_x = good_feature_behaviour.iloc[:,0:10]
g_y = good_feature_behaviour.iloc[:,10]

good_feature_emotional.info()
col = good_feature_emotional.columns


#for Emotional


log_class=LogisticRegression()
grid={'C':10.0 **np.arange(-2,3),'penalty':['l1','l2']}
cv=KFold(n_splits=5,random_state=None,shuffle=False)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(g_x , g_y , train_size=0.7)
clf=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred)) #0.93

print(classification_report(y_test,y_pred))

y_pred=clf.predict(X_train)
print(accuracy_score(y_train,y_pred)) #0.95
print(classification_report(y_train,y_pred))





