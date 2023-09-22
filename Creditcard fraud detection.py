#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data =pd.read_csv('creditcard.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


print("Number of rows",data.shape[0])
print("Number of cols",data.shape[1])


# In[8]:


data.info()


# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


sc = StandardScaler()
data['Amount']= sc.fit_transform(pd.DataFrame(data['Amount']))


# In[11]:


data.head()


# In[12]:


data= data.drop(['Time'],axis=1)


# In[13]:


data.head()


# In[14]:


data.duplicated().any()


# In[15]:


data= data.drop_duplicates()


# In[16]:


data.shape


# In[17]:


data['Class'].value_counts()


# In[18]:


import seaborn as sns


# In[19]:


sns.countplot(data['Class'])


# In[20]:


X=data.drop('Class',axis=1)
y=data['Class']


# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# In[22]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)


# In[23]:


y_pred1= log.predict(X_test)


# In[24]:


from sklearn.metrics import accuracy_score


# In[25]:


accuracy_score(y_test,y_pred1)


# In[26]:


from sklearn.metrics import precision_score,recall_score,f1_score


# In[27]:


precision_score(y_test,y_pred1)


# In[28]:


recall_score(y_test,y_pred1)


# In[29]:


f1_score(y_test,y_pred1)


# In[30]:


normal = data[data['Class']==0]
fraud = data[data['Class']==1]


# In[31]:


fraud.shape


# In[32]:


normal.shape


# In[33]:


normal_sample=normal.sample(n=473)


# In[34]:


normal_sample.shape


# In[35]:


new_data =pd.concat([normal_sample,fraud],ignore_index=True)


# In[36]:


new_data['Class'].value_counts()


# In[37]:


new_data.head()


# In[38]:


X = new_data.drop('Class',axis=1)
y= new_data['Class']


# In[39]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# In[40]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)


# In[41]:


y_pred1= log.predict(X_test)


# In[42]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)


# In[43]:


from sklearn.metrics import precision_score,recall_score,f1_score


# In[44]:


precision_score(y_test,y_pred1)


# In[45]:


recall_score(y_test,y_pred1)


# In[46]:


f1_score(y_test,y_pred1)


# In[47]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[48]:


y_pred2= dt.predict(X_test)


# In[49]:


accuracy_score(y_test,y_pred2)


# In[50]:


precision_score(y_test,y_pred2)


# In[51]:


recall_score(y_test,y_pred2)


# In[52]:


f1_score(y_test,y_pred2)


# In[53]:


from sklearn.ensemble import RandomForestClassifier
rf =RandomForestClassifier()
rf.fit(X_train,y_train)


# In[54]:


y_pred3= rf.predict(X_test)


# In[55]:


accuracy_score(y_test,y_pred3)


# In[56]:


precision_score(y_test,y_pred2)


# In[57]:


recall_score(y_test,y_pred3)


# In[58]:


f1_score(y_test,y_pred2)


# In[59]:


pd.DataFrame({'Models': ['LR','DT','RF'],"ACC":[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred2)*100,accuracy_score(y_test,y_pred3)*100]})


# In[60]:


final_data = pd.DataFrame({'Models': ['LR','DT','RF'],"ACC":[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred2)*100,accuracy_score(y_test,y_pred3)*100]})


# In[61]:


final_data


# In[62]:


import seaborn as sns


# In[63]:


sns.barplot(x=final_data['Models'],y=final_data['ACC'])


# In[64]:


X = data.drop('Class',axis=1)
y = data['Class']


# In[65]:


X.shape


# In[66]:


y.shape


# In[67]:


pip install imbalanced-learn


# In[68]:


from imblearn.over_sampling import SMOTE


# In[69]:


X_res, y_res = SMOTE().fit_resample(X, y)


# In[70]:


y_res.value_counts()


# In[71]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X_res,y_res,test_size=0.20,random_state=42)


# In[72]:


log = LogisticRegression()
log.fit(X_train,y_train)


# In[73]:


y_pred1 = log.predict(X_test)


# In[74]:


accuracy_score(y_test,y_pred1)


# In[75]:


precision_score(y_test,y_pred1)


# In[76]:


recall_score(y_test,y_pred1)


# In[77]:


f1_score(y_test,y_pred1)


# In[79]:


dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[80]:


y_pred2= dt.predict(X_test)


# In[81]:


accuracy_score(y_test,y_pred2)


# In[82]:


precision_score(y_test,y_pred2)


# In[83]:


recall_score(y_test,y_pred2)


# In[84]:


f1_score(y_test,y_pred2)


# In[85]:


rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[87]:


y_pred3 = rf.predict(X_test)


# In[88]:


accuracy_score(y_test,y_pred3)


# In[89]:


precision_score(y_test,y_pred3)


# In[90]:


recall_score(y_test,y_pred3)


# In[91]:


f1_score(y_test,y_pred3)


# In[92]:


final_data = pd.DataFrame({'Models': ['LR','DT','RF'],"ACC":[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred2)*100,accuracy_score(y_test,y_pred3)*100]})


# In[93]:


final_data


# In[94]:


sns.barplot(x=final_data['Models'],y=final_data['ACC'])


# In[95]:


rf1 = RandomForestClassifier()
rf1.fit(X_res,y_res)


# In[96]:


import joblib


# In[97]:


joblib.dump(rf1,"credit_card_model")


# In[98]:


model = joblib.load("credit_card_model")


# In[117]:


pred = model.predict([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


# In[107]:


if pred == 0:
    print("Normal Transcation")
else:
    print("Fraudalent Transaction")


# In[ ]:


from tkinter import *
import joblib

def show_entry_fields():
    v1 = float(e1.get())
    v2 = float(e2.get())
    v3 = float(e3.get())
    v4 = float(e4.get())
    v5 = float(e5.get())
    v6 = float(e6.get())
    
    v7 = float(e7.get())
    v8 = float(e8.get())
    v9 = float(e9.get())
    v10= float(e10.get()) 
    v11= float(e11.get())
    v12= float(e12.get())
    
    v13= float(e13.get())
    v14= float(e14.get())
    v15= float(e15.get())
    v16= float(e16.get())
    v17= float(e17.get())
    v18= float(e18.get())
    
    v19= float(e19.get())
    v20= float(e20.get()) 
    v21= float(e21.get())
    v22= float(e22.get())
    v23= float(e23.get())
    v24= float(e24.get())
    
    v25= float(e25.get())
    v26= float(e26.get())
    v27= float(e27.get())
    v28= float(e28.get())
    
    model = joblib.load('model_credit.pkl')
    y_pred = model.predict([[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29]])
    list1  = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29]  
    result = []
    if y_pred ==0:
        result.append("Normal Transaction")
    else:
        result.append("Fraudelent Transaction")
    print("###################################################")
    print("Credit Card Fraud Detection System",result)
    print("####################################################")


    Label(master,text="Final prediction from the model -credit card fraudelent system")
    Label(master,text=result).grid(row=32)

master =Tk()
master.title("Credit Card Fraud Detection System")

label = Label (master, text = "Credit Card Fraud Detection System"
                 ,bg = "black", fg = "white", width = 30).grid(row=0)

Label (master, text="Enter value of V1").grid(row=1)
Label (master, text="Enter value of V2").grid(row=2)
Label (master, text="Enter value of V3").grid(row=3)
Label (master, text="Enter value of V4").grid(row=4)
Label (master, text="Enter value of V5").grid(row=5) 
Label (master, text="Enter value of V6").grid(row=6)

Label(master, text="Enter value of V7").grid(row=7)
Label (master, text="Enter value of V8").grid(row=8)
Label (master, text="Enter value of V9").grid(row=9)
Label (master, text="Enter value of V10").grid(row=10) 
Label (master, text="Enter value of V11").grid(row=11)
Label(master, text="Enter value of V12").grid(row=12)

Label (master, text="Enter value of V13").grid(row=13)
Label (master, text="Enter value of V14").grid(row=14)
Label (master, text="Enter value of V15").grid(row=15) 
Label (master, text="Enter value of V16").grid(row=16)
Label(master, text="Enter value of V17").grid(row=17)
Label (master, text="Enter value of V18").grid(row=18)

Label (master, text="Enter value of V19").grid(row=19)
Label (master, text="Enter value of V20").grid(row=20) 
Label (master, text="Enter value of V21").grid(row=21)
Label(master, text="Enter value of V22").grid(row=22)
Label (master, text="Enter value of V23").grid(row=23)
Label (master, text="Enter value of V24").grid(row=24)

Label (master, text="Enter value of V25").grid(row=25) 
Label (master, text="Enter value of V26").grid(row=26)
Label(master, text="Enter value of V27").grid(row=27)
Label (master, text="Enter value of V28").grid(row=28)
Label (master, text="Enter value of V29").grid(row=29)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)

e7  = Entry(master)
e8  = Entry(master)
e9  = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)

e13 = Entry(master)
e14 = Entry(master)
e15 = Entry(master)
e16 = Entry(master)
e17 = Entry(master)
e18 = Entry(master)

e19 = Entry(master)
e20 = Entry(master)
e21 = Entry(master)
e22 = Entry(master)
e23 = Entry(master)
e24 = Entry(master)

e25 = Entry(master)
e26 = Entry(master)
e27 = Entry(master)
e28 = Entry(master)
e29 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1) 
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)

e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1) 
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)

e13.grid(row=13, column=1) 
e14.grid(row=14, column=1)
e15.grid(row=15, column=1)
e16.grid(row=16, column=1)
e17.grid(row=17, column=1)
e18.grid(row=18, column=1)

e19.grid(row=19, column=1) 
e20.grid(row=20, column=1)
e21.grid(row=21, column=1)
e22.grid(row=22, column=1)
e23.grid(row=23, column=1) 
e24.grid(row=24, column=1)

e25.grid(row=25, column=1)
e26.grid(row=26, column=1)
e27.grid(row=27, column=1)
e28.grid(row=28, column=1)
e29.grid(row=29, column=1) 

Button(master, text='Predict', command=show_entry_fields).grid(row=30, column=30)
                                                               
mainloop()


# In[ ]:





# In[ ]:




