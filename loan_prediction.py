#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[11]:


#Read Data
df = pd.read_csv('C:/Users/LENOVO/Documents/RAKAMIN/VIX/Idx Partner/Final Task/loan.csv')
pd.set_option('display.max_columns',100)
df.head()


# In[12]:


df.info()


# # 1. DESCRIPTIVE STATISTICS

# In[13]:


df.select_dtypes(include=np.number).describe().T


# Terdapat beberapa variabel pada data numerik yang missing dan bernilai 0/NaN dan kolom kolom yang tidak dibutuhkan seperti ("unnammed, memeber id, policy_code & id") sehingga harus dihapus

# In[14]:


df.drop(columns=['Unnamed: 0', 'member_id', 'id', 'policy_code'], inplace=True)


# In[15]:


#menghapus kolom yang nilai seluruhnya NaN
df1 = df.dropna(axis=1,how='all')


# In[16]:


df1.select_dtypes(include=['object']).describe().T


# **berdasarkan penjealasan diatas kolom yang akan dihapus :**
# - sub grade tidak diperlukan karena sudah terwakilkan oleh grade
# - emp_tittle terlalu banyak values uniq sudah tergantikan dengan emp_length
# - issue_d, url, desc, title, zip_code, earliest_cr_line, last_pymnt_d, next_pymnt_d & last_credit_pull_d terlalu banyak nilai uniq
# - application type tidak diperlukan

# In[17]:


df1.drop(columns=['sub_grade', 'emp_title','issue_d', 'url', 'desc', 'title', 'zip_code', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'application_type'], inplace=True)


# In[18]:


df1.info()


# # Perbaikan Label Feature (Status Pinjaman) untuk keperluan EDA
# - Menemukan beberapa values yang tidak diperlukan didalam kolom loan_status 

# In[19]:


plt.figure(figsize=(10,5))
sns.countplot(y= "loan_status", data = df1, palette = 'Spectral', lw = 1, ec = 'k')
plt.show()

value_counts = df1["loan_status"].value_counts()
percentage = value_counts / value_counts.sum()
percentage = percentage.apply("{:.2%}".format)
print(percentage)


# **Berdasarkan Penjelasan diatas :**
# 
# Berhasil / Disetujui = Fully Paid
# 
# Gagal / Ditolak = Charged Off, Default dan does not meet the credit policy
# 
# Late, In Grace Period dan Current tidak bisa digunakan karena status pinjaman masih berlangsung

# In[20]:


# Menghapus values pada kolom loan_status
succes = ["Fully Paid"]
fail = ["Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged off",
        "Does not meet the credit policy. Status:Fully Paid"]
def loan(status):
    if status in fail:
        return 'Gagal Bayar'
    return 'Berhasil Bayar'


# In[21]:


df_loan = df1[df1["loan_status"].isin(succes + fail)].copy()
df_loan["loan_status"] = df_loan["loan_status"].apply(loan)


# In[22]:


df_loan['loan_status'].value_counts()


# In[23]:


plt.figure(figsize=(10,5))
sns.countplot(y= "loan_status", data = df_loan, palette = "Spectral")
plt.show()

value_counts = df_loan["loan_status"].value_counts()
percentage = value_counts / value_counts.sum()
percentage = percentage.apply("{:.2%}".format)
print(percentage)


# # 2. EDA

# ## Annual Income Borrowers

# In[24]:


import matplotlib.ticker as mticker

# Visualisasi status pelanggan berdasarkan tenure
plt.figure(figsize=(12, 6))
sns.boxplot(x='annual_inc', y='loan_status', data=df_loan, orient='h')


# Mengatur format label sumbu x
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))  # Mengubah label x menjadi format ribuan

plt.show()


# Persebaran gagal bayar hanya terjadi pada annual_income < 2.0000.000. <br>
# Hal ini menunjukan bahwa peminjam dengan pendapatan tahunan yang rendah lebih cenderung mengalami kesulitan dalam membayar pinjaman mereka.

# ## Grade

# In[25]:


plt.figure(figsize=(8,5))
sns.countplot(data=df_loan,x='grade',hue='loan_status',   palette = ['lightgreen', 'tomato'], 
            lw = 1, ec = 'k')
#plt.xticks(rotation=180, ha='right')
plt.show()

df1 = df_loan.groupby(['grade', 'loan_status']).agg(Jumlah_Peminjam=( 'loan_status', 'count')).reset_index()
df2 =  df_loan.groupby('grade').agg(Jumlah_Total_Peminjam=('grade', 'count')).reset_index()
df_merge = df1.merge(df2, on = 'grade')

# Menambah Kolom Rasio
df_merge['Rasio'] = round (df_merge['Jumlah_Peminjam'] / df_merge['Jumlah_Total_Peminjam'] * 100,2)

#ubah nama data
df_merge['loan_status'] = df_merge['loan_status'].replace({0: 'berhasil bayar', 1: 'gagal bayar'})
df_merge


# In[26]:


import seaborn as sns
# Membuat plot
plt.figure (figsize = (15, 10))
sns.barplot(x ='grade', y = 'Rasio', hue = 'loan_status', data = df_merge, palette = [ 'lightgreen', 'tomato'], lw = 1, ec = 'k')

# Memberi judul pada plot dan sumbu-sumbunya
plt.xlabel('grade', fontsize = 12, fontname='monospace')
plt.ylabel('Rasio Persentase(%)', fontsize = 12,fontname='monospace')
plt.text(x=-0.20, y=94.23, s='93.23%', ha='center', va='center', fontsize=10)
plt.text(x=0.20, y=7.77, s='6.77%', ha='center', va='center', fontsize=10)
plt.text(x=0.80, y=87.43, s='86.43%', ha='center', va='center', fontsize=10)
plt.text(x=1.20, y=14.57, s='13.57%', ha='center', va='center', fontsize=10)
plt.text(x=1.80, y=79.35, s='78.35%', ha='center', va='center', fontsize=10)
plt.text(x=2.20, y=22.65, s='21.65%', ha='center', va='center', fontsize=10)
plt.text(x=2.80, y=72.39, s='71.39%', ha='center', va='center', fontsize=10)
plt.text(x=3.20, y=29.61, s='28.61%', ha='center', va='center', fontsize=10)
plt.text(x=3.80, y=63.96, s='62.96%', ha='center', va='center', fontsize=10)
plt.text(x=4.20, y=38.04, s='37.04%', ha='center', va='center', fontsize=10)
plt.text(x=4.80, y=58.72, s='57.72%', ha='center', va='center', fontsize=10)
plt.text(x=5.20, y=43.28, s='42.28%', ha='center', va='center', fontsize=10)
plt.text(x=5.20, y=43.28, s='42.28%', ha='center', va='center', fontsize=10)
plt.text(x=5.80, y=52.51, s='51.51%', ha='center', va='center', fontsize=10)
plt.text(x=6.20, y=49.49, s='48.49%', ha='center', va='center', fontsize=10)
plt.yticks(range(0, 110, 10)) 


# Menampilkan plot
plt.show()


# ## Term

# In[27]:


plt.figure(figsize=(8,5))
sns.countplot(data=df_loan,x='term',hue='loan_status',   palette = ['lightgreen', 'tomato'], 
            lw = 1, ec = 'k')
plt.xticks(rotation=90, ha='right')
plt.show()

df1 = df_loan.groupby(['term', 'loan_status']).agg(Jumlah_Peminjam=( 'loan_status', 'count')).reset_index()
df2 =  df_loan.groupby('term').agg(Jumlah_Total_Peminjam=('term', 'count')).reset_index()
df_merge = df1.merge(df2, on = 'term')

# Menambah Kolom Rasio
df_merge['Rasio'] = round (df_merge['Jumlah_Peminjam'] / df_merge['Jumlah_Total_Peminjam'] * 100,2)

#ubah nama data
df_merge['loan_status'] = df_merge['loan_status'].replace({0: 'berhasil bayar', 1: 'gagal bayar'})
df_merge


# Banyak gagal bayar terjadi di jangka waktu 36 bulan <br>
# jika dilihat dalam bentuk persentase jumlah gagal bayar terbanyak terjadi pada jangka waktu 60 bulan

# # Tujuan Menggunakan Pinjaman (Purpose)

# In[28]:


plt.figure(figsize=(10,10))
sns.countplot(y= "purpose",order=value_counts.index, data = df_loan, palette = "Spectral")
plt.show()

value_counts = df_loan["purpose"].value_counts()
percentage = value_counts / value_counts.sum()
percentage = percentage.apply("{:.2%}".format)
print(percentage)


# Hampir 60% Peminjaman digunakan untuk menutupi hutang sebelumnya

# In[70]:


df_pre = df_loan.copy()


# # 3. Preprocessing

# # Feature Encoding

# In[71]:


# Drop column ymnt_plan
df_pre = df_pre.drop(['pymnt_plan'], axis=1)


# ### one Hot Encoding

# In[72]:


df_pre = pd.get_dummies(df_pre, columns=['term'], prefix=['term'])


# In[73]:


df_pre = pd.get_dummies(df_pre, columns=['purpose'], prefix=['purpose'])


# In[74]:


df_pre = pd.get_dummies(df_pre, columns=['grade'], prefix=['grade'])


# In[75]:


df_pre = pd.get_dummies(df_pre, columns=['home_ownership'], prefix=['home_ownership'])


# In[76]:


df_pre.info()


# In[77]:


df_pre


# ## Feature Selection
# menghapus fitur yang tidak diperlukan

# Uji Statistik

# In[78]:


import numpy as np
from scipy import stats
# Pisahkan data menjadi dua kelompok berdasarkan 'loan_status'
group1 = df_pre[df_pre['loan_status'] == 'Berhasil Bayar']['installment']
group2 = df_pre[df_pre['loan_status'] == 'Gagal Bayar']['installment']

# Lakukan t-test independen
t_statistic, p_value = stats.ttest_ind(group1, group2)

# Tampilkan hasil uji
print("T-statistic:", t_statistic)
print("P-value:", p_value)

# Interpretasi hasil
alpha = 0.05  # Tingkat signifikansi yang Anda tentukan
if p_value < alpha:
    print("Ada perbedaan yang signifikan dalam rata-rata 'acc_now_delinq' antara 'Berhasil Bayar' dan 'Gagal Bayar'.")
else:
    print("Tidak ada perbedaan yang signifikan dalam rata-rata 'acc_now_delinq' antara 'Berhasil Bayar' dan 'Gagal Bayar'.")


# In[79]:


df_pre.drop(columns=['funded_amnt_inv','emp_length','addr_state', 'verification_status', 'delinq_2yrs','inq_last_6mths','mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec','revol_bal','revol_util','total_acc','mths_since_last_major_derog', 'initial_list_status','out_prncp','out_prncp_inv','total_pymnt_inv','collection_recovery_fee', 'last_pymnt_amnt','collections_12_mths_ex_med','tot_coll_amt', 'tot_cur_bal','total_rev_hi_lim', 'acc_now_delinq'], inplace=True)


# In[80]:


df_pre['loan_status'] = df_pre['loan_status'].replace({'Gagal Bayar': '1', 'Berhasil Bayar': '0'})


# In[81]:


df_pre['loan_status'] = df_pre['loan_status'].astype(int)


# In[82]:


df_pre.info()


# ### Corelation heatmap (untuk mencegah terjadinya feature redundan)

# In[83]:


plt.figure(figsize=(30, 15))
sns.heatmap(df_pre.corr(), cmap='Blues', annot=True)


# **Berdasarkan Keterangan Heatmap diatas:**
# - loan_amnt,installment dan funded_amnt memiliki korelasi yang tinggi sehingga 2 feature lainnya harus di drop (loan_amnt & installment)
# - total_rec_prncp & total_pymnt juga memiliki korelasi yang tinggi sehingga harus di drop (total_rec_prncp) untuk menghindari feature redundant

# In[84]:


df_pre = df_pre.drop(['loan_amnt', 'installment','total_rec_prncp'], axis=1)


# In[85]:


df_pre.info()


# ###  Handle Values yang tidak diperlukan dan Missing Values

# In[86]:


import pandas as pd

# Menghitung jumlah nilai yang hilang per kolom
missing_values = df_pre.isnull().sum()

# Menghitung total baris dalam DataFrame
total_rows = len(df_pre)

# Menghitung persentase missing values per kolom
missing_percentage = (missing_values / total_rows) * 100

# Menampilkan hasil
print(missing_percentage)


# hanya kolom annual_inc yang terdapat missing values dan dilihat dari persentase missing values sangat kecil kurang dari 1% sehingga akan di drop missing values pada kolom annual_inc

# In[87]:


# Hapus missing values
df_pre.dropna(subset=['annual_inc'], inplace=True)


# In[88]:


import pandas as pd

# Menghitung jumlah nilai yang hilang per kolom
missing_values = df_pre.isnull().sum()

# Menghitung total baris dalam DataFrame
total_rows = len(df_pre)

# Menghitung persentase missing values per kolom
missing_percentage = (missing_values / total_rows) * 100

# Menampilkan hasil
print(missing_percentage)


# In[89]:


df = df_pre.copy()


# In[90]:


df.info()


# ### Handle Outlier

# In[91]:


nums = ['annual_inc', 'total_pymnt']


# In[92]:


from scipy import stats
print(f'Jumlah baris sebelum memfilter outlier: {len(df)}')

filtered_entries = np.array([True] * len(df))

for col in nums:
    zscore = abs(stats.zscore(df[col])) # hitung absolute z-scorenya
    filtered_entries = (zscore < 3) & filtered_entries # keep yang kurang dari 3 absolute z-scorenya
    
df = df[filtered_entries] # filter, cuma ambil yang z-scorenya dibawah 3

print(f'Jumlah baris setelah memfilter outlier: {len(df)}')


# In[93]:


df= df.drop(['total_rec_int', 'total_rec_late_fee', 'recoveries'], axis=1)


# In[94]:


df


# ## Feature Transform (StandardScaller)

# In[95]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
df['annual_inc_std'] = StandardScaler().fit_transform(df['annual_inc'].values.reshape(len(df), 1))
df['total_pymnt_std'] = StandardScaler().fit_transform(df['total_pymnt'].values.reshape(len(df), 1))


# In[96]:


df= df.drop(['annual_inc', 'total_pymnt'], axis=1)


# In[97]:


df.info()


# # 4. Machine Learning

# In[98]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# from google.colab import drive # import csv data from gdrive
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


# ### split test & train

# In[99]:


#split test & train
y = df['loan_status']
X = df.drop('loan_status', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ### Class Imbalance

# In[100]:


#class imbalance
from imblearn import over_sampling
X_train_over, y_train_over = over_sampling.SMOTE(sampling_strategy=0.5).fit_resample(X_train, y_train)


# In[65]:


df


# # DecisionTreeClasifier

# In[101]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

def eval_classification(model):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred_proba_train = model.predict_proba(X_train)
    
    print("Accuracy (Test Set): %.2f" % accuracy_score(y_test, y_pred))
    print("Precision (Test Set): %.2f" % precision_score(y_test, y_pred))
    print("Recall (Test Set): %.2f" % recall_score(y_test, y_pred))
    print("F1-Score (Test Set): %.2f" % f1_score(y_test, y_pred))
    
    print("roc_auc (test-proba): %.2f" % roc_auc_score(y_test, y_pred_proba[:, 1]))
    print("roc_auc (train-proba): %.2f" % roc_auc_score(y_train, y_pred_proba_train[:, 1]))
    # 
    score = cross_validate(DecisionTreeClassifier(), X, y, cv=5, scoring='recall', return_train_score=True)
    print('Recall (crossval train): '+ str(score['train_score'].mean()))
    print('Recall (crossval test): '+ str(score['test_score'].mean()))

def show_feature_importance(model):
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 8))
    ax.invert_yaxis()

    plt.xlabel('score')
    plt.ylabel('feature')
    plt.title('feature importance score')

def show_best_hyperparameter(model):
    print(model.best_estimator_.get_params())


# In[102]:


# decision tree
from sklearn.tree import DecisionTreeClassifier # import decision tree dari sklearn
dt = DecisionTreeClassifier() # inisiasi object dengan nama dt
dt.fit(X_train_over, y_train_over) # fit model decision tree dari data train
eval_classification(dt)


# nilai dari matrik evalaluasi cukup bagus akan tetapi pada rata rata cross validation data test & data train terlihat model cukup overfitting

# ## Logistic Regression

# In[110]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

def eval_classification(model):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred_proba_train = model.predict_proba(X_train)
    
    print("Accuracy (Test Set): %.2f" % accuracy_score(y_test, y_pred))
    print("Precision (Test Set): %.2f" % precision_score(y_test, y_pred))
    print("Recall (Test Set): %.2f" % recall_score(y_test, y_pred))
    print("F1-Score (Test Set): %.2f" % f1_score(y_test, y_pred))
    
    print("roc_auc (test-proba): %.2f" % roc_auc_score(y_test, y_pred_proba[:, 1]))
    print("roc_auc (train-proba): %.2f" % roc_auc_score(y_train, y_pred_proba_train[:, 1]))
    # 
    score = cross_validate(LogisticRegression(), X, y, cv=5, scoring='recall', return_train_score=True)
    print('recall (crossval train): '+ str(score['train_score'].mean()))
    print('recall (crossval test): '+ str(score['test_score'].mean()))

def show_feature_importance(model):
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 8))
    ax.invert_yaxis()

    plt.xlabel('score')
    plt.ylabel('feature')
    plt.title('feature importance score')

def show_best_hyperparameter(model):
    print(model.best_estimator_.get_params())


# In[111]:


from sklearn.linear_model import LogisticRegression # import logistic regression dari sklearn
logreg = LogisticRegression() # inisiasi object dengan nama logreg
logreg.fit(X_train_over, y_train_over) # fit model regression dari data train
eval_classification(logreg)


# pada model logistic regression model terlihat cukup baik terlihat dari skor matrik evaluasi juga tidak ada tanda overfitting ataupun underfitting

# In[105]:


def show_feature_importance(model, feature_names):
    coef = model.coef_[0]
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(), x='Coefficient', y='Feature', palette='viridis')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.title('Top 5 Feature Importances')
    plt.show()

show_feature_importance(logreg, X_train_over.columns)


# In[106]:


from sklearn.metrics import confusion_matrix

y_pred = logreg.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=0, suppress=True, threshold=100000)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel(" labels")
plt.show()


# ## LGBM

# In[107]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

def eval_classification(model):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred_proba_train = model.predict_proba(X_train)
    
    print("Accuracy (Test Set): %.2f" % accuracy_score(y_test, y_pred))
    print("Precision (Test Set): %.2f" % precision_score(y_test, y_pred))
    print("Recall (Test Set): %.2f" % recall_score(y_test, y_pred))
    print("F1-Score (Test Set): %.2f" % f1_score(y_test, y_pred))
    print("roc_auc-Score (Test Set): %.2f" % roc_auc_score(y_test, y_pred))
    
    print("roc_auc (test-proba): %.2f" % roc_auc_score(y_test, y_pred_proba[:, 1]))
    print("roc_auc (train-proba): %.2f" % roc_auc_score(y_train, y_pred_proba_train[:, 1]))
    # 
    score = cross_validate(LGBMClassifier(), X, y, cv=5, scoring='recall', return_train_score=True)
    print('Recall (crossval train): '+ str(score['train_score'].mean()))
    print('Recall (crossval test): '+ str(score['test_score'].mean()))

def show_feature_importance(model):
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    ax = feat_importances.nlargest(6).plot(kind='barh', figsize=(10, 8))
    ax.invert_yaxis()

    plt.xlabel('score')
    plt.ylabel('feature')
    plt.title('feature importance score')

def show_best_hyperparameter(model):
    print(model.best_estimator_.get_params())


# In[108]:


from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
lgbm.fit(X_train_over,y_train_over)

y_pred = dt.predict(X_test)
eval_classification(lgbm)


# ## XGBOOST

# In[445]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

def eval_classification(model):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred_proba_train = model.predict_proba(X_train)
    
    print("Accuracy (Test Set): %.2f" % accuracy_score(y_test, y_pred))
    print("Precision (Test Set): %.2f" % precision_score(y_test, y_pred))
    print("Recall (Test Set): %.2f" % recall_score(y_test, y_pred))
    print("F1-Score (Test Set): %.2f" % f1_score(y_test, y_pred))
    
    print("roc_auc (test-proba): %.2f" % roc_auc_score(y_test, y_pred_proba[:, 1]))
    print("roc_auc (train-proba): %.2f" % roc_auc_score(y_train, y_pred_proba_train[:, 1]))
    # 
    score = cross_validate(XGBClassifier(), X, y, cv=5, scoring='f1', return_train_score=True)
    print('f1 (crossval train): '+ str(score['train_score'].mean()))
    print('f1 (crossval test): '+ str(score['test_score'].mean()))

def show_feature_importance(model):
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    ax = feat_importances.nlargest(6).plot(kind='barh', figsize=(10, 8))
    ax.invert_yaxis()

    plt.xlabel('score')
    plt.ylabel('feature')
    plt.title('feature importance score')

def show_best_hyperparameter(model):
    print(model.best_estimator_.get_params())


# In[446]:


from xgboost import XGBClassifier

xg = XGBClassifier()
xg.fit(X_train_over, y_train_over)
eval_classification(xg)


# ## Random Forest

# In[447]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

def eval_classification(model):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred_proba_train = model.predict_proba(X_train)
    
    print("Accuracy (Test Set): %.2f" % accuracy_score(y_test, y_pred))
    print("Precision (Test Set): %.2f" % precision_score(y_test, y_pred))
    print("Recall (Test Set): %.2f" % recall_score(y_test, y_pred))
    print("F1-Score (Test Set): %.2f" % f1_score(y_test, y_pred))
    
    print("roc_auc (test-proba): %.2f" % roc_auc_score(y_test, y_pred_proba[:, 1]))
    print("roc_auc (train-proba): %.2f" % roc_auc_score(y_train, y_pred_proba_train[:, 1]))
    # 
    score = cross_validate(RandomForestClassifier(), X, y, cv=5, scoring='recall', return_train_score=True)
    print('recall (crossval train): '+ str(score['train_score'].mean()))
    print('recall (crossval test): '+ str(score['test_score'].mean()))

def show_feature_importance(model):
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 8))
    ax.invert_yaxis()

    plt.xlabel('score')
    plt.ylabel('feature')
    plt.title('feature importance score')

def show_best_hyperparameter(model):
    print(model.best_estimator_.get_params())


# In[448]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_over, y_train_over)
eval_classification(rf)


# In[ ]:




