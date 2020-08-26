import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('X:/CS2/kaggle/titanic/train.csv')
test_df = pd.read_csv('X:/CS2/kaggle/titanic/test.csv')
combine = [data,test_df]
# data.info() # Cabin，Sex为object类型，且重复数据较多考虑转换为catergory，还需要填充缺失数据
data.Cabin.value_counts()
'''
# data.Cabin = data.Cabin.astype('category')
data.Cabin.dtype
# data.info() # 发现内存占用反而增多，Cabin unique数据太多的原因，将其注释掉
data.Sex = data.Sex.astype('category')
data.Sex.dtype
# data.info()
data.shape
print('nan data:\n',data.isnull().sum().sort_values(ascending=False)) # 查看数据缺失值情况
data.Embarked.value_counts()
data.Embarked = data.Embarked.astype('category')
data.head()
# 画出各个性别存活情况的图，三种方法
survived_0 = data[data.Survived==0].Sex.value_counts()
survived_1 = data[data.Survived==1].Sex.value_counts()
df1 = pd.DataFrame({'survived':survived_1,'unsurvived':survived_0})
df1.plot(kind='bar',stacked=True,rot=0)
plt.ylabel('Number')
survived_f = data[data.Sex=='female'].Survived.value_counts()
survived_m = data[data.Sex=='male'].Survived.value_counts()
df2 = pd.DataFrame({'female':survived_f,'male':survived_m})
df2.index=['unsurvived','survived']
df2.plot(kind='bar',stacked=True,rot=0)
plt.ylabel('Number')
df3 = data.groupby('Sex').Survived.value_counts().unstack('Survived')
df3.columns = ['unsurvived','survived']
df3.plot(kind='bar',stacked=True,rot=0)
plt.ylabel('Number')
plt.title('Survived_Gender')
# 可以看出女性的获救率比男性高出很多
data.head()
# 看船舱等级对获救的影响
data.Pclass.value_counts()
df_class = data.groupby('Pclass').Survived.value_counts().unstack('Survived')
df_class.columns=['unsurvived','survived']
df_class.plot(kind='bar',rot=0,stacked=True)
plt.title('Pclass_Survived')
plt.ylabel('Number')
# 看登船口是都对获救有影响,把之前的代码封装成一个函数，便于后续调用
def pt(col):
    df_0 = data[data.Survived==0][col].value_counts()
    df_1 = data[data.Survived==1][col].value_counts()
    df = pd.DataFrame({'survived':df_1,'unsurvived':df_0})
    df.plot(kind='bar',rot=0,stacked=True)
    plt.ylabel('Number')
    plt.title(col + '_Survived',y=1.01)
    plt.xlabel(col)
pt('Embarked') # 发现C登船口获救率高一些

data['Companion'] = data.SibSp + data.Parch
pt('Companion') # 可以看出当同行者有1，2，3个是存活率更高，而为0时存活率较低
df_age = data[['Age','Survived']]
df_Age = df_age.groupby('Age').Survived.value_counts().unstack('Survived').fillna(0)
df_Age.plot(kind='area',figsize=(14,5))
plt.title('Age_Survived')
df_Age['survival_rate'] = df_Age[1]/df_Age[1]
df_rate = df_Age['survival_rate']
plt.show()
total = df_Age[[0,1]].sum().sum()
df_Age['percent0'] = df_Age[0]/total
df_Age['percent1'] = df_Age[1]/total
col_svive = df_Age[df_Age[1]>0].index
col_unsvive = df_Age[df_Age[1]==0].index
df_new = pd.DataFrame({'survived':col_svive})
df_new.plot(kind='kde')
df_new2 = pd.DataFrame({'unsurvived':col_unsvive})
df_new2.plot(kind='kde')
'''
# 删除没用的特征,保留test的ID
data = data.drop(['Ticket','Cabin','PassengerId'],axis=1)
test_df = test_df.drop(['Ticket','Cabin'],axis=1)
# 提取出Name中的特征，比如
test_df['Title'] = test_df.Name.str.extract('([A-Za-z]+)\.',expand=False)
data['Title'] = data.Name.str.extract('([A-Za-z]+)\.')
data.groupby('Title').Survived.value_counts().unstack('Survived').plot(kind='bar',stacked=True)
data.groupby('Title').Survived.mean()
combine = [data,test_df]
for dataset in combine:
    dataset['Title'] = dataset.Title.replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Other')
    dataset['Title']=dataset.Title.replace('Mlle','Miss')
    dataset['Title']=dataset.Title.replace('Ms','Miss')
    dataset['Title']=dataset.Title.replace('Mme','Mrs')
data.groupby('Title').Survived.mean()
title_mapping = {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Other':5}
for dataset in combine:
    dataset.Title=dataset.Title.map(title_mapping)
    dataset.Title=dataset.Title.fillna(0)
data.head()
# Master is a title used to speak to a boy who is too young to be called Mr.
data = data.drop('Name',axis=1)
test_df = test_df.drop('Name',axis=1)
combine = [data,test_df]
for i in combine:
    i.Sex = i.Sex.map({'female':1,'male':0}).astype(int)
data.head()
# 通过Pclass(1,2,3)和Sex(0,1)组合来猜测缺失年龄。填充缺失值！
ages = np.zeros((2,3))
ages
for df in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess=df[(df.Sex==i)&(df.Pclass==j+1)]['Age'].dropna() # 找出所有有效的组合,即组合且有对应的Age值
            final_guess = guess.median()
            ages[i,j] = int(final_guess/0.5 + 0.5)*0.5 #将浮点值转换为最小为0.5的值
    for i in range(2):
        for j in range(3):
            df.loc[(df.Age.isnull())&(df.Sex==i)&(df.Pclass==j+1),'Age'] = ages[i,j]
            # 把找到所有组合结果的median赋值给P，S组合对应缺失值
data['AgeRange'] = pd.cut(data.Age,5) # 将年龄分成五块,查看各年龄段的生还率
#data.groupby('AgeRange')['Survived'].mean().sort_values('AgeRange',ascending=True)
data.groupby(['AgeRange'])['Survived'].mean()
combine = [data,test_df]
for i in combine:
    i.loc[i.Age<=16,'Age']=0
    i.loc[(i.Age<=32)&(i.Age>16),'Age']=1
    i.loc[(i.Age<=48)&(i.Age>32),'Age']=2
    i.loc[(i.Age<=64)&(i.Age>48),'Age']=3
    i.loc[i.Age>64,'Age']=4
# 将年龄分为几个区间并将其归于其中
data.head()
data = data.drop('AgeRange',axis=1)
combine = [data,test_df]
data.head()
data['Companion'] = data.SibSp + data.Parch
test_df['Companion'] = test_df.SibSp + test_df.Parch
data = data.drop(['SibSp','Parch'],axis=1)
data.head()
data.groupby('Companion').Survived.mean()
combine = [data,test_df]
# Create a new feature isalone to replace Companion feature
for i in combine:
    i['IsAlone'] = 0
    i.loc[i.Companion==0,'IsAlone']=1
data = data.drop('Companion',axis=1)
test_df = test_df.drop('Companion',axis=1)
combine = [data,test_df]
data.head()
data[['IsAlone','Survived']].groupby('IsAlone').mean()
# Pclass*Age to create a new feature
for i in combine:
    i['Age*class'] = (i.Age * i.Pclass).astype(int)
data.loc[:,['Age*class','Pclass','Age']].head(10)
freq_port = data.Embarked.dropna().mode()[0]
freq_port
for i in combine:
    i['Embarked'] = i.Embarked.fillna(freq_port)
data[['Embarked','Survived']].groupby('Embarked').mean()
# 将分类特征转换为数值特征
for i in combine:
    i.Embarked = i.Embarked.map({'S':0,'C':1,'Q':2}).astype(int)
data.head()
test_df['Fare'].fillna(test_df.Fare.dropna().median(),inplace=True)
data['FareRange'] = pd.qcut(data.Fare,4)
data[['FareRange','Survived']].groupby('FareRange').mean()
#可以看出来，票价越高生还率越高。类似于Age，把Fare转换成有序的几个变量
for i in combine:
    i.loc[i.Fare<=7.91,'Fare'] = 0
    i.loc[(i.Fare>7.91)&(i.Fare<=14.454),'Fare'] = 1
    i.loc[(i.Fare>14.454)&(i.Fare<=31),'Fare'] = 2
    i.loc[i.Fare>31,'Fare'] = 3
    i['Fare'] = i['Fare'].astype(int)
data = data.drop('FareRange',axis=1)
combine = [data,test_df]
data.Age = data.Age.astype(int)
data.head()
test_df.Age = test_df.Age.astype('int')
test_df = test_df.drop(['SibSp','Parch'],axis=1)
test_df.head()
X_train = data.drop('Survived',axis=1)
Y_train = data['Survived']
# Set X AND Y
X_test = test_df.drop('PassengerId',axis=1).copy() # 这样才不会影响test_df
X_train.shape,Y_train.shape,X_test.shape


def normalize(array):
    return (array - np.mean(array))/(np.std(array))

# 先尝试使用Logistic Regression 模型
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
logreg = LogisticRegression(solver='lbfgs')
# for small datasets, 'liblinear' is a good choice
logreg.fit(X_train,Y_train)
Y_predic_logreg = logreg.predict(X_test)
acc_logreg = round(logreg.score(X_train,Y_train)*100,2)
acc_logreg
# 用logistic回归验证特征船舰和完成目标的假设和决策。这可以通过计算决策函数中特征的系数来实现
coeff_df = pd.DataFrame(data.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation',ascending=False)

# use support vector machine
from sklearn.svm import SVC
svc = SVC(gamma='auto',max_iter=1000)
svc.fit(X_train,Y_train)
Y_predic_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train,Y_train)*100,2)
acc_svc

# use KNN
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
Y_predic_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train,Y_train)*100,2)

# Gaussian Naive Bayes 需要一些参数和变量feature的数量成线性关系
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train,Y_train)
Y_predic_bayes = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train,Y_train)*100,2)

# Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train,Y_train)
Y_predic_perceptron = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train,Y_train)*100,2)

# Linear SVC
from sklearn.svm import LinearSVC
l_svc = LinearSVC()
l_svc.fit(X_train,Y_train)
Y_predic_lsvc = l_svc.predict(X_test)
acc_l_svc = round(l_svc.score(X_train,Y_train)*100,2)

# Stochastic Graident Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train,Y_train)
Y_predic_sgd = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train,Y_train)*100,2)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,Y_train)
Y_predic_dt = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train,Y_train)*100,2)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,Y_train)
Y_predic_rf = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train,Y_train)*100,2)



# compare score got by different models
rf_scores = cross_val_score(random_forest,X_train,Y_train,cv=10).mean()
logreg_scores = cross_val_score(logreg,X_train,Y_train,cv=10).mean()
bayes_scores = cross_val_score(gaussian,X_train,Y_train,cv=10).mean()
perceptron_scores = cross_val_score(perceptron,X_train,Y_train,cv=10).mean()
dt_scores = cross_val_score(decision_tree,X_train,Y_train,cv=10).mean()
sgd_scores = cross_val_score(sgd,X_train,Y_train,cv=10).mean()
lsvc_scores = cross_val_score(l_svc,X_train,Y_train,cv=10).mean()
knn_scores = cross_val_score(knn,X_train,Y_train,cv=10).mean()
svc_scores = cross_val_score(svc,X_train,Y_train,cv=10).mean()
models = pd.DataFrame({'Model':['SVM','KNN','LogisticR','Random Forest','Naive Bayes',
                              'Perceptron','SGD','Linear SVC','Decision Tree'],
                     'Score':[acc_svc,acc_knn,acc_logreg,acc_random_forest,acc_gaussian,
                             acc_perceptron,acc_sgd,acc_l_svc,acc_decision_tree],
                      'Cross_Val_Score':[svc_scores,knn_scores,logreg_scores,rf_scores,bayes_scores,
                                        perceptron_scores,sgd_scores,lsvc_scores,dt_scores]})
models.sort_values(by='Score')
submission = pd.DataFrame({'PassengerId':test_df.PassengerId,
                         'Survived':Y_predic_dt}) #choose Y_predict from models with highest score.
# 可以选择随机森林来纠正决策树队训练集过拟合的倾向
path='X:/CS2/kaggle/submission_decision_tree.csv'
submission.to_csv(path)
models
# test_df.head()
