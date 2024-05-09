import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz

df2_valid = pd.read_excel('文物样品信息汇总-填补缺失值.xlsx')
# 将风化样本的无风化采样点归到无风化组
df_whe = df2_valid[df2_valid.表面风化 == '风化'].drop([23, 25, 29, 30, 44, 45, 48, 53, 56, 60])
df_no_whe = pd.concat(
    [df2_valid[df2_valid.表面风化 == '无风化'], df2_valid.loc[[23, 25, 29, 30, 44, 45, 48, 53, 56, 60]]]).sort_index()
Y_whe = df_whe.类型
Y_no_whe = df_no_whe.类型
X_whe = df_whe.drop(['文物编号', '文物采样点', '类型', '表面风化', '纹饰', '颜色'], axis=1)
X_no_whe = df_no_whe.drop(['文物编号', '文物采样点', '类型', '表面风化', '纹饰', '颜色'], axis=1)
Xtrain_whe, Xtest_whe, Ytrain_whe, Ytest_whe = train_test_split(X_whe, Y_whe, test_size=0.2, random_state=400)
Xtrain_no_whe, Xtest_no_whe, Ytrain_no_whe, Ytest_no_whe = train_test_split(X_no_whe, Y_no_whe, test_size=0.2,
                                                                            random_state=2022)
print('决策权拟合，有风化：')
clf_whe = tree.DecisionTreeClassifier(random_state=400).fit(Xtrain_whe, Ytrain_whe)
print('测试集准确率为：')
print(clf_whe.score(Xtest_whe, Ytest_whe))
feature_name = ['二氧化硅', '氧化钠', '氧化钾', '氧化钙', '氧化镁', '氧化铝', '氧化铁', '氧化铜', '氧化铅', '氧化钡', '五氧化二磷', '氧化锶', '氧化锡', '二氧化硫']
dot_data = tree.export_graphviz(clf_whe
                                , feature_names=feature_name
                                , class_names=["铅钡", "高钾"]
                                , filled=True
                                , rounded=True
                                )
graph = graphviz.Source(dot_data)  # 可以通过ipython进行展示
print('决策树的可视化可以通过ipython展示')
print('*' * 30)
print('决策权拟合，无风化：')
clf_no_whe = tree.DecisionTreeClassifier(random_state=1000).fit(Xtrain_no_whe, Ytrain_no_whe)
print('测试集准确率为：')
print(clf_no_whe.score(Xtest_no_whe, Ytest_no_whe))
feature_name = ['二氧化硅', '氧化钠', '氧化钾', '氧化钙', '氧化镁', '氧化铝', '氧化铁', '氧化铜', '氧化铅', '氧化钡', '五氧化二磷', '氧化锶', '氧化锡', '二氧化硫']
dot_data = tree.export_graphviz(clf_no_whe
                                , feature_names=feature_name
                                , class_names=["铅钡", "高钾"]
                                , filled=True
                                , rounded=True
                                )
graph = graphviz.Source(dot_data)  # 可以通过ipython进行展示
print('决策树的可视化可以通过ipython展示')
print('*' * 30)
print('决策权拟合，有风化，去除铅：')
X_whe_dropPb = X_whe.drop('氧化铅(PbO)', axis=1)
Xtrain_whe_dropPb, Xtest_whe_dropPb, Ytrain_whe_dropPb, Ytest_whe_dropPb = train_test_split(X_whe_dropPb, Y_whe,
                                                                                            test_size=0.2,
                                                                                            random_state=2022)
clf_whe_dropPb = tree.DecisionTreeClassifier(random_state=2022).fit(Xtrain_whe_dropPb, Ytrain_whe_dropPb)
print('测试集准确率为：')
print(clf_whe_dropPb.score(Xtest_whe_dropPb, Ytest_whe_dropPb))
feature_name = ['二氧化硅', '氧化钠', '氧化钾', '氧化钙', '氧化镁', '氧化铝', '氧化铁', '氧化铜', '氧化钡', '五氧化二磷', '氧化锶', '氧化锡', '二氧化硫']
dot_data = tree.export_graphviz(clf_whe_dropPb
                                , feature_names=feature_name
                                , class_names=["铅钡", "高钾"]
                                , filled=True
                                , rounded=True
                                )
graph = graphviz.Source(dot_data)  # 可以通过ipython进行展示
print('决策树的可视化可以通过ipython展示')
print('*' * 30)
print('决策权拟合，无风化，去除铅：')
X_no_whe_dropPb = X_no_whe.drop('氧化铅(PbO)',axis=1)
Xtrain_no_whe_dropPb, Xtest_no_whe_dropPb, Ytrain_no_whe_dropPb, Ytest_no_whe_dropPb = train_test_split(X_no_whe_dropPb, Y_no_whe, test_size=0.2,random_state=800)
clf_no_whe_dropPb = tree.DecisionTreeClassifier(random_state=2022).fit(Xtrain_no_whe_dropPb, Ytrain_no_whe_dropPb)
print('测试集准确率为：')
print(clf_no_whe_dropPb.score(Xtest_no_whe_dropPb, Ytest_no_whe_dropPb))
feature_name = ['二氧化硅', '氧化钠', '氧化钾', '氧化钙', '氧化镁', '氧化铝', '氧化铁', '氧化铜', '氧化钡', '五氧化二磷', '氧化锶', '氧化锡', '二氧化硫']
dot_data = tree.export_graphviz(clf_no_whe_dropPb
                                , feature_names=feature_name
                                , class_names=["铅钡", "高钾"]
                                , filled=True
                                , rounded=True
                                )
graph = graphviz.Source(dot_data)  # 可以通过ipython进行展示
print('决策树的可视化可以通过ipython展示')
print('*' * 30)
print('随机森林拟合：')
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
Xtrain_whe, Xtest_whe, Ytrain_whe, Ytest_whe = train_test_split(X_whe, Y_whe, test_size=0.2,random_state=400)
Xtrain_no_whe, Xtest_no_whe, Ytrain_no_whe, Ytest_no_whe = train_test_split(X_no_whe, Y_no_whe, test_size=0.2,random_state=300)
le1 = LabelEncoder().fit(Ytrain_whe)
rfr1 = RandomForestRegressor(random_state=2022, n_estimators=1000)
rfr1 = rfr1.fit(Xtrain_whe, le1.transform(Ytrain_whe))
score1 = rfr1.score(Xtest_whe, le1.transform(Ytest_whe))
print('有风化，测试集准确率：')
print(score1)
le12 = LabelEncoder().fit(Ytrain_no_whe)
rfr2 = RandomForestRegressor(random_state=2022, n_estimators=1000)
rfr2 = rfr2.fit(Xtrain_no_whe, le1.transform(Ytrain_no_whe))
score2 = rfr2.score(Xtest_no_whe, le1.transform(Ytest_no_whe))
print('无风化，测试集准确率：')
print(score2)
print('有风化，化学成分在分类中的重要性：')
print(pd.Series(rfr1.feature_importances_,index=X_whe.columns).sort_values(ascending=False))
print('无风化，化学成分在分类中的重要性：')
print(pd.Series(rfr2.feature_importances_,index=X_no_whe.columns).sort_values(ascending=False))
