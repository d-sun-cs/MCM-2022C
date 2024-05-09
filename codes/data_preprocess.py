import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

df1 = pd.read_excel('附件.xlsx', sheet_name=0)
df2_valid = pd.read_excel('文物样品信息汇总.xlsx')
print('缺失值比例情况：\n', df1.isnull().sum() / df1.shape[0])
filter_df1 = df2_valid[(df2_valid.纹饰 == 'A') & (df2_valid.类型 == '铅钡') & (df2_valid.表面风化 == '风化')].drop(
    ['文物编号', '文物采样点'], axis=1)
filter_df2 = df2_valid[(df2_valid.纹饰 == 'C') & (df2_valid.类型 == '铅钡') & (df2_valid.表面风化 == '风化')].drop(
    ['文物编号', '文物采样点'], axis=1)
X_train1 = filter_df1.drop(['纹饰', '类型', '颜色', '表面风化'], axis=1).fillna(0)[~filter_df1.颜色.isna().values]
Y_train1 = filter_df1.颜色.dropna()
X_pre1 = filter_df1.drop(['纹饰', '类型', '颜色', '表面风化'], axis=1).fillna(0)[filter_df1.颜色.isna().values]
X_train2 = filter_df2.drop(['纹饰', '类型', '颜色', '表面风化'], axis=1).fillna(0)[~filter_df2.颜色.isna().values]
Y_train2 = filter_df2.颜色.dropna()
X_pre2 = filter_df2.drop(['纹饰', '类型', '颜色', '表面风化'], axis=1).fillna(0)[filter_df2.颜色.isna().values]
le1 = LabelEncoder().fit(Y_train1)
rfr1 = RandomForestRegressor(random_state=2022, n_estimators=100)
rfr1 = rfr1.fit(X_train1, le1.transform(Y_train1))
res1 = le1.inverse_transform(rfr1.predict(X_pre1).round().astype('int'))
score1 = rfr1.score(X_train1, le1.transform(Y_train1))
print('铅钡A纹饰颜色预测结果及准确率：')
print(res1)
print(score1)
le2 = LabelEncoder().fit(Y_train2)
rfr2 = RandomForestRegressor(random_state=2022, n_estimators=100)
rfr2 = rfr2.fit(X_train2, le2.transform(Y_train2))
res2 = le2.inverse_transform(rfr2.predict(X_pre2).round().astype('int'))
score2 = rfr2.score(X_train2, le2.transform(Y_train2))
print('铅钡C纹饰颜色预测结果及准确率：')
print(res2)
print(score2)
df2_valid.iloc[19, 18] = res1[0]
df2_valid.iloc[42, 18] = res2[0]
df2_valid.iloc[52, 18] = res1[1]
df2_valid.iloc[66, 18] = res2[1]
df2_valid.fillna(0, inplace=True)
df2_valid.to_excel('文物样品信息汇总-填补缺失值.xlsx', index=False)
