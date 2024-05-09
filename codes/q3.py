import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import zscore
import pylab as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei"] #解决中文字符乱码的问题
plt.rcParams["axes.unicode_minus"] = False #正常显示负号

# 读取数据
df2_valid = pd.read_excel('文物样品信息汇总-填补缺失值.xlsx')
# 填补缺失值
df2_valid_fill0 = df2_valid.fillna(0)
# 转换虚拟变量
df2_valid_fill0_dummies = pd.get_dummies(df2_valid_fill0[['纹饰','类型','颜色','表面风化']])
df2_valid_fill0 = df2_valid_fill0.drop(['纹饰','类型','颜色','表面风化'],axis=1)
df2_valid_fill0_dummies = pd.concat([df2_valid_fill0,df2_valid_fill0_dummies],axis=1)
df2_valid_fill0_dummies
df2_valid_fill0_dummies.to_excel('表1和表2数据（含虚拟变量）.xlsx')

# 保留与化学成分信息和类别信息
df = pd.read_excel('表1和表2数据（含虚拟变量）.xlsx')
feature = df[['二氧化硅(SiO2)','氧化钠(Na2O)','氧化钾(K2O)','氧化钙(CaO)','氧化镁(MgO)','氧化铝(Al2O3)','氧化铁(Fe2O3)','氧化铜(CuO)','氧化铅(PbO)','氧化钡(BaO)','五氧化二磷(P2O5)','氧化锶(SrO)','氧化锡(SnO2)','二氧化硫(SO2)']]
label = df['类型_铅钡']
all_f = df[['二氧化硅(SiO2)','氧化钠(Na2O)','氧化钾(K2O)','氧化钙(CaO)','氧化镁(MgO)','氧化铝(Al2O3)','氧化铁(Fe2O3)','氧化铜(CuO)','氧化铅(PbO)','氧化钡(BaO)','五氧化二磷(P2O5)','氧化锶(SrO)','氧化锡(SnO2)','二氧化硫(SO2)','类型_铅钡']]
all_f.to_excel('指标汇总（所有化学成分）.xlsx')

# 正态分布检验（JB检验）
jb_p = []
jb_values = []
for i in feature.columns:
    res = stats.jarque_bera(feature[i])
    jb_p.append(res.pvalue)
    jb_values.append(res.statistic)

# 计算斯皮尔曼相关系数
res_r = []
res_p = []
for x in feature.columns:
    r, p = stats.spearmanr(feature[x], label)
    res_r.append(r)
    res_p.append(p)
res_sp = pd.DataFrame({'相关系数(绝对值)':abs(np.array(res_r)),
                       'P值':res_p})
res_sp.index = feature.columns
res_sp = res_sp.sort_values(by='相关系数(绝对值)',ascending=False)
res_sp.to_excel('各化学成分与玻璃类型的spearman相关系数(绝对值).xlsx')

# 需要降维的指标 ['氧化钠(Na2O)', '氧化钙(CaO)', '氧化镁(MgO)', '氧化铝(Al2O3)', '氧化铁(Fe2O3)','氧化铜(CuO)', '五氧化二磷(P2O5)', '氧化锡(SnO2)', '二氧化硫(SO2)']
# 保留的指标 ['二氧化硅(SiO2)', '氧化钾(K2O)', '氧化铅(PbO)', '氧化钡(BaO)', '氧化锶(SrO)']

df = pd.read_excel('指标汇总（所有化学成分）.xlsx')
X = df.drop(['类型_铅钡'],axis=1)
y = df['类型_铅钡']
X_reduce = X[['氧化钠(Na2O)', '氧化钙(CaO)', '氧化镁(MgO)', '氧化铝(Al2O3)', '氧化铁(Fe2O3)','氧化铜(CuO)', '五氧化二磷(P2O5)', '氧化锡(SnO2)', '二氧化硫(SO2)']]

# 降维保留个数的确认
contribute = []
X_ = zscore(X_reduce, axis=0)  # X需要标准化
for i in range(1,9):
    pca = PCA(n_components=i)
    result_pca = pca.fit_transform(X_)    # 这里的数据实际上已经被标准化了
    t = np.array(pca.explained_variance_ratio_).cumsum()[-1]
    contribute.append(t)
plt.plot(range(1,9),contribute,'-r',range(1,9),contribute,'b*',range(1,9),[0.85,0.85,0.85,0.85,0.85,0.85,0.85,0.85],'--k')
plt.xlabel('特征数')
plt.ylabel('累计贡献率')
plt.savefig('特征数与累计贡献率的关系.png')
plt.show()

# 确认保留6个维度，求系数矩阵
pca = PCA(n_components=6)
result_pca = pca.fit_transform(X_)
np.savetxt('特征向量(系数矩阵).txt',np.array(pca.components_.T),)

# 指标选取（保存相关性大的指标，其余进行降维）
'''
函数名：deal_sheet3
函数作用：对附件表三进行指标提取
输入：要改变含量的列col，增加幅度i，权中系数w1，w2
输出：11个指标
'''
def deal_sheet3(col,i,w1,w2):
    # print(w1,w2)
    df3 = pd.read_excel('附件.xlsx', sheet_name=2)
    df3 = df3.drop(['文物编号','表面风化'],axis=1)
    df3 = df3.fillna(0)
    # print(df3)
    df3[col] = (1+i)*df3[col]
    temp = df3.sum(axis=1)[:,np.newaxis]
    temp = np.tile(temp,(1,df3.shape[1]))
    df3 = (df3/temp)*100
    df3.to_excel('预测数据'+col+'增加'+str(i)+'.xlsx')
    df3_X = df3[['氧化钠(Na2O)', '氧化钙(CaO)', '氧化镁(MgO)', '氧化铝(Al2O3)', '氧化铁(Fe2O3)','氧化铜(CuO)', '五氧化二磷(P2O5)', '氧化锡(SnO2)', '二氧化硫(SO2)']]
    df3_X_ = zscore(df3_X, axis=0)  # X需要标准化
    result_pca = df3_X_ @ np.loadtxt('特征向量(系数矩阵).txt')
    df3_all_feature = pd.concat([w1*df3[['二氧化硅(SiO2)', '氧化钾(K2O)', '氧化铅(PbO)', '氧化钡(BaO)', '氧化锶(SrO)']],pd.DataFrame(w2*result_pca)],axis=1)
    return df3_all_feature
'''
函数名：deal_X
函数作用：对数据进行指标提取
输入：原始数据X，权中系数w1，w2
输出：11个指标
'''
def deal_X(X,w1,w2):
    # print(w1,w2)
    temp = X.sum(axis=1)[:,np.newaxis]
    temp = np.tile(temp,(1,X.shape[1]))
    X = (X/temp)*100
    X_reduce = X[['氧化钠(Na2O)', '氧化钙(CaO)', '氧化镁(MgO)', '氧化铝(Al2O3)', '氧化铁(Fe2O3)','氧化铜(CuO)', '五氧化二磷(P2O5)', '氧化锡(SnO2)', '二氧化硫(SO2)']]
    X_reduce_ = zscore(X_reduce, axis=0)  # X需要标准化
    result_pca = X_reduce_ @ np.loadtxt('特征向量(系数矩阵).txt')
    all_feature = pd.concat([w1*X[['二氧化硅(SiO2)', '氧化钾(K2O)', '氧化铅(PbO)', '氧化钡(BaO)', '氧化锶(SrO)']],pd.DataFrame(w2*result_pca)],axis=1)
    return all_feature

# 读取数据并提取指标
w1 = 0.7
w2 = 0.3
X = deal_X(X,w1,w2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=8)#8,12,2,6,7  9
X_predict = deal_sheet3('二氧化硅(SiO2)',0,w1,w2)

# 预测
clf_sk = KNeighborsClassifier(n_neighbors=5)
clf_sk.fit(X_train, y_train)
print('knn得分:',clf_sk.score(X_test, y_test))
print('预测结果',clf_sk.predict(X_predict))

# 敏感性分析
'''
函数名：Sensitive_Analysis
函数作用：改变某一化学成分含量，输出结果变化
输入：模型model，要改变的化学成分col，每次增加的幅度delta，增加次数i
输出：准确度得分数组，和预测结果数组
'''
def Sensitive_Analysis(model,col,delta,i):
    model.fit(X_train,y_train)
    X_test_ = X_test.copy()
    score = []
    predict_res = []
    for k in range(i):
        X_predict_ = deal_sheet3(col,k*delta,w1,w2)
        X_test_[col] = (k * delta + 1) * X_test[col]
        score_temp = model.score(X_test_,y_test)
        predict_temp = model.predict(X_predict_)
        score.append(score_temp)
        predict_res.append(predict_temp)
        print('\n'+col+'增加'+str(k*delta)+'倍:')
        print('得分',score_temp)
        print('预测结果',predict_temp)
    return score,predict_res

##%%
# knn算法敏感度分析
prefict_res = Sensitive_Analysis(KNeighborsClassifier(n_neighbors=5),'二氧化硅(SiO2)',0.1,8)
# 绘制图片
plt.plot(np.arange(0,0.8,0.1),prefict_res[0],'-r')
plt.xlabel("二氧化硅增加量")
plt.ylabel('训练集得分')
plt.savefig('训练集得分与二氧化硅增加量的关系')