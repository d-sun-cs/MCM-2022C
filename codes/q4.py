
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"] #解决中文字符乱码的问题
plt.rcParams["axes.unicode_minus"] = False #正常显示负号

# 读取数据
df = pd.read_excel('表1和表2数据（含虚拟变量）.xlsx')
feature = df[['二氧化硅(SiO2)','氧化钠(Na2O)','氧化钾(K2O)','氧化钙(CaO)','氧化镁(MgO)','氧化铝(Al2O3)','氧化铁(Fe2O3)','氧化铜(CuO)','氧化铅(PbO)','氧化钡(BaO)','五氧化二磷(P2O5)','氧化锶(SrO)','氧化锡(SnO2)','二氧化硫(SO2)','类型_铅钡']]
feature.to_excel("化学成分含量及类型.xlsx")
feature_group = feature.groupby(by='类型_铅钡')
feature_group = [x[1].drop('类型_铅钡',axis=1) for x in feature_group]
feature_group[0].to_excel('高钾玻璃数据的特征.xlsx',index=False)
feature_group[1].to_excel("铅钡玻璃数据的特征.xlsx",index=False)

# 正态分布检验（JB检验）
def JB_Check(data,mess):
    jb_p = []
    jb_values = []
    plt.figure(14,figsize=(10,100))
    i = 1
    for item in data.columns:
        # JB检验
        res = stats.jarque_bera(data[item])
        jb_p.append(res.pvalue)
        jb_values.append(res.statistic)
        # 画直方图
        plt.subplot(14,1,i)
        plt.hist(data[item], bins=20, density=True)
        plt.title(item)
        # plt.savefig('./图片/'+mess+item+'直方图.png')
        i += 1
    plt.show()
    return jb_values,jb_p

# 铅钡玻璃正态分布检验
df1 = pd.read_excel('铅钡玻璃数据的特征.xlsx')
jb_values, jb_p = JB_Check(df1,'铅钡')
print(jb_p)
np.array(jb_p)>0.05
print(df1.columns[np.array(jb_p)>0.05])

# 高钾玻璃正态分布检验
df2 = pd.read_excel('高钾玻璃数据的特征.xlsx')
jb_values, jb_p = JB_Check(df2,'高钾')
print(jb_p)
np.array(jb_p)>0.05
print(df2.columns[np.array(jb_p)>0.05])

# 绘制热力图
def graph(data,mess):
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(111)
    ax = sns.heatmap(data,vmax = 1,vmin = -1,annot = True,annot_kws = {"size":10,"weight":"bold"},linewidths = 0.05,cmap="RdBu")
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.savefig(mess+'玻璃各化学成分spearman相关系数热力图.png',dpi=200, bbox_inches='tight')
    plt.show()
# 斯皮尔曼相关系数
col = ['二氧化硅','氧化钠','氧化钾','氧化钙','氧化镁','氧化铝','氧化铁','氧化铜','氧化铅','氧化钡','五氧化二磷','氧化锶','氧化锡','二氧化硫']
corr1 = df1.corr('spearman')
corr2 = df2.corr('spearman')
corr1.columns = col
corr1.index = col
corr2.columns = col
corr2.index = col
graph(corr1,'铅钡')
graph(corr2,'高钾')

# 秩检验
res_statistic = []
res_p = []
for i in col:
    res = stats.wilcoxon(corr1[i],corr2[i],correction=True)
    res_statistic.append(res[0])
    res_p.append(res[1])
wilcoxon_df = pd.DataFrame({'统计量':res_statistic,
                            'P值':res_p,
                            '是否有差异(95%)':['有差异' if x < 0.05 else '无差异' for x in res_p],
                            '是否有差异(90%)':['有差异' if x < 0.10 else '无差异' for x in res_p]})
wilcoxon_df.index = col
wilcoxon_df.to_excel('wilcoxon符号秩检验(spearman).xlsx')