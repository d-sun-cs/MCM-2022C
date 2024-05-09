import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings

warnings.filterwarnings("ignore",category=Warning)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

df2_valid = pd.read_excel('文物样品信息汇总-填补缺失值.xlsx')
print('*'*50)
print('高钾亚类划分')
df2_k_cc = df2_valid[df2_valid.类型=='高钾'].drop(['文物编号','文物采样点','纹饰','类型','颜色','表面风化'],axis=1)
df2_k_cc.iloc[1] = df2_k_cc.iloc[1:3].mean()
df2_k_cc.iloc[5] = df2_k_cc.iloc[5:7].mean()
df2_k_cc = df2_k_cc.drop([3,7])
print('各化学成分标准差：')
print(df2_k_cc.std().sort_values(ascending=False))
k_no_zero_count = (df2_k_cc.values != 0).sum(axis=0)
print('各化学成分排零均值：')
print((df2_k_cc.sum()/k_no_zero_count).sort_values(ascending=False))
df2_k_cc_cluster = df2_k_cc[['二氧化硅(SiO2)','氧化铝(Al2O3)','氧化钙(CaO)']]
print('最终选择的用于亚类划分的化学成分：')
print(df2_k_cc_cluster)
print('根据轮廓系数图像选择合适的n_clusters：')
X_k = df2_k_cc_cluster
for n_clusters in [2,3,4]:
    n_clusters = n_clusters
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(20, 3.5)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, X_k.shape[0] + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=2022).fit(X_k)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(X_k, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X_k, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i)/n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper)
                          ,ith_cluster_silhouette_values
                          ,facecolor=color
                          ,alpha=0.7
                         )
        ax1.text(-0.05
                 , y_lower + 0.5 * size_cluster_i
                 , str(i))
        y_lower = y_upper + 10

    # ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("各样本轮廓系数的取值",fontsize=14)
    ax1.set_ylabel("聚类簇的标签",fontsize=14)
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()
clusterer = KMeans(n_clusters=2, random_state=2022).fit(df2_k_cc_cluster)
pri_k_cluster_res = clusterer.labels_
k_cluster_res = clusterer.labels_
# 输出结果
k_cluster_res = list(k_cluster_res)
k_cluster_res.insert(1,k_cluster_res[1])
k_cluster_res.insert(5,k_cluster_res[5])
k_cluster_info_all = df2_valid[df2_valid.类型=='高钾']
k_cluster_info_all['亚类标号'] = k_cluster_res
k_cluster_info_all.to_excel('高钾亚类划分结果.xlsx',index=False)
# 分析结果
sub_cl0_mean = k_cluster_info_all[k_cluster_info_all.亚类标号 == 0].drop(
    ['文物编号', '亚类标号', '文物采样点', '颜色', '类型', '表面风化', '纹饰'], axis=1).mean()
sub_cl1_mean = k_cluster_info_all[k_cluster_info_all.亚类标号 == 1].drop(
    ['文物编号', '亚类标号', '文物采样点', '颜色', '类型', '表面风化', '纹饰'], axis=1).mean()
print('亚类化学成分均值对比：')
print(pd.DataFrame([sub_cl0_mean, sub_cl1_mean], columns=sub_cl0_mean.index))
print('灵敏度分析：')
ccs = ['二氧化硅(SiO2)','氧化铝(Al2O3)','氧化钙(CaO)']
pre_labels = pri_k_cluster_res
pre_data = np.array(df2_k_cc_cluster)
wav_data = pre_data[pre_labels==0]
record = np.zeros((20,5),dtype=object)
rd_idx = 0
for i in range(len(ccs)):
    for wav in [-0.01,-0.02,-0.05,-0.1,0.5,1,3,5,7,8,9,10]:
        if (i == 0) & (wav > 0):
            continue
        if (i != 0) & (wav < 0):
            continue
        new_data = np.array(pre_data)
        new_wave_data = np.array(wav_data)
        new_wave_data[:,i] = new_wave_data[:,i] * (1 + wav)
        new_data[pre_labels==0] = new_wave_data
        clusterer = KMeans(n_clusters=2, random_state=2022).fit(new_data)
        cl_res = clusterer.labels_
        if cl_res[0]==0:
            cl_res = np.where(cl_res==0,1,0)
        record[rd_idx,:] = [ccs[i],wav,pre_labels,cl_res,np.all(pre_labels==cl_res)]
        rd_idx += 1
print(pd.DataFrame(record,columns=['化学成分', '波动率', '波动前的聚类结果', '波动后的聚类结果', '聚类结果是否保持不变']))
print('*'*50)
print('铅钡亚类划分')
df2_pb_cc = df2_valid[df2_valid.类型 == '铅钡'].drop(['文物编号', '文物采样点', '纹饰', '类型', '颜色', '表面风化'], axis=1)
df2_pb_cc.iloc[13] = df2_pb_cc.iloc[13:15].mean()
df2_pb_cc.iloc[26] = df2_pb_cc.iloc[26:28].mean()
df2_pb_cc.iloc[28] = df2_pb_cc.iloc[28:30].mean()
df2_pb_cc.iloc[39] = df2_pb_cc.iloc[39:41].mean()
df2_pb_cc = df2_pb_cc.drop([32,45,47,58])
print('各化学成分标准差：')
print(df2_pb_cc.std().sort_values(ascending=False))
print('各化学成分排零均值：')
pb_no_zero_count = (df2_pb_cc.values != 0).sum(axis=0)
print((df2_pb_cc.sum()/pb_no_zero_count).sort_values(ascending=False))
print('最终选择的用于亚类划分的化学成分：')
df2_pb_cc_cluster = df2_pb_cc[['二氧化硅(SiO2)','五氧化二磷(P2O5)','二氧化硫(SO2)','氧化铝(Al2O3)','氧化铜(CuO)']]
print(df2_pb_cc_cluster)
print('根据轮廓系数图像选择合适的n_clusters：')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
X_pb = df2_pb_cc_cluster
for n_clusters in [2, 3, 4]:
    n_clusters = n_clusters
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(20,4.5)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, X_pb.shape[0] + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=2022).fit(X_pb)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(X_pb, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X_pb, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper)
                          , ith_cluster_silhouette_values
                          , facecolor=color
                          , alpha=0.7
                          )
        ax1.text(-0.05
                 , y_lower + 0.5 * size_cluster_i
                 , str(i))
        y_lower = y_upper + 10

    # ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("各样本轮廓系数的取值",fontsize=14)
    ax1.set_ylabel("聚类簇的标签",fontsize=14)
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()
clusterer = KMeans(n_clusters=2, random_state=2022).fit(X_pb)
pri_pb_clu_res = clusterer.labels_
pb_cluster_res = clusterer.labels_
pb_cluster_res = list(pb_cluster_res)
pb_cluster_res.insert(13,pb_cluster_res[13])
pb_cluster_res.insert(26,pb_cluster_res[26])
pb_cluster_res.insert(28,pb_cluster_res[28])
pb_cluster_res.insert(39,pb_cluster_res[39])
# 输出结果
pb_cluster_info_all = df2_valid[df2_valid.类型=='铅钡']
pb_cluster_info_all['亚类标号'] = pb_cluster_res
pb_cluster_info_all.to_excel('铅钡亚类划分结果.xlsx',index=False)
# 分析结果
sub_cl0_mean = pb_cluster_info_all[pb_cluster_info_all.亚类标号==0].drop(['文物编号','亚类标号','文物采样点','颜色','类型','表面风化','纹饰'],axis=1).mean()
sub_cl1_mean = pb_cluster_info_all[pb_cluster_info_all.亚类标号==1].drop(['文物编号','亚类标号','文物采样点','颜色','类型','表面风化','纹饰'],axis=1).mean()
print('亚类化学成分均值对比：')
print(pd.DataFrame([sub_cl0_mean,sub_cl1_mean],columns=sub_cl0_mean.index))
print('灵敏度分析：')
ccs = ['二氧化硅(SiO2)','五氧化二磷(P2O5)','氧化铝(Al2O3)']
pre_labels = pri_pb_clu_res
pre_data = np.array(df2_pb_cc_cluster)
wav_data = pre_data[pre_labels==0]
record = np.zeros((100,5),dtype=object)
rd_idx = 0
wav_range = [
    [-0.01,-0.02,-0.05,-0.1,-0.15,-0.2,-0.3],
    [1,3,5,7,8,9,10,16,20],
    [-0.1,-0.3,-0.5,-0.7,-0.8,-0.9],
]
for i in range(3):
    for wav in wav_range[i]:
        new_data = np.array(pre_data)
        new_wave_data = np.array(wav_data)
        new_wave_data[:,i] = new_wave_data[:,i] * (1 + wav)
        new_data[pre_labels==0] = new_wave_data
        clusterer = KMeans(n_clusters=2, random_state=2022).fit(new_data)
        cl_res = clusterer.labels_
        if cl_res[0]==0:
            cl_res = np.where(cl_res==0,1,0)
        record[rd_idx,:] = [ccs[i],wav,pre_labels,cl_res,np.all(pre_labels==cl_res)]
        rd_idx += 1
print(pd.DataFrame(record,columns=['化学成分', '波动率', '波动前的聚类结果', '波动后的聚类结果', '聚类结果是否保持不变']))
print('*'*50)
