import pandas as pd
import numpy as np
df1_valid = pd.read_excel('表1-填补缺失值.xlsx')
df2_valid = pd.read_excel('文物样品信息汇总-填补缺失值.xlsx')
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
rel_y = df1_valid.表面风化
rel_cls = LabelEncoder().fit(df1_valid.类型).transform(df1_valid.类型)
rel_wen = LabelEncoder().fit(df1_valid.纹饰).transform(df1_valid.纹饰)
rel_col = LabelEncoder().fit(df1_valid.颜色).transform(df1_valid.颜色)
rel_X = np.array([rel_cls,rel_wen,rel_col]).T
chi2s, chi2_p_values = chi2(rel_X, rel_y)
print('类型、纹饰、颜色的卡方检验p值：')
print(chi2_p_values)
# 高钾类风化前后化学成分统计和预测
df2_sta = df2_valid.drop(['文物编号', '纹饰', '颜色'],axis=1)
chemical_component_K = df2_sta[(df2_sta.类型 == '高钾')].drop(['类型'],axis=1)
cck_whe_mean = chemical_component_K[chemical_component_K.表面风化 == '风化'].drop(['文物采样点','表面风化'],axis=1).mean()
cck_no_whe_mean = chemical_component_K[chemical_component_K.表面风化 == '无风化'].drop(['文物采样点','表面风化'],axis=1).mean()
cck_whe_ratio = (cck_whe_mean - cck_no_whe_mean)/cck_no_whe_mean
pd.DataFrame([cck_no_whe_mean,cck_whe_mean,cck_whe_ratio],index=['无风化化学成分含量占比','风化化学成分含量占比','风化后化学成分含量变化率']).to_excel('高钾玻璃风化化学成分含量的变化.xlsx')
cck_whe = chemical_component_K[chemical_component_K.表面风化 == '风化'].drop(['文物采样点','表面风化'],axis=1)
cck_ratio_idx = 0
cck_whe = chemical_component_K[chemical_component_K.表面风化 == '风化'].drop(['文物采样点','表面风化'],axis=1)
cck_whe_pred = np.zeros(cck_whe.shape)
for cck in cck_whe_mean:
    cck_ratio = cck_whe_ratio[cck_ratio_idx]
    cck_whe_each = cck_whe.iloc[:,cck_ratio_idx]
    if cck_ratio == -1:
        # 风化后化学成分直接消失了的，按均值预测
        cck_whe_pred[:,cck_ratio_idx] = cck_no_whe_mean[cck_ratio_idx]
    else:
        # 风化后化学成分直接消失了的，按均值预测，没有消失的按比例预测
        tmp = cck_whe.iloc[:,cck_ratio_idx]
        cck_whe_pred[:,cck_ratio_idx] = np.where(tmp==0,cck_no_whe_mean[cck_ratio_idx],tmp/(1+cck_ratio))
    cck_ratio_idx += 1
cck_pre_res = cck_whe_pred / cck_whe_pred.sum(axis=1).reshape(-1,1)
# 生成预测结果
k_number = df2_valid.loc[cck_whe.index]['文物编号']
k_pre_df = pd.DataFrame(cck_pre_res,index=k_number,columns=cck_whe.columns)
k_pre_df.to_excel('高钾风化点预测结果.xlsx')
# 检验预测结果
print('高钾类预测风化前的均值')
print(k_pre_df.mean())
print('高钾类无风化样本的均值')
print(cck_no_whe_mean)
# 高钾类风化前后化学成分统计和预测
chemical_component_PB = df2_sta[(df2_sta.类型 == '铅钡')].drop(['类型'], axis=1)
ccpb_whe = chemical_component_PB[(chemical_component_PB.表面风化 == '风化')]
ccpb_whe = ccpb_whe.drop([23,25,29,30,44,45,48,53,56,60])
ccpb_whe_mean = ccpb_whe.drop(['文物采样点', '表面风化'], axis=1).mean()
ccpb_no_whe = chemical_component_PB[(chemical_component_PB.表面风化 == '无风化')]
ccpb_no_whe = pd.concat([ccpb_no_whe,chemical_component_PB.loc[[23,25,29,30,44,45,48,53,56,60]]]).sort_index()
ccpb_no_whe_mean = ccpb_no_whe.drop(['文物采样点', '表面风化'], axis=1).mean()
ccpb_whe_ratio = (ccpb_whe_mean - ccpb_no_whe_mean) / ccpb_no_whe_mean
pd.DataFrame([ccpb_no_whe_mean,ccpb_whe_mean,ccpb_whe_ratio],index=['无风化化学成分含量占比','风化化学成分含量占比','风化后化学成分含量变化率']).to_excel('铅钡玻璃风化化学成分含量的变化.xlsx')
ccpb_whe_to_pre = ccpb_whe.drop(['文物采样点', '表面风化'], axis=1)
ccpb_ratio_idx = 0
ccpb_whe_pred = np.zeros(ccpb_whe_to_pre.shape)
for ccpb in ccpb_whe_mean:
    ccpb_ratio = ccpb_whe_ratio[ccpb_ratio_idx]
    tmp = ccpb_whe_to_pre.iloc[:, ccpb_ratio_idx]
    # 风化后化学成分直接消失了的，按均值预测，没有消失的按比例预测
    ccpb_whe_pred[:, ccpb_ratio_idx] = np.where(tmp == 0, ccpb_no_whe_mean[ccpb_ratio_idx], tmp / (1 + ccpb_ratio))
    ccpb_ratio_idx += 1
ccpb_whe_res = ccpb_whe_pred / ccpb_whe_pred.sum(axis=1).reshape(-1,1)
# 生成预测结果
pb_number = df2_valid.loc[ccpb_whe.index]['文物编号']
pb_pre_df = pd.DataFrame(ccpb_whe_res,index=pb_number,columns=cck_whe.columns)
pb_pre_df.to_excel('铅钡风化点预测结果.xlsx')
# 检验预测结果
print('铅钡类预测风化前的均值')
print(pb_pre_df.mean())
print('铅钡类无风化样本的均值')
print(ccpb_no_whe_mean)
