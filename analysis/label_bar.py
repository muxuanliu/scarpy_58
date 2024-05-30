import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']


df = pd.read_csv('../employ_info.csv',encoding='GB2312')

# 检查是否有缺失值
print(df.isnull().sum())

# 生成柱状图查看每种类别数据的个数-----------------------------------------------------------------
print('数据总量：%d .' %len(df))
# 创建字典d，分别有两个键，一个是label一个是count
d = {'label': df['label'].value_counts().index,'count':df['label'].value_counts()}
# 创建一个dataFrame data传入一个字典
df_label = pd.DataFrame(data=d).reset_index(drop=True)
df_label.plot(x='label',y='count',kind = 'bar',legend=False,figsize=(8,5))
plt.title('类目分布')
plt.ylabel('数目',fontsize=18)
plt.xlabel('类目',fontsize=18)
plt.savefig('label_bar.png', dpi=300, bbox_inches='tight')
plt.show()