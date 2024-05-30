import pandas as pd
import chardet
# 尝试读取csv文件发现编码格式不匹配，使用chardet获取文件编码格式
# data = pd.read_csv("employ_info.csv",encoding='utf-8')
# print(data)

# 用with打开文件的好处是可以确保文件在操作完成后会被正确关闭
# with open('employ_info.csv','rb') as f:
#     result = chardet.detect(f.read())
#     encoding = result['encoding']
# print(encoding)
# 获知编码格式为GB2312

df = pd.read_csv('../employ_info.csv', encoding='GB2312')
# 先查看label列有几种类别，每种类别都是什么
unique_category = df['label'].unique()
print("Unique values in 'label' column:")
for label in unique_category:
    print(label)
    # print(df['label'].dtype)


# 替换词典
replace_dict = {'yiliaojk/':'医疗健康','shengchanzhz/':'生产制造','fuwuy/':'服务业',
                'renshixzhcw/':'人事行政财务','yunyingkf/':'运营客服','anbaoxf/':'安保消防',
                'mendianlsh/':'门店零售','ncanyin/':'餐饮','nxiaoshou/':'销售','sijiwl/':'司机物流'
                ,'nchuanmei':'传媒'}

# 将label中的拼音替换为对应的汉字
df['label'] = df['label'].replace(replace_dict)
# 将修改后的数据写回csv
df.to_csv('employ_info.csv',index = False,encoding = 'GB2312')
# print(df)

# 合并job_name和description列为一列，即训练数据
df['data'] = df['job_name']+ ' ' +df['description']
# print(df.head())
df.to_csv('employ_info.csv',index = False,encoding='GB2312')
# 通过打印可以观察到已经将两列合并为一列data
print(df)


# 需要将原来的job_name和description列删除
df = df.drop(columns = ['job_name','description'])
df.to_csv('employ_info.csv',index = False,encoding='GB2312')
# 打印后观察到job_name和description列已经删除
print(df.columns)
