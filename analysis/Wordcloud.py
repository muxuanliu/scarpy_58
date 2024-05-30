# 词云图
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# 检查文本编码格式
import chardet
plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_csv('../employ_info.csv', encoding='GB2312')

# 按类别分组并准备文本数据
def generate_text_by_category(category_data):
    category_text = " ".join(category_data['data'])
    return category_text

# 创建一个空的DataFrame来保存每个类别的文本
category_texts = pd.DataFrame()

# 按类别迭代并准备文本
for category,group in df.groupby('label'):
    # 迭代中category为label的值，group是一个dataframe，包含了所有label值为categrouy的行
    category_text = generate_text_by_category(group)
    category_texts = category_texts.append({'category':category,'text':category_text},ignore_index=True)

font_path = 'C:\\Windows\\Fonts\\SimHei.ttf'
# 为每个类别生成词云图
# .iterrows()是DataFrame的一个方法，它以索引和行数据为单位迭代DataFrame。使用iterrows()时，每次迭代会返回两个对象，index和row
for index,row in category_texts.iterrows():
    text = row['text']
    category = row['category']
    # # 查看text的编码格式
    # detected = chardet.detect(text.encode())
    # print(detected)

    # 生成词云图
    wordcloud = WordCloud(width=800,height=400,background_color='white',font_path=font_path).generate(text)

    # 使用matplotlib显示词云图
    # 定义图形大小
    plt.figure(figsize = (10,5))
    # interpolation是一个参数，指定图像插值方法，双线性插值(bilinear)是一种平滑插值方法，可以使图像在放大时边缘更平滑
    plt.imshow(wordcloud,interpolation='bilinear')
    # 关闭坐标轴显示
    plt.axis("off")
    plt.title(f'Category:{category}')
    # 保存词云图到文件
    plt.savefig(f'wordcloud_{category}.png', dpi=300, bbox_inches='tight')
    plt.show()
