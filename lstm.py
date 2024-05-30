import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
# 检查文本编码格式
import chardet
# 中文文本分词库
import jieba as jb
plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



# df = pd.read_csv('employ_info.csv',encoding='GB2312')

# # 检查是否有缺失值
# # print(df.isnull().sum())
#
# 生成柱状图查看每种类别数据的个数-----------------------------------------------------------------
# print('数据总量：%d .' %len(df))
# # 创建字典d，分别有两个键，一个是label一个是count
# d = {'label': df['label'].value_counts().index,'count':df['label'].value_counts()}
# # 创建一个dataFrame data传入一个字典
# df_label = pd.DataFrame(data=d).reset_index(drop=True)
# df_label.plot(x='label',y='count',kind = 'bar',legend=False,figsize=(8,5))
# plt.title('类目分布')
# plt.ylabel('数目',fontsize=18)
# plt.xlabel('类目',fontsize=18)
# plt.show()
# ----------------------------------------------------------------------------------------------
# # 定义删除除字母，数字，汉字以外的所有符号的函数
# def remove_punctuation(line):
#     line = str(line)
#     # 如果使用strip()方法移除字符串两端的空白字符后字符串为空，则返回空
#     if line.strip() == '':
#         return ''
#     # 编译一个正则表达式，用于匹配英文字母、数字和中文字符
#     # regex是Python标准库中用于正则表达式操作的模块,要import re
#     rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
#     # 按照rule1规则移除line中所有不匹配正则表达式的字符
#     line = rule.sub('',line)
#     return line
#
# def stopwordslist(filepath):
#     stopwords = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
#     return stopwords
#
# # 加载停用词
# stopwords = stopwordslist("D:\Pycharm\data_58\ChineseStopWords.txt")
#
#
# # 删除除字母，数字，汉字以外的所有符号
# df['clean_data'] = df['data'].apply(remove_punctuation)
# print(df['clean_data'])
#
# # lambda 表达式创建一个匿名函数，接收clean_data中的每个元素作为输入， join字符串连接操作，[w...]是一个列表推导式，用于生成一个词语列表
# # 列表推导式：[expression for item in iterable if condition]
# # expression  对每个元素执行的表达式或操作
# # item        来自iterable的每个元素的变量名
# # iterable    可迭代的对象
# # condition   用于过滤元素
# # 去除停用词
# df['cut_review'] = df['clean_data'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
# # print(df['cut_review'])


# 读取CSV文件
data = pd.read_csv('employ_info.csv',encoding='gb2312')

# 假设CSV文件中有一个名为'data'的列，包含文本数据，和一个名为'label'的列，包含分类标签
texts = data['data'].astype(str).values
labels = data['label'].astype(str).values

# 文本数据预处理和分词
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 填充序列以确保它们具有相同的长度
maxlen = 100
X = pad_sequences(sequences, maxlen=maxlen)
# 使用LabelEncoder转换标签
label_encoder = LabelEncoder()
label_indices = label_encoder.fit_transform(labels)
y = pd.get_dummies(label_indices).values  # 将标签转换为独热编码

print(y)

# 创建从标签索引到原始标签的映射字典
index_to_label = {int(index): label for index, label in enumerate(label_encoder.classes_)}

# 将映射字典保存为JSON文件
with open('label_index_json.json', 'w') as f:
    # 在序列化之前，将字典的键转换为字符串
    json.dump({str(k): v for k, v in index_to_label.items()}, f)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 构建LSTM模型
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=maxlen))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))  # 输出层的神经元数量应与类别数量相匹配

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
epochs = 30
batch_size = 32
# 训练模型时记录损失和准确率
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1  # 显示训练过程的进度
)

# 训练模型后，保存Tokenizer的word_index
with open('tokenizer_word_index.json', 'w') as f:
    json.dump(tokenizer.word_index, f)

# 绘制训练和验证的损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和验证的准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 保存图表为图片
plt.savefig('loss_and_accuracy.png', dpi=300, bbox_inches='tight')
# 显示图表
plt.tight_layout()
plt.show()

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# 保存模型
model.save('text_classification_model.keras')


