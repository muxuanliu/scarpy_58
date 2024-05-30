from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

# 加载模型
model = load_model('text_classification_model.keras')

# 加载Tokenizer的word_index
with open('tokenizer_word_index.json', 'r') as f:
    word_index = json.load(f)

# 加载Tokenizer（假设你已经保存了Tokenizer的状态）
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.word_index = word_index  # 这里需要加载之前训练模型时使用的word_index

# 待预测的文本数据
new_texts = ["健身教练 岗位职责：1、负责门店顾客接待；2、为来店客户提供专业的美容/纤体咨询服务；3、达成销售意向，成功*所经营产品；4、维护好老客户。任职资格：1、普通话标准，形象气质佳；2、美容行业工作经验1年以上，具备团队合作精神；3、爱岗敬业，工作认真，具有较强的沟通能力和亲和力。工作时间：",
             "5K包吃住聘护理员 工作内容：服务半自理、卧床，失能失智老人生活起居及其他护理工作要求：1.有耐心可以照顾老人2.有过经验，能够独立完成护理工作，或者学习能力较强者条件可适当放宽3.供吃供住工作长期稳定，工资待遇4000-10000。年龄：57周岁以内，**不限"]


# 文本预处理
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=100)  # 假设maxlen=100

# 使用模型进行预测
predicted_probabilities = model.predict(new_padded_sequences)
predicted_classes = np.argmax(predicted_probabilities, axis=1)

# 加载JSON文件中的映射字典
with open('label_index_json.json', 'r') as f:
    index_to_label = json.load(f)

# 将字符串类型的键转换为整数类型
index_to_label = {int(k): v for k, v in index_to_label.items()}

# 解释输出
# 假设你有一个将标签从数值转换回文本的字典
predicted_labels = [index_to_label[class_idx] for class_idx in predicted_classes]

print(predicted_labels)