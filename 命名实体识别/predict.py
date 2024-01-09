from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.preprocessing import sequence
import numpy as np

model = models.load_model("model", compile=False)

trans_params = model.get_layer('crf').get_weights()[0]
# print(trans_params)

trans_params = model.get_layer('crf').get_weights()[0]
# 获得BiLSTM的输出logits
sub_model = models.Model(inputs=model.get_layer('input_ids').input,
                        outputs=model.get_layer('dense').output)

def predict(model, inputs, input_lens):
    logits = sub_model.predict(inputs)
    # 获取CRF层的转移矩阵
    # crf_decode：viterbi解码获得结果
    pred_seq, viterbi_score = tfa.text.crf_decode(logits, trans_params, input_lens)
    return pred_seq

maxlen = 100
char_vocab_path = "./data/char_vocabs.txt" # 字典文件
train_data_path = "./data/train_data" # 训练数据
test_data_path = "./data/test_data" # 测试数据

special_words = ['<PAD>', '<UNK>'] # 特殊词表示

# "BIO"标记的标签
label2idx = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }
# 索引和BIO标签对应
idx2label = {idx: label for label, idx in label2idx.items()}

# 读取字符词典文件
with open(char_vocab_path, "r", encoding="utf8") as fo:
    char_vocabs = [line.strip() for line in fo]
char_vocabs = special_words + char_vocabs
# 字符和索引编号对应
idx2vocab = {idx: char for idx, char in enumerate(char_vocabs)}
vocab2idx = {char: idx for idx, char in idx2vocab.items()}

sentence = "陈伟坐飞机从夏威夷前往中国北京，到北京后路过了中国社会科学院"
sent_chars = list(sentence)
sent2id = [vocab2idx[word] if word in vocab2idx else vocab2idx['<UNK>'] for word in sent_chars]
sent2id_new = np.array([[0] * (maxlen-len(sent2id)) + sent2id[:maxlen]])
test_lens = np.array([100])

pred_seq = predict(model, sent2id_new, test_lens)

y_label = pred_seq.numpy().reshape(1, -1)[0]

y_ner = [idx2label[i] for i in y_label][-len(sent_chars):]

# 对预测结果进行命名实体解析和提取
def get_valid_nertag(input_data, result_tags):
    result_words = []
    start, end =0, 1 # 实体开始结束位置标识
    tag_label = "O" # 实体类型标识
    for i, tag in enumerate(result_tags):
        if tag.startswith("B"):
            if tag_label != "O": # 当前实体tag之前有其他实体
                result_words.append((input_data[start: end], tag_label)) # 获取实体
            tag_label = tag.split("-")[1] # 获取当前实体类型
            start, end = i, i+1 # 开始和结束位置变更
        elif tag.startswith("I"):
            temp_label = tag.split("-")[1]
            if temp_label == tag_label: # 当前实体tag是之前实体的一部分
                end += 1 # 结束位置end扩展
        elif tag == "O":
            if tag_label != "O": # 当前位置非实体 但是之前有实体
                result_words.append((input_data[start: end], tag_label)) # 获取实体
                tag_label = "O"  # 实体类型置"O"
            start, end = i, i+1 # 开始和结束位置变更
    if tag_label != "O": # 最后结尾还有实体
        result_words.append((input_data[start: end], tag_label)) # 获取结尾的实体
    return result_words

result_words = get_valid_nertag(sent_chars, y_ner)
for (word, tag) in result_words:
    print("".join(word), tag)