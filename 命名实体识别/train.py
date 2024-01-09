from tensorflow.keras.preprocessing import sequence
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
print(tf.__version__)
print(tfa.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[0]
tf.config.set_visible_devices(gpu, 'GPU')

EPOCHS = 20
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_SIZE = 64
MAX_LEN = 100

# 读取训练语料
def read_corpus(corpus_path, vocab2idx, label2idx):
    datas, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            char, label = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            sent_ids = [vocab2idx[char] if char in vocab2idx else vocab2idx['<UNK>'] for char in sent_]
            tag_ids = [label2idx[label] if label in label2idx else 0 for label in tag_]
            datas.append(sent_ids)
            labels.append(tag_ids)
            sent_, tag_ = [], []
    return datas, labels

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

# 加载训练集
train_datas, train_labels = read_corpus(train_data_path, vocab2idx, label2idx)
# 加载测试集
test_datas, test_labels = read_corpus(test_data_path, vocab2idx, label2idx)

train_datas = sequence.pad_sequences(train_datas, maxlen=MAX_LEN)
train_labels = sequence.pad_sequences(train_labels, maxlen=MAX_LEN)
train_seq_lens = np.array([MAX_LEN] * len(train_labels))
labels = np.ones(len(train_labels))
# train_labels = keras.utils.to_categorical(train_labels, CLASS_NUMS)

# print(np.shape(train_datas), np.shape(train_labels))

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

class CRF(layers.Layer):
    def __init__(self, label_size):
        super(CRF, self).__init__()
        self.trans_params = tf.Variable(
            tf.random.uniform(shape=(label_size, label_size)), name="transition")
    
    @tf.function
    def call(self, inputs, labels, seq_lens):
        log_likelihood, self.trans_params = tfa.text.crf_log_likelihood(
                                                inputs, labels, seq_lens,
                                                transition_params=self.trans_params)
        loss = tf.reduce_sum(-log_likelihood)
        return loss

K.clear_session()

VOCAB_SIZE = len(vocab2idx)
CLASS_NUMS = len(label2idx)

inputs = layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
targets = layers.Input(shape=(MAX_LEN,), name='target_ids', dtype='int32')
seq_lens = layers.Input(shape=(), name='input_lens', dtype='int32')

x = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=True)(inputs)
x = layers.Bidirectional(layers.LSTM(HIDDEN_SIZE, return_sequences=True))(x)
logits = layers.Dense(CLASS_NUMS)(x)
loss = CRF(label_size=CLASS_NUMS)(logits, targets, seq_lens)

model = models.Model(inputs=[inputs, targets, seq_lens], outputs=loss)

# print(model.summary())

class CustomLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        loss, pred = y_pred
        return loss

model.compile(loss=lambda y_true, y_pred: y_pred, optimizer='adam')

# 训练
model.fit(x=[train_datas, train_labels, train_seq_lens], y=labels, 
        validation_split=0.1, batch_size=BATCH_SIZE, epochs=EPOCHS)

# 保存
model.save("model")