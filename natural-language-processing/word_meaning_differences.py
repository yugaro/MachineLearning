import matplotlib.pyplot as plt
import json
import numpy as np
from keras.layers import Input, Dense, Dropout, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import dot, concatenate
from keras.layers.core import Activation
from keras.layers.pooling import AveragePooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import mpl_toolkits.axes_grid1

batch_size = 32 # Batch size
embedding_dim = 100 # Dimensions of the word vector
seq_length1 = 20 # Length of the question
seq_length2 = 10 # Length of response
lstm_units = 200 # The number of dimensions of the hidden state vector of LSTM
hidden_dim = 200 # The number of dimensions of the final output vector

with open("../data/preprocessed_val.json", "r") as f:
    val = json.load(f)

questions = []
answers = []
outputs = []
for t in val:
    for i, ans in t["answerChoices"].items():
        if i == t["correctAnswer"]:
            outputs.append([1, 0])
        else:
            outputs.append([0, 1])
        questions.append(t["question"])
        answers.append(ans)

questions = pad_sequences(questions, maxlen=seq_length1,dtype=np.int32, padding='post', truncating='post', value=0)
answers = pad_sequences(answers, maxlen=seq_length2,dtype=np.int32, padding='post', truncating='post', value=0)

with open("../data/word2id.json", "r") as f:
    word2id = json.load(f)

vocab_size = len(word2id) # The number of vocabulary to handle
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

input1 = Input(shape=(seq_length1,))
embed1 = embedding(input1)
bilstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed1)
h1 = Dropout(0.2)(bilstm1)

input2 = Input(shape=(seq_length2,))
embed2 = embedding(input2)
bilstm2 = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed2)
h2 = Dropout(0.2)(bilstm2)

# Calculating the product of each element
product = dot([h2, h1], axes=2)
a = Activation('softmax')(product)

c = dot([a, h1], axes=[2, 1])
c_h2 = concatenate([c, h2], axis=2)
h = Dense(hidden_dim, activation='tanh')(c_h2)

mean_pooled_1 = AveragePooling1D(pool_size=seq_length1, strides=1, padding='valid')(h1)
mean_pooled_2 = AveragePooling1D(pool_size=seq_length2, strides=1, padding='valid')(h)
con = concatenate([mean_pooled_1, mean_pooled_2], axis=-1)
con = Reshape((lstm_units * 2 + hidden_dim,))(con)
output = Dense(2, activation='softmax')(con)

prob_model = Model(inputs=[input1, input2], outputs=[a, output])

prob_model.load_weights("../data/model.hdf5")

question = np.array([[2945, 1752, 2993, 1099, 122, 2717, 100,200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]])
answer = np.array([[2841, 830, 2433, 500, 1500, 1600, 1700, 1800, 1900, 2000]])

att, pred = prob_model.predict([question, answer])

id2word = {v: k for k, v in word2id.items()}

q_words = [id2word[w] for w in question[0]]
a_words = [id2word[w] for w in answer[0]]

f = plt.figure(figsize=(8, 8.5))
ax = f.add_subplot(1, 1, 1)

# add image
i = ax.imshow(att[0], interpolation='nearest', cmap='gray')

# add labels
ax.set_yticks(range(att.shape[1]))
ax.set_yticklabels(a_words)
ax.set_xticks(range(att.shape[2]))
ax.set_xticklabels(q_words, rotation=45)
ax.set_xlabel('Question')
ax.set_ylabel('Answer')

# add colorbar
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes('right', '5%', pad='3%')
plt.colorbar(i, cax=cax)
plt.show()
