import os
import json
import pandas as pd
import re
from janome.tokenizer import Tokenizer
from collections import Counter
import itertools
import networkx as nx
import matplotlib.pyplot as plt

# Creation of an utterance data set that is not a break
file_path = "../projectnextnlp-chat-dialogue-corpus/json/init100/"
file_dir = os.listdir(file_path)

label_text = []
for file in file_dir[:100]:
    r = open(file_path + file, 'r', encoding='utf-8')
    json_data = json.load(r)
    for turn in json_data['turns']:
        turn_index = turn['turn-index']
        speaker = turn['speaker']
        utterance = turn['utterance']
        if turn_index != 0:
            if speaker == 'U':
                u_text = ''
                u_text = utterance
            else:
                a = ''
                for annotate in turn['annotations']:
                    a = annotate['breakdown']
                    tmp1 = str(a) + '\t' + u_text
                    tmp2 = tmp1.split('\t')
                    label_text.append(tmp2)

df_label_text = pd.DataFrame(label_text)
df_label_text = df_label_text.drop_duplicates()
df_label_text_O = df_label_text[df_label_text[0] == 'O']

# Separate writing and remove unwanted strings in regular expressions
t = Tokenizer()
wakatiO = []
tmp1 = []
tmp2 = ''
for row in df_label_text_O.values.tolist():
    reg_row = re.sub('[0-9a-zA-Z]+', '', row[1])
    reg_row = reg_row.replace('\n', '')
    tmp1 = t.tokenize(reg_row, wakati=True)
    wakatiO.append(tmp1)
    tmp1 = []

# Add to dic, counting and sorting the number of occurrences of the word
word_freq = Counter(itertools.chain(*wakatiO))
dic = []
for word_uniq in word_freq.most_common():
    dic.append(word_uniq[0])

# Assign ID to words and create a dictionary
dic_inv = {}
for i, word_uniq in enumerate(dic, start=1):
    dic_inv.update({word_uniq: i})

# Convert the word to ID
wakatiO_n = [[dic_inv[word] for word in waka] for waka in wakatiO]

# Create a 2-gram list
tmp = []
bigramO = []
for i in range(0, len(wakatiO_n)):
    # Creation of 2-grams
    row = wakatiO_n[i]
    for j in range(len(row)-1):
        tmp.append([row[j], row[j+1]])
    bigramO.extend(tmp)
    tmp = []

# Convert the array `bigramO` to DataFrame and set the column
df_bigramO = pd.DataFrame(bigramO)
df_bigramO = df_bigramO.rename(columns={0: 'node1', 1: 'node2'})

# Add the `weight` column and unify the values at 1
df_bigramO['weight'] = 1

# 2-grams counting
df_bigramO = df_bigramO.groupby(['node1', 'node2'], as_index=False).sum()

# Extract the list of occurrences greater than 1
df_bigramO = df_bigramO[df_bigramO['weight'] > 1]

# Creation of a directed graph
G_bigramO = nx.from_pandas_edgelist(df_bigramO, 'node1', 'node2', ['weight'], nx.DiGraph)

# speech network not broken
# Degrees of Intake
indegree = sorted([d for n, d in G_bigramO.in_degree(weight='weight')], reverse=True)
indegreeCount = Counter(indegree)
indeg, cnt = zip(*indegreeCount.items())

# Degrees of Outtake
outdegree = sorted([d for n, d in G_bigramO.out_degree(weight='weight')], reverse=True)
outdegreeCount = Counter(outdegree)
outdeg, cnt = zip(*outdegreeCount.items())

# Creating the order distribution
plt.subplot(1, 2, 1)
plt.bar(indeg, cnt, color='r')
plt.title('in_degree')
plt.subplot(1, 2, 2)
plt.bar(outdeg, cnt, color='b')
plt.title('out_degree')
plt.show()
