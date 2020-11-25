import json
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
nltk.download('punkt')

with open("./word2id.json", "r") as f:
    word2id = json.load(f)

with open("../tqa_train_val_test/train/tqa_v1_train.json", "r") as f:
    train = json.load(f)

def preprocess(s):
    s = s.lower()
    s = word_tokenize(s)
    return s

train_data=[]

for Train in train:
    Target=Train["questions"]["nonDiagramQuestions"]
    for num in Target.keys():
        train_d={}
        answer_choice={}
        for answer,text in Target[num]["answerChoices"].items():
            target_a= preprocess(text["processedText"])
            answer_choice[answer] = [word2id.get(w, 0) for w in target_a]
        train_d["answerChoices"]=answer_choice

        target_ca = Target[num]["correctAnswer"]["processedText"]
        train_d["correctAnswer"]=target_ca

        target_q = preprocess(Target[num]["beingAsked"]["processedText"])
        train_d["question"] = [word2id.get(w, 0) for w in target_q]
        train_data.append(train_d)

print(train_data)
with open('./preprocessed_train.json', 'w') as f:
    json.dump(train_data, f, ensure_ascii=False)
