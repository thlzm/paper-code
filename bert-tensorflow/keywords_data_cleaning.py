import json
import re
import jieba
import pandas as pd
from tqdm import tqdm

def remove_punctuation(_line):
    """只保留str中的汉字，数字，以及英文字符"""
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    _line = rule.sub('', _line)
    return _line


# 读入stopwords表
with open(r'D:\桌面\research\new2016zh\stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = [x.strip() for x in f]

train_data = []
with open(r'D:\桌面\research\new2016zh\news2016zh_valid.json', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        line_dt = json.loads(line)
        keywords = line_dt['keywords']
        content = line_dt['content']
        # keywords中部分keywords是句子，应该予以排除
        # 排除依据：用','或'，'分割后，并不存在与content之中
        keywords_lst = re.split(r'[, ，]', keywords)
        if sum([keyword in content for keyword in keywords_lst]) == 0:
            continue
        # content进行分词处理，并去除停用词stopwords
        pure_content = remove_punctuation(content)
        content_lst = jieba.lcut(pure_content)
        content_lst = [x for x in content_lst if x not in stopwords]

        # 句子与每个词都生成一个分类样本

        for word in content_lst:
            if word in keywords_lst:
                train_data.append([1, content, word])
            else:
                train_data.append([0, content, word])

df_train = pd.DataFrame.from_records(train_data)
df_train.to_pickle(r'D:\桌面\research\new2016zh\news2016zh_valid.pkl')
print(df_train.iloc[:, 2].sum())
