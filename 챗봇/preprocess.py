# 📖 출처: 텐서플로 2와 머신러닝으로 시작하는 자연어 처리
# 데이터를 불러오고 가공하는 파일

import os
import re
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from konlpy.tag import Okt

FILTERS = "([~.,!?\"':;)(])"
# 어떤 의미도 없는 패딩 토큰
PAD = "<PAD>"
# 시작 토큰
STD = "<SOS>"
# 종료 토큰
END = "<END>"
# 사전에 없는 단어
UNK = "<UNK>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQUENCE = 25

# 데이터를 불러오는 함수
def load_data(path):
    data_df = pd.read_csv(path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])

    return question, answer

# 데이터 전처리 한 후 단어 리스트로 만드는 함수
def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)
    return [word for word in words if word]

# 형태소로 분리하는 함수
def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)

    return result_data

# 단어 사전 만드는 함수
def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    vocabulary_list = []
    if not os.path.exists(vocab_path):
        if (os.path.exists(path)):
            data_df = pd.read_csv(path, encoding='utf-8')
            question, answer = list(data_df['Q']), list(data_df['A'])

            if tokenize_as_morph:  # 형태소에 따른 토크나이져 처리
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)

            data = []
            data.extend(question) # extend() : 리스트 끝에 가장 바깥쪽 iterable의 모든 항목을 넣음
            data.extend(answer)
            words = data_tokenizer(data)
            words = list(set(words)) # 공통 단어 제거
            words[:0] = MARKER # words 리스트 처음에 MARKER 데이터 넣음

        # 사전에 데이터 사전 존재x
        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')

    # 사전에 데이터 사전 존재
    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    char2idx, idx2char = make_vocabulary(vocabulary_list)
    return char2idx, idx2char, len(char2idx)

def make_vocabulary(vocabulary_list):
    # 키가 단어이고 값이 인덱스인 딕셔너리
    char2idx = {char: idx for idx, char in enumerate(vocabulary_list)}
    # 키가 인덱스이고 값이 단어인 딕셔너리
    idx2char = {idx: char for idx, char in enumerate(vocabulary_list)}
    return char2idx, idx2char

# 인코더 전처리 함수
def enc_processing(value, dictionary, tokenize_as_morph=False):
    sequences_input_index = []
    sequences_length = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        # 1. 특수문자 모두 제거
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []

        # 2. 잘려진 단어를 단어 사전을 이용해 단어 인덱스로 바꿈
        for word in sequence.split():
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            # 잘려진 단어가 딕셔너리에 존재 하지 않는 경우 UNK(2)를 넣어 준다.
            else:
                sequence_index.extend([dictionary[UNK]])

        # 3. 문장 제한 길이보다 길어질 경우 뒤를 자름
        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]

        # 하나의 문장에 대한 길이
        sequences_length.append(len(sequence_index))
        # max_sequence_length보다 문장 길이가 작을 경우 빈 부분에 PAD(0)를 넣어줌
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을 sequences_input_index에 넣어줌
        sequences_input_index.append(sequence_index)

    # 인덱스화된 일반 배열을 넘파이 배열로 변경함 : 텐서플로우 dataset에 넣어 주기 위해
    return np.asarray(sequences_input_index), sequences_length

# 디코더 전처리 함수
def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    sequences_output_index = []
    sequences_length = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # 디코딩 입력의 처음에는 START가 와야함
        sequence_index = [dictionary[STD]] + [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        # 문장 제한 길이보다 길어질 경우 뒤를 자름
        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]

        # 하나의 문장에 대한 길이
        sequences_length.append(len(sequence_index))
        # max_sequence_length보다 문장 길이가 작을 경우 빈 부분에 PAD(0)를 넣어줌
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을 sequences_input_index에 넣어줌
        sequences_output_index.append(sequence_index)
    # 인덱스화된 일반 배열을 넘파이 배열로 변경함 : 텐서플로우 dataset에 넣어 주기 위해
    return np.asarray(sequences_output_index), sequences_length

# 디코더의 타깃값을 만드는 전처리 함수
# 디코더의 입력값을 만드는 함수와의 차이점은 문장이 시작하는 부분에 시작 토큰을 넣지 않고 마지막에 종료 토큰을 넣는다는 점
def dec_target_processing(value, dictionary, tokenize_as_morph=False):
    sequences_target_index = []
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]

        if len(sequence_index) >= MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE - 1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]

        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_target_index.append(sequence_index)

    return np.asarray(sequences_target_index)