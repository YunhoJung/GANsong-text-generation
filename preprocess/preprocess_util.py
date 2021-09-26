import os
import re
import pickle

from konlpy.tag import Twitter, Kkma
import collections
import numpy as np

from utils.save_pickle import save_pickle

np.random.seed(100)


def sentence2pos(train_text, tag):
    if tag == "kkma":
        analyzer = Kkma()
    elif tag == "twitter":
        analyzer = Twitter()

    sentences = list()
    for line in train_text:
        sentence = re.sub(r"[^ㄱ-힣]+", ' ', line)
        if sentence:
            sentence = analyzer.pos(sentence, norm=True)
            sentences.append(sentence)

    pos_counter = [['UNK', -1]]
    pos_counter.extend(collections.Counter([word[0] for words in sentences for word in words]).most_common())
    print(pos_counter)

    pos_list = list()
    for pos, _ in pos_counter:
        pos_list.append(pos)

    return sentences, pos_list


def load_sentence(path):
    sentences = list()
    with open(path, 'r') as fout:
        for line in fout:
            pos_line = list()
            for pos in line.split():
                pos_line.append(pos)
            sentences.append(pos_line)
    return sentences


def get_sentence_pos(data_path, input):
    print("Sentence to pos in progressing...")
    data, pos_list = sentence2pos(input, tag="twitter")

    save_pickle(os.path.join(data_path, '1_sg_real_data.pkl'), data)
    save_pickle(os.path.join(data_path, 'sg_pos_list.pkl'), pos_list)

    f = open(os.path.join(data_path, '1_sg_real_data.txt'), 'w')
    for token in data:
        for word in token:
            word = str(word) + ' '
            f.write(word)
        f.write('\n')
    f.close()


def get_preprocess_data(data_path, embedpath):
    print("Loading embedding vector...")
    i = 0
    with open(embedpath, 'r', encoding='utf-8') as fout:
        embed_pos_list = list()
        embedding_list = list()
        for line in fout:
            line = line.strip()
            if i == 0:
                embedding_size = int(line.split(" ")[1])
                i += 1
                continue
            vector_list = list()
            line_sp = line.split(" ")
            for j in range(len(line_sp)):
                if j == 0:
                    tmp = line_sp[j]
                elif j == 1:
                    embed_pos_list.append(tmp + " " + line_sp[j])
                else:
                    vector_list.append(line_sp[j])
            embedding_list.append(vector_list)

    pos2idx = dict()
    for pos in embed_pos_list:
        pos2idx[pos] = len(pos2idx)
    idx2pos = dict(zip(pos2idx.values(), pos2idx.keys()))
    print(pos2idx)
    print(idx2pos)

    embedding_vec = np.array(embedding_list, dtype=np.float32)
    print("before embed: ", np.shape(embedding_vec))

    pos_size = len(pos2idx)

    save_pickle(os.path.join(data_path, 'sg_pos2idx.pkl'), pos2idx)
    save_pickle(os.path.join(data_path, 'sg_idx2pos.pkl'), idx2pos)
    save_pickle(os.path.join(data_path, 'pretrain_embedding_vec.pkl'), embedding_vec)
    print("Save all data as pkl !!")

    return pos_size, embedding_size


def pkl_loading_test(data_path):
    # load sentences separated by pos (pkl)
    sents_pickle = open(os.path.join(data_path, '1_sg_real_data.pkl'), 'rb')
    sents = pickle.load(sents_pickle)

    # load pos_list (pkl)
    pos_list_pickle = open(os.path.join(data_path, 'sg_pos_list.pkl'), 'rb')
    pos_list = pickle.load(pos_list_pickle)

    # load pos2idx (pkl)
    pos2idx_pickle = open(os.path.join(data_path, 'sg_pos2idx.pkl'), 'rb')
    pos2idx = pickle.load(pos2idx_pickle)

    # load idx2pos (pkl)
    idx2pos_pickle = open(os.path.join(data_path, 'sg_idx2pos.pkl'), 'rb')
    idx2pos = pickle.load(idx2pos_pickle)

    # load embedding_vec (pkl)
    embedding_vec_pickle = open(os.path.join(data_path, 'pretrain_embedding_vec.pkl'), 'rb')
    embedding_vec = pickle.load(embedding_vec_pickle)

    print(sents)
    print(len(pos_list))
    print(pos2idx)
    print(idx2pos)
    print(np.shape(embedding_vec))


if __name__ == "__main__":

    data_path = "../model/data/"
    embed_path = os.path.join(data_path, "pos_vec.txt")

    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    song_desc = []
    for i in range(1, 8):
        data = open(os.path.join(data_path, 'sg_{}.pickle'.format(i)), 'rb')
        song_data = pickle.load(data)
        song_desc.append(song_data['lyrics'])

    inp = []
    for desc in song_desc:
        for d in desc:
            for sent in d.split("\n"):
                inp.append(sent)

    print("Data Loading and indexing...")
    get_sentence_pos(inp)

    sents_pickle = open(os.path.join(data_path, '1_sg_real_data.pkl'), 'rb')
    sents = pickle.load(sents_pickle)

    pos_list_pickle = open(os.path.join(data_path, 'sg_pos_list.pkl'), 'rb')
    pos_list = pickle.load(pos_list_pickle)

    print("Data preprocessing in progress..")
    pos_size, embedding_size = get_preprocess_data(embed_path, pos_list)
    print("pos_size: ", pos_size)
    print("embedding_size: ", embedding_size)

    print("#### test ####")
    pkl_loading_test(data_path)


