import os
import pickle

import re
from konlpy.tag import Twitter, Kkma

from utils.save_pickle import save_pickle


def get_before_dataset(data_path):
    max_len = 0

    song_desc = []
    for i in range(1, 8):
        data = open(data_path + 'sg_{}.pickle'.format(i), 'rb')
        song_data = pickle.load(data)
        song_desc.append(song_data['lyrics'])

    lyrics_total = list()
    for desc in song_desc:
        lyrics_artist = list()
        for d in desc:
            lyric = list()
            for sent in d.split("\n"):
                if sent == '':
                    continue
                sent = re.sub(r"[^ㄱ-힣]+", ' ', sent)
                if sent:
                    sent = analyzer.pos(sent, norm=True)
                    if len(sent) > max_len:
                        max_len = len(sent)
                        max_sent = sent
                lyric.append(sent)
            lyrics_artist.append(lyric)
        lyrics_total.append(lyrics_artist)

    print(max_len)
    print(max_sent)

    save_pickle(os.path.join(data_path, "sg_before_input_data.pkl"), lyrics_total)


def create_sequence(data_path, seq_length, lyrics_total):
    data = list()

    for lyrics_artist in lyrics_total:
        for lyric in lyrics_artist:
            for i in range(len(lyric)):
                seq_data = list()

                for word in lyric[i]:
                    if len(seq_data) < seq_length:
                        seq_data.append(word)

                while seq_length > len(seq_data):
                    seq_data.append(('UNK', 'Alpha'))

                data.append(seq_data)

    save_pickle(os.path.join(data_path, "2_sg_preprocessed_data.pkl"), data)

    f = open(os.path.join(data_path, '2_sg_preprocessed_data.txt'), 'w')
    for tokens in data:
        for word in tokens:
            word = str(word) + ' '
            f.write(word)
        f.write('\n')
    f.close()


def data_to_index(data_path, dataset, pos2idx):
    print(pos2idx)
    idx_dataset = list()
    for sent in dataset:
        print(sent)
        idx_sentence = list()
        for word in sent:
            idx_sentence.append(pos2idx[str(word).strip()])
        idx_dataset.append(idx_sentence)

    save_pickle(os.path.join(data_path, "3_sg_data_index.pkl"), idx_dataset)

    f = open(os.path.join(data_path, '3_sg_data_index.txt'), 'w')
    for idx_sent in idx_dataset:
        for word in idx_sent:
            word = str(word) + ' '
            f.write(word)
        f.write('\n')
    f.close()


if __name__ == "__main__":
    data_path = "../model/data/"
    seq_length = 10  # max 52
    analyzer = Twitter()

    get_before_dataset(data_path)

    lyrics_pickle = open(os.path.join(data_path, "sg_before_input_data.pkl"), 'rb')
    lyrics = pickle.load(lyrics_pickle)

    print("Create Sequence in a length of seq_length...")
    create_sequence(data_path, seq_length, lyrics)

    print("Complete Creating sequence !!")

    # load after dataset
    dataset_pickle = open(os.path.join(data_path, "2_sg_preprocessed_data.pkl"), 'rb')
    dataset = pickle.load(dataset_pickle)

    # load pos to index
    pos2idx_pickle = open(os.path.join(data_path, "sg_pos2idx.pkl"), 'rb')
    pos2idx = pickle.load(pos2idx_pickle)

    print("Replace Sequence to Index...")
    data_to_index(dataset, pos2idx)

    print("Complete Creating sequence to index !!")
