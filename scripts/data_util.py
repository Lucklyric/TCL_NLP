from tensorflow.python.platform import gfile
import os
import re

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_DIGIT_RE = re.compile("\d")


def create_vocabulary(vocabulary_path, data_path, word2vec=False):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="r") as f:
            print "Filed readed"
            counter = 0
            for line in f:
                counter += 1
                if counter % 100 == 0:
                    print ("processing line line %d" % counter)
                for w in line.split():
                    word = _DIGIT_RE.sub("number", w)
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")
    return


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab


def sentence_to_token_ids(sentence, vocabulary):
    words = sentence.split()
    return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s " % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="r") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_tcl_data(data_dir, vocab_dir):
    in_vocab_path = os.path.join(vocab_dir, "vocab.in")
    ner_vocab_path = os.path.join(vocab_dir, "vocab.ner")
    create_vocabulary(in_vocab_path, os.path.join(data_dir, "train_in.txt"))
    create_vocabulary(ner_vocab_path, os.path.join(data_dir, "train_ner.txt"))

    # Create token ids for training data
    in_train_ids_path = os.path.join(data_dir, "train_in.ids")
    ner_train_ids_path = os.path.join(data_dir, "train_ner.ids")
    data_to_token_ids(os.path.join(data_dir, "train_in.txt"), in_train_ids_path, in_vocab_path)
    data_to_token_ids(os.path.join(data_dir, "train_ner.txt"), ner_train_ids_path, ner_vocab_path)

    # Create token ids for testing data
    in_test_ids_path = os.path.join(data_dir, "test_in.ids")
    ner_test_ids_path = os.path.join(data_dir, "test_ner.ids")
    data_to_token_ids(os.path.join(data_dir, "test_in.txt"), in_test_ids_path, in_vocab_path)
    data_to_token_ids(os.path.join(data_dir, "test_ner.txt"), ner_test_ids_path, ner_vocab_path)

    return (in_train_ids_path, ner_train_ids_path,
            in_test_ids_path, ner_test_ids_path,
            in_vocab_path, ner_vocab_path)


# prepare_tcl_data("../data/format_data", "../data/vocabulary")

# create_vocabulary('./vocabulary/train_voc.txt', 'tmp/train_raw.txt')
# train_vocab, train_rev_vocab = initialize_vocabulary("./vocabulary/train_voc.txt")
# print train_vocab
