import numpy as np
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from random import shuffle
import operator


def build_vocabulary(sentences,savedir):
    print ("begin to build vocab")

    pad_str = '<pad>'.decode('utf-8')
    unk_str = '<unk>'.decode('utf-8')

    #just for count_words
    all_words = []
    for sen in sentences:
        for word in sen:
            if " " != word:
                all_words.append(word)
    print  ("all_words.len::", len(all_words))
    #sort wordlist with word's count
    word_set = list(set(all_words))
    print ("word_set.len::", len(word_set))

    count_set_list = {}
    for word in all_words:
        if count_set_list.has_key(word):
            count_set_list[word] += 1
        else:
            count_set_list[word] = 1
        #count_set_list[item] = all_words.count(item)

    sorted_list_word = sorted(count_set_list.items(), key=operator.itemgetter(1), reverse=False)

    word_list = []
    for item in sorted_list_word[::-1]:
        word_list.append(item[0])


    word_list.append(unk_str)
    if pad_str in word_list:
        word_list.remove(pad_str)
    word_list.insert(0,pad_str)

    word_dic = dict()

    with open(savedir,'w') as f:
        for word in word_list:
            word_dic[word]=int(word_list.index(word))
            line = str(word)+ "\n"
            f.write(line)

    return word_dic,len(word_dic.keys())


def read_vocabulary(voc_dir):

    voc = dict()
    lines = open(voc_dir,'r').readlines()

    for i in range(len(lines)):
        key = lines[i].decode('utf-8').split('\n')[0]
        voc[key] = i

    print('adsfaf')
    print 'read vocabulary len: %f' % len(voc.keys())
    return voc,len(voc.keys())


def sentence2matrix(sentences,max_length,vocs):
    sentences_num = len(sentences)
    data_dict = np.zeros((sentences_num,max_length),dtype='int32')

    for index,sentence in enumerate(sentences):
        data_dict[index,:]=map2id(sentence,vocs,max_length)

    return data_dict


def map2id(sentence,voc,max_len):
    array_int = np.zeros((max_len,),dtype='int32')
    min_range = min(max_len,len(sentence))

    for i in range(min_range):
        item = sentence[i]
        array_int[i] = voc.get(item,voc['<unk>'])

    return array_int


def clean_str(string):
    return string.strip().lower()

def mkdir_if_not_exist(dirpath):

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    return dirpath


def seperate_line(line):
    return ''.join([word + ' ' for word in line])


def read_and_clean_file(input_file, output_cleaned_file = None):
    lines = list(open(input_file, "r").readlines())
    lines = [clean_str(seperate_line(line.decode('utf-8'))) for line in lines]

    if output_cleaned_file is not None:
        with open(output_cleaned_file, 'w') as f:
            for line in lines:
                f.write((line + '\n').encode('utf-8'))

    return lines


def load_data_and_labels(caijing_data_file, caipiao_data_file, fangchan_data_file, gupiao_data_file,jiaju_data_file,
                         jiaoyu_data_file,shishang_data_file,shizheng_data_file,tiyu_data_file,yule_data_file):
    caijing_examples = read_and_clean_file(caijing_data_file)
    caipiao_examples = read_and_clean_file(caipiao_data_file)
    fangchan_examples = read_and_clean_file(fangchan_data_file)
    gupiao_examples = read_and_clean_file(gupiao_data_file)
    jiaju_examples = read_and_clean_file(jiaju_data_file)
    jiaoyu_examples = read_and_clean_file(jiaoyu_data_file)
    shishang_examples = read_and_clean_file(shishang_data_file)
    shizheng_examples = read_and_clean_file(shizheng_data_file)
    tiyu_examples = read_and_clean_file(tiyu_data_file)
    yule_examples = read_and_clean_file(yule_data_file)

    x_text= caijing_examples + caipiao_examples + fangchan_examples + gupiao_examples + jiaju_examples +\
             jiaoyu_examples + shishang_examples + shizheng_examples + tiyu_examples + yule_examples

    caijing_labels = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in caijing_examples]
    caipiao_labels = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0] for _ in caipiao_examples]
    fangchan_labels =[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0] for _ in fangchan_examples]
    gupiao_labels =  [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0] for _ in gupiao_examples]
    jiaju_labels =   [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0] for _ in jiaju_examples]
    jiaoyu_labels =  [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0] for _ in jiaoyu_examples]
    shishang_labels =[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0] for _ in shishang_examples]
    shizheng_labels =[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0] for _ in shizheng_examples]
    tiyu_labels =    [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0] for _ in tiyu_examples]
    yule_labels =    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1] for _ in yule_examples]

    y = np.concatenate([caijing_labels, caipiao_labels, fangchan_labels, gupiao_labels, jiaju_labels,jiaoyu_labels,
                        shishang_labels, shizheng_labels, tiyu_labels, yule_labels], 0)

    return [x_text, y]


def load_testfile_and_labels(input_text_file,input_label_file,num_labels):
    x_text = read_and_clean_file(input_text_file)

    y = None if not os.path.exists(input_label_file) else map(int, list(open(input_label_file, "r").readlines()))

    return (x_text, y)


def padding_sentences(input_sentences, padding_token, padding_sentence_length = None):
    sentences = [sentence.split(' ') for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in sentences])

    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))

    return sentences

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]


def read_data_from_strs(lines,max_sentence_length):
    data_line = []

    for line in lines:
        line = line.decode('utf-8')
        line = ''.join([word + ' ' for word in line])
        line = line.strip().lower()
        line=line.split(' ')

        if len(line) > max_sentence_length:
            line = line[:max_sentence_length]
        else:
            line.extend(['<pad>'] * (max_sentence_length - len(line)))

        data_line.append(line)

    return data_line


def read_data_from_str(line,max_sentence_length):
    line = line.decode('utf-8')
    line = ''.join([word + ' ' for word in line])
    line = line.strip().lower()
    line=line.split(' ')

    if len(line) > max_sentence_length:
        line = line[:max_sentence_length]
    else:
        line.extend(['<pad>'] * (max_sentence_length - len(line)))

    return [line]

if __name__ == '__main__':
    pass
