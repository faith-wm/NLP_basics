
import pandas as pd
import re, collections, csv,time
from _collections import OrderedDict

# this code is copied from:  https://leimao.github.io/blog/Byte-Pair-Encoding/


start_time=time.time()

def get_vocab(filename):
    vocab = collections.defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            line=line.lower()
            words = line.strip().split()

            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out



def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens


def get_tokens_from_vocab(vocab):
    tokens_frequencies = collections.defaultdict(int)
    vocab_tokenization = {}
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq
        vocab_tokenization[''.join(word_tokens)] = word_tokens
    return tokens_frequencies, vocab_tokenization


def measure_token_length(token):
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    else:
        return len(token)


def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token]

    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token.replace('.', '[.]'))

        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]

        substring_start_position = 0
        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position:substring_end_position]
            string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i + 1:],
                                           unknown_token=unknown_token)
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i + 1:],
                                       unknown_token=unknown_token)
        break
    return string_tokens





train_file='train.en'
vocab = get_vocab(train_file)
# for key,val in vocab.items():
#     print(key,val)

print('=====Tokens before BPE=======')
token_frequencis, vocab_tokenization=get_tokens_from_vocab(vocab)
print('Tokens:{}'.format(token_frequencis.keys()))
print('Number of tokens: {}'.format(len(token_frequencis.keys())))


num_merges=1000
from _collections import OrderedDict
for i in range(num_merges):
    pairs=get_stats(vocab)
    # sorted_dict = OrderedDict(sorted(pairs.items(), key=lambda x: x[1]))
    # for k, v in sorted_dict.items():
    #     print(k,v)
    if not pairs:
        break
    best=max(pairs,key=pairs.get)
    vocab=merge_vocab(best,vocab)
    print('Iter: {} out of {}: time taken:{} secs'.format(i, num_merges, time.time()-start_time))
    print('Best pair: {}'.format(best))
    token_frequencis, vocab_tokenization=get_tokens_from_vocab(vocab)
    print('number of tokens: {}'.format(len(token_frequencis)))
    print('=================')


#encoding and decoding
known_word='1-[(R)-1-ブロモペンチル]ピリジニウム</w>'
unknown_wod='1-メチル-3-エチル-1H-イミダゾール-3-イウム イソシアナート</w>'


sorted_tokens_tuple = sorted(token_frequencis.items(), key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
sorted_tokens=[token for (token,frequency) in sorted_tokens_tuple]


tes_file='test.en'
savefile='v2-Train_test_1000_EN.csv'


unknown_word=[]
test_file=pd.read_csv(tes_file,sep='\t',header=None)
# fl=fl.head()
for i in test_file[0]:
    unknown_word.append(i)

print('----------------------')
count=0
for word in unknown_word:
    count+=1
    print('=============count {}: time till now{}========'.format(count,time.time()-start_time))
    # if word in vocab_tokenization:
    #     print('tokenizing known word')
    #     # print(vocab_tokenization[word])
        # print('treating known word as unknown')
        # print(tokenize_word(word, sorted_tokens, '</u>'))
    # else:
    #     print('tokenizing unknown word')
        # print(tokenize_word(word, sorted_tokens, '</u>'))

    word=str(word.lower())+'</w>'
    my_dict={count:[word,tokenize_word(word, sorted_tokens, '</u>')]}
    with open(savefile, 'a') as outfile:
        writer = csv.writer(outfile)
        for k, v in my_dict.items():
            writer.writerow([k] + v)


print('total time: {} seconds'.format(time.time()-start_time))





