import numpy as np
import HMM
import importlib
import random

def remove_punctuation(words, dictionary):
    punctuation = [',','.',':','?',';','!',"'",'"', '(', ')']
    for i, word in enumerate(words):
        word = word.lower()

        while word not in dictionary:

            if word[-1] in punctuation:
                word = word[:-1]

            if word[0] in punctuation and word not in dictionary:
                word = word[1:]

        words[i] = word

    return words

def load_data(filename, dictionary):
    data = []
    with open(filename) as f:
        start = 0
        current = []
        for i, line in enumerate(f):
            words = line.split()
            if len(words) < 1:
                continue

            if len(words) == 1:
                start = i
                continue

            words = remove_punctuation(words, dictionary)
            current.extend(words)

            if i == start + 4 or i == start + 8 or i == start + 12 or \
            i == start + 14:
                data.append(current)
                current = []

    return data

def encode_data_HMM(data, word_list):
    encoding = {}
    for i, word in enumerate(word_list):
        encoding[word] = i

    encoded_data = []
    for x in data:
        encoded_x = []
        for word in x:
            encoded_x.append(encoding[word])
        encoded_data.append(encoded_x)

    return encoded_data

def decode_emission(emission, word_list):
    decoded = []
    for word in emission:
        decoded.append(word_list[word])

    return ' '.join(decoded)

def get_syllable_count(line, dictionary):
    words = line.split(" ")
    syllables = []
    for i, word in enumerate(words):
        syl_info = dictionary[word]
        final_syl_c = 0
        for num in syl_info:
            try:
                final_syl_c = int(num)
            except:
                if i == len(words)-1:
                    final_syl_c = int(num[-1])
                    break
        syllables.append(final_syl_c)
    return sum(syllables)



def generate_sonnet(model, word_list, syllable_dict):

    def generate_sonnet_line():
        num_words_l = [5, 6, 7, 8, 9]
        line = decode_emission(model.generate_emission(7)[0], word_list)
        while get_syllable_count(line, syllable_dict) != 10:
            num_words = random.choice(num_words_l)
            emission = model.generate_emission(num_words)[0]
            line = decode_emission(emission, word_list)
        return line

    for i in range(3):
        for j in range(4):
            print(generate_sonnet_line())
        print()

    for k in range(2):
        print(generate_sonnet_line())

def main():
    word_list = []
    dictionary = set()
    syllable_dict = {}
    with open('data/Syllable_dictionary.txt') as f:
        for line in f:
            word_list.append(line.split()[0])
            dictionary.add(line.split()[0])
            syllable_dict[line.split()[0]] = line.split()[1:]

    data = load_data('data/shakespeare.txt', dictionary)
    data_HMM = encode_data_HMM(data, word_list)

    model = HMM.load_from_file('HMM.npz')

    generate_sonnet(model, word_list, syllable_dict)

main()
