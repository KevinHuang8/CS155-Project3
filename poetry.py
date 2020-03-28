import numpy as np
import HMM
import importlib
import random

# Removes punctuation from beginning and end of word
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

# Loads data in line-by-line and in reverse
def load_data_rhyme(filename, dictionary):
    data = []
    with open(filename) as f:
        #sonnet_c = 0
        for i, line in enumerate(f):
            words = line.split()
            if len(words) <= 1:
                continue

            '''
            if len(words) == 1:
                sonnet_c += 1
                continue

            # Sonnet 99 has 15 lines for some reason
            # Sonnet 126 has 12 lol
            if sonnet_c == 99 or sonnet_c == 126:
                continue
            '''

            words = remove_punctuation(words, dictionary)
            words.reverse()
            data.append(words)

    return data

# Converts words into an integer format suitable for HMM training
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

# Converts integer representation of words into strings
def decode_emission(emission, word_list):
    decoded = []
    for word in emission:
        decoded.append(word_list[word])

    return ' '.join(decoded)

# Gets the number of syllables in a line, accounting for nuances.
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

# Generates a normal sonnet without rhyming
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

# Generates a rhyming sonnet
def generate_rhyming_sonnet(model, word_list, syllable_dict, rhyme_dict):
    print("### Rhyming Sonnet ###")
    # Model must be trained in reverse
    def generate_sonnet_line(seed):
        num_words_l = [5, 6, 7, 8, 9]
        line = decode_emission(model.generate_emission(7)[0], word_list)
        while get_syllable_count(line, syllable_dict) != 10:
            num_words = random.choice(num_words_l)
            emission = model.generate_emission_seed(num_words, word_list.index(seed))[0]
            emission.reverse()
            line = decode_emission(emission, word_list)
        return line

    for i in range(3):
        word1, word2 = random.choice(list(rhyme_dict.items()))
        word3, word4 = random.choice(list(rhyme_dict.items()))
        print(generate_sonnet_line(word1))
        print(generate_sonnet_line(word3))
        print(generate_sonnet_line(word2))
        print(generate_sonnet_line(word4))
        print()

    for k in range(1):
        word1, word2 = random.choice(list(rhyme_dict.items()))
        print(generate_sonnet_line(word1))
        print(generate_sonnet_line(word2))

# Generates a haiku
def generate_haiku(model, word_list, syllable_dict):
    def generate_haiku_line(syl_count):
        num_words_l = [3, 4, 5, 6, 7]
        line = decode_emission(model.generate_emission(4)[0], word_list)
        while get_syllable_count(line, syllable_dict) != syl_count:
            num_words = random.choice(num_words_l)
            emission = model.generate_emission(num_words)[0]
            line = decode_emission(emission, word_list)
        return line

    print(generate_haiku_line(5))
    print(generate_haiku_line(7))
    print(generate_haiku_line(5))

# generwates sonnets using different models
def poems_from_various_models(word_list, syllable_dict):
    models = [1, 3, 6, 10, 15, 20, 25]
    for i in models:
        print("### Model " + str(i) + " ####")
        model = HMM.load_from_file('HMM' + str(i) + '.npz')
        generate_sonnet(model, word_list, syllable_dict)
        print()

# Searches training data for rhyming pairs
def generate_rhyme_pairs(data):
    #Assumes data is reversed
    rhyme_dict = {}
    for i, x in enumerate(data):
        if (i%14) in [0,1,4,5,8,9]:
            # get the other pair 2 lines ahead
            word1 = x[0]
            word2 = data[i+2][0]
            rhyme_dict[word1] = word2
            rhyme_dict[word2] = word1

        if (i%14) == 12:
            # get the other pair 1 line ahead
            word1 = x[0]
            word2 = data[i+1][0]
            rhyme_dict[word1] = word2
            rhyme_dict[word2] = word1

    return rhyme_dict

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
    data_rhyme = load_data_rhyme('data/shakespeare.txt', dictionary)

    data_HMM = encode_data_HMM(data, word_list)
    data_HMM_rhyme = encode_data_HMM(data_rhyme, word_list)

    rhyme_dict = generate_rhyme_pairs(load_data_rhyme('data/shakespeare_no_dumb_poems.txt', dictionary))

    '''
    model = HMM.unsupervised_HMM(data_HMM_rhyme, 20, 100)
    model.save("reversedHMM")
    '''
    modelRhyme = HMM.load_from_file("reversedHMM.npz")
    modelRegular = HMM.load_from_file("HMM20.npz")

    poems_from_various_models(word_list, syllable_dict)

    generate_rhyming_sonnet(modelRhyme, word_list, syllable_dict, rhyme_dict)

    generate_haiku(modelRegular, word_list, syllable_dict)

main()
