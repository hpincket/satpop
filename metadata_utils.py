import csv

import numpy as np

import constants as C


# import matplotlib.pyplot as plt

def __get_list_of_pops():
    populations = []
    with open(C.SATPOP_MAIN_DATA_FILE, "r") as fd:
        tsvreader = csv.reader(fd, delimiter='\t', quotechar='|')
        for line in tsvreader:
            populations.append(float(line[3]))
    return populations


def generate_histogram(divisions, populations=None):
    if not populations:
        populations = __get_list_of_pops()
    print(np.histogram(populations, bins=divisions))


def generate_even_divisions(num_of_divisions):
    populations = __get_list_of_pops()
    populations.sort()
    bucket_size = len(populations) / num_of_divisions
    upper_barriers = []
    for i in range(1, num_of_divisions):
        upper_barriers.append(populations[i * bucket_size])
    upper_barriers.append(populations[-1] + 1)
    return upper_barriers


if __name__ == "__main__":
    bins = [0, 10, 100, 1000, 10000, 100000]
    generate_histogram(bins)
    bins = generate_even_divisions(10)
    print(bins)
    generate_histogram(bins)
