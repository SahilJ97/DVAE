import csv
import numpy as np
from params import *
import matplotlib.pyplot as plt


noises = np.arange(0., MAX_N_LEVEL, NOISE_STEP)

log_file = 'log.csv'
#log_file = 'toy_log.csv'


performance_data = dict()
for row in csv.DictReader(open(log_file)):
    if row[' epoch'] != ' final':
        continue
    tr_n_and_it = row['training noise-level and iteration']
    tr_n = tr_n_and_it.split(' ')[0]
    te_n = row[' test noise-level']
    cost = row[' cost']
    if str(tr_n) not in performance_data.keys():
        performance_data[str(tr_n)] = []
    performance_data[str(tr_n)].append({'test_noise': te_n, 'cost': cost})

tr_noise_levels = []

for tr_noise, l in performance_data.items():
    costs = []
    # for each test noise level, delete the two worst performances.
    for test_noise in noises:
        best = 1e20
        for i in range(len(l)):
            item_test_noise = float(l[i]['test_noise'].replace(' ', ''))
            item_cost = float(l[i]['cost'].replace(' ', ''))
            if item_test_noise == test_noise:
                if item_cost < best:
                    best = item_cost
        costs.append(best)
    tr_noise_levels.append(costs)


if __name__ == '__main__':
    for c, tr_level in zip(tr_noise_levels, noises):
        plt.scatter(noises, c, label='{}'.format(tr_level), s=5)
        plt.plot(noises, c)
    plt.legend()
    plt.xlabel('Test noise')
    plt.ylabel('Loss')
    plt.show()
