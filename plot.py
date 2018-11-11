import csv
import numpy as np
from params import *
import matplotlib.pyplot as plt


performance_data = dict()
for row in csv.DictReader(open("log.csv")):
    if row[' epoch'] == 'final':
        continue
    tr_n_and_it = row['training noise-level and iteration']
    tr_n = tr_n_and_it.split(' ')[0]
    te_n = row[' test noise-level']
    cost = row[' cost']
    if str(tr_n) not in performance_data.keys():
        performance_data[str(tr_n)] = []
    performance_data[str(tr_n)].append({'test_noise': te_n, 'cost': cost})
for tr_noise, l in performance_data.items():
    # for each test noise level, delete the two worst performances.
    for test_noise in np.arange(0., MAX_N_LEVEL, NOISE_STEP):
        best = 1e20
        best_index = None
        for i in range(len(l)):
            item_test_noise = float(l[i]['test_noise'].replace(' ', ''))
            item_cost = float(l[i]['cost'].replace(' ', ''))
            if item_test_noise == test_noise:
                if item_cost < best and best_index is not None:
                    l.remove(l[best_index])
                    best_index = i
                    best = l[i]['cost']
                elif item_cost >= best:
                    l.remove(l[i])

tr_noise_levels = []
costs = []
for tr_noise, l in performance_data.items():
    new_noise = []
    new_c = []
    for item in l:
        new_noise.append(float(item['test_noise'].replace(' ', '')))
        new_c.append(float(item['cost'].replace(' ', '')))
    tr_noise_levels.append(new_noise)
    costs.append(new_c)


if __name__ == '__main__':
    for c, t in zip(costs, tr_noise_levels):
        print(np.min(c))
        plt.scatter(t, c, label = '{}'.format(t))
        plt.plot(t, c, label='{}'.format(t))
    plt.show()
