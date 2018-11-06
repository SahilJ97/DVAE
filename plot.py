import csv


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
    for 

if __name__ == '__main__':
    print(performance_data)
