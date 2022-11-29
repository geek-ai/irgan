capplies = dict()

f = open("/Users/fzafari/Projects/innovation/irgan/item_recommendation/SEEK_AU_202109_100_5K/applies.csv.train")
for l in f:
    line = l.split(",")
    c = line[2]
    j = line[3]
    if c not in capplies:
        capplies[c] = []
    capplies[c].append(j)

ftrain = open("/Users/fzafari/Projects/innovation/irgan/item_recommendation/SEEK_AU_202109_100_5K/train", "w")
ftest = open("/Users/fzafari/Projects/innovation/irgan/item_recommendation/SEEK_AU_202109_100_5K/test", "w")

train_i = 0
test_i = 0
jobs = dict()

train_tuples = []
test_tuples = []
for k in sorted(capplies, key=lambda k: len(capplies[k])):
    applies = capplies[k]
    if len(applies) < 5:
        continue
    train_applies = applies[:int(4*len(applies)/5)]
    test_applies = applies[int(4*len(applies)/5)+1:]
    if train_i < 10000:
        for a in train_applies:
            if a not in jobs:
                jobs[a] = 1
            train_tuples.append((str(k), str(a)))
            train_i += 1
    if test_i < 1000:
        for a in test_applies:
            if a not in jobs:
                jobs[a] = 1
            test_tuples.append((str(k), str(a)))
            test_i += 1

candidate_index_map = dict()
job_index_map = dict()

for apps in [train_tuples, test_tuples]:
    for c, j in apps:
        if c not in candidate_index_map:
            candidate_index_map[c] = str(len(candidate_index_map))
        if j not in job_index_map:
            job_index_map[j] = str(len(job_index_map))


print("Num candidates:", len(candidate_index_map))
print("Num jobs:", len(job_index_map))
print("Num interactions train set:", len(train_tuples))
print("Num interactions test set:", len(test_tuples))


for (file, tuples) in [(ftrain, train_tuples), (ftest, test_tuples)]:
    for (c, j) in tuples:
        file.write(candidate_index_map[c] + "   " + job_index_map[j] + "    " + "1" + "\n")
        file.flush()