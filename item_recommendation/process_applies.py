capplies = dict()

NUM_USER_TRAIN = 1000
NUM_USER_TEST = 100

f = open("/Users/fzafari/Projects/innovation/irgan/item_recommendation/SEEK_AU_202109_100_5K/applies.csv.train")
candidate_index_map = dict()
job_index_map = dict()

for l in f:
    line = l.split(",")
    c = line[2]
    j = line[3]
    if c not in capplies:
        capplies[c] = []
    capplies[c].append(j)

ftrain = open("/Users/fzafari/Projects/innovation/irgan/item_recommendation/SEEK_AU_202109_100_5K/train", "w")
ftest = open("/Users/fzafari/Projects/innovation/irgan/item_recommendation/SEEK_AU_202109_100_5K/test", "w")

train_cands_num = 0
test_cands_num = 0
jobs = dict()

train_tuples = []
test_tuples = []
for k in sorted(capplies, key=lambda k: len(capplies[k])):
    applies = capplies[k]
    if len(applies) < 7:
        continue
    train_applies = applies[:int(4*len(applies)/5)]
    test_applies = applies[int(4*len(applies)/5)+1:]
    print(k, "train:", len(train_applies), "test:", len(test_applies))
    # We only include candidates that have at least 1 apply to include in test set:
    if len(train_applies) > 0 and len(test_applies) > 0:
        # make sure we have 1000 candidates in the train set.
        if train_cands_num < NUM_USER_TRAIN:
            for a in train_applies:
                if a not in jobs:
                    jobs[a] = 1
                train_tuples.append((str(k), str(a)))
                if a not in job_index_map:
                    job_index_map[a] = str(len(job_index_map))
            train_cands_num += 1
            if k not in candidate_index_map:
                candidate_index_map[k] = str(len(candidate_index_map))
        # make sure we have 100 candidates in the test set.
        if test_cands_num < NUM_USER_TEST:
            for a in test_applies:
                if a not in jobs:
                    jobs[a] = 1
                test_tuples.append((str(k), str(a)))
                if a not in job_index_map:
                    job_index_map[a] = str(len(job_index_map))
            test_cands_num += 1
            if k not in candidate_index_map:
                candidate_index_map[k] = str(len(candidate_index_map))

print("Num candidates:", len(candidate_index_map))
print("Num jobs:", len(job_index_map))
print("Num interactions train set:", len(train_tuples))
print("Num interactions test set:", len(test_tuples))


for (file, tuples) in [(ftrain, train_tuples), (ftest, test_tuples)]:
    for (c, j) in tuples:
        file.write(candidate_index_map[c] + "   " + job_index_map[j] + "    " + "1" + "\n")
        file.flush()