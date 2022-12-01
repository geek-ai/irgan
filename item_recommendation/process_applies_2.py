candidates_map = dict()
jobs_map = dict()
capplies_train = dict()

ftrain_input = open("/item_recommendation/seek/applies.csv.train")
candidate_index_map = dict()
job_index_map = dict()


num_cands = 0
num_jobs = 0
for l in ftrain_input:
    line = l.split(",")
    c = line[2]
    j = line[3]
    if c not in capplies_train:
        capplies_train[c] = []
    capplies_train[c].append(j)
    if c not in candidates_map:
        candidates_map[c] = str(num_cands)
        num_cands += 1
    if j not in jobs_map:
        jobs_map[j] = str(num_jobs)
        num_jobs += 1

capplies_test = dict()

ftest_input = open("/item_recommendation/seek/applies.csv.test")

for l in ftest_input:
    line = l.split(",")
    c = line[2]
    j = line[3]
    if c not in capplies_test:
        capplies_test[c] = []
    capplies_test[c].append(j)
    if c not in candidates_map:
        candidates_map[c] = str(num_cands)
        num_cands += 1
    if j not in jobs_map:
        jobs_map[j] = str(num_jobs)
        num_jobs += 1

ftrain = open("/item_recommendation/seek/train", "w")
ftest = open("/item_recommendation/seek/test", "w")

for c, applies in capplies_train.items():
    for j in applies:
        ftrain.write(candidates_map[c] + "  " + jobs_map[j] + " " + "1\n")

for c, applies in capplies_test.items():
    for j in applies:
        ftest.write(candidates_map[c] + "  " + jobs_map[j] + " " + "1\n")

ftrain.flush()
ftest.flush()