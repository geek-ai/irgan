candidates_map = dict()
jobs_map = dict()
capplies_train = dict()

ftrain_input = open("/Users/fzafari/Projects/innovation/irgan/item_recommendation/SEEK_AU_202109_100_5K/applies.csv.train")
candidate_index_map = dict()
job_index_map = dict()

for l in ftrain_input:
    line = l.split(",")
    c = line[2]
    j = line[3]
    if c not in capplies_train:
        capplies_train[c] = []
    capplies_train[c].append(j)
    if c not in candidates_map:
        candidates_map[c] = str(len(candidates_map.keys()))
    if j not in jobs_map:
        jobs_map[j] = str(len(jobs_map.keys()))

capplies_test = dict()

ftest_input = open("/Users/fzafari/Projects/innovation/irgan/item_recommendation/SEEK_AU_202109_100_5K/applies.csv.test")

for l in ftest_input:
    line = l.split(",")
    c = line[2]
    j = line[3]
    if c not in capplies_test:
        capplies_test[c] = []
    capplies_test[c].append(j)
    if c not in candidates_map:
        candidates_map[c] = str(len(candidates_map.keys()))
    if j not in jobs_map:
        jobs_map[j] = str(len(jobs_map.keys()))

ftrain = open("/Users/fzafari/Projects/innovation/irgan/item_recommendation/SEEK_AU_202109_100_5K/train", "w")
ftest = open("/Users/fzafari/Projects/innovation/irgan/item_recommendation/SEEK_AU_202109_100_5K/test", "w")

for c, applies in capplies_train.items():
    for j in applies:
        ftrain.write(candidates_map[c] + "  " + jobs_map[j] + " " + "1\n")

for c, applies in capplies_test.items():
    for j in applies:
        ftest.write(candidates_map[c] + "  " + jobs_map[j] + " " + "1\n")

ftrain.flush()
ftest.flush()