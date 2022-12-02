candidates_map = dict()
jobs_map = dict()
capplies_train = dict()

ftrain_input = open("applies.csv.train")
candidate_index_map = dict()
job_index_map = dict()


num_cands = 0
num_jobs = 0
max_cands = 1000
for x, l in enumerate(ftrain_input):
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

ftest_input = open("applies.csv.test")

for x, l in enumerate(ftest_input):
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

ftrain = open("train", "w")
ftest = open("test", "w")

num_cands_count = 0
new_jobs_map = {}
new_cands_map = {}
job_index = 0
cand_index = 0

for c, c_index in candidates_map.items():
    if c in capplies_train and c in capplies_test:
        if c not in new_cands_map:
            new_cands_map[c] = str(cand_index)
            cand_index += 1
        for j in capplies_train[c]:
            if j not in new_jobs_map:
                new_jobs_map[j] = str(job_index)
                job_index += 1
            ftrain.write(new_cands_map[c] + "  " + new_jobs_map[j] + " " + "1\n")
        for j in capplies_test[c]:
            if j not in new_jobs_map:
                new_jobs_map[j] = str(job_index)
                job_index += 1
            ftest.write(new_cands_map[c] + "  " + new_jobs_map[j] + " " + "1\n")

        num_cands_count += 1
    if num_cands_count == max_cands:
        break

print("num users:", len(candidates_map.keys()))
print("num jobs:", len(jobs_map.keys()))

ftrain.flush()
ftest.flush()