import pickle,os
path=""
def load(file_name):
  return pickle.load(open(os.path.join(path, file_name), 'rb'))

answers=load("original/answers")
print ("have %d answers" % len(answers))

vocabulary=load("original/vocabulary")
print ("have %d words" % len(vocabulary))

def convert2TSV(rawFilename):
	test1=load("original/"+rawFilename)
	
	lines=[]
	print (len(test1))
	for item in test1:
		
		question=item["question"]
		bad=item["bad"]
		good=item["good"]
		q_words= " ".join([ vocabulary[w]  for w in question])
		for sen in good:
			correct_words=" ".join( [ vocabulary[w]  for w in answers[sen]])
			line= "\t".join( (q_words, correct_words, "1" ))
			lines.append(line)
		for sen in bad:
			uncorrect_words= " ".join( [ vocabulary[w]  for w in answers[sen]])
			line= "\t".join( (q_words, uncorrect_words, "0" ))
			lines.append(line)
		# print lines

	filename= "original/insurance_%s.tsv" %(rawFilename)
	with open(filename, "w") as f:
		f.write("\n".join(lines) )
	return filename

def convertAll(subset_size=0):
	for rawFilename in ["dev","test1","test2"]:
		filename=convert2TSV(rawFilename)
		temp_file=format_file(filename,subset_size)
		os.remove(filename)


def parseTrain():
	train= load ("train")
	lines=[]
	for item in train:
		question=item["question"]
		q_words= " ".join([ vocabulary[w]  for w in question])
		answerIDs=item["answers"]
		for sen in answerIDs:
			correct_words=" ".join( [ vocabulary[w]  for w in answers[sen]])
			line= "\t".join( (q_words, correct_words, "1" ))
			lines.append(line)
			
	filename= "original/insurance_%s.tsv" %("train")
	with open(filename, "w") as f:
		f.write("\n".join(lines) )

def format_file(filename="insurance_dev.tsv",subset_size=1800):
	temp_file="insuranceQA"+"/"+filename[filename.index("_")+1:filename.index(".")]
	with open(filename) as f, open(temp_file,"w") as out:
		for index, line in enumerate(f):
			question,answer,label=line.strip().split("\t")
			newline="%s qid:%s" %(label, index/500)
			if subset_size!=0 and index / 500 >=subset_size:
				break
			for sen in [question,answer]:
				tokens=sen.split()
				fill=max(0,200-len(tokens))
				tokens.extend(['<a>']*fill)
				newline+=" "+"_".join( tokens)+"_" 
			out.write(newline+"\n")
	return temp_file

			


if __name__ == "__main__":
	# parseTrain()
	convertAll(subset_size=0)                #


