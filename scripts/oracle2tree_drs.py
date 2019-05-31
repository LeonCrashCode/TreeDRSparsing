import sys
import re

def get_in(filename):
	lemmas = []
	lines = []
	for line in open(filename):
		line = line.strip()
		if line == "":
			new_lemma = [[]]
			lemma = lines[1].split()
			past_lemmas = []
			for i in range(len(lemma)):
				if lemma[i] == "|||":
					new_lemma.append([])
					past_lemmas = []
					continue
				if lemma[i] in past_lemmas:
					continue
				past_lemmas.append(lemma[i])
				new_lemma[-1].append(lemma[i])
			lemmas.append(new_lemma)
			lines = []
		else:
			lines.append(line)
	return lemmas

def get_out(filename):
	trees = []
	for line in open(filename):
		line = line.strip()
		trees.append(line.split())
	return trees

def out_tree(lemmas, trees):
	assert len(lemmas) == len(trees)

	i = 0
	cur = 0
	while i < len(lemmas):
		j = 0
		while j < len(trees[i]):
			if re.match("^DRS-[0-9]+\($", trees[i][j]):
				cur = int(trees[i][j][4:-1])
				assert cur < len(lemmas[i])
			elif re.match("^\$[0-9]+\(",trees[i][j]):
				idx = int(trees[i][j][1:-1])
				trees[i][j] = lemmas[i][cur][idx]+"("
			j += 1
		i += 1

if __name__ == "__main__":
	lemmas = get_in(sys.argv[1])
	trees = get_out(sys.argv[2])

	out_tree(lemmas, trees)

	print "\n".join([" ".join(tree) for tree in trees])
