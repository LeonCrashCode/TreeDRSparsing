import os
import sys
import re

v_p = re.compile("^[XESTPK][0-9]+$")
d_p = re.compile("^DRS-[0-9]+\($")
pb = re.compile("^P[0-9]+\($")
kb = re.compile("^K[0-9]+\($")

def correct(tree):
	# Here we correct some weired things

	#e.g. :( K1 K2 )
	for i in range(len(tree)):
		if tree[i] == ":(" and tree[i+1][0] == "K":
			tree[i] = "THAT("
		if tree[i] == "-(" and tree[i+1][0] == "K":
			tree[i] = "THAT("
		if tree[i] == "((" and tree[i+1][0] == "K":
			tree[i] = "THAT("

def bracket2list(bracket):
	stack = []
	for tok in bracket:
		if tok[-1] == "(":
			stack.append([tok])
		elif tok == ")":
			if len(stack) != 1:
				back = stack.pop()
				stack[-1].append(back)
		else:
			stack[-1].append(tok)
	assert len(stack) == 1
	return stack[0]

def tree2oracle(lemmas, tree, out_action):
	lemmas = lemmas.split()
	new_lemmas = [[]]
	past_lemmas = []
	for i in range(len(lemmas)):
		if lemmas[i] == "|||":
			new_lemmas.append([])
			past_lemmas = []
			continue

		if lemmas[i] in past_lemmas:
			continue
		past_lemmas.append(lemmas[i])
		new_lemmas[-1].append(lemmas[i])
	lemmas = new_lemmas
	v = ["X","E","S","T","P","K"]
	vl = [ [] for i in range(6)]

	for tok in tree:
		if pb.match(tok):
			assert tok[:-1] not in vl[-2]
			vl[-2].append(tok[:-1])
		if kb.match(tok):
			assert tok[:-1] not in vl[-1]
			vl[-1].append(tok[:-1])


	root = bracket2list(tree)

	def is_struct(tok):
		if tok in ["DRS(", "SDRS(", "NOT(", "POS(", "NEC(", "IMP(", "OR(", "DUP("]:
			return True
		if d_p.match(tok):
			return True
		if re.match("^[PK][0-9]+\($", tok):
			return True
		return False

	def travel(root):
		#global v
		#global vl
		parent = root[0]
		child = root[1:]
		if parent[-1] == "(":
			if is_struct(parent):
				if (parent in ["DRS(", "SDRS("]) or d_p.match(parent):
					for c in child:
						if not is_struct(c[0]):
							for cc in c[1:]:
								if v_p.match(cc):
									idx = v.index(cc[0])
									if cc not in vl[idx]:
										vl[idx].append(cc)
				for c in child:
					travel(c)
			else:
				pass
				
		else:
			print parent
			assert False
	travel(root)
	#for tok in tree:
	#	if v_p.match(tok):
	#		idx = v.index(tok[0])
	#		if tok not in vl[idx]:
	#			vl[idx].append(tok)
	#print vl[0]

	correct(tree)
	i = 0
	cur = 0
	while i < len(tree):
		tok = tree[i]
		if re.match("DRS-[0-9]+\(", tok): #DRS-10(
			cur = int(tok[4:-1])
			assert cur < len(lemmas)
		if pb.match(tok):
			idx = vl[-2].index(tok[:-1])
			tree[i] = "P"+str(idx+1)+"("
		elif kb.match(tok):
			idx = vl[-1].index(tok[:-1])
			tree[i] = "K"+str(idx+1)+"("
		elif v_p.match(tok):
			vl_idx = v.index(tok[0])
			idx = vl[vl_idx].index(tok)
			tree[i] = v[vl_idx] + str(idx+1)
		else:
			if tok == "Card(":
				assert tree[i+3] == ")" # Card( X0 2 )
				tree[i+2] = "CARD_NUMBER"
			elif tok == "Timex(":
				assert tree[i+3] == ")"
				tree[i+2] = "TIME_NUMBER"
			elif tok[-1] == "(" and tok[:-1] in lemmas[cur]:
				idx = lemmas[cur].index(tok[:-1])
				tree[i] = "$"+str(idx)+"("
		i += 1
	
	out_action.write(" ".join(tree)+"\n")

def filter(illform, tree):
	#filter two long sentences, actually only one
	"""
	cnt = 0
	for item in tree:
		if item == "DRS(":
			cnt += 1
	if cnt >= 21:
		return True
	"""
	for item in illform:
		if item in tree:
			return True
	return False
if __name__ == "__main__":
	
	illform = []
	"""
	for line in open("illform"):
		line = line.strip()
		if line == "":
			continue
		illform.append(line.split()[-2])
	"""
	if os.path.exists("manual_correct"):
		for line in open("manual_correct"):
			line = line.strip()
			if line == "" or line[0] == "#":
				continue
			illform.append(line.split()[0])
	
	lines = []
	filename = ""
	out_input = open(sys.argv[1]+".oracle.doc.in", "w")
	out_action = open(sys.argv[1]+".oracle.doc.out", "w")
	for line in open(sys.argv[1]):
		line = line.strip()
		if line == "":

			idx = lines.index("TREE")
			
			assert idx % 2 == 0 and idx != 0
			lemmas = " ||| ".join([ lines[i*2+1] for i in range(idx/2) ])
			lemmas_copy = " ".join([ lines[i*2+1] for i in range(idx/2) ])
			words = " ||| ".join([ lines[i*2] for i in range(idx/2) ])

			tree = lines[idx+1].split()
			if filter(illform, tree):
				lines = []
				continue

			out_input.write("\n".join([words, lemmas]))
			out_input.write("\n\n")
			tree2oracle(lemmas, tree, out_action)
			out_input.flush()
			out_action.flush()
			lines = []
		else:
			if line[0] == "#":
				filename = line.split()[-1]
				continue
			lines.append(line)
	out_input.close()
	out_action.close()

			
