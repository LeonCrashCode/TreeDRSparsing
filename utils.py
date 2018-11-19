# -*- coding: utf-8 -*-
def read_input(filename):
	data = [[]]
	for line in open(filename):
		line = line.strip()
		if line == "":
			data.append([])
		else:
			if line[0] == "#":
				continue
			data[-1].append(["<s>"] + line.split() + ["</s>"])
	if len(data[-1]) == 0:
		data.pop()
	return data

def read_input_doc(filename):
	data = []
	List = []
	for line in open(filename):
		line = line.strip()
		if line == "":
			data.append(zip(List[0], List[1]))
		else:
			if line[0] == "#":
				continue
			List.append([ ["<s>"] + x.strip().split() + ["</s>"] for x in line.split("|||")])

	return data

def get_singleton_dict(train_input, word_v):
	d = {}
	singleton_idx_dict = {}
	word_dict = {}
	for instance in train_input:
		for w in instance[0]:
			if w in d:
				d[w] += 1
			else:
				d[w] = 1
			word_v.toidx(w)

	for key in d.keys():
		if d[key] == 1:
			singleton_idx_dict[word_v.toidx(key)] = 1
		else:
			word_dict[key] = 1
	return singleton_idx_dict, word_dict, word_v

def unkify(token, word_dict, lang):
	if len(token.rstrip()) == 0:
		return '<UNK>'
	elif not(token.rstrip() in word_dict):
		if lang == "ch":
			return '<UNK>'
		numCaps = 0
		hasDigit = False
		hasDash = False
		hasLower = False
		for char in token.rstrip():
			if char.isdigit():
				hasDigit = True
			elif char == '-':
				hasDash = True
			elif char.isalpha():
				if char.islower():
					hasLower = True
				elif char.isupper():
					numCaps += 1
		result = '<UNK>'
		lower = token.rstrip().lower()
		ch0 = token.rstrip()[0]
		if ch0.isupper():
			if numCaps == 1:
				result = result + '-INITC'
				if lower in word_dict:
					result = result + '-KNOWNLC'
			else:
				result = result + '-CAPS'
		elif not(ch0.isalpha()) and numCaps > 0:
			result = result + '-CAPS'
		elif hasLower:
			result = result + '-LC'
		if hasDigit:
			result = result + '-NUM'
		if hasDash:
			result = result + '-DASH'
		if lower[-1] == 's' and len(lower) >= 3:
			ch2 = lower[-2]
			if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
				result = result + '-s'
		elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
			if lower[-2:] == 'ed':
				result = result + '-ed'
			elif lower[-3:] == 'ing':
				result = result + '-ing'
			elif lower[-3:] == 'ion':
				result = result + '-ion'
			elif lower[-2:] == 'er':
				result = result + '-er'
			elif lower[-3:] == 'est':
				result = result + '-est'
			elif lower[-2:] == 'ly':
				result = result + '-ly'
			elif lower[-3:] == 'ity':
				result = result + '-ity'
			elif lower[-1] == 'y':
				result = result + '-y'
			elif lower[-2:] == 'al':
				result = result + '-al'
		return result
	else:
		return token.rstrip() 

def input2instance(train_input, word_v, char_v, pretrain, extra_vl, word_dict, args, op):
	train_instance = []
	for instance in train_input:
		#lexicon representation
		train_instance.append([])
		train_instance[-1].append([]) # word
		train_instance[-1].append([]) # char
		train_instance[-1].append([]) # pretrain
		train_instance[-1].append([]) # typed UNK
		for w in instance[0]:
			idx = word_v.toidx(w)
			if op == "train":
				train_instance[-1][0].append(idx)
				idx = word_v.toidx(unkify(w, word_dict, "en"))
				train_instance[-1][3].append(idx)
			elif op == "dev":
				if idx == 0: # UNK
					idx = word_v.toidx(unkify(w, word_dict, "en"))
				train_instance[-1][0].append(idx)

			if args.use_char:
				train_instance[-1][1].append([])
				for c in list(w):
					idx = char_v.toidx(c)
					train_instance[-1][1][-1].append(idx)
			if args.pretrain_path:
				idx = pretrain.toidx(w.lower())
				train_instance[-1][2].append(idx)
		#extra representatoin
		for i, extra_info in enumerate(instance[1:], 0):
			train_instance[-1].append([])
			for t in extra_info:
				idx = extra_vl[i].toidx(t)
				train_instance[-1][-1].append(idx)

	return train_instance, word_v, char_v, extra_vl

def input2instance_doc(train_input, word_v, char_v, pretrain, extra_vl, word_dict, args, op):
	train_instances = []

	for instances in train_input:
		train_instance = []
		for instance in instances:
			#lexicon representation
			train_instance.append([])
			train_instance[-1].append([]) # word
			train_instance[-1].append([]) # char
			train_instance[-1].append([]) # pretrain
			train_instance[-1].append([]) # typed UNK
			for w in instance[0]:
				idx = word_v.toidx(w)
				if op == "train":
					train_instance[-1][0].append(idx)
					idx = word_v.toidx(unkify(w, word_dict, "en"))
					train_instance[-1][3].append(idx)
				elif op == "dev":
					if idx == 0: # UNK
						idx = word_v.toidx(unkify(w, word_dict, "en"))
					train_instance[-1][0].append(idx)

				if args.use_char:
					train_instance[-1][1].append([])
					for c in list(w):
						idx = char_v.toidx(c)
						train_instance[-1][1][-1].append(idx)
				if args.pretrain_path:
					idx = pretrain.toidx(w.lower())
					train_instance[-1][2].append(idx)
			#extra representatoin
			for i, extra_info in enumerate(instance[1:], 0):
				train_instance[-1].append([])
				for t in extra_info:
					idx = extra_vl[i].toidx(t)
					train_instance[-1][-1].append(idx)
		train_instances.append(train_instance)
	return train_instances, word_v, char_v, extra_vl

def read_tree(filename):
	data = []
	for line in open(filename):
		line = line.strip()
		if line[0] == "#":
			continue
		data.append(bracket2list(line.split()))
	return data

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

def tree2action(output, actn_v):
	actions = []
	for line in output:
		#print line
		actions.append(get_struct_rel_var(line, actn_v))
	return actions
import re
def is_struct(tok):
	if tok in ["DRS(", "SDRS(", "NOT(", "POS(", "NEC(", "IMP(", "OR(", "DUP("]:
		return True
	if re.match("^[PK][0-9]+\($", tok):
		return True
	return False

def get_struct_rel_var(tree, actn_v):
	"""
	input list of string
	return list of index
	"""
	struct = [actn_v.toidx("<START>")]
	relation = []
	variable = []
	def travel(root):
		parent = root[0]
		child = root[1:]
		if parent[-1] == "(":
			if is_struct(parent):
				struct.append(actn_v.toidx(parent))
				if parent in ["DRS(", "SDRS("]:
					relation.append([])
					for c in child:
						if not is_struct(c[0]):
							if re.match("^\$[0-9]+\($",c[0]):
								relation[-1].append(c[0])
							else:
								relation[-1].append(actn_v.toidx(c[0]))
							variable.append([actn_v.toidx(cc) for cc in c[1:]] + [actn_v.toidx(")")])
					relation[-1].append(actn_v.toidx(")"))
				for c in child:
					travel(c)
				struct.append(actn_v.toidx(")"))
			else:
				pass
				#if re.match("^\$[0-9]+\($",parent):
				#	relation[-1].append(parent)
				#else:
				#	relation[-1].append(actn_v.toidx(parent))
				
		else:
			print parent
			assert False
	
	travel(tree)
	#struct.append(actn_v.toidx("<END>"))
	return [struct, relation, variable]

	
def output2action(train_output, actn_v):
	"""
	get action index in actn_v
	"""
	train_action = []
	for output in train_output:
		train_action.append([])
		for a in output:
			idx = actn_v.toidx(a)
			train_action[-1].append(idx)
	return train_action, actn_v



def output2action_cpy(train_output, actn_v):
	"""
	get action index in actn_v if action is not for copy
	remain action string  if action is for copy,
	"""
	import re
	cpy1_p = re.compile("^\$[0-9]+$")
	train_action = []
	for output in train_output:
		train_action.append([])
		for a in output:
			if a in ["B", "X", "E", "S", "T", "P"]:
				train_action[-1].append([a, actn_v.toidx(a)])
			elif cpy1_p.match(a):
				train_action[-1].append(a)
			else:
				idx = actn_v.toidx(a)
				train_action[-1].append(idx)
	return train_action

def get_same_lemma(lemmas):
	"""
	get matrix that indicates which lemmas are the same
	"""
	comb = []
	for vi in lemmas:
		comb.append([])
		for j, vj in enumerate(lemmas):
			if vi == vj:
				comb[-1].append(j)
	return comb

def get_same_lemma_doc(lemmas):
	comb = []
	all_lemmas = []
	for lemma in lemmas:
		all_lemmas += lemma[1:-1] # alway not for <s> and </s>

	past_lemmas = []
	for vi in all_lemmas:
		if vi in past_lemmas:
			continue
		past_lemmas.append(vi)
		comb.append([])
		for i in range(len(lemmas)):
			for j, vj in enumerate(lemmas[i]):
				if vi == vj:
					comb[-1].append((i,j))
	return comb

def get_k_scope(output, actn_v):
	stack = []
	k_scope = {}
	sdrs_idx = 0
	for act in output:
		act_s = actn_v.totok(act)
		if act_s[-1] == "(":
			if act_s == "SDRS(":
				stack.append([sdrs_idx,[]])
				sdrs_idx += 1
			elif re.match("^K[0-9]+\($", act_s):
				stack.append([1000+int(act_s[1:-1])-1, []])
			else:
				stack.append([-1,[]])
		elif actn_v.totok(act) == ")":
			b = stack.pop()
			if b[0] != -1 and b[0] < 1000:
				k_scope[b[0]] = b[1]
			if len(stack) > 0:
				stack[-1][1] = stack[-1][1] + b[1]
			if b[0] >= 1000:
				stack[-1][1].append(b[0]%1000)
	return k_scope

def get_p_max(output, actn_v):
	p_max = -1
	for act in output:
		if re.match("^P[0-9]+\($", actn_v.totok(act)):
			p_max = max(p_max, int(actn_v.totok(act)[1:-1])-1)

	return p_max
