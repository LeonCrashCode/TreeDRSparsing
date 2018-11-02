
class vocabulary:
	def __init__(self, UNK=True):
		if UNK:
			self._tti = {"<UNK>":0}
			self._itt = ["<UNK>"]
		else:
			self._tti = {}
			self._itt = []
		self._frozen = False
	def freeze(self):
		self._frozen = True
	def unfreeze(self):
		self._frozen = False
	def read_file(self, filename):
		with open(filename, "r") as r:
			while True:
				l = r.readline().strip()
				if l.startswith("###"):
					continue
				if not l:
					break
				self.toidx(l)
	def toidx(self, tok):
		if tok in self._tti:
			return self._tti[tok]

		if self._frozen == False:
			self._tti[tok] = len(self._itt)
			self._itt.append(tok)
			return len(self._itt) - 1
		else:
			return 0
	def totok(self, idx):
		assert idx < self.size, "Out of Vocabulary"
		return self._itt[idx]
	def size(self):
		return len(self._tti)
	def dump(self, filename):
		with open(filename, "w") as w:
			for tok in self._itt:
				w.write(tok+"\n")
			w.flush()
			w.close()

		#for i, tok in enumerate(self._itt, 0):
		#	print i, tok

