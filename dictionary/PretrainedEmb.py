from vocabulary import vocabulary
class PretrainedEmb:
	def __init__(self, filename):
		self._ttv = [[]] #leave a space for UNK
		self._v = vocabulary()
		self.read_file(filename)
	def read_file(self, filename):
		if not filename:
			return
		with open(filename, "r") as r:
			while True:
				l = r.readline().strip()
				if not l:
					break
				l = l.split()
				idx = self._v.toidx(l[0])
				assert idx == len(self._ttv)
				self._ttv.append([float(t) for t in l[1:]])
				if len(self._ttv[0]) == 0:
					self._ttv[0] = [ 0.0 for i in range(len(l)-1)]
				for i in range(len(self._ttv[0])):
					self._ttv[0][i] += float(l[i+1])
		for i in range(len(self._ttv[0])):
			self._ttv[0][i] /= (len(self._ttv) - 1)
		self._v.freeze()
	def toidx(self, tok):
		return self._v.toidx(tok)
	def size(self):
		return self._v.size()
	def vectors(self):
		return self._ttv