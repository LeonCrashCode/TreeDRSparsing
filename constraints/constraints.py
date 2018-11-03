
class struct_constraints:
	def __init__(self, actn_v, args):
		self.actn_v = actn_v
		self.size = actn_v.size()
		self.args = args

		self.start = actn_v.toidx("<START>")
		self.end = actn_v.toidx("<END>")
		self.sep = actn_v.toidx(")")

		self.DRS = actn_v.toidx("DRS(")
		self.SDRS = actn_v.toidx("SDRS(")
		self.NOT = actn_v.toidx("NOT(")
		self.NEC = actn_v.toidx("NEC(")
		self.POS = actn_v.toidx("POS(")
		self.IMP = actn_v.toidx("IMP(")
		self.DUP = actn_v.toidx("DUP(")
		self.OR = actn_v.toidx("OR(")

		self.pb_start = actn_v.toidx("P1(")
		self.pb_end = actn_v.toidx("P"+str(args.P_l)+"(")
		self.kb_start = actn_v.toidx("K1(")
		self.kb_end = actn_v.toidx("K"+str(args.K_l)+"(")

		self.k_relation_offset = 0
		self.drs_offset = 1

		self.reset()
	def reset(self):
		self.stack = [self.actn_v.toidx("<START>")]
		self.stack_ex = [[0 for i in range(2)]]

		self.k = 0
		self.p = 0
		self.drs_c = 0

	def isterminal(self):
		return len(self.stack) == 1

	def get_step_mask(self):
		if self.stack[-1] == 0:
			#SOS
			return self._get_sos_mask()
		elif self.stack[-1] == self.SDRS:
			#SDRS
			return self._get_sdrs_mask()
		elif self.stack[-1] == self.DRS:
			#DRS
			return self._get_drs_mask()
		elif self.stack[-1] in [self.NOT, self.NEC, self.POS]:
			#not, nec, pos
			return self._get_1_mask()
		elif self.stack[-1] in [self.OR, self.IMP, self.DUP]:
			#or, imp, duplex
			return self._get_2_mask()
		elif self.stack[-1] == self.kb_start or self.stack[-1] == self.pb_start:
			#k p
			return self._get_1_mask()
		else:
			assert False
	def _get_sos_mask(self):
		re = self._get_zero(self.size)
		self._assign(re, self.DRS, self.DRS, 1)
		self._assign(re, self.SDRS, self.SDRS, 1)
		return re
	def _get_sdrs_mask(self):
		#SDRS
		if self.stack_ex[-1][self.k_relation_offset] < 2:
			#only k
			re = self._get_zero(self.size)
			self._assign(re, self.kb_start + self.k, self.kb_start + self.k, 1)
			return re
		else:
			#only reduce
			re = self._get_zero(self.size)
			self._assign(re, self.sep, self.sep, 1)
			#soft
			if self.args.drs_l - self.drs_c >= 1:
				cnt = 0
				for i in range(len(self.stack)-1):
					if self.stack[i] == self.SDRS:
						cnt += max(0, 1 - self.stack_ex[i][self.k_relation_offset])
				if self.k < self.args.K_l - cnt:
					self._assign(re, self.kb_start + self.k, self.kb_start + self.k, 1)
			return re
	def _get_drs_mask(self):
		re = self._get_zero(self.size)
		self._assign(re, self.sep, self.sep, 1)

		if self.args.drs_l - self.drs_c >= 1:
			if self.p < self.args.P_l:
				self._assign(re, self.pb_start + self.p, self.pb_start + self.p, 1)
			self._assign(re, self.NOT, self.NOT, 1)
			self._assign(re, self.NEC, self.NEC, 1)
			self._assign(re, self.POS, self.POS, 1)
		if self.args.drs_l - self.drs_c >= 2:
			self._assign(re, self.IMP, self.IMP, 1)
			self._assign(re, self.DUP, self.DUP, 1)
			self._assign(re, self.OR, self.OR, 1)
		return re

	def _get_1_mask(self):
		re = self._get_zero(self.size)
		if self.stack_ex[-1][self.drs_offset] == 0:
			self._assign(re, self.DRS, self.DRS, 1)
			if self.args.drs_l - self.drs_c >= 2:
				cnt = 0
				for i in range(len(self.stack)-1):
					if self.stack[i] == self.SDRS:
						cnt += max(0, 1 - self.stack_ex[i][self.k_relation_offset])
				if self.args.K_l - self.k - 2 >= cnt:
					self._assign(re, self.SDRS, self.SDRS, 1)
		else:
			self._assign(re, self.sep, self.sep, 1)
		return re
	def _get_2_mask(self):
		re = self._get_zero(self.size)
		if self.stack_ex[-1][self.drs_offset] == 0:
			self._assign(re, self.DRS, self.DRS, 1)
			if self.args.drs_l - self.drs_c >= 3:
				cnt = 0
				for i in range(len(self.stack)-1):
					if self.stack[i] == self.SDRS:
						cnt += max(0, 1 - self.stack_ex[i][self.k_relation_offset])
				if self.args.K_l - self.k - 2 >= cnt:
					self._assign(re, self.SDRS, self.SDRS, 1)
		elif self.stack_ex[-1][self.drs_offset] == 1:
			self._assign(re, self.DRS, self.DRS, 1)
			if self.args.drs_l - self.drs_c >= 2:
				cnt = 0
				for i in range(len(self.stack)-1):
					if self.stack[i] == self.SDRS:
						cnt += max(0, 1 - self.stack_ex[i][self.k_relation_offset])
				if self.args.K_l - self.k - 2 >= cnt:
					self._assign(re, self.SDRS, self.SDRS, 1)
		else:
			self._assign(re, self.sep, self.sep, 1)
		return re
	def update(self, ix):
		if ix in [self.DRS, self.SDRS, self.NOT, self.NEC, self.POS, self.IMP, self.DUP, self.OR]:
			self.stack.append(ix)
			if ix == self.DRS:
				self.drs_c += 1
			self.stack_ex.append([0 for i in range(2)])
		elif ix >= self.kb_start and ix <= self.kb_end:
			self.stack.append(self.kb_start)
			self.stack_ex.append([0 for i in range(2)])
			self.k += 1
		elif ix >= self.pb_start and ix <= self.pb_end:
			self.stack.append(self.pb_start)
			self.stack_ex.append([0 for i in range(2)])
			self.p += 1
		elif ix == self.sep:
			self.stack_ex.pop()
			if self.stack[-1] == self.DRS or self.stack[-1] == self.SDRS:
				self.stack_ex[-1][self.drs_offset] += 1
			elif self.stack[-1] in [self.NOT, self.NEC, self.POS, self.IMP, self.DUP, self.OR]:
				pass
			elif self.stack[-1] == self.kb_start:
				self.stack_ex[-1][self.k_relation_offset] += 1
			elif self.stack[-1] == self.pb_start:
				pass
			else:
				assert False
			self.stack.pop()
		elif ix == 1:
			pass
		else:
			assert False

	def _print_state(self):
		print "stack", [self.actn_v.totok(x) for x in self.stack]
		print "stack_ex", self.stack_ex
		print "kp", self.k, self.p
		print "drs", self.drs_c
	def _get_one(self, size):
		return [1 for i in range(size)]
	def _get_zero(self, size):
		return [0 for i in range(size)]
	def _assign(self, m, s, e, v):
		i = s
		while i <=e:
			m[i] = v
			i += 1


class relation_constraints:
	def __init__(self, actn_v, args, starts, ends):
		self.actn_v = actn_v
		self.size = actn_v.size()
		### BOX DISCOURSE RELATION PREDICATE
		self.starts = starts
		self.ends = ends
		self.args = args

		self.DRS = actn_v.toidx("DRS(")
		self.SDRS = actn_v.toidx("SDRS(")
		self.sep = actn_v.toidx(")")

	def reset_length(self, copy_length):
		self.copy_length = copy_length
		self.rel_g = 0
		self.d_rel_g = 0
	def reset_condition(self, cond):
		self.cond = cond
		self.rel = 0
		self.d_rel = 0
		self.sep_exist = False

	def isterminal(self):
		return self.sep_exist

	def get_step_mask(self):
		BOX = 0
		DISCOURSE = 1
		RELATION = 2
		PREDICATE = 3

		re = self._get_zero(self.size + self.copy_length)
		if self.cond == self.DRS:
			if self.rel == 0:
				self._assign(re, self.starts[RELATION], self.ends[RELATION], 1)
				self._assign(re, self.starts[PREDICATE], self.ends[PREDICATE], 1)
				self._assign(re, self.size, self.size+self.copy_length-1, 1)
			else:
				if self.args.rel_l - self.rel > 0 and self.args.rel_g_l - self.rel_g > 0:
					self._assign(re, self.starts[RELATION], self.ends[RELATION], 1)
					self._assign(re, self.starts[PREDICATE], self.ends[PREDICATE], 1)
					self._assign(re, self.size, self.size+self.copy_length-1, 1)
				self._assign(re, self.sep, self.sep, 1)
		elif self.cond == self.SDRS:
			if self.d_rel == 0:
				self._assign(re, self.starts[DISCOURSE], self.ends[DISCOURSE], 1)
			else:
				if self.args.d_rel_l - self.d_rel > 0 and self.args.d_rel_g_l - self.d_rel_g > 0:
					self._assign(re, self.starts[DISCOURSE], self.ends[DISCOURSE], 1)
				self._assign(re, self.sep, self.sep, 1)
		else:
			assert False
		return re
	
	def update(self, ix):
		if ix == self.sep:
			self.sep_exist = True
		elif self.cond == self.DRS:
			self.rel += 1
			self.rel_g += 1
		elif self.cond == self.SDRS:
			self.d_rel += 1
			self.d_rel_g += 1
		else:
			assert False

	def _print_state(self):
		print "cond", self.actn_v.totok(self.cond)
		print "rel g_rel", self.rel, self.rel_g
		print "d_rel d_rel_g", self.d_rel, self.d_rel_g
	def _get_one(self, size):
		return [1 for i in range(size)]
	def _get_zero(self, size):
		return [0 for i in range(size)]
	def _assign(self, m, s, e, v):
		i = s
		while i <=e:
			m[i] = v
			i += 1

class variable_constraints:
	def __init__(self, actn_v, args):
		self.actn_v = actn_v
		self.size = actn_v.size()
		self.args = args

		self.DRS = actn_v.toidx("DRS(")
		self.SDRS = actn_v.toidx("SDRS(")
		self.sep = actn_v.toidx(")")

		self.x_start = actn_v.toidx("X1")
		self.e_start = actn_v.toidx("E1")
		self.s_start = actn_v.toidx("S1")
		self.t_start = actn_v.toidx("T1")
		self.p_start = actn_v.toidx("P1")
		self.k_start = actn_v.toidx("K1")

		self.equ = actn_v.toidx("Equ(")
		self.CARD = actn_v.toidx("CARD_NUMBER")
		self.TIME = actn_v.toidx("TIME_NUMBER")
		self.CARD_b = actn_v.toidx("Card(")
		self.TIME_b = actn_v.toidx("Timex(")

	def reset(self, p_max):
		self.p_max = p_max
		self.x = 0
		self.e = 0
		self.s = 0
		self.t = 0

	def reset_condition(self, cond, k_scope=None):
		self.cond = cond
		self.k_scope = k_scope

	def reset_relation(self, rel):
		self.rel = rel
		self.prev_v = -1
		self.prev_prev_v = -1
		self.sep_exist = False

	def isterminal(self):
		return self.sep_exist

	def get_step_mask(self):
		if self.cond == self.DRS:
			return self.get_drs_mask()
		elif self.cond == self.SDRS:
			return self.get_sdrs_mask()
		else:
			assert False

	def _assign_all_v(self,re):
		self._assign(re, self.x_start, self.x_start + self.x, 1)
		self._assign(re, self.e_start, self.e_start + self.e, 1)
		self._assign(re, self.s_start, self.s_start + self.s, 1)
		self._assign(re, self.t_start, self.t_start + self.t, 1)
	def get_drs_mask(self):
		if self.prev_v == -1:
			re = self._get_zero(self.size)
			self._assign_all_v(re)
			if self.p_max >= 0:
				self._assign(re, self.p_start, self.p_start + self.p_max, 1)
			return re
		elif self.prev_prev_v == -1:
			re = self._get_zero(self.size)
			if self.rel == self.CARD_b:
				self._assign(re, self.CARD, self.CARD, 1)
			elif self.rel == self.TIME_b:
				self._assign(re, self.TIME, self.TIME, 1)
			else:
				self._assign_all_v(re)
				if self.p_max >= 0:
					self._assign(re, self.p_start, self.p_start + self.p_max, 1)
				self._assign(re, self.sep, self.sep, 1)

			if self.rel != self.equ:
				self._assign(re, self.prev_v, self.prev_v, 0)
			return re
		else:
			re = self._get_zero(self.size)
			self._assign(re, self.sep, self.sep, 1)
			return re
	def get_sdrs_mask(self):
		if self.prev_v == -1:
			re = self._get_zero(self.size)
			for k in self.k_scope:
				self._assign(re, self.k_start + k, self.k_start + k, 1)
			return re
		elif self.prev_prev_v == -1:
			re = self._get_zero(self.size)
			for k in self.k_scope:
				self._assign(re, self.k_start + k, self.k_start + k, 1)
			self._assign(re, self.prev_v, self.prev_v, 0)
			return re
		else:
			re = self._get_zero(self.size)
			self._assign(re, self.sep, self.sep, 1)
			return re
	def update(self, ix):
		if ix == self.sep:
			self.sep_exist = True
		elif ix == self.x_start + self.x:
			self.x += 1
		elif ix == self.e_start + self.e:
			self.e += 1
		elif ix == self.s_start + self.s:
			self.s += 1
		elif ix == self.t_start + self.t:
			self.t += 1
		self.prev_prev_v = self.prev_v
		self.prev_v = ix

	def _print_state(self):
		print "cond", self.actn_v.totok(self.cond)
		if self.rel >= self.size:
			print "rel", "$"+str(self.rel - self.size)+"("
		else:
			print "rel", self.actn_v.totok(self.rel)
		print "p_max", self.p_max
		print "k_scope", self.k_scope
		print "xest", self.x, self.e, self.s, self.t
		print "prev_prev_v prev_v", self.prev_prev_v, self.prev_v
	def _get_one(self, size):
		return [1 for i in range(size)]
	def _get_zero(self, size):
		return [0 for i in range(size)]
	def _assign(self, m, s, e, v):
		i = s
		while i <=e:
			m[i] = v
			i += 1
