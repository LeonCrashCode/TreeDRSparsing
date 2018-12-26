class struct_constraints_state:
	def __init__(self):
		self.reset()
	def reset(self):
		self.stack = [0]
		self.stack_ex = [[0 for i in range(3)]]

		self.k = 0
		self.p = 0
		self.b = 0
		self.drs_c = 0

		self.bracket = 0
		self.init = True
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

		self.pb = actn_v.toidx("@P(")
		self.kb = actn_v.toidx("@K(")

		self.b_0 = actn_v.toidx("B0")
		self.b_s = actn_v.toidx("B1")
		self.b_e = actn_v.toidx("B"+str(args.B_l))

		self.k_relation_offset = 0
		self.drs_offset = 1
		self.pointer_offset = 2
		
	def isterminal(self, state):
		return len(state.stack) == 1 and state.init == False

	def bracket_completed(self, state):
		return state.bracket == 0 and state.init == False

	def get_step_mask(self, state):
		if state.stack[-1] == 0:
			#SOS
			return self._get_sos_mask(state)
		elif state.stack[-1] == self.SDRS:
			#SDRS
			return self._get_sdrs_mask(state)
		elif state.stack[-1] == self.DRS:
			#DRS
			return self._get_drs_mask(state)
		elif state.stack[-1] in [self.NOT, self.NEC, self.POS]:
			#not, nec, pos
			return self._get_1p_mask(state)
		elif state.stack[-1] in [self.OR, self.IMP, self.DUP]:
			#or, imp, duplex
			return self._get_2p_mask(state)
		elif state.stack[-1] == self.kb:
			#k
			return self._get_1_mask(state)
		elif state.stack[-1] == self.pb:
			#p
			return self._get_1p_mask(state)
		else:
			pass
			#assert False
	def _get_sos_mask(self, state):
		re = self._get_zero(self.size)
		if state.init:
			self._assign(re, self.DRS, self.DRS, 1)
			self._assign(re, self.SDRS, self.SDRS, 1)
		else:
			self._assign(re, self.end, self.end, 1)
		return re
	def _get_sdrs_mask(self, state):
		#SDRS
		if state.stack_ex[-1][self.k_relation_offset] < 2:
			#only k
			re = self._get_zero(self.size)
			self._assign(re, self.kb, self.kb, 1)
			return re
		else:
			#only reduce
			re = self._get_zero(self.size)
			self._assign(re, self.sep, self.sep, 1)
			#soft
			if self.args.drs_l - state.drs_c >= 1:
				cnt = 0
				for i in range(len(state.stack)-1):
					if state.stack[i] == self.SDRS:
						cnt += max(0, 1 - state.stack_ex[i][self.k_relation_offset])
				if state.k < self.args.K_l - cnt:
					self._assign(re, self.kb, self.kb, 1)
			return re
	def _get_drs_mask(self, state):
		re = self._get_zero(self.size)
		self._assign(re, self.sep, self.sep, 1)

		if self.args.drs_l - state.drs_c >= 1:
			if state.p < self.args.P_l:
				self._assign(re, self.pb, self.pb, 1)
			self._assign(re, self.NOT, self.NOT, 1)
			self._assign(re, self.NEC, self.NEC, 1)
			self._assign(re, self.POS, self.POS, 1)
		if self.args.drs_l - state.drs_c >= 2:
			self._assign(re, self.IMP, self.IMP, 1)
			self._assign(re, self.DUP, self.DUP, 1)
			self._assign(re, self.OR, self.OR, 1)
		return re
	def _get_1p_mask(self, state):
		if state.stack_ex[-1][self.pointer_offset] == 0:
			re = self._get_zero(self.size)
			self._assign(re, self.b_0, self.b_0, 1)
			self._assign(re, self.b_s, self.b_s + state.b, 1)
			return re
		else:
			return self._get_1_mask(state)
	def _get_2p_mask(self, state):
		if state.stack_ex[-1][self.pointer_offset] == 0:
                        re = self._get_zero(self.size)
                        self._assign(re, self.b_0, self.b_0, 1)
                        self._assign(re, self.b_s, self.b_s + state.b, 1)
                        return re
                else:
                        return self._get_2_mask(state)
	def _get_1_mask(self, state):
		re = self._get_zero(self.size)
		if state.stack_ex[-1][self.drs_offset] == 0:
			self._assign(re, self.DRS, self.DRS, 1)
			if self.args.drs_l - state.drs_c >= 2:
				cnt = 0
				for i in range(len(state.stack)-1):
					if state.stack[i] == self.SDRS:
						cnt += max(0, 1 - state.stack_ex[i][self.k_relation_offset])
				if self.args.K_l - state.k - 2 >= cnt:
					self._assign(re, self.SDRS, self.SDRS, 1)
		else:
			self._assign(re, self.sep, self.sep, 1)
		return re
	def _get_2_mask(self, state):
		re = self._get_zero(self.size)
		if state.stack_ex[-1][self.drs_offset] == 0:
			self._assign(re, self.DRS, self.DRS, 1)
			if self.args.drs_l - state.drs_c >= 3:
				cnt = 0
				for i in range(len(state.stack)-1):
					if state.stack[i] == self.SDRS:
						cnt += max(0, 1 - state.stack_ex[i][self.k_relation_offset])
				if self.args.K_l - state.k - 2 >= cnt:
					self._assign(re, self.SDRS, self.SDRS, 1)
		elif state.stack_ex[-1][self.drs_offset] == 1:
			self._assign(re, self.DRS, self.DRS, 1)
			if self.args.drs_l - state.drs_c >= 2:
				cnt = 0
				for i in range(len(state.stack)-1):
					if state.stack[i] == self.SDRS:
						cnt += max(0, 1 - state.stack_ex[i][self.k_relation_offset])
				if self.args.K_l - state.k - 2 >= cnt:
					self._assign(re, self.SDRS, self.SDRS, 1)
		else:
			self._assign(re, self.sep, self.sep, 1)
		return re
	def update(self, ix, state):
		state.init = False
		if ix in [self.DRS, self.SDRS, self.NOT, self.NEC, self.POS, self.IMP, self.DUP, self.OR]:
			state.stack.append(ix)
			if ix == self.DRS:
				state.drs_c += 1
			state.stack_ex.append([0 for i in range(3)])
		elif ix == self.kb:
			state.stack.append(self.kb)
			state.stack_ex.append([0 for i in range(3)])
			state.k += 1
		elif ix == self.pb:
			state.stack.append(self.pb)
			state.stack_ex.append([0 for i in range(3)])
			state.p += 1
		elif (ix >= self.b_s and ix <= self.b_e) or ix == self.b_0:
			state.stack_ex[-1][self.pointer_offset] = 1
			if ix == self.b_s + state.b:
				state.b += 1
		elif ix == self.sep:
			state.stack_ex.pop()
			if state.stack[-1] == self.DRS or state.stack[-1] == self.SDRS:
				state.stack_ex[-1][self.drs_offset] += 1
			elif state.stack[-1] in [self.NOT, self.NEC, self.POS, self.IMP, self.DUP, self.OR]:
				pass
			elif state.stack[-1] == self.kb:
				state.stack_ex[-1][self.k_relation_offset] += 1
			elif state.stack[-1] == self.pb:
				pass
			else:
				pass
				#assert False
			state.stack.pop()
		elif ix == 1:
			pass
		else:
			pass
			#assert False

		if ix == self.sep:
			state.bracket -= 1
		else:
			state.bracket += 1

	def _print_state(self, state):
		print "stack", [self.actn_v.totok(x) for x in state.stack]
		print "stack_ex", state.stack_ex
		print "kp", state.k, state.p
		print "drs", state.drs_c
	def _get_one(self, size):
		return [1 for i in range(size)]
	def _get_zero(self, size):
		return [0 for i in range(size)]
	def _assign(self, m, s, e, v):
		i = s
		while i <=e:
			m[i] = v
			i += 1

class relation_constraints_state:
	def __init__(self):
		pass
	def reset(self):
		self.rel_g = 0
		self.d_rel_g = 0
	def reset_length(self, copy_length):
		self.copy_length = copy_length
	def reset_condition(self, cond, cond2):
		self.cond = cond
		self.cond2 = cond2
		self.rel = 0
		self.d_rel = 0
		self.sep_exist = False

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

	def isterminal(self, state):
		return state.sep_exist

	def get_step_mask(self, state):
		BOX = 0
		DISCOURSE = 1
		RELATION = 2
		PREDICATE = 3

		re = self._get_zero(self.size + state.copy_length)
		if state.cond == self.DRS:
			if state.rel == 0:
				self._assign(re, self.starts[RELATION], self.ends[RELATION], 1)
				self._assign(re, self.starts[PREDICATE], self.ends[PREDICATE], 1)
				self._assign(re, self.size, self.size+state.copy_length-1, 1)
				if state.cond2 != None and state.cond2 == self.sep:
					pass
				else:
					self._assign(re, self.sep, self.sep, 1)
			else:
				if self.args.rel_l - state.rel > 0 and self.args.rel_g_l - state.rel_g > 0:
					self._assign(re, self.starts[RELATION], self.ends[RELATION], 1)
					self._assign(re, self.starts[PREDICATE], self.ends[PREDICATE], 1)
					self._assign(re, self.size, self.size+state.copy_length-1, 1)
				self._assign(re, self.sep, self.sep, 1)
		elif state.cond == self.SDRS:
			if state.d_rel == 0:
				self._assign(re, self.starts[DISCOURSE], self.ends[DISCOURSE], 1)
			else:
				if self.args.d_rel_l - state.d_rel > 0 and self.args.d_rel_g_l - state.d_rel_g > 0:
					self._assign(re, self.starts[DISCOURSE], self.ends[DISCOURSE], 1)
				self._assign(re, self.sep, self.sep, 1)
		else:
			pass
			#assert False
		return re
	
	def update(self, ix, state):
		if ix == self.sep:
			state.sep_exist = True
		elif state.cond == self.DRS:
			state.rel += 1
			state.rel_g += 1
		elif state.cond == self.SDRS:
			state.d_rel += 1
			state.d_rel_g += 1
		else:
			pass
			#assert False

	def _print_state(self, state):
		print "cond", self.actn_v.totok(state.cond)
		print "rel g_rel", state.rel, state.rel_g
		print "d_rel d_rel_g", state.d_rel, state.d_rel_g
	def _get_one(self, size):
		return [1 for i in range(size)]
	def _get_zero(self, size):
		return [0 for i in range(size)]
	def _assign(self, m, s, e, v):
		i = s
		while i <=e:
			m[i] = v
			i += 1
class variable_constraints_state:
	def __init__(self):
		pass
	def reset(self, p_max, b_max, length):
		self.p_max = p_max
		self.x = 0
		self.e = 0
		self.s = 0
		self.t = 0
		self.b = b_max
		self.copy_length = length
	def reset_condition(self, cond, k_scope=None):
		self.cond = cond
		self.k_scope = k_scope

	def reset_relation(self, rel):
		self.rel = rel
		self.prev_v = -1
		self.sep_exist = False

		# 0: (empty)
		# 1: B0
		# 2: B0 var1
		# 3: B0 var1 var2
		# 4: B0 var1 $1
		# 5: B0 var1 sense
		if self.k_scope == None:
			self.format = 0
		else:
			self.format = 6

class variable_constraints:
	def __init__(self, actn_v, args, sense_s, sense_e):
		self.actn_v = actn_v
		self.size = actn_v.size()
		self.args = args

		self.DRS = actn_v.toidx("DRS(")
		self.SDRS = actn_v.toidx("SDRS(")
		self.sep = actn_v.toidx(")")

		self.x_s = actn_v.toidx("X1")
		self.e_s = actn_v.toidx("E1")
		self.s_s = actn_v.toidx("S1")
		self.t_s = actn_v.toidx("T1")
		self.p_s = actn_v.toidx("P1")
		self.k_s = actn_v.toidx("K1")
		self.b_s = actn_v.toidx("B1")

		self.b_0 = actn_v.toidx("B0")

		self.equ = actn_v.toidx("Equ(")
		#self.CARD = actn_v.toidx("CARD_NUMBER")
		#self.TIME = actn_v.toidx("TIME_NUMBER")
		#self.CARD_b = actn_v.toidx("Card(")
		#self.TIME_b = actn_v.toidx("Timex(")

		self.sense_s = sense_s
		self.sense_e = sense_e

	def isterminal(self, state):
		return state.sep_exist

	def get_step_mask(self, state):
		#if self.args.soft_const:
		#	return self.get_mask(state)

		if state.cond == self.DRS:
			return self.get_drs_mask(state)
		elif state.cond == self.SDRS:
			return self.get_sdrs_mask(state)
		else:
			pass
			#assert False

	def _assign_all_v(self, re, state):
		self._assign(re, self.x_s, self.x_s + state.x, 1)
		self._assign(re, self.e_s, self.e_s + state.e, 1)
		self._assign(re, self.s_s, self.s_s + state.s, 1)
		self._assign(re, self.t_s, self.t_s + state.t, 1)
	"""
	def get_mask(self,state):
		re = self._get_zero(self.size)
		if state.prev_v == -1:
			self._assign(re, self.x_start, self.x_start + self.args.X_l - 1, 1)
			self._assign(re, self.e_start, self.e_start + self.args.E_l - 1, 1)
			self._assign(re, self.s_start, self.s_start + self.args.S_l - 1, 1)
			self._assign(re, self.t_start, self.t_start + self.args.T_l - 1, 1)
			self._assign(re, self.p_start, self.p_start + self.args.P_l - 1, 1)
			self._assign(re, self.k_start, self.k_start + self.args.K_l - 1, 1)
			self._assign(re, self.CARD, self.CARD, 1)
			self._assign(re, self.TIME, self.TIME, 1)
		elif state.prev_prev_v == -1:
			self._assign(re, self.x_start, self.x_start + self.args.X_l - 1, 1)
                        self._assign(re, self.e_start, self.e_start + self.args.E_l - 1, 1)
                        self._assign(re, self.s_start, self.s_start + self.args.S_l - 1, 1)
                        self._assign(re, self.t_start, self.t_start + self.args.T_l - 1, 1)
                        self._assign(re, self.p_start, self.p_start + self.args.P_l - 1, 1)
                        self._assign(re, self.k_start, self.k_start + self.args.K_l - 1, 1)
			self._assign(re, self.CARD, self.CARD, 1)
			self._assign(re, self.TIME, self.TIME, 1)
			self._assign(re, self.sep, self.sep, 1)
		else:
			self._assign(re, self.sep, self.sep, 1)
		return re
	"""
	def get_drs_mask(self, state):
		re = self._get_zero(self.size + state.copy_length)
		if state.format == 0:
			self._assign(re, self.b_0, self.b_0, 1)
			self._assign(re, self.b_s, self.b_s + state.b, 1)
		elif state.format == 1:
			self._assign_all_v(re, state)
			if state.p_max >= 0:
				self._assign(re, self.p_s, self.p_s + state.p_max, 1)
		elif state.format == 2:
			self._assign_all_v(re, state)
			if state.p_max >= 0:
				self._assign(re, self.p_s, self.p_s + state.p_max, 1)
			self._assign(re, self.size, self.size+state.copy_length-1, 1)
			self._assign(re, self.sense_s, self.sense_e, 1)
			if state.rel != self.equ:
				self._assign(re, state.prev_v, state.prev_v, 0)
		elif state.format == 3 or state.format == 5:
			self._assign(re, self.sep, self.sep, 1)
		elif state.format == 4:
			self._assign(re, self.size, self.size+state.copy_length-1, 1)
			self._assign(re, self.sep, self.sep, 1)
		else:
			assert False, "unrecognized format"
		return re
	def get_sdrs_mask(self, state):
		re = self._get_zero(self.size + state.copy_length)
		if state.format == 6:
			for k in state.k_scope:
				self._assign(re, self.k_s + k, self.k_s + k, 1)
		elif state.format == 7:
			for k in state.k_scope:
				self._assign(re, self.k_s + k, self.k_s + k, 1)
			self._assign(re, state.prev_v, state.prev_v, 0)
		elif state.format == 8:
			self._assign(re, self.sep, self.sep, 1)
		else:
			assert False, "unrecognized format"
		return re
	def update(self, ix, state):
		if ix == self.sep:
			state.sep_exist = True
		elif ix == self.x_s + state.x and state.x + 1 < self.args.X_l:
			state.x += 1
		elif ix == self.e_s + state.e and state.e + 1 < self.args.E_l:
			state.e += 1
		elif ix == self.s_s + state.s and state.s + 1 < self.args.S_l:
			state.s += 1
		elif ix == self.t_s + state.t and state.t + 1 < self.args.T_l:
			state.t += 1
		elif ix == self.b_s + state.b and state.b + 1 < self.args.B_l:
			state.b += 1
		state.prev_v = ix

		if state.format <= 1:
			state.format += 1
		elif state.format == 2:
			if ix >= self.sense_s and ix <= self.sense_e:
				state.format = 5
			elif ix < self.size:
				state.format = 3
			else:
				state.format = 4
		elif state.format in [3, 4, 5]:
			pass
		elif state.format <= 7:
			state.format += 1
		elif state.format == 8:
			pass
		else:
			assert False, "unrecognized format"

	def _print_state(self, state):
		print "cond", self.actn_v.totok(state.cond)
		if state.rel >= self.size:
			print "rel", "$"+str(state.rel - self.size)+"("
		else:
			print "rel", self.actn_v.totok(state.rel)
		print "p_max", state.p_max
		print "k_scope", state.k_scope
		print "xestb", state.x, state.e, state.s, state.t, state.b
		print "prev_v",  state.prev_v
		print "format", state.format
	def _get_one(self, size):
		return [1 for i in range(size)]
	def _get_zero(self, size):
		return [0 for i in range(size)]
	def _assign(self, m, s, e, v):
		i = s
		while i <=e:
			m[i] = v
			i += 1
