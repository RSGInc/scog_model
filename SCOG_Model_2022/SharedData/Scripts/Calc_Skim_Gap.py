#update the skim gap for lov to test stopping criteria...

import numpy
import VisumPy.helpers as h

def calcGap(m1, m2):
	cur_mat = h.GetSkimMatrixRaw(Visum, m1)
	avg_mat = h.GetSkimMatrixRaw(Visum, m2)
	gap = numpy.max(numpy.abs(cur_mat - avg_mat) / avg_mat)
	Visum.Net.SetAttValue("PMPK_SKIM_GAP", gap)

calcGap(103, 100) #new matrix, old matrix
