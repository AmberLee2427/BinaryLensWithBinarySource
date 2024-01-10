import numpy as np
import VBBinaryLensing

def VBBL_magnification(self,t,p=None,u=None,LD=None):

	#if self.use_lens_orbital_motion:
	#	raise NotImplementedError('Orbital motion not implemented for VBBL.')

	if self.debug:
		print('va')

	if LD==None:
		LD = self.limb_constant

	if self.VBBL is None:
		self.VBBL = VBBinaryLensing.VBBinaryLensing()
		self.VBBL.RelTol = self.VBBL_RelTol
		self.VBBL.a1 = LD

	self.VBBL.a1 = LD
	if self.use_limb_darkening:
		self.VBBL.a1 = p[self.limb_index]
	
	if self.debug:
		print('vb')

	if p is None:
		p = self.p.copy()

	logd, logq, logrho, u0, phi, t0, tE = p[:7]
	s = 10.0**logd
	q = 10.0**logq
	rho = 10.0**logrho

	#ln10 = np.log(10)
	if self.debug:
		print('vc')

	if u is not None:
		u1, u2 = u
		delta_d = np.zeros_like(t)
	else:
		u1, u2, delta_phi, delta_d = self.trajectory(t,p)

	#params = [ln10*logd, ln10*logq, u0, np.pi-phi, ln10*logrho, np.log(tE), t0]
	#mag = self.VBBL.BinaryLightCurve(params, t, u1, u2)

	mag = np.zeros_like(t)

	for i in range(len(t)):
		mag[i] = self.VBBL.BinaryMag2(s+delta_d[i], q, u1[i], u2[i], rho)

	return np.array(mag)

