import numpy as np


def limb_darkening_ln_prior_prob(self,p):
	lp = 0.0
	f = p[self.limb_index]
	if  f < 0.0:
		return -np.inf
	if f > 1.0:
		return -np.inf
	return lp



def add_limb_darkening(self,limb_constant=None):

	""" Setup for modelling limb darkening. """

	if limb_constant is None:
		limb_constant = self.limb_constant

	self.limb_index = self.dims
	self.dims += 1
	self.use_limb_darkening = True
	self.parameter_labels.append(r"$\Gamma$")
	self.p = np.hstack((self.p,limb_constant))
	self.p_sig = np.hstack((self.p_sig,0.001))
	self.freeze = np.hstack((self.freeze,np.zeros(1,dtype=int)))

def add_flux_parameters(self):

	""" Treat the linear flux parameters as MCMC variables. """

	self.flux_index = {}
	chi2, a0, a1, _, _, _ = self.chi2_calc(self.p)

	self.treat_flux_parameters_as_nonlinear = True

	for site in self.data:

		self.flux_index[site] = self.dims

		if self.data_type[site] == 0:
			self.p = np.hstack((self.p,a0[site]))
			self.p_sig = np.hstack((self.p_sig,0.001))
			self.freeze = np.hstack((self.freeze,np.zeros(1,dtype=int)))
			self.parameter_labels.append(r"$f_S[%s]$"%site)
			self.dims += 1
		else:
			self.p = np.hstack((self.p,a0[site],a1[site]))
			self.p_sig = np.hstack((self.p_sig,0.001,0.001))
			self.freeze = np.hstack((self.freeze,np.zeros(2,dtype=int)))
			self.parameter_labels.append(r"$f_S[%s]$"%site)
			self.parameter_labels.append(r"$f_B[%s]$"%site)
			self.dims += 2

