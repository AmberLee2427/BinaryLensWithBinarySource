import numpy as np

def add_lens_orbital_motion(self,theta_E=None,delta_theta_E=None,source_distance=8.0):

	""" Setup for modelling 2-parameter lens orbital motion. """

	self.dddt_index = self.dims
	self.dphidt_index = self.dims+1
	self.dims += 2
	self.use_lens_orbital_motion = True
	self.parameter_labels.append(r"$\dot{s} \, (\rm{yr}^{-1})$")
	self.parameter_labels.append(r"$\dot{\alpha} \, (\rm{yr}^{-1})$")
	self.p = np.hstack((self.p,np.zeros(2)))
	self.p_sig = np.hstack((self.p_sig,0.0001*np.ones(2)))
	self.freeze = np.hstack((self.freeze,np.zeros(2,dtype=int)))
	self.lens_orbital_motion_plot_delta = None
	self.lens_orbital_motion_reference_date = None
	self.source_distance = source_distance
	if theta_E is not None:
		self.theta_E = theta_E
		if delta_theta_E is None:
			print('Error: value required for delta_theta_E in add_lens_orbital_motion.')
			sys.exit()
		self.delta_theta_E = delta_theta_E
		self.source_distance = source_distance
		self.use_lens_orbital_motion_energy_constraint = True


def add_lens_orbital_acceleration(self):

	""" Setup for modelling 2-parameter lens orbital motion. """
	if not self.use_lens_orbital_motion:
		self.add_lens_orbital_motion()
	self.d2ddt2_index = self.dims
	self.d2phidt2_index = self.dims+1
	self.dims += 2
	self.use_lens_orbital_acceleration = True
	self.parameter_labels.append(r"$\ddot{s} \, (\rm{yr}^{-2})$")
	self.parameter_labels.append(r"$\ddot{\alpha} \, (\rm{yr}^{-2})$")
	self.p = np.hstack((self.p,np.zeros(2)))
	self.p_sig = np.hstack((self.p_sig,0.00001*np.ones(2)))
	self.freeze = np.hstack((self.freeze,np.zeros(2,dtype=int)))

def lens_orbital_motion_energy_prior(self,p):
	pi_E2 = p[self.Pi_EE_index]**2 + p[self.Pi_EN_index]**2
	pi_E = np.sqrt(pi_E2)
	d = 10.0**p[0]
	pi_S = 1.0/self.source_distance
	gamma2 = (p[self.dddt_index]/d)**2 + p[self.dphidt_index]**2
	gamma2_max = 9.644 * (self.theta_E/(d**3 *pi_E)) * (pi_E + pi_S/self.theta_E)**3
	grad = -3.0 * 9.644 * (self.theta_E/(d**3 * pi_E)) * (pi_S/self.theta_E**2) * (pi_E + pi_S/self.theta_E)**2 + \
					(9.644/(d**3 * pi_E)) *  (pi_E + pi_S/self.theta_E)**3
	var_gamma2_max = grad**2 * self.delta_theta_E**2
	if gamma2 > gamma2_max:
		chi2_penalty = (gamma2 - gamma2_max)**2 / var_gamma2_max
		print('energy constraint chi2 penalty = ',chi2_penalty,'for',p)
		return -chi2_penalty/2.0
	return 0.0


def grid_search_lens_orbital_motion(self,dddt,dphidt):


	n_dddt = dddt.shape[0]
	n_dphidt = dphidt.shape[0]

	chi2 = np.empty((n_dddt,n_dphidt))

	for i in range(n_dddt):

		self.p[self.dddt_index] = dddt[i]

		for j in range(n_dphidt):

			self.p[self.dphidt_index] = dphidt[j]

			chi2[i,j], _, _, _, _, _ = self.chi2_calc()

	return chi2

