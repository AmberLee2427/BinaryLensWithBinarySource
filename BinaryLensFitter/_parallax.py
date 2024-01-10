import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from pylab import subplots_adjust
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable, inset_locator

def add_parallax(self, right_ascension, declination, year):

	""" Setup for modelling annual parallax. """

	if self.use_parallax:

		print('Error: parallax already set up. Returning.')
		return

	self.use_parallax = True
	self.Pi_EE_index = self.dims
	self.Pi_EN_index = self.dims + 1
	self.dims += 2
	self.parameter_labels.append(r"$\pi_{E,E}$")
	self.parameter_labels.append(r"$\pi_{E,N}$")
	self.p = np.hstack((self.p,np.zeros(2)))
	self.p_sig = np.hstack((self.p_sig,0.0001*np.ones(2)))
	self.freeze = np.hstack((self.freeze,np.zeros(2,dtype=int)))
	self.right_ascension = right_ascension
	self.declination = declination
	self.vernal, self.peri = self.get_vernal_peri(year)
	self.earth_ecc = 0.0167

	north = np.array([0.0, 0.0, 1.0])
	spring = np.array([1.0, 0.0 ,0.0])		
	summer = np.array([0.0, 0.9174, 0.3971]) 
	
	self.parallax_offset = self.vernal - self.peri
	
	self.parallax_rad = np.array([np.cos(self.right_ascension) * np.cos(self.declination), \
								  np.sin(self.right_ascension) * np.cos(self.declination), \
								  np.sin(self.declination)])
	
	# East direction on lens plane
	self.parallax_east = np.cross(north,self.parallax_rad)
	self.parallax_east /= np.linalg.norm(self.parallax_east)

	# North direction on lens plane
	self.parallax_north = np.cross(self.parallax_rad,self.parallax_east)

	# Perihelion phase
	parallax_phi = (1.0-self.parallax_offset/365.25)*2.0*np.pi
	parallax_psi= self.get_parallax_psi(parallax_phi)

	costh = (np.cos(parallax_psi) - self.earth_ecc) / ( 1.0 - self.earth_ecc*np.cos(parallax_psi) )
	sinth = -np.sqrt(1.0-costh*costh)
	
	# Earth position at perihelion?
	self.parallax_xpos = spring*costh + summer*sinth
	self.parallax_ypos = -spring*sinth + summer*costh

	if not(self.state is None):
		self.state = [np.hstack((self.state[i],np.zeros(2))) for i in range(len(self.state))]

	# Galactic rotation
	galaxy_rotation_ra = 5.550227
	galaxy_rotation_dec = 0.8435095
	self.galaxy_rotation_direction  = np.array([np.cos(galaxy_rotation_ra) * np.cos(galaxy_rotation_dec), \
								  np.sin(galaxy_rotation_ra) * np.cos(galaxy_rotation_dec), \
								  np.sin(galaxy_rotation_dec)])
	self.galaxy_rotation_east = np.dot(self.galaxy_rotation_direction,self.parallax_east)
	self.galaxy_rotation_north = np.dot(self.galaxy_rotation_direction,self.parallax_north)
	galaxy_north_ra = 3.157
	galaxy_north_dec = 0.473 
	self.galaxy_north_direction  = np.array([np.cos(galaxy_north_ra) * np.cos(galaxy_north_dec), \
								  np.sin(galaxy_north_ra) * np.cos(galaxy_north_dec), \
								  np.sin(galaxy_north_dec)])
	solar_pecular_velocity_kms = 12.24 * self.galaxy_rotation_direction + 7.25 * self.galaxy_north_direction
	solar_pecular_velocity_east = np.dot(solar_pecular_velocity_kms,self.parallax_east)
	solar_pecular_velocity_north = np.dot(solar_pecular_velocity_kms,self.parallax_north)
	self.solar_pecular_velocity_kms = np.array([solar_pecular_velocity_north,solar_pecular_velocity_east])

def get_vernal_peri(self, year):

	years = list(range(2005,2021))
	perihelion = [3372.540499087, 3740.165261868, 4104.332037400, 4468.498813419, 4836.123576385, 5199.498764492, \
					5565.290377501, 5931.540278206, 6294.707154200, 6661.998616987, 7026.790330067, 7390.457156155, \
					7758.081919031, 8121.748744610, 8486.707107861, 8853.831920967]
	vernal = [3450.017141061, 3815.262279952, 4180.499085511, 4545.735891070, 4910.983113294, 5276.224779964, \
					5641.467141080, 6006.712279971, 6371.953946642, 6737.200474423, 7102.442141095, 7467.681724434, \
					7832.931029993, 8198.171307778, 8563.409502230, 8928.653946680]

	return vernal[years==year],perihelion[years==year]


def get_parallax_psi(self, parallax_phi):

	parallax_psi = parallax_phi

	for i in range(4):
		fun	= parallax_psi - self.earth_ecc * np.sin(parallax_psi)
		dif	= parallax_phi - fun
		der	= 1.0 - self.earth_ecc * np.cos(parallax_psi)
		parallax_psi += dif / der

	return parallax_psi



def compute_parallax_terms(self, ts_in, t_peak=None):

	"""Return the projected position of the Sun in the lens frame.

	   The mathematical description is given in Gould, A., 2004, ApJ, 606, 319.

	   sun = Earth - Sun cartesian vector in heliocentric frame in units of AU

	   S_n, S_e = N & E components of sun vector projected onto lens plane in AU units

	   v_Earth_perp = N & E components of Earth velocity projected onto lens plane in AU/day units

	   (q_n, q_e) is the projected Sun position in the lens frame in AU, Delta(s_n, s_e) in the above paper


	"""

	if t_peak is None:
		t_peak = self.parallax_t_ref

	sun = np.zeros([3], np.float64)
	S_e_arr	= np.zeros([3], np.float64)
	S_n_arr	= np.zeros([3], np.float64)
						
	# Calculates the eccentric anomaly
	int_adjust	= np.array([0.0, -1.0, 1.0])
	
	for j in range(2, -1, -1):

		parallax_phi = ( (t_peak + int_adjust[j] - self.peri) / 365.25 ) * 2.0*np.pi
		parallax_psi = self.get_parallax_psi(parallax_phi)

		for i in range(3):

			sun[i] = self.parallax_xpos[i] * (np.cos(parallax_psi)-self.earth_ecc) + \
						self.parallax_ypos[i] * np.sin(parallax_psi) * np.sqrt(1.0-self.earth_ecc*self.earth_ecc)
			S_n_arr[j] += sun[i] * self.parallax_north[i]
			S_e_arr[j] += sun[i] * self.parallax_east[i]

	vn0	= (S_n_arr[2] - S_n_arr[1]) / 2.0
	ve0	= (S_e_arr[2] - S_e_arr[1]) / 2.0
	
	self.v_Earth_perp = np.array([-vn0, -ve0])

	sun = np.zeros(([3, len(ts_in)]), np.float64)

	#	Calculate S_n and S_e for the actual time value
	parallax_phi = ( (ts_in - self.peri) / 365.25 ) * (2.0*np.pi)
	parallax_psi = self.get_parallax_psi(parallax_phi)
	q_n = -S_n_arr[0] - vn0 * (ts_in - t_peak)
	q_e = -S_e_arr[0] - ve0 * (ts_in - t_peak)
	
	for i in range(3):

		sun[i,:] = self.parallax_xpos[i] * ( np.cos(parallax_psi) - self.earth_ecc ) + \
					self.parallax_ypos[i] * np.sin(parallax_psi) * np.sqrt( 1-self.earth_ecc*self.earth_ecc)
		q_n += sun[i,:]*self.parallax_north[i]
		q_e += sun[i,:]*self.parallax_east[i]

	return q_n, q_e


def parallax_trajectory(self,piEN,piEE,t1,t2):

	"""Assuming a parallax (piEN,piEE), compute a revised u0, alpha, t0, tE that gives the same source position
	at times t1, t2 as a non-parallax model with self.u0, self.alpha, self.t0, self.tE."""

	logd, logq, logrho, u0, alpha_ref, t0, tE_ref = self.p[:7]

	d = 10.0**logd
	q = 10.0**logq
	a = 0.5*d
	b = a*(1.0-q)/(1.0+q)
	#u0_ref = u0 - np.sin(alpha_ref)*b
	#t0_ref = t0 + tE_ref*b*np.cos(alpha_ref)
	u0_ref = u0
	t0_ref = t0

	if self.parallax_t_ref is None:
		self.parallax_t_ref = t0

	#self.p[self.Pi_EN_index], self.p[self.Pi_EE_index] = self.spitzer_delta_beta_tau_to_parallax(db, dt)

	self.p[self.Pi_EN_index] = piEN
	self.p[self.Pi_EE_index] = piEE

	q_n, q_e = self.compute_parallax_terms(np.array([t1,t2]))
	delta_tau = q_n*self.p[self.Pi_EN_index] + q_e*self.p[self.Pi_EE_index]
	delta_beta = -q_n*self.p[self.Pi_EE_index] + q_e*self.p[self.Pi_EN_index]
	tau1 = delta_tau[0]
	tau2 = delta_tau[1]
	beta1 = delta_beta[0]
	beta2 = delta_beta[1]

	E1 = (t1-t0_ref)*np.cos(alpha_ref)/tE_ref + u0_ref*np.sin(alpha_ref)
	F1 = -(t1-t0_ref)*np.sin(alpha_ref)/tE_ref + u0_ref*np.cos(alpha_ref)
	E2 = (t2-t0_ref)*np.cos(alpha_ref)/tE_ref + u0_ref*np.sin(alpha_ref)
	F2 = -(t2-t0_ref)*np.sin(alpha_ref)/tE_ref + u0_ref*np.cos(alpha_ref)

	print('self.p', self.p)
	print('tau1, tau2', tau1, tau2)
	print('beta1, beta2', beta1, beta2)
	print('E1, F1', E1, F1)
	print('E2, F2', E2, F2)
	print('q_n, q_e', q_n, q_e)

	sqrt_term = np.sqrt((E1-E2)**2+(F1-F2)**2-(beta1-beta2)**2)
	tEarray = np.array([(t1-t2)/(tau2-tau1+sqrt_term),(t1-t2)/(tau2-tau1-sqrt_term)])
	print('tEarray', tEarray)
	tE = tEarray[tEarray>0]

	if len(tE) == 0:
		return -1.0, -1.0, -1.0, -1.0


	tE = tE[0]

	G1 = E1*tE*(beta1-beta2) - F1*(t1-t2+tE*(tau1-tau2))
	G2 = F1*tE*(beta1-beta2) + E1*(t1-t2+tE*(tau1-tau2))
	G3 = 0.5*tE*(E1**2-E2**2+F1**2-F2**2+(beta1-beta2)**2+(tau1-tau2)**2) + 0.5*(t1-t2)**2/tE + (t1-t2)*(tau1-tau2)
	G4 = np.sqrt(G1**2+G2**2-G3**2)

	alpha_array = np.empty((2,))
	alpha_array[0] = np.arctan((G1*G3+G2*G4)/(G2*G3-G1*G4))
	alpha_array[1] = np.arctan((G1*G3-G2*G4)/(G2*G3+G1*G4))
	print('alpha_array', alpha_array)
	#alpha = alpha_array[alpha_array/alpha_ref > 0][0]

	u0_array = np.empty((2,))
	t0_array = np.empty((2,))

	t0_array = ( ( 0.5*tE**2*(E1**2-E2**2+F1**2-F2**2+(beta1-beta2)**2-tau1**2+tau2**2)  - 0.5*(t1**2-t2**2) - tE*(t1*tau1 - t2*tau2) )*np.sin(alpha_array) + \
				(t1*tE + tE**2*tau1)*(beta1-beta2)*np.cos(alpha_array) - E1*tE**2*(beta1-beta2) ) / \
				 ( tE*(beta1-beta2)*np.cos(alpha_array) - (t1 - t2 + tE*(tau1-tau2))*np.sin(alpha_array) )

	if (beta1 - beta2)**2 > 1.e-10:
		u0_array = ( E1**2-E2**2+F1**2-F2**2 -tau1**2+tau2**2 -beta1**2+beta2**2 + (t2**2-t1**2)/tE**2 + 2*(t2*tau2-t1*tau1)/tE + 2*t0_array*(t1-t2)/tE**2 + 2*t0*(tau1-tau2)/tE ) / \
					(2*(beta1-beta2))
	else:
		u0_array = 0.0*alpha_array

	print('t0_array', t0_array)
	print('u0_array', u0_array)

	resid_array = (((t1-t0_array)/tE + tau1)*np.cos(alpha_array) + (u0_array+beta1)*np.sin(alpha_array) - E1)**2 + \
					(-((t1-t0_array)/tE + tau1)*np.sin(alpha_array) + (u0_array+beta1)*np.cos(alpha_array) - F1)**2 + \
					(((t2-t0_array)/tE + tau2)*np.cos(alpha_array) + (u0_array+beta2)*np.sin(alpha_array) - E2)**2 + \
					(-((t2-t0_array)/tE + tau2)*np.sin(alpha_array) + (u0_array+beta2)*np.cos(alpha_array) - F2)**2 

	print('resid_array', resid_array)

	print(np.array([ [ ((t1-t0_array)/tE + tau1)*np.cos(alpha_array) + (u0_array+beta1)*np.sin(alpha_array) - E1, \
							-((t1-t0_array)/tE + tau1)*np.sin(alpha_array) + (u0_array+beta1)*np.cos(alpha_array) - F1 ], \
							[ ((t2-t0_array)/tE + tau2)*np.cos(alpha_array) + (u0_array+beta2)*np.sin(alpha_array) - E2, \
							-((t2-t0_array)/tE + tau2)*np.sin(alpha_array) + (u0_array+beta2)*np.cos(alpha_array) - F2 ] ]))

	pts = np.where(resid_array < 0.002)[0] 

	if len(pts) < 1:
		return -1.0, -1.0, -1.0, -1.0

	alpha = alpha_array[pts]
	u0 = u0_array[pts]
	t0 = t0_array[pts]

	print('alpha, t0, u0', alpha, t0, u0) 


	#residual = np.array([ [ ((t1-t0)/tE + tau1)*np.cos(alpha) + (u0+beta1)*np.sin(alpha) - E1, \
	#						-((t1-t0)/tE + tau1)*np.sin(alpha) + (u0+beta1)*np.cos(alpha) - F1 ], \
		#					[ ((t2-t0)/tE + tau2)*np.cos(alpha) + (u0+beta2)*np.sin(alpha) - E2, \
	#						-((t2-t0)/tE + tau2)*np.sin(alpha) + (u0+beta2)*np.cos(alpha) - F2 ] ])

	return u0, alpha, t0, tE


def search_parallax_trajectory(self,grid_u0,grid_alpha,grid_t0,u_ref,t_ref):

	"""Perform a 3D grid search over (u_0,alpha,t0), optimizing t_0 to minimize the distance
		between the model trajectory evaluated at t_ref and the coordinates u_ref."""

	def dist(p,u_ref,t_ref):

		_, _, _, _ = self.trajectory(t_ref,p=p)
		return np.sum((self.u1-u_ref[0])**2 + (self.u2-u_ref[1])**2)

	p = self.p

	min_dist = 1.e6

	for u0 in grid_u0.tolist():

		p[3] = u0

		for alpha in grid_alpha.tolist():

			p[4] = alpha

			for t0 in grid_t0.tolist():

				p[5] = t0

				dist_p = dist(p,u_ref,t_ref)

				if dist_p < min_dist:

					min_dist = dist_p
					min_u0 = u0
					min_alpha = alpha
					min_t0 = t0

	print('min at u0 =', min_u0, ' alpha =', min_alpha, ' t0 =', min_t0) 

	return min_u0, min_alpha, min_t0


def grid_search_parallax(self,params=None,range_piEE=(-0.5,0.5),range_piEN=(-0.5,0.5),n_piEE=501,n_piEN=501,t_ref=None,residual_tolerance=0.02,refine_all_parameters=False,max_dist=0.03):

	from scipy.optimize import fsolve, root, least_squares, minimize, Bounds
	from scipy.signal import argrelmin
	from time import time
	from Metropolis_Now import Adaptive_Sampler


	def func(pars, pi_EE, pi_EN, b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref):
		if len(pars) == 3:
			u0, alpha, t0 = pars
			tE = tE_ref
		else:
			u0, alpha, t0, tE = pars
		self.p[3] = u0 + b*np.sin(alpha)
		self.p[4] = alpha
		self.p[5] = t0 - tE*b*np.cos(alpha)
		self.p[6] = tE
		q_n, q_e = self.compute_parallax_terms(np.array(t_ref),self.p[5])
		delta_tau = q_n*pi_EN + q_e*pi_EE
		delta_beta = -q_n*pi_EE + q_e*pi_EN

		return [cref*(t_ref[0]-t0_mid_ref)/tE_ref + sref*u0_mid_ref - \
							( np.cos(alpha)*((t_ref[0]-t0)/tE + delta_tau[0]) + np.sin(alpha)*(u0+delta_beta[0]) ), \
				-sref*(t_ref[0]-t0_mid_ref)/tE_ref + cref*u0_mid_ref -  \
							( -np.sin(alpha)*((t_ref[0]-t0)/tE + delta_tau[0]) + np.cos(alpha)*(u0+delta_beta[0]) ), \
				cref*(t_ref[1]-t0_mid_ref)/tE_ref + sref*u0_mid_ref - \
							( np.cos(alpha)*((t_ref[1]-t0)/tE + delta_tau[1]) + np.sin(alpha)*(u0+delta_beta[1]) ), \
				-sref*(t_ref[1]-t0_mid_ref)/tE_ref + cref*u0_mid_ref - \
							( -np.sin(alpha)*((t_ref[1]-t0)/tE + delta_tau[1]) + np.cos(alpha)*(u0+delta_beta[1]) ) ]

	print('grid_search_parallax')

	start = time()

	if params is None:
		params = self.p

	params_ref = params.copy()
	print('params_ref', params_ref)

	self.freeze = np.zeros_like(self.p,dtype=np.int)
	self.freeze[0] = 1
	self.freeze[1] = 1
	self.freeze[2] = 1
	self.freeze[-3] = 1
	self.freeze[-2] = 1
	self.freeze[-1] = 1


	self.plotprefix = 'grid_search_parallax'

	if t_ref is not None:

		logd, logq, logrho, u0_ref, alpha_ref, t0_ref, tE_ref = params[:7]
		cref = np.cos(alpha_ref)
		sref = np.sin(alpha_ref)
		params_ref[5] -= t0_ref

		d = 10.0**logd
		q = 10.0**logq
		rho = 10.0**logrho
		a = 0.5*d
		b = a*(1.0-q)/(1.0+q)
		u0_mid_ref = u0_ref - b*sref
		t0_mid_ref = t0_ref + tE_ref*b*cref

		u0_grid = np.zeros([n_piEE,n_piEN])
		alpha_grid = np.zeros([n_piEE,n_piEN])
		t0_grid = np.zeros([n_piEE,n_piEN])
		tE_grid = np.zeros([n_piEE,n_piEN])
		piEE_grid = np.zeros([n_piEE,n_piEN])
		piEN_grid = np.zeros([n_piEE,n_piEN])

		if refine_all_parameters:
			logd_grid = np.zeros([n_piEE,n_piEN])
			logq_grid = np.zeros([n_piEE,n_piEN])
			logrho_grid = np.zeros([n_piEE,n_piEN])
			pi_EE_grid = np.zeros([n_piEE,n_piEN])
			pi_EN_grid = np.zeros([n_piEE,n_piEN])

		t_ref_array = np.array(t_ref)

		# For searching for other solutions
		t_mid = np.mean(t_ref)
		t_range = t_ref[1] - t_ref[0]
		t_search = np.hstack((np.linspace(t_ref[0]-5.0*tE_ref,t_ref[0]-0.1*tE_ref,200),np.linspace(t_ref[1]+0.1*tE_ref,t_ref[1]+5.0*tE_ref,200)))

		print('reference u0, alpha, t0:', u0_ref, alpha_ref, t0_ref)
		print('reference u0_mid, t0_mid:', u0_mid_ref, t0_mid_ref)

		print('reference u1, u2:', cref*(t_ref[0]-t0_mid_ref)/tE_ref + sref*u0_mid_ref, \
									-sref*(t_ref[0]-t0_mid_ref)/tE_ref + cref*u0_mid_ref, \
									cref*(t_ref[1]-t0_mid_ref)/tE_ref + sref*u0_mid_ref, \
									-sref*(t_ref[1]-t0_mid_ref)/tE_ref + cref*u0_mid_ref)

	chi2 = np.zeros([n_piEE,n_piEN])
	chi2_constrained = np.zeros([n_piEE,n_piEN])

	pi_EE = np.linspace(range_piEE[0],range_piEE[1],n_piEE)
	pi_EN = np.linspace(range_piEN[0],range_piEN[1],n_piEN)

	min_chi2 = 1.e10

	for iE in range(n_piEE):

		for iN in range(n_piEN):

			params = params_ref.copy()
			print('elapsed time', time() - start) 
			print('p', self.p)
			print('iE, iN, pi_EE[iE], pi_EN[iN]', iE, iN, pi_EE[iE], pi_EN[iN])

			scale = np.array([1.0,1.0,t0_mid_ref])

			result = least_squares(func, np.array([u0_mid_ref, alpha_ref, t0_mid_ref]),method='lm', \
						args=(pi_EE[iE], pi_EN[iN], b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref))

			print('3-parameter least squares:')
			print(result)

			if not result.success or result.cost > max_dist:

				chi2[iE,iN] = np.nan
				print('-'*30)
				continue

			u0, alpha, t0 = result.x.tolist()

			scale = np.array([1.0,1.0,t0,tE_ref])

			result = least_squares(func, np.array([u0_mid_ref, alpha_ref, t0_mid_ref, tE_ref]),method='lm', \
						args=(pi_EE[iE], pi_EN[iN], b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref))
			u0, alpha, t0, tE = result.x.tolist()

			print('elapsed time', time() - start) 

			print('4-parameter least squares:')
			print(result)

			if not result.success or result.cost > max_dist:

				chi2[iE,iN] = np.nan
				print('-'*30)
				continue

			print('ref:', [u0_mid_ref, alpha_ref, t0_mid_ref, tE_ref])
			print('solved:', [u0, alpha, t0, tE])
			print('residual:', result.fun)
			print('elapsed time', time() - start) 

			# converting back to CoM coordinates
			params[3] = u0 + b*np.sin(alpha)
			params[4] = alpha
			params[5] = t0 - tE*b*np.cos(alpha)
			params[6] = tE
			print('u0, alpha, t0, tE:', params[3], params[4], params[5], params[6])
			u0_grid[iE,iN] = params[3]
			alpha_grid[iE,iN] = params[4]
			t0_grid[iE,iN] = params[5]
			tE_grid[iE,iN] = params[6]

			params[self.Pi_EE_index] = pi_EE[iE]
			params[self.Pi_EN_index] = pi_EN[iN]
			print('piEE, piEN:', params[self.Pi_EE_index], params[self.Pi_EN_index])
			print('params:', params)

			chi2[iE,iN], _ , _, _, _, _ = self.chi2_calc(params)
			print('chi2:', chi2[iE,iN])

			if refine_all_parameters and chi2[iE,iN] < 1.e5 and tE < 500.0:

				self.p = params
				self.freeze[0] = 0
				self.freeze[1] = 0
				self.freeze[2] = 0
				self.freeze[7] = 1
				self.freeze[8] = 1
				self.freeze[9] = 1
				pv = np.where(self.freeze == 0)[0]

				pp = np.atleast_2d(np.array([params[i] for i in pv.tolist()]))

				iterations = 50
				nsteps = 100
				temperature = 0.03

				sampler = Adaptive_Sampler(ndim=self.dims-np.sum(self.freeze),sigma=self.p_sig[pv]/1.e2,nchains=1, initial_temperature=temperature, \
											ln_prob_fn=self.lnprob,parameter_labels=[self.parameter_labels[i] for i in pv.tolist()])

				for i in range(iterations):

					sampler.iterate_chains(nsteps,start=pp, plot_progress=False,scale_individual_chains=True,minimum_temperature=temperature, \
											min_diag_steps_ratio=5.0)

				max_lnp = np.max(sampler.ln_prob)
				index_max_lnp = np.where(sampler.ln_prob.ravel() == max_lnp)[0][0]

				params[pv] = sampler.chains[index_max_lnp,0,:]


				print('full minimization:')
				print('elapsed time', time() - start) 

				print('refined params:', params)

				chi2[iE,iN], _ , _, _, _, _ = self.chi2_calc(params)
				print('chi2:', chi2[iE,iN])

				logd_grid[iE,iN] = params[0]
				logq_grid[iE,iN] = params[1]
				logrho_grid[iE,iN] = params[2]

				pi_EE_grid[iE,iN] = params[self.Pi_EE_index]
				pi_EN_grid[iE,iN] = params[self.Pi_EN_index]

			u0_grid[iE,iN] = params[3]
			alpha_grid[iE,iN] = params[4]
			t0_grid[iE,iN] = params[5]
			tE_grid[iE,iN] = params[6]

			if chi2[iE,iN] < min_chi2:
				min_chi2 = chi2[iE,iN]
				self.plot_caustic_and_trajectory(p=params,plot_data=False)
				self.plot_lightcurve(p=params)

			print(pi_EE[iE], pi_EN[iN], chi2[iE,iN], min_chi2)
			print('-'*30)

	if refine_all_parameters:

		return chi2, logd_grid, logq_grid, logrho_grid, u0_grid, alpha_grid, t0_grid, tE_grid, pi_EE_grid, pi_EN_grid

	return chi2, u0_grid, alpha_grid, t0_grid, tE_grid

def grid_search_parallax3(self,params=None,range_piEE=(-0.5,0.5),range_piEN=(-0.5,0.5),n_piEE=501,n_piEN=501,t_ref=None,residual_tolerance=0.02,refine_all_parameters=False,max_dist=0.03):

	from scipy.optimize import fsolve, root, least_squares, minimize, Bounds
	from scipy.signal import argrelmin
	from time import time


	def func(pars, pi_EE, pi_EN, b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref):
		if len(pars) == 3:
			u0, alpha, t0 = pars
			tE = tE_ref
		else:
			u0, alpha, t0, tE = pars
		self.p[3] = u0 + b*np.sin(alpha)
		self.p[4] = alpha
		self.p[5] = t0 - tE*b*np.cos(alpha)
		self.p[6] = tE
		q_n, q_e = self.compute_parallax_terms(np.array(t_ref),self.p[5])
		delta_tau = q_n*pi_EN + q_e*pi_EE
		delta_beta = -q_n*pi_EE + q_e*pi_EN

		return [cref*(t_ref[0]-t0_mid_ref)/tE_ref + sref*u0_mid_ref - \
							( np.cos(alpha)*((t_ref[0]-t0)/tE + delta_tau[0]) + np.sin(alpha)*(u0+delta_beta[0]) ), \
				-sref*(t_ref[0]-t0_mid_ref)/tE_ref + cref*u0_mid_ref -  \
							( -np.sin(alpha)*((t_ref[0]-t0)/tE + delta_tau[0]) + np.cos(alpha)*(u0+delta_beta[0]) ), \
				cref*(t_ref[1]-t0_mid_ref)/tE_ref + sref*u0_mid_ref - \
							( np.cos(alpha)*((t_ref[1]-t0)/tE + delta_tau[1]) + np.sin(alpha)*(u0+delta_beta[1]) ), \
				-sref*(t_ref[1]-t0_mid_ref)/tE_ref + cref*u0_mid_ref - \
							( -np.sin(alpha)*((t_ref[1]-t0)/tE + delta_tau[1]) + np.cos(alpha)*(u0+delta_beta[1]) ) ]

	print('grid_search_parallax')

	start = time()

	if params is None:
		params = self.p

	params_ref = params.copy()
	print('params_ref', params_ref)

	self.plotprefix = 'grid_search_parallax'

	if t_ref is not None:

		logd, logq, logrho, u0_ref, alpha_ref, t0_ref, tE_ref = params[:7]
		cref = np.cos(alpha_ref)
		sref = np.sin(alpha_ref)
		params_ref[5] -= t0_ref

		d = 10.0**logd
		q = 10.0**logq
		rho = 10.0**logrho
		a = 0.5*d
		b = a*(1.0-q)/(1.0+q)
		u0_mid_ref = u0_ref - b*sref
		t0_mid_ref = t0_ref + tE_ref*b*cref

		u0_grid = np.zeros([n_piEE,n_piEN])
		alpha_grid = np.zeros([n_piEE,n_piEN])
		t0_grid = np.zeros([n_piEE,n_piEN])
		tE_grid = np.zeros([n_piEE,n_piEN])
		piEE_grid = np.zeros([n_piEE,n_piEN])
		piEN_grid = np.zeros([n_piEE,n_piEN])

		if refine_all_parameters:
			logd_grid = np.zeros([n_piEE,n_piEN])
			logq_grid = np.zeros([n_piEE,n_piEN])
			logrho_grid = np.zeros([n_piEE,n_piEN])
			pi_EE_grid = np.zeros([n_piEE,n_piEN])
			pi_EN_grid = np.zeros([n_piEE,n_piEN])

		t_ref_array = np.array(t_ref)

		# For searching for other solutions
		t_mid = np.mean(t_ref)
		t_range = t_ref[1] - t_ref[0]
		t_search = np.hstack((np.linspace(t_ref[0]-5.0*tE_ref,t_ref[0]-0.1*tE_ref,200),np.linspace(t_ref[1]+0.1*tE_ref,t_ref[1]+5.0*tE_ref,200)))

		print('reference u0, alpha, t0:', u0_ref, alpha_ref, t0_ref)
		print('reference u0_mid, t0_mid:', u0_mid_ref, t0_mid_ref)

		print('reference u1, u2:', cref*(t_ref[0]-t0_mid_ref)/tE_ref + sref*u0_mid_ref, \
									-sref*(t_ref[0]-t0_mid_ref)/tE_ref + cref*u0_mid_ref, \
									cref*(t_ref[1]-t0_mid_ref)/tE_ref + sref*u0_mid_ref, \
									-sref*(t_ref[1]-t0_mid_ref)/tE_ref + cref*u0_mid_ref)

	chi2 = np.zeros([n_piEE,n_piEN])
	chi2_constrained = np.zeros([n_piEE,n_piEN])

	pi_EE = np.linspace(range_piEE[0],range_piEE[1],n_piEE)
	pi_EN = np.linspace(range_piEN[0],range_piEN[1],n_piEN)

	min_chi2 = 1.e10

	for iE in range(n_piEE):

		for iN in range(n_piEN):

			params = params_ref.copy()
			print('elapsed time', time() - start) 
			print('p', self.p)
			print('iE, iN, pi_EE[iE], pi_EN[iN]', iE, iN, pi_EE[iE], pi_EN[iN])

			scale = np.array([1.0,1.0,t0_mid_ref])

			result = least_squares(func, np.array([u0_mid_ref, alpha_ref, t0_mid_ref]),method='lm', \
						args=(pi_EE[iE], pi_EN[iN], b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref))

			print('3-parameter least squares:')
			print(result)

			if not result.success or result.cost > max_dist:

				chi2[iE,iN] = np.nan
				print('-'*30)
				continue

			u0, alpha, t0 = result.x.tolist()

			scale = np.array([1.0,1.0,t0,tE_ref])

			result = least_squares(func, np.array([u0_mid_ref, alpha_ref, t0_mid_ref, tE_ref]),method='lm', \
						args=(pi_EE[iE], pi_EN[iN], b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref))
			u0, alpha, t0, tE = result.x.tolist()

			print('elapsed time', time() - start) 

			print('4-parameter least squares:')
			print(result)

			if not result.success or result.cost > max_dist:

				chi2[iE,iN] = np.nan
				print('-'*30)
				continue

			print('ref:', [u0_mid_ref, alpha_ref, t0_mid_ref, tE_ref])
			print('solved:', [u0, alpha, t0, tE])
			print('residual:', result.fun)
			print('elapsed time', time() - start) 

			# converting back to CoM coordinates
			params[3] = u0 + b*np.sin(alpha)
			params[4] = alpha
			params[5] = t0 - tE*b*np.cos(alpha)
			params[6] = tE
			print('u0, alpha, t0, tE:', params[3], params[4], params[5], params[6])
			u0_grid[iE,iN] = params[3]
			alpha_grid[iE,iN] = params[4]
			t0_grid[iE,iN] = params[5]
			tE_grid[iE,iN] = params[6]

			params[self.Pi_EE_index] = pi_EE[iE]
			params[self.Pi_EN_index] = pi_EN[iN]
			print('piEE, piEN:', params[self.Pi_EE_index], params[self.Pi_EN_index])
			print('params:', params)

			chi2[iE,iN], _ , _, _, _, _ = self.chi2_calc(params)
			print('chi2:', chi2[iE,iN])

			if refine_all_parameters and chi2[iE,iN] < 1.e5:

				pp = np.array([params[i] for i in np.where(1-self.freeze)[0].tolist()])


				# lower_bounds = 0.95*pp
				# lower_bounds[5] = pp[5] - 10.0
				# upper_bounds = 1.05*pp
				# upper_bounds[5] = pp[5] + 10.0

				# bounds = Bounds(lower_bounds,upper_bounds)

				result = minimize(self.neg_lnprob,pp)

				print('full minimization:')
				print(result)
				print('elapsed time', time() - start) 

				if result.success:

					params[np.where(1-self.freeze)[0]] = result.x
					print('refined params:', params)

					chi2[iE,iN], _ , _, _, _, _ = self.chi2_calc(params)
					print('chi2:', chi2[iE,iN])

					logd_grid[iE,iN] = params[0]
					logq_grid[iE,iN] = params[1]
					logrho_grid[iE,iN] = params[2]

				pi_EE_grid[iE,iN] = params[self.Pi_EE_index]
				pi_EN_grid[iE,iN] = params[self.Pi_EN_index]

			u0_grid[iE,iN] = params[3]
			alpha_grid[iE,iN] = params[4]
			t0_grid[iE,iN] = params[5]
			tE_grid[iE,iN] = params[6]

			if chi2[iE,iN] < min_chi2:
				min_chi2 = chi2[iE,iN]
				self.plot_caustic_and_trajectory(p=params,plot_data=False)
				self.plot_lightcurve(p=params)

			print(pi_EE[iE], pi_EN[iN], chi2[iE,iN], min_chi2)
			print('-'*30)

	if refine_all_parameters:

		return chi2, logd_grid, logq_grid, logrho_grid, u0_grid, alpha_grid, t0_grid, tE_grid, pi_EE_grid, pi_EN_grid

	return chi2, u0_grid, alpha_grid, t0_grid, tE_grid



def grid_search_parallax2(self,params=None,range_piEE=(-0.5,0.5),range_piEN=(-0.5,0.5),n_piEE=501,n_piEN=501,t_ref=None,residual_tolerance=0.02,
							refine_all_parameters=False,max_dist=0.03,refine_chi2_threshold=5.0):

	from scipy.optimize import fsolve, root, least_squares, minimize, Bounds
	from scipy.signal import argrelmin
	from time import time


	def func2(pars, logrho, limb, pi_EE, pi_EN):

		p = np.array([pars[0],pars[1],logrho,pars[2],pars[3],pars[4],pars[5],limb,pi_EE,pi_EN]) 
		return self.neg_lnprob(p)


	def func(pars, pi_EE, pi_EN, b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref):
		if len(pars) == 3:
			u0, alpha, t0 = pars
			tE = tE_ref
		else:
			u0, alpha, t0, tE = pars
		self.p[3] = u0 + b*np.sin(alpha)
		self.p[4] = alpha
		self.p[5] = t0 - tE*b*np.cos(alpha)
		self.p[6] = tE
		q_n, q_e = self.compute_parallax_terms(np.array(t_ref),self.p[5])
		delta_tau = q_n*pi_EN + q_e*pi_EE
		delta_beta = -q_n*pi_EE + q_e*pi_EN

		return [cref*(t_ref[0]-t0_mid_ref)/tE_ref + sref*u0_mid_ref - \
							( np.cos(alpha)*((t_ref[0]-t0)/tE + delta_tau[0]) + np.sin(alpha)*(u0+delta_beta[0]) ), \
				-sref*(t_ref[0]-t0_mid_ref)/tE_ref + cref*u0_mid_ref -  \
							( -np.sin(alpha)*((t_ref[0]-t0)/tE + delta_tau[0]) + np.cos(alpha)*(u0+delta_beta[0]) ), \
				cref*(t_ref[1]-t0_mid_ref)/tE_ref + sref*u0_mid_ref - \
							( np.cos(alpha)*((t_ref[1]-t0)/tE + delta_tau[1]) + np.sin(alpha)*(u0+delta_beta[1]) ), \
				-sref*(t_ref[1]-t0_mid_ref)/tE_ref + cref*u0_mid_ref - \
							( -np.sin(alpha)*((t_ref[1]-t0)/tE + delta_tau[1]) + np.cos(alpha)*(u0+delta_beta[1]) ) ]

	print('grid_search_parallax')

	start = time()

	if params is None:
		params = self.p

	params_ref = params.copy()
	print('params_ref', params_ref)

	self.plotprefix = 'grid_search_parallax'

	if t_ref is not None:

		logd, logq, logrho, u0_ref, alpha_ref, t0_ref, tE_ref = params[:7]
		cref = np.cos(alpha_ref)
		sref = np.sin(alpha_ref)
		params_ref[5] -= t0_ref

		d = 10.0**logd
		q = 10.0**logq
		rho = 10.0**logrho
		a = 0.5*d
		b = a*(1.0-q)/(1.0+q)
		u0_mid_ref = u0_ref - b*sref
		t0_mid_ref = t0_ref + tE_ref*b*cref

		u0_grid = np.zeros([n_piEE,n_piEN])
		alpha_grid = np.zeros([n_piEE,n_piEN])
		t0_grid = np.zeros([n_piEE,n_piEN])
		tE_grid = np.zeros([n_piEE,n_piEN])
		piEE_grid = np.zeros([n_piEE,n_piEN])
		piEN_grid = np.zeros([n_piEE,n_piEN])

		if refine_all_parameters:
			logd_grid = np.zeros([n_piEE,n_piEN])
			logq_grid = np.zeros([n_piEE,n_piEN])
			logrho_grid = np.zeros([n_piEE,n_piEN])
			pi_EE_grid = np.zeros([n_piEE,n_piEN])
			pi_EN_grid = np.zeros([n_piEE,n_piEN])
			n_data = np.sum([len(mc.data[d][0]) for d in data])

		t_ref_array = np.array(t_ref)

		u1_ref = cref*(t_ref-t0_mid_ref)/tE_ref + sref*u0_mid_ref
		u2_ref = -sref*(t_ref-t0_mid_ref)/tE_ref + cref*u0_mid_ref
		u_ref = [u1_ref, u2_ref]

		# For searching for other solutions
		t0_search = np.linspace(t0_mid_ref-tE_ref,t0_mid_ref+tE_ref,21)
		u0_search = np.linspace(u0_mid_ref*0.6,u0_mid_ref*1.4,11)
		alpha_search = np.linspace(alpha_ref-0.03,alpha_ref+0.03,11)

		print('reference u0, alpha, t0:', u0_ref, alpha_ref, t0_ref)
		print('reference u1, u2:', u1_ref, u2_ref)

	chi2 = np.zeros([n_piEE,n_piEN])
	chi2_constrained = np.zeros([n_piEE,n_piEN])

	pi_EE = np.linspace(range_piEE[0],range_piEE[1],n_piEE)
	pi_EN = np.linspace(range_piEN[0],range_piEN[1],n_piEN)

	min_chi2 = 1.e10

	for iE in range(n_piEE):

		for iN in range(n_piEN):

			params = params_ref.copy()
			print('elapsed time', time() - start) 
			print('p', self.p)
			print('iE, iN, pi_EE[iE], pi_EN[iN]', iE, iN, pi_EE[iE], pi_EN[iN])

			min_u0, min_alpha, min_t0 = self.search_parallax_trajectory(u0_search,alpha_search,t0_search,u_ref,t_ref_array)

			print('min_u0, min_alpha, min_t0 = ',min_u0, min_alpha, min_t0)
			print('elapsed time', time() - start) 

			result = least_squares(func, np.array([min_u0, min_alpha, min_t0]),method='lm', \
						args=(pi_EE[iE], pi_EN[iN], b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref))

			print('3-parameter least squares:')
			print(result)

			if not result.success or result.cost > max_dist:

				chi2[iE,iN] = np.nan
				print('-'*30)
				continue

			u0, alpha, t0 = result.x.tolist()

			result = least_squares(func, np.array([u0, alpha, t0, tE_ref]),method='lm', \
						args=(pi_EE[iE], pi_EN[iN], b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref))

			u0, alpha, t0, tE = result.x.tolist()

			print('elapsed time', time() - start) 

			print('4-parameter least squares:')
			print(result)

			if not result.success or result.cost > max_dist:

				chi2[iE,iN] = np.nan
				print('-'*30)
				continue

			print('ref:', [u0_mid_ref, alpha_ref, t0_mid_ref, tE_ref])
			print('solved:', [u0, alpha, t0, tE])
			print('residual:', result.fun)
			print('elapsed time', time() - start) 

			# converting back to CoM coordinates
			params[3] = u0 + b*np.sin(alpha)
			params[4] = alpha
			params[5] = t0 - tE*b*np.cos(alpha)
			params[6] = tE
			print('u0, alpha, t0, tE:', params[3], params[4], params[5], params[6])
			u0_grid[iE,iN] = params[3]
			alpha_grid[iE,iN] = params[4]
			t0_grid[iE,iN] = params[5]
			tE_grid[iE,iN] = params[6]

			params[self.Pi_EE_index] = pi_EE[iE]
			params[self.Pi_EN_index] = pi_EN[iN]
			print('piEE, piEN:', params[self.Pi_EE_index], params[self.Pi_EN_index])
			print('params:', params)

			chi2[iE,iN], _ , _, _, _, _ = self.chi2_calc(params)
			print('chi2:', chi2[iE,iN])


			if refine_all_parameters and chi2[iE,iN]/n_data < refine_chi2_threshold:

				freeze_save = self.freeze
				pp = np.array([params[0],params[1],params[3],params[4],params[5],params[6]])


				# lower_bounds = 0.95*pp
				# lower_bounds[5] = pp[5] - 10.0
				# upper_bounds = 1.05*pp
				# upper_bounds[5] = pp[5] + 10.0

				# bounds = Bounds(lower_bounds,upper_bounds)

				result = minimize(func2, pp, args=(params[2],params[7],pi_EE[iE], pi_EN[iN]),methd='Nelder-Mead')

				print('full minimization:')
				print(result)
				print('elapsed time', time() - start) 

				pp = params.copy()
				pp[0] = result.x[0]
				pp[1] = result.x[1]
				pp[3] = result.x[2]
				pp[4] = result.x[3]
				pp[5] = result.x[4]
				pp[6] = result.x[5]
				chi2_new , _ , _, _, _, _ = self.chi2_calc(pp)

				if result.success or chi2_new < chi2[iE,iN]:

					params[0] = result.x[0]
					params[1] = result.x[1]
					params[3] = result.x[2]
					params[4] = result.x[3]
					params[5] = result.x[4]
					params[6] = result.x[5]
					print('refined params:', params)

					chi2[iE,iN], _ , _, _, _, _ = self.chi2_calc(params)
					print('chi2:', chi2[iE,iN])

					logd_grid[iE,iN] = params[0]
					logq_grid[iE,iN] = params[1]

				pi_EE_grid[iE,iN] = params[self.Pi_EE_index]
				pi_EN_grid[iE,iN] = params[self.Pi_EN_index]
				logrho_grid[iE,iN] = params[2]

				self.freeze = freeze_save

			u0_grid[iE,iN] = params[3]
			alpha_grid[iE,iN] = params[4]
			t0_grid[iE,iN] = params[5]
			tE_grid[iE,iN] = params[6]

			if chi2[iE,iN] < min_chi2:
				min_chi2 = chi2[iE,iN]
				print('params:', params)
				self.plot_caustic_and_trajectory(p=params,plot_data=False)
				self.plot_lightcurve(p=params)

			print(pi_EE[iE], pi_EN[iN], chi2[iE,iN], min_chi2)
			print('-'*30)

	if refine_all_parameters:

		return chi2, logd_grid, logq_grid, logrho_grid, u0_grid, alpha_grid, t0_grid, tE_grid, pi_EE_grid, pi_EN_grid

	return chi2, u0_grid, alpha_grid, t0_grid, tE_grid




def plot_map(self,chi2_arr,x,y,plot_file='parallax_grid.png',x_label=r'$\pi_{EE}$',y_label=r'$\pi_{EN}$'):

	fig = plt.figure(figsize=(13.50,11.50))
	ax = fig.add_subplot(111,aspect=1.0)

	q = np.where(np.isnan(chi2_arr))
	p = np.where(np.isfinite(chi2_arr))
	chi2_arr[q] = np.max(chi2_arr[p])

	min_chi2 = np.min(chi2_arr)

	plt.pcolor(np.sqrt(chi2_arr.T - min_chi2), cmap='autumn')
	plt.colorbar()


	x_tick_arr = np.round(x, 2)
	x_tick_labels = ['']*len(x_tick_arr)
	for i in np.arange(0,len(x_tick_arr),10):
		x_tick_labels[i] = str(x_tick_arr[i])

	y_tick_arr = np.round(y, 2)
	y_tick_labels = ['']*len(y_tick_arr)
	for i in np.arange(0,len(y_tick_arr),20):
		y_tick_labels[i] = str(y_tick_arr[i])

	ax.set_xticks(np.arange(chi2_arr.shape[0])+0.5, minor=False)
	ax.set_yticks(np.arange(chi2_arr.shape[1])+0.5, minor=False)
	ax.set_xticklabels(x_tick_labels, minor=False)
	ax.set_yticklabels(y_tick_labels, minor=False)


	plt.xlabel(r'$\pi_{EE}$')
	plt.ylabel(r'$\pi_{EN}$')
	min_chi2 = np.min(chi2_arr)
	plt.title(r"$\sqrt{\Delta \chi^2} \quad \quad  (\chi^2_{min} = %d)$"%min_chi2)

	plt.savefig(plot_file)


def grid_search_parallax_simple(self,piEE,piEN,plot=True,save=True,file='chi2_parallax.npy'):


	n_piEE = piEE.shape[0]
	n_piEN = piEN.shape[0]

	chi2 = np.zeros((n_piEE,n_piEN))

	for i in range(n_piEE):

		self.p[self.Pi_EE_index] = piEE[i]

		for j in range(n_piEN):

			self.p[self.Pi_EN_index] = piEN[j]

			chi2[i,j], _, _, _, _, _ = self.chi2_calc()


	if save:
		np.save(file,chi2)

	if plot:
		self.plot_map(chi2,piEE,piEN)

	return chi2
