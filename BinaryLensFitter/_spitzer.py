import numpy as np


def add_spitzer(self,position_file,data,right_ascension=None,declination=None,year=None):

	""" Set up for modelling data from the Spitzer spacecraft. 

		inputs:

			position_file: 		textfile containing columns of HJD-6830, RA, DEC, distance for Spitzer

			data:				tupple of Spitzer observations (date, flux, flux error)

			right_ascension:
			declination:		coordinates of the microlensing event in degrees 	

	"""

	self.spitzer_flux_ratio_reference_site = 'dummmy'
	self.spitzer_pos = np.loadtxt(position_file)
	self.spitzer_pos[:,0] += 6830.0 
	self.spitzer_pos[:,1] = np.unwrap(np.deg2rad(self.spitzer_pos[:,1]))
	self.spitzer_pos[:,2] = np.deg2rad(self.spitzer_pos[:,2])

	self.spitzer_data = data

	self.use_spitzer = True


	if self.scale_error_bars_multiplicative:

		site = 'spitzer'
		self.error_bar_scale_index[site] = self.dims
		self.p = np.hstack((self.p,1.0))
		self.p_sig = np.hstack((self.p_sig,0.001))
		self.freeze = np.hstack((self.freeze,np.zeros(1,dtype=int)))
		self.parameter_labels.append("s-"+site)
		self.dims += 1


	if self.Pi_EE_index is None:

		if right_ascension is not None and declination is not None and year is not None:
			self.right_ascension = right_ascension
			self.declination = declination
			self.vernal, self.peri = self.get_vernal_peri(year)
		else:
			print('Error: must provide right ascension and declination when adding Spitzer without Earth-orbit parallax.')
			sys.exit(0)

		self.Pi_EE_index = self.dims
		self.Pi_EN_index = self.dims + 1

		self.dims += 2
		self.parameter_labels.append(r"$\pi_{E,E}$")
		self.parameter_labels.append(r"$\pi_{E,N}$")

		self.p = np.hstack((self.p,np.zeros(2)))
		self.p_sig = np.hstack((self.p_sig,0.00001*np.ones(2)))

		self.freeze = np.hstack((self.freeze,np.zeros(2,dtype=int)))

		north = np.array([0.0, 0.0, 1.0])
		self.parallax_rad = np.array([np.cos(self.right_ascension) * np.cos(self.declination), \
								  np.sin(self.right_ascension) * np.cos(self.declination), \
								  np.sin(self.declination)])

		self.parallax_east = np.cross(north,self.parallax_rad)
		self.parallax_east /= np.linalg.norm(self.parallax_east)
		self.parallax_north = np.cross(self.parallax_rad,self.parallax_east)
		

def spitzer_delta_beta_tau_to_parallax(self,delta_beta, delta_tau, p=None, t0=None, debug=False):

	"""Convert satellite trajactory offset (delta_beta, delta_tau) to microlensing parallax (piE_E, piE_N)."""

	if t0 is None:

		if p is None:
			p = self.p

		logd, logq, logrho, u0, phi, t0, tE = p[:7]

	q_n = np.array([0.0])
	q_e = np.array([0.0])

	t = np.atleast_1d(t0)

	if self.use_parallax:

		if self.parallax_t_ref is None:
			self.parallax_t_ref = t0

		q_n, q_e = self.compute_parallax_terms(t)

	sq_n, sq_e, sq_r = self.compute_spitzer_parallax_terms(t)

	C = np.array([[q_n[0]+sq_n[0], q_e[0]+sq_e[0]],[q_e[0]+sq_e[0],-(q_n[0]+sq_n[0])]])
	b = np.array([delta_tau,delta_beta]).T

	x = np.linalg.solve(C,b)

	pi_EN = x[0]
	pi_EE = x[1]

	if debug:
		print()
		print('spitzer_delta_beta_tau_to_parallax')
		print('(sq_n, sq_e)', sq_n[0], sq_e[0])
		print('(qn+sq_n, qe+sq_e)', q_n[0]+sq_n[0], q_e[0]+sq_e[0])
		print('(pi_EN, pi_EE)', (pi_EN, pi_EE))
		print('(delta_tau, delta_beta)', (delta_tau, delta_beta))

	return pi_EN, pi_EE 


def compute_spitzer_parallax_terms(self, ts_in):

	ra = np.interp(ts_in,self.spitzer_pos[:,0],self.spitzer_pos[:,1])
	dec = np.interp(ts_in,self.spitzer_pos[:,0],self.spitzer_pos[:,2])
	self.spitzer_distance = np.interp(ts_in,self.spitzer_pos[:,0],self.spitzer_pos[:,3])

	sun = np.zeros(([3, len(ts_in)]), np.float64)
	sun[0,:] = -np.cos(ra)*np.cos(dec)
	sun[1,:] = -np.sin(ra)*np.cos(dec)
	sun[2,:] = -np.sin(dec)

	q_n = np.zeros_like(ts_in)
	q_e = np.zeros_like(ts_in)
	q_r = np.zeros_like(ts_in)

	for i in range(3):

		q_n += sun[i,:]*self.parallax_north[i]*self.spitzer_distance
		q_e += sun[i,:]*self.parallax_east[i]*self.spitzer_distance
		q_r += sun[i,:]*self.parallax_rad[i]*self.spitzer_distance

	return q_n, q_e, q_r


def spitzer_magnification(self, t, p=None, LD=None):

	"""Compute magnification for satellite trajectory."""

	from scipy.spatial import cKDTree

	if LD==None:
		LD = self.spitzer_limb_constant

	if p is None:
		p = self.p.copy()

	logd, logq, logrho, u0, phi, t0, tE = p[:7]
	d = 10.0**logd
	q = 10.0**logq
	rho = 10.0**logrho
	a = 0.5*d
	b = a*(1.0-q)/(1.0+q)
	
	if self.spitzer_u0_sign < 0:
		u0 *= -1.0
		phi = 2.0*np.pi - phi

	delta_tau = np.zeros_like(t)
	delta_beta = np.zeros_like(t)
	delta_phi = np.zeros_like(t)
	delta_d = np.zeros_like(t)

	q_n = np.zeros_like(t)
	q_e = np.zeros_like(t)

	pi_EN = np.float64(p[self.Pi_EN_index])
	pi_EE = np.float64(p[self.Pi_EE_index])

	if self.use_parallax:

		if self.parallax_t_ref is None:
			self.parallax_t_ref = t0

		q_n, q_e = self.compute_parallax_terms(t)

	sq_n, sq_e, sq_r = self.compute_spitzer_parallax_terms(t)

	delta_tau	= (q_n+sq_n)*pi_EN + (q_e+sq_e)*pi_EE
	delta_beta	= -(q_n+sq_n)*pi_EE + (q_e+sq_e)*pi_EN

	if self.use_lens_orbital_motion:

		if self.lens_orbital_motion_reference_date is None:
			t_ref = t0
		else:
			t_ref = self.lens_orbital_motion_reference_date

		dphidt = np.float64(p[self.dphidt_index])
		dddt = np.float64(p[self.dddt_index])

		delta_phi = dphidt*(t-t_ref)/365.2425
		delta_d = dddt*(t-t_ref)/365.2425

		if self.use_lens_orbital_acceleration:

			d2phidt2 = np.float64(p[self.d2phidt2_index])
			d2ddt2 = np.float64(p[self.d2ddt2_index])
			
			delta_phi += 0.5 * d2phidt2 * ((t-t_ref)/365.2425)**2
			delta_d += 0.5 * d2ddt2 *  ((t-t_ref)/365.2425)**2


	if self.use_VBBL:
		# tau = (t - t0)/tE
		# u1 = -u0*np.sin(phi+delta_phi) + (tau+delta_tau)*np.cos(phi+delta_phi)
		# u2 = u0*np.cos(phi+delta_phi) + (tau+delta_tau)*np.sin(phi+delta_phi)
		phi = - phi
		u1 = ((t - t0)/tE + delta_tau)*np.cos(phi+delta_phi) - (u0+delta_beta)*np.sin(phi+delta_phi)
		u2 = -((t - t0)/tE + delta_tau)*np.sin(phi+delta_phi) - (u0+delta_beta)*np.cos(phi+delta_phi)
		return self.VBBL_magnification(t,p,u=(u1,u2),LD=LD)


	u0n = u0 - np.sin(phi)*b
	t0n = t0 + tE*b*np.cos(phi)

	u1 = ((t- t0n)/tE + delta_tau)*np.cos(phi+delta_phi) + (u0n+delta_beta)*np.sin(phi+delta_phi)
	u2 = -((t - t0n)/tE + delta_tau)*np.sin(phi+delta_phi) + (u0n+delta_beta)*np.cos(phi+delta_phi)

	import pycuda.autoinit
	import pycuda.driver as drv
	from ._gpu_image_centred import gpu_image_centred
	image_centred = gpu_image_centred.get_function("data_magnifications")

	# First compute point-source mag for all epochs
	A = self.point_source_magnification(u1,u2,delta_d,p=p,LD=LD)

	high_points = np.where(A > self.hexadecapole_magnification_threshold)[0]
	if len(high_points) > 0:

		n_points = len(high_points)
		blockshape = (int(128),1, 1)
		gridshape = (n_points, 1)
		
		A_high = np.atleast_1d(np.zeros(n_points,np.float64))
		u1_f = np.atleast_1d(np.float64(u1[high_points]))
		u2_f = np.atleast_1d(np.float64(u2[high_points]))
		d_f = np.atleast_1d(np.float64(d+delta_d[high_points]))
		
		Gamma = np.float64(LD)

		#if self.use_limb_darkening:
		#	Gamma = np.float64(p[self.limb_index])

		zeta_real, zeta_imag = self.caustic(params=p)
		zeta_real_f = np.float64(zeta_real)
		zeta_imag_f = np.float64(zeta_imag)

		image_centred(np.float64(q), np.float64(rho), Gamma, np.int32(1), np.float64(self.hexadecapole_approximation_threshold), 
						np.int32(len(zeta_real)), drv.In(zeta_real_f), drv.In(zeta_imag_f),
						drv.In(d_f), drv.In(u1_f), drv.In(u2_f), drv.Out(A_high), block=blockshape, grid=gridshape)

		A[high_points] = A_high



	# zeta_real, zeta_imag = self.caustic(params=p)
	# caustic_tree = cKDTree(np.stack((zeta_real,zeta_imag),axis=-1))
	# dist, ind = caustic_tree.query(np.stack((u1,u2),axis=-1))

	# A = np.zeros_like(t)

	# points = np.where(dist > rho*self.point_source_approximation_threshold)[0]
	# if len(points)>0:
	# 	A[points] = self.point_source_magnification(u1[points],u2[points],delta_d[points],p=p)

	# points = np.where((dist <= rho*self.point_source_approximation_threshold) & \
	# 				(dist > rho*self.hexadecapole_approximation_threshold))[0]
	# if len(points)>0:
	# 	A[points] = self.hexadecapole_magnification(u1[points],u2[points],delta_d[points],p=p,LD=LD)

	# points = np.where(dist <= rho*self.hexadecapole_approximation_threshold)[0]
	# if len(points)>0:
	# 	A[points] = self.finite_source_magnification(u1[points],u2[points],delta_d[points],p=p,LD=LD)

	return A

def compute_spitzer_perpendicular_distance(self,t,lens_distance):

	spitzer_ra = np.interp(t,self.spitzer_pos[:,0],self.spitzer_pos[:,1])
	spitzer_dec = np.interp(t,self.spitzer_pos[:,0],self.spitzer_pos[:,2])
	spitzer_distance = np.interp(t,self.spitzer_pos[:,0],self.spitzer_pos[:,3])

	spitzer_direction = np.zeros(3, np.float64)
	spitzer_direction[0] = np.cos(spitzer_ra)*np.cos(spitzer_dec)
	spitzer_direction[1] = np.sin(spitzer_ra)*np.cos(spitzer_dec)
	spitzer_direction[2] = np.sin(spitzer_dec)

	print('spitzer_distance:', spitzer_distance)
	print('spitzer_direction:', spitzer_direction)
	print('spitzer_norm:', np.linalg.norm(spitzer_direction,axis=0))
	print('lens_distance:', lens_distance)
	print('lens_direction:', self.parallax_rad)
	print('lens_norm:', np.linalg.norm(self.parallax_rad,axis=0))
	print('angle:', np.arccos(np.dot(spitzer_direction,self.parallax_rad)))


	s1 = 0.0
	s2 = 0.0
	for k in range(3):
		s1 += lens_distance*spitzer_direction[k]*self.parallax_rad[k] - spitzer_distance*spitzer_direction[k]**2
		s2 += lens_distance*self.parallax_rad[k]**2 - spitzer_distance*self.parallax_rad[k]*spitzer_direction[k]

		print(k, spitzer_direction[k], self.parallax_rad[k]) 

	print('s1:', s1)
	print('s2:', s2)

	ab_ratio = -s1/s2
	print('ab_ratio:', ab_ratio)

	v = ab_ratio*self.parallax_rad + spitzer_direction

	print(ab_ratio*self.parallax_rad)
	print(spitzer_direction)
	print('v:', v)
	print('norm v:', np.linalg.norm(v,axis=0))

	v /= np.linalg.norm(v,axis=0)

	print('v:', v)

	A = np.array([[v[0], spitzer_distance*spitzer_direction[0]-lens_distance*self.parallax_rad[0]],[v[1], spitzer_distance*spitzer_direction[1]-lens_distance*self.parallax_rad[1]]])
	c = spitzer_distance*spitzer_direction[:2]

	x = np.linalg.solve(A,c)
	print('x:', x)

	d_perp = x[0]

	check = x[0]*v[2] - spitzer_distance*spitzer_direction[2] - x[1]*(lens_distance*self.parallax_rad[2] - spitzer_distance*spitzer_direction[2])

	return d_perp, check


def compute_spitzer_parallax(self):

	pi_EN = np.float64(self.p[self.Pi_EN_index])
	pi_EE = np.float64(self.p[self.Pi_EE_index])

	t0 = self.p[5]
	t = np.zeros(1)
	t[0] = t0

	q_n = np.zeros_like(t)
	q_e = np.zeros_like(t)

	if self.use_parallax:

		if self.parallax_t_ref is None:
			self.parallax_t_ref = t0

		q_n, q_e = self.compute_parallax_terms(t)

	sq_n, sq_e, sq_r = self.compute_spitzer_parallax_terms(t)

	self.spitzer_delta_tau	= (q_n+sq_n)*pi_EN + (q_e+sq_e)*pi_EE
	self.spitzer_delta_beta	= -(q_n+sq_n)*pi_EE + (q_e+sq_e)*pi_EN

	print()
	print('compute_spitzer_parallax')
	print('(sq_n, sq_e)', sq_n, sq_e)
	print('(qn+sq_n, qe+sq_e)', q_n+sq_n, q_e+sq_e)
	print('(pi_EN, pi_EE)', (pi_EN, pi_EE))
	print('(spitzer_delta_tau, spitzer_delta_beta)', (self.spitzer_delta_tau, self.spitzer_delta_beta))


	self.spitzer_d_perp, check = self.compute_spitzer_perpendicular_distance(t0)

	return check


def lnprob_spitzer(self,p,params,index):

	params[index] = p
	chi2, _ , _, _, _, _ = self.chi2_calc(params,use_spitzer_only=True)

	if np.isfinite(chi2):
		return -chi2/2.0
	else:
		return -np.inf


def grid_search_spitzer(self,params=None,range_dbeta=(-0.5,0.5),range_dtau=(-0.5,0.5),n_dbeta=1001,n_dtau=1001,t_ref=None,residual_tolerance=0.02):

	from scipy.optimize import fsolve, root, least_squares





	def func(pars, db, dt, b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref):
		if len(pars) == 3:
			u0, alpha, t0 = pars
			tE = tE_ref
		else:
			u0, alpha, t0, tE = pars
		self.p[3] = u0 + b*np.sin(alpha)
		self.p[4] = alpha
		self.p[5] = t0 - tE*b*np.cos(alpha)
		self.p[6] = tE
		self.p[self.Pi_EN_index], self.p[self.Pi_EE_index] = self.spitzer_delta_beta_tau_to_parallax(db, dt)
		q_n, q_e = self.compute_parallax_terms(np.array(t_ref),self.p[5])
		delta_tau = q_n*self.p[self.Pi_EN_index] + q_e*self.p[self.Pi_EE_index]
		delta_beta = -q_n*self.p[self.Pi_EE_index] + q_e*self.p[self.Pi_EN_index]

		return [cref*(t_ref[0]-t0_mid_ref)/tE_ref + sref*u0_mid_ref - \
							( np.cos(alpha)*((t_ref[0]-t0)/tE + delta_tau[0]) + np.sin(alpha)*(u0+delta_beta[0]) ), \
				-sref*(t_ref[0]-t0_mid_ref)/tE_ref + cref*u0_mid_ref -  \
							( -np.sin(alpha)*((t_ref[0]-t0)/tE + delta_tau[0]) + np.cos(alpha)*(u0+delta_beta[0]) ), \
				cref*(t_ref[1]-t0_mid_ref)/tE_ref + sref*u0_mid_ref - \
							( np.cos(alpha)*((t_ref[1]-t0)/tE + delta_tau[1]) + np.sin(alpha)*(u0+delta_beta[1]) ), \
				-sref*(t_ref[1]-t0_mid_ref)/tE_ref + cref*u0_mid_ref - \
							( -np.sin(alpha)*((t_ref[1]-t0)/tE + delta_tau[1]) + np.cos(alpha)*(u0+delta_beta[1]) ) ]

	print('grid_search_spitzer')

	if params is None:
		params = self.p

	print('params_ref', params)

	params_ref = params.copy()

	self.plotprefix = 'grid_search_spitzer'

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

		u0_grid = np.zeros([n_dbeta,n_dtau])
		alpha_grid = np.zeros([n_dbeta,n_dtau])
		t0_grid = np.zeros([n_dbeta,n_dtau])
		tE_grid = np.zeros([n_dbeta,n_dtau])
		piEE_grid = np.zeros([n_dbeta,n_dtau])
		piEN_grid = np.zeros([n_dbeta,n_dtau])


		t_ref_array = np.array(t_ref)


		print('reference u0, alpha, t0:', u0_ref, alpha_ref, t0_ref)
		print('reference u0_mid, t0_mid:', u0_mid_ref, t0_mid_ref)

		print('reference u1, u2:', cref*(t_ref[0]-t0_mid_ref)/tE_ref + sref*u0_mid_ref, \
									-sref*(t_ref[0]-t0_mid_ref)/tE_ref + cref*u0_mid_ref, \
									cref*(t_ref[1]-t0_mid_ref)/tE_ref + sref*u0_mid_ref, \
									-sref*(t_ref[1]-t0_mid_ref)/tE_ref + cref*u0_mid_ref)

		#self.plot_date_range = t_ref

	chi2 = np.zeros([n_dbeta,n_dtau])
	chi2_constrained = np.zeros([n_dbeta,n_dtau])

	dbeta = np.linspace(range_dbeta[0],range_dbeta[1],n_dbeta)
	dtau = np.linspace(range_dtau[0],range_dtau[1],n_dtau)

	if self.spitzer_has_colour_constraint:
		_, a0 , _, _, _, _ = self.chi2_calc(params)
		a0_ref = a0[self.spitzer_flux_ratio_reference_site]


	min_chi2 = 1.e10

	for ib in range(n_dbeta):

		for it in range(n_dtau):

			print('p', self.p)
			print('ib, it, dbeta[ib], dtau[ib]', ib, it, dbeta[ib], dtau[it])

			if t_ref is not None:

				result = least_squares(func, np.array([u0_mid_ref, alpha_ref, t0_mid_ref]),method='lm', \
							args=(dbeta[ib], dtau[it], b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref))

				print(result.success)
				print(result.message)

				if not result.success:

					chi2[ib,it] = np.nan
					chi2_constrained[ib,it] = np.nan
					continue

				u0, alpha, t0 = result.x.tolist()

				result = least_squares(func, np.array([u0, alpha, t0, tE_ref]),method='lm', \
							args=(dbeta[ib], dtau[it], b, sref, cref, u0_mid_ref, t0_mid_ref, tE_ref, t_ref))
				u0, alpha, t0, tE = result.x.tolist()

				print(result.success)
				print(result.message)

				if not result.success:

					chi2[ib,it] = np.nan
					chi2_constrained[ib,it] = np.nan
					continue

				print('ref:', [u0_mid_ref, alpha_ref, t0_mid_ref, tE_ref])
				print('solved:', [u0, alpha, t0, tE])
				print('residual:', result.fun)


				# converting back to CoM coordinates
				params[3] = u0 + b*np.sin(alpha)
				params[4] = alpha
				params[5] = t0 - tE*b*np.cos(alpha)
				params[6] = tE
				print('u0, alpha, t0, tE:', params[3], params[4], params[5], params[6])
				u0_grid[ib,it] = params[3]
				alpha_grid[ib,it] = params[4]
				t0_grid[ib,it] = params[5]
				tE_grid[ib,it] = params[6]

				self.p[5] = params[5]
				piEN_grid[ib,it], piEE_grid[ib,it] = self.spitzer_delta_beta_tau_to_parallax(dbeta[ib], dtau[it])

				params[self.Pi_EE_index] = piEE_grid[ib,it]
				params[self.Pi_EN_index] = piEN_grid[ib,it]
				print('piEN, piEE:', params[self.Pi_EN_index], params[self.Pi_EE_index])
				print('params:', params)

			else:

				params[self.Pi_EN_index], params[self.Pi_EE_index] = self.spitzer_delta_beta_tau_to_parallax(dbeta[ib], dtau[it])

			chi2[ib,it], a0 , _, _, _, _ = self.chi2_calc(params,use_spitzer_only=True)

			if self.spitzer_has_colour_constraint:

				flux_ratio = a0_ref/a0['spitzer']
				chi2_constrained[ib,it] = chi2[ib,it] + (2.5 * np.log10(flux_ratio/self.spitzer_flux_ratio))**2 / self.spitzer_colour_uncertainty**2

			if a0['spitzer'] < 0.0:

				chi2[ib,it] = np.nan
				chi2_constrained[ib,it] = np.nan

			elif t_ref is not None:

				if np.max(result.fun**2) > residual_tolerance**2:

					chi2[ib,it] = np.nan
					chi2_constrained[ib,it] = np.nan

				if chi2_constrained[ib,it] < min_chi2:
					min_chi2 = chi2_constrained[ib,it]
					self.plot_caustic_and_trajectory(p=params,plot_data=False)
					self.plot_lightcurve(p=params)

			print(dbeta[ib], dtau[it], params[self.Pi_EE_index], params[self.Pi_EN_index], chi2[ib,it], chi2_constrained[ib,it], min_chi2)
			print('-'*30)

	if t_ref is not None:

		return chi2, chi2_constrained, u0_grid, alpha_grid, t0_grid, tE_grid, piEE_grid, piEN_grid

	return chi2, chi2_constrained


def emcee_converge_spitzer(self, covariance=None, state=None):

	"""Compute parallax assuming a fixed standard solution, plus satellita data."""

	import emcee

	print("Running burn-in ...")

	p_index = np.array([self.Pi_EE_index,self.Pi_EN_index])

	parameter_labels = [self.parameter_labels[self.Pi_EE_index]]
	parameter_labels.append(self.parameter_labels[self.Pi_EN_index])

	sampler = emcee.EnsembleSampler(self.emcee_walkers, 2, self.lnprob_spitzer,args=[self.p,p_index])

	state = [self.p[p_index] + self.p_sig[p_index] * np.random.randn(2) \
						for i in range(self.emcee_walkers)]

	state, lnp , _ = sampler.run_mcmc(state, 3*self.emcee_burnin_steps)
	kmax = np.argmax(sampler.flatlnprobability)
	p = sampler.flatchain[kmax,:]
	
	self.plot_chain(sampler,suffix='-pre-burnin.png',parameter_labels=parameter_labels)

	sampler.reset()
	state = [p + self.p_sig[p_index] * np.random.randn(2) \
						for i in range(self.emcee_walkers)]


	converged = False
	steps = 0

	while not converged and steps < self.emcee_max_burnin_steps:

		state, lnp , _ = sampler.run_mcmc(state, self.emcee_burnin_steps)
		steps = sampler.chain.shape[1]

		self.plot_chain(sampler,suffix='-burnin.png',parameter_labels=parameter_labels)
		np.save(self.plotprefix+'-state-burnin',np.asarray(self.state))
		
		np.save(self.plotprefix+'-burnin-chain',sampler.flatchain)
		np.save(self.plotprefix+'-burnin-chi2',-2.0*sampler.flatlnprobability)

		kmax = np.argmax(sampler.flatlnprobability)
		p = sampler.flatchain[kmax,:]
		self.p[self.Pi_EE_index] = p[0]
		self.p[self.Pi_EN_index] = p[1]

		np.save(self.plotprefix+'-burnin-min_chi2',-2.0*np.asarray(sampler.flatchain[kmax]))
		print('Minimum chi2 so far:', sampler.flatchain[kmax])

		self.plot_caustic_and_trajectory()
		self.plot_lightcurve()

		converged = self.emcee_has_converged(sampler,n_steps=self.emcee_burnin_steps)


	if steps >= self.emcee_max_burnin_steps:
		print('Maximum number of steps reached. Terminating burn-in.')

	print('lowest chi2 at', sampler.flatchain[np.argmax(sampler.flatlnprobability)])

	self.plot_chain(sampler,suffix='-burnin.png',parameter_labels=parameter_labels)

	sampler.reset()

	print("Running production...")

	state, lnp, _ = sampler.run_mcmc(state, self.emcee_production_steps, lnprob0=lnp)

	print('lowest chi2 at', sampler.flatchain[np.argmax(sampler.flatlnprobability)].tolist())

	np.save(self.plotprefix+'-chain',sampler.flatchain)
	np.save(self.plotprefix+'-chi2',-2.0*sampler.flatlnprobability)

	self.samples = sampler.flatchain
	self.samples_lnp = sampler.flatlnprobability

	params = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(self.samples, \
						[16, 50, 84], axis=0))]

	print('parameter means and uncertainties:')
	print(params)

	p = np.asarray(params)[:,0]
	self.p[self.Pi_EE_index] = p[0]
	self.p[self.Pi_EN_index] = p[1]

	self.plot_chain(sampler,parameter_labels=parameter_labels)
	self.plot_chain_corner(parameter_labels=parameter_labels)
	np.save(self.plotprefix+'-state-production',np.asarray(self.state))
	np.save(self.plotprefix+'-min_chi2-production',np.asarray(sampler.flatchain[np.argmax(sampler.flatlnprobability)]))

	return params

