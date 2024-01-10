import numpy as np

def add_error_bar_scaling_multiplicative(self):

	""" Add a factor per data set to multiply each data uncertainty."""

	self.scale_error_bars_multiplicative = True

	self.error_bar_scale_index = {}

	for site in self.data:

		self.error_bar_scale_index[site] = self.dims
		self.p = np.hstack((self.p,1.0))
		self.p_sig = np.hstack((self.p_sig,0.001))
		self.freeze = np.hstack((self.freeze,np.zeros(1,dtype=int)))
		self.parameter_labels.append("s-"+site[3])
		self.dims += 1
		
def add_spitzer_error_bar_scaling_multiplicative(self):

	""" Add a factor for Spitzer data to multiply each data uncertainty by (defaults to 1.0)."""

	self.scale_spitzer_error_bars_multiplicative = True

	self.spitzer_error_bar_scale_index = self.dims
	self.p = np.hstack((self.p,1.0)) # defaults to 1.0
	self.p_sig = np.hstack((self.p_sig,0.001))
	self.freeze = np.hstack((self.freeze,np.zeros(1,dtype=int)))
	self.parameter_labels.append(r"$s_{\rm Spitzer}$")
	self.dims += 1
	self.spitzer_error_bar_scale_min = 0.
	self.spitzer_error_bar_scale_max = 1e6


def add_model_bad_data(self,bad_sigma=None,bad_mean=None,bad_prob=1.e-4,bad_prob_sigma=1.e-8):

	""" Model the data outliers using a Gaussian mixture model with fixed mean and variance.
		See Hogg, Bovy & Lang, https://arxiv.org/pdf/1008.4686.pdf Equation 17 """

	self.model_bad_data = True

	self.bad_data_probability_index  = {}
	self.bad_data_variance = {}
	self.bad_data_mean = {}

	for site in self.data:

		self.bad_data_probability_index[site] = self.dims
		self.p = np.hstack((self.p,bad_prob))
		self.p_sig = np.hstack((self.p_sig,bad_prob_sigma))
		self.freeze = np.hstack((self.freeze,np.zeros(1,dtype=int)))
		self.parameter_labels.append(r"$%s$"%site[:4])
		self.dims += 1

		if bad_sigma is None:
			self.bad_data_variance[site] = (np.max(self.data[site][1]) - np.min(self.data[site][1]))**2
		else:
			self.bad_data_variance[site] = bad_sigma**2

		if bad_mean is None:
			self.bad_data_mean[site] = np.median(self.data[site][1])
		else:
			self.bad_data_mean[site] = bad_mean

def reformat_data(self):

	self.data_range = {}
	self.data_type = {}
	self.ts = np.arange(0,dtype=np.float64)
	self.mags = np.arange(0,dtype=np.float32)
	self.sigs = np.arange(0,dtype=np.float32)
	self.data_sets = 0

	if not(self.data is None):

		dcount = 0
		for site in self.data:

			ndata = len(self.data[site][0])
			self.data_range[site] = (dcount,dcount+ndata)
			dcount += ndata
			self.data_type[site] = 1
			self.ts = np.hstack((self.ts,self.data[site][0]))
			self.mags = np.hstack((self.mags,self.data[site][1]))
			self.sigs = np.hstack((self.sigs,self.data[site][2]))
			self.data_sets += 1

		self.mags = np.float32(self.mags)
		self.sigs = np.float32(self.sigs)


def renormalise_data_uncertainties_simple(self,p=None,source=None,excl_fraction=None):

	"""Adjust data uncertainties to force reduced chi^2 to 1 for each data source."""

	print('Renormalising data uncertainties using fluxes')

	if p is None:
		p = self.p.copy()

	if excl_fraction is None:

		for site in self.data:

			if source is None or site == source:

				chi2, fs, fb, _, _, _ = self.chi2_calc(p,source=site)
				scale = np.sqrt(chi2/(len(self.data[site][0])))

				sigma = self.data[site][2]
				sigma *= scale
				self.data[site] = (self.data[site][0],self.data[site][1],sigma)

				print('Scale factor for',site,'is',scale)

	else:

		chi2, fs, fb, _, _, chi2_points = self.chi2_calc(p)

		for site in self.data:

			if source is None or site == source:

				threshold = np.percentile(chi2_points[site],100*(1.0-excl_fraction))
				q = np.where(chi2_points[site]<threshold)[0]
				scale = np.sqrt(np.sum(chi2_points[site][q])/(len(q)))

				sigma = self.data[site][2]
				sigma *= scale
				self.data[site] = (self.data[site][0],self.data[site][1],sigma)

				print('Scale factor for',site,'is',scale)


	self.reformat_data()


def renormalise_data_uncertainties(self,p=None,source=None,threshold=0.9,min_high_points=16,iterations=10,chi2_tolerance=1,coefficients={}):

	"""Adjust data uncertainties by an added amount and a scale factor to force reduced chi^2 to 
	   approximately 1 for each data source, with no trend with magnification. See Yee et al., 2012, ApJ, 755, 102.
	   Return a dictionary of renormalisation coefficients."""

	from scipy.optimize import least_squares, minimize_scalar

	def chi2_scale(x, p, site, mag, err_mag, low_points, high_points, n_low, n_high):
		k = x[0]
		eps2 = x[1]**2
		sigma = k*np.sqrt(err_mag**2+eps2)
		print('mean sigma', np.mean(sigma))
		y = 10.0**(0.4*(self.zp-mag))
		dy = np.abs(10.0**(0.4*(self.zp-mag+sigma))-y)
		if site == 'spitzer':
			self.spitzer_data = (self.spitzer_data[0],y,dy)
		else:
			self.data[site] = (self.data[site][0],y,dy)
			self.reformat_data()
		_, _, _, _, _, chi2 = self.chi2_calc(p)
		eqns = [(np.sum(chi2[site][low_points])-4.0)/np.float(n_low_points) - 1.0, (np.sum(chi2[site][high_points])-4.0)/np.float(n_high_points) - 1.0]
		print(site, x, n_low_points-4, n_high_points-4, np.sum(chi2[site][low_points]), np.sum(chi2[site][high_points]), eqns)
		return eqns


	def chi2_eps(x, p, site, mag, err_mag, high_points, n_high, k):
		eps2 = x**2
		sigma = k*np.sqrt(err_mag**2+eps2)
		#print 'mean sigma', np.mean(sigma)
		y = 10.0**(0.4*(self.zp-mag))
		dy = np.abs(10.0**(0.4*(self.zp-mag+sigma))-y)
		if site == 'spitzer':
			self.spitzer_data = (self.spitzer_data[0],y,dy)
		else:
			self.data[site] = (self.data[site][0],y,dy)
			self.reformat_data()
		_, _, _, _, _, chi2 = self.chi2_calc(p)
		eqn = np.sum(chi2[site][high_points])/np.float(n_high_points) - 1.0
		#print site, x, n_high_points, np.sum(chi2[site][high_points]), eqn
		return eqn

	def chi2_k(p, site, mag, err_mag, low_points, n_low,eps):
		eps2 = eps**2
		sigma = np.sqrt(err_mag**2+eps2)
		y = 10.0**(0.4*(self.zp-mag))
		dy = np.abs(10.0**(0.4*(self.zp-mag+sigma))-y)
		dy_original = dy
		if site == 'spitzer':
			self.spitzer_data = (self.spitzer_data[0],y,dy)
			_, _, _, _, _, chi2 = self.chi2_calc(p,use_spitzer_only=True)
		else:
			self.data[site] = (self.data[site][0],y,dy)
			self.reformat_data()
			_, _, _, _, _, chi2 = self.chi2_calc(p,source=site)

		scale = np.sqrt(np.sum(chi2[site][low_points])/n_low)
		# sigma = scale*np.sqrt(err_mag**2+eps2)
		# y = 10.0**(0.4*(self.zp-mag))
		# dy = np.abs(10.0**(0.4*(self.zp-mag+sigma))-y)
		# if site == 'spitzer':
		# 	self.spitzer_data = (self.spitzer_data[0],y,dy)
		# else:
		# 	self.data[site] = (self.data[site][0],y,dy)
		# 	self.reformat_data()
		# _, _, _, _, _, chi2 = self.chi2_calc(p)
		#print 'Scale factor for',site,'is',scale
		# print 'n_low =',n_low_points,'  chi2_low =', np.sum(chi2[site][low_points])
		return scale


	def fh(x, p, site, mag, err_mag, high_points, n_high, low_points, n_low):
		k = np.min([1.5,np.max([0.5,chi2_k(p,site,magnitude,err_mag,low_points,n_low_points,x)])])
		f_high = chi2_eps(x, p, site, mag, err_mag, high_points, n_high, k)
		return f_high**2


	print('Renormalising data uncertainties')
	print('source:', source)

	if p is None:
		p = self.p.copy()
		
	if not isinstance(self.limb_constant, float):
		if self.spitzer_limb_constant != self.limb_constant['spitzer']:
			print('\n\n\n Replacing Spitzer limb constant (%f) with %f.\n\n\n' %(self.spitzer_limb_constant,self.limb_constant['spitzer']))
			self.spitzer_limb_constant = self.limb_constant['spitzer']

	sites = [site for site in self.data]
	if self.use_spitzer:
		sites.append('spitzer')

	results = {}

	for site in sites:

		if (source is None) or (site == source):

			print('site:', site)

			if site == 'spitzer':
				magnitude_0 = self.zp - 2.5*np.log10(self.spitzer_data[1])
				err_magnitude_0 = np.abs(self.zp - 2.5*np.log10(self.spitzer_data[1]+self.spitzer_data[2]) - magnitude_0)
				chi2_0, _, _, _, _, _ = self.chi2_calc(p,use_spitzer_only=True)
			else:
				magnitude_0 = self.zp - 2.5*np.log10(self.data[site][1])
				err_magnitude_0 = np.abs(self.zp - 2.5*np.log10(self.data[site][1]+self.data[site][2]) - magnitude_0)
				chi2_0, _, _, _, _, _ = self.chi2_calc(p,source=site)
			print('site, chi2_0', site, chi2_0)

			if site in coefficients:

				print('using supplied coefficients', coefficients[site])
				k = coefficients[site][0]
				eps = coefficients[site][1]

				sigma = k*np.sqrt(err_magnitude_0**2+eps**2)

				y = 10.0**(0.4*(self.zp-magnitude_0))
				dy = 10.0**(0.4*(self.zp-magnitude_0+sigma))-y
				if site == 'spitzer':
					self.spitzer_data = (self.spitzer_data[0],self.spitzer_data[1],dy)
				else:
					self.data[site] = (self.data[site][0],self.data[site][1],dy)
					self.reformat_data()

			else:

				if site == 'spitzer':
					
					mag = self.spitzer_magnification(self.spitzer_data[0], p=p)
				elif isinstance(self.limb_constant, float):
					mag = self.magnification(self.data[site][0])
				else:
					mag = self.magnification(self.data[site][0], LD=self.limb_constant[site])

				ind = np.argsort(mag)


				print('mean error',np.mean(err_magnitude_0))
				print('min error',np.min(err_magnitude_0))

				thresh = threshold
				n_high_points = 0
				while n_high_points < min_high_points:
					thresh *= 0.9
					high_points = np.where(mag > thresh*(np.max(mag)-1.0) + 1)[0]
					n_high_points = len(high_points)
					print(thresh, thresh*(np.max(mag)-1.0) + 1, n_high_points)

				low_points = np.where(mag <= thresh*(np.max(mag)-1.0) + 1)[0]
				n_low_points = len(low_points)

				# result = least_squares(chi2_scale, np.array([1.0,np.min(err_magnitude)]),method='dogbox', bounds=([0.1, 0.1*np.min(err_magnitude)],[10.0, 0.2]),  \
				# 			args=(p,site,magnitude,err_magnitude,low_points,high_points,n_low_points,n_high_points))

				# print result
				# k = result.x[0]
				# eps2 = result.x[1]**2

				# k = 0.3
				# for i in range(3):
				# 	result = minimize_scalar(chi2_eps,bracket=(0.002,0.003),bounds=(0.001,0.5),method='Bounded',args=(p,site,magnitude,err_magnitude,high_points,n_high_points,k))
				# 	print result
				# 	eps = result.x
				# 	k = chi2_k(p,site,magnitude,err_magnitude,low_points,n_low_points,eps)

				k_total = 1.0
				eps_2_total = 0.0
				for i in range(iterations):

					if site == 'spitzer':
						magnitude = self.zp - 2.5*np.log10(self.spitzer_data[1])
						err_magnitude = np.abs(self.zp - 2.5*np.log10(self.spitzer_data[1]+self.spitzer_data[2]) - magnitude)
					else:
						magnitude = self.zp - 2.5*np.log10(self.data[site][1])
						err_magnitude = np.abs(self.zp - 2.5*np.log10(self.data[site][1]+self.data[site][2]) - magnitude)

					result = minimize_scalar(fh,bracket=(0.000002,0.003),bounds=(0.001,0.5),method='Brent',args=(p,site,magnitude,err_magnitude,high_points,n_high_points,low_points,n_low_points))
					print('iteration', i)
					#print result
					eps = result.x

					k = chi2_k(p,site,magnitude,err_magnitude,low_points,n_low_points,eps)

					eps_2_total += eps**2/k_total**2
					k_total *= k

					print('eps_2_total', eps_2_total)
					print('k_total', k_total)

					sigma = k*np.sqrt(err_magnitude**2+eps**2)

					y = 10.0**(0.4*(self.zp-magnitude))
					dy = 10.0**(0.4*(self.zp-magnitude+sigma))-y
					if site == 'spitzer':
						self.spitzer_data = (self.spitzer_data[0],self.spitzer_data[1],dy)
					else:
						self.data[site] = (self.data[site][0],self.data[site][1],dy)
						self.reformat_data()

					k = k_total
					eps = np.sqrt(eps_2_total)


				# Revert to simple scaling if the above procedure has failed
				if site == 'spitzer':
					chi2_total, _, _, _, _, _ = self.chi2_calc(p,use_spitzer_only=True)
					npts = len(self.spitzer_data[0])
				else:
					chi2_total, _, _, _, _, _ = self.chi2_calc(p,source=site)
					npts = len(self.data[site][0])
				print('chi2 total = ',chi2_total,'for',npts,'data points')
				if np.abs(chi2_total - npts) > chi2_tolerance:
					print('Using simple scaling')
					eps = 0.0

					k_total = 1.0
					for i in range(iterations):


						k = np.sqrt(chi2_0/npts)
						sigma = k*err_magnitude_0
						y = 10.0**(0.4*(self.zp-magnitude_0))
						dy = 10.0**(0.4*(self.zp-magnitude_0+sigma))-y

						if site == 'spitzer':
							print('iteration', i)
							print(y)
							print(self.spitzer_data[1])
							print(err_magnitude_0)
							print(k, chi2_0, npts)
							print(dy)
							print()


						if site == 'spitzer':
							self.spitzer_data = (self.spitzer_data[0],self.spitzer_data[1],dy)
						else:
							self.data[site] = (self.data[site][0],self.data[site][1],dy)
						self.reformat_data()

						if site == 'spitzer':
							magnitude_0 = self.zp - 2.5*np.log10(self.spitzer_data[1])
							err_magnitude_0 = np.abs(self.zp - 2.5*np.log10(self.spitzer_data[1]+self.spitzer_data[2]) - magnitude_0)
						else:
							magnitude_0 = self.zp - 2.5*np.log10(self.data[site][1])
							err_magnitude_0 = np.abs(self.zp - 2.5*np.log10(self.data[site][1]+self.data[site][2]) - magnitude_0)
						if site == 'spitzer':
							chi2_0, _, _, _, _, _ = self.chi2_calc(p,use_spitzer_only=True)
						else:
							chi2_0, _, _, _, _, _ = self.chi2_calc(p,source=site)

						k_total *= k

					k = k_total


			print(site)
			print('scale factor:', k)
			print('added magnitude uncertainty:', eps)

			results[site] = (k,eps)

	return results


def set_zero_blending(self,reference_flux):
	"""
	reference_flux is a disctionary of reference fluxes for each site, to be added to the data
	Typically this would be zero for OGLE (since f_ref is already included), and large positive
	numbers for KMT.
	"""

	if self.treat_flux_parameters_as_nonlinear:
		raise ValueError("Cannot set zero blending when treat_flux_parameters_as_nonlinear is True.")

	for site in self.data:
		i0 = self.data_range[site][0]
		i1 = self.data_range[site][1]
		self.mags[i0:i1] += reference_flux[site]
		self.data[site] = (self.data[site][0],self.data[site][1]+reference_flux[site],self.data[site][2])

	self.use_zero_blending = True

	return



def bad_data_ln_prior_prob(self,p):
	lp = 0.0
	for site in self.data:
		f = p[self.bad_data_probability_index[site]]
		if  f < 0.0:
			return -np.inf
		if f > 1.0:
			return -np.inf
		lp += np.log((1.0-f)/(1.0+f))
	return lp


def error_bar_scale_ln_prior_prob(self,p):
	lp = 0.0
	f_mean = 1.0
	f_sig = 3.0
	for site in self.data:
		f = p[self.error_bar_scale_index[site]]
		if  f < 0.0:
			return -np.inf
		lp -= (f-f_mean)**2/(2*f_sig**2) # additional chi2 compenent from scaling
	return lp
	
def lnprior_scale_spitzer_multiplicative(self,p):
	lp = 0.0
	S_sig = 5.0 # 1 sig error scaling estimate
	S = p[self.spitzer_error_bar_scale_index].copy()
	Smin = self.spitzer_error_bar_scale_min 
	Smax = self.spitzer_error_bar_scale_max
	if (S <= Smin) or (S >= Smax):
		lp += -np.inf	
	else:
	    lp -= (S-1.)**2/(2*S_sig**2) # scaling prior guess
	return lp
	
def known_parameters_prior(self,p):
	lp = 0.0
	for prior in self.known_parameters:
		index, mu, sig = prior
		x = p[index] #fit parameter x of index
		lp -= (x-mu)**2/(2*sig**2) # additional chi2 compenent from scaling
	return lp
	
def murelgeo_lnprior(self,p):
	lp = 0.0
	
	#prior
	murel, murel_error = self.murelgeo
	murelN, murelE = murel
	theta_star = self.thetastar_prior
	
	#sample
	mu_geo, _, mu_err = self.compute_relative_proper_motion(theta_star,p=p,p_sig=p_sig,quiet=quiet)
	murelN_sample,murelE_sample = self.mu_geo
	
	#combined error
	murelN_error =  np.sqrt(murel_error[0]**2+mu_err[0]**2)
	murelE_error = np.sqrt(murel_error[1]**2+mu_err[1]**2)
	
	# additional chi2 component
	lp -= (murelN_sample-murelN)**2/(2*murelN_error**2) #from N
	lp -= (murelE_sample-murelE)**2/(2*murelE_error**2) #from E
	return lp
	
