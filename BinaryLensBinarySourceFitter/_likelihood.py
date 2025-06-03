import numpy as np
import sys

def chi2_calc(self, p_in=None, source=None, use_spitzer_only=False, compute_uncertainties=False):

	"""Compute chi^2 for all data sources or for a single source."""

	if source is None:
		source = []

	if p_in is None:
		p_in = self.p.copy()
	else:
		self.p = p_in.copy()
	self.match_p()
	p1 = self.primary.p.copy()
	p2 = self.secondary.p.copy()

	#if d <= 0.0 or q <= 0.0 or rho <= 1e-7 or d > 20:
	#	return np.inf, 0, 0, 0, 0, 0
	
	logd, logq, logrho1, u01, phi1, t01, tE1, piEE1, piEN1, logrho2, deltau0, deltaphi, deltat0, tE2 = p_in[:14]
	u02 = u01 + deltau0
	phi2 = phi1 + deltaphi
	t02 = t01 + deltat0
	d		=	10.0**np.float64(logd)  # s
	q		=	10.0**np.float64(logq)
	rho1		=	10.0**np.float64(logrho1)
	rho2		=	10.0**np.float64(logrho2)
	piEN1 = np.float64(piEN1)
	piEE1 = np.float64(piEE1)

	a0 = {}
	a1 = {}
	a2 = {}
	sigma_a0 = {}
	sigma_a1 = {}
	sigma_a2 = {}
	chi2_sum = 0.0
	chi2_elements = {}
		
	if not use_spitzer_only:
		if isinstance(self.limb_constant, float):  	# no site specific 
			if self.debug:									# limb darkening
				print('A')
				print(p_in,p1)
			A_all1 = self.primary.magnification(self.primary.ts,p=p1)
			if self.debug:
				print('B')
				print(p_in,p2)
			A_all2 = self.secondary.magnification(self.primary.ts,p=p2)  
						# I have been using self.primary only for data calls
			if self.debug:
				print('C')
		else:
			A_all1 = A_all2 = None  # site specific limb darkening - to be 
									# added later

		for k, site in enumerate(self.primary.data): 
			if site in source or not source:  # e.g. CTIO-43, but not Spitzer

				# A 
				if A_all1 is not None:  # then A_all2 is also not None - no
										# specific limb darkening
					A1 = A_all1[self.primary.data_range[site][0]\
								:self.primary.data_range[site][1]]  
								# A for a single site
					A2 = A_all2[self.primary.data_range[site][0]\
								:self.primary.data_range[site][1]]
				else:  # using site specific limb darkening
					if self.debug:
						print('C')
					LD 	= self.limb_constant[site]  
					A1 	= self.primary.magnification(\
							self.primary.ts[self.primary.data_range[site][0]\
							:self.primary.data_range[site][1]]\
													, p=p1, LD=LD)
					A2 	= self.secondary.magnification(\
							self.primary.ts[self.primary.data_range[site][0]\
							:self.primary.data_range[site][1]]\
														, p=p2, LD=LD)
				
				# sig^2
				if self.scale_error_bars_multiplicative:
					sig2 = (p_in[self.error_bar_scale_index[site]]\
						*self.primary.data[site][2])**2
				else:
					sig2 = self.primary.data[site][2]**2
					
				if self.debug:
					print('D')

				# fit a0, a1, a2
				if self.treat_flux_parameters_as_nonlinear:
					a0[site] = p_in[self.flux_index[site]]
					a1[site] = p_in[self.flux_index[site]+1]
					if self.use_zero_blending:
						a2[site] = 0
					else:
						a2[site] = p_in[self.flux_index[site]+2]
					chi2_elements[site] = (A1*a0[site] + A2*a1[site] + a2[site] - self.primary.data[site][1])**2 / sig2 

				# solve for a0, a1, a2
				else:
					a0[site] = 0.0
					a1[site] = 0.0
					a2[site] = 0.0
					
					for repeat in (0,1):	# repeat if -50<fs2<0, with fs2=0
						accept = np.arange(len(A1)) # A1 and A2 are the same  
								     				# length, because they are
								     				# the length of the data
						if ((repeat == 1) and (a1[site] == 0.0)) or repeat==0:
						
							for iteration in range(3):  
								# 3 itterative LS regressions or a0, a1, a2,
								# recursively removing outliers beyond the
								# self.data_outlier_threshold x sigma limit,
								# where sigma is the standard deviation of
								# resisuals.
						
								# from requiring the partial derivatives of chi2
								# to = 1 (i.e. minimising chi2):
								#         M         *    x     =    V
								# |  a    b    c  |   |  a0  |   |  d  |
								# |  e    f    g  | x |  a1  | = |  h  | ,
								# |  i    j    k  |   |  a2  |   |  l  |
								# where a0 is the primary source flux,
								# a1 is the secondary source flux,
								# and s2 is the blend flux.
						
								# M elements
								ma 	= np.sum( A1[accept]*A1[accept]\
									/ sig2[accept] )  
								# matrix equaiton, element a
								mb 	= np.sum( A1[accept]*A2[accept]\
									/ sig2[accept] )
								mc 	= np.sum( A1[accept]\
									/ sig2[accept] )             
								# mc == mi
								me 	= np.sum( A2[accept]*A1[accept]\
									/ sig2[accept] )  
								# me == mb
								mf 	= np.sum( A2[accept]*A2[accept]\
									/ sig2[accept] )
								mg 	= np.sum( A2[accept] \
									/ sig2[accept] )             
								# mg == mj
								mi 	= np.sum( A1[accept]/sig2[accept] )
								mj 	= np.sum( A2[accept]/sig2[accept] )
								mk 	= np.sum( 1.0/sig2[accept] )
						
								# V elements
								md = np.sum((self.primary.data[site][1][accept]\
									* A1[accept]) / sig2[accept] )
								mh = np.sum((self.primary.data[site][1][accept]\
								 	* A2[accept]) / sig2[accept] )
								ml = np.sum(self.primary.data[site][1][accept]\
									/ sig2[accept] )
							
								if self.use_zero_blending:
									if repeat == 0:
										M 	= np.array([ [ma,mb], [me,mf] ])
										V 	= np.array([md,mh])
									elif repeat == 1:
										M	= np.array([ [ma] ])
										V	= np.array([ md ])
								else:
									if repeat == 0:
										M 	= np.array([ [ma,mb,mc], [me,mf,mg]\
											, [mi,mj,mk] ])  		# M - matrix
										V 	= np.array([md,mh,ml])	# V - vector
									elif repeat == 1:
										M 	= np.array([ [ma, mc], [mi, mk] ]) 
										V 	= np.array([ md, ml ])	
						
								# Solve the matrix equation	
								try:
									x = np.linalg.solve(M,V)
								except:
									print('Warning: numpy.linalg.solve failed')
									print('a must be square and of full-rank,'\
											+' i.e., all rows (or, '\
											+'equivalently, columns) must be '\
											+'linearly independent; if either '\
											+'is not true, use lstsq for the '\
											+'least-squares best “solution” of'\
											+' the system/equation.')
									x = np.linalg.lstsq(M,V,rcond=None)[0]  
									# https://numpy.org/doc/stable/reference/
									#generated/numpy.linalg.lstsq.html#numpy.
									#linalg.lstsq  
									# Note: the rcond default has changed from
									# old versions.
								else:
									if self.use_zero_blending:
										if repeat == 0:
											a0[site], a1[site] = x 
											a2[site] = 0.
										if repeat == 1:
											a1[site] = a2[site] = 0.
											a2[site] = md/ma	
									else:
										if repeat == 0:
											a0[site], a1[site], a2[site] = x
										if repeat == 1:
											a0[site], a2[site] = x
											a1[site] = 0.

								delta 	= ( A1[accept]*a0[site] + A2[accept] \
										* a1[site] + a2[site] \
										- self.primary.data[site][1][accept]\
										 ) / np.sqrt(sig2[accept])

								sig_delta = np.std(delta)
								accept 	= accept[np.abs(delta/sig_delta) \
												< self.data_outlier_threshold]

							if compute_uncertainties:
								M_inv = np.linalg.inv(M)
								sigma_a0[site] = np.sqrt(M_inv[0,0])
								sigma_a1[site] = np.sqrt(M_inv[1,1])
								if not self.use_zero_blending:
									sigma_a2[site] = np.sqrt(M_inv[2,2])
				
							if (a0[site] < 0.) or (a1[site] < 0.):
								chi2_elements[site] = A1*1e10	
								# when shit is hitting the fan for some reason
								if self.debug or repeat==1:
									print('Warning: source flux less than 0')
									print('site =',site)
									print('Fs1, Fs2, Fb:', a0[site], a1[site]\
										, a2[site])
								if a1[site] < 0.0 and a1[site] > -50.0 \
															and repeat==0:
									print('Fixing %s Fs2 at 0' %site)
									a1[site] = 0.0
							else:
								chi2_elements[site] = ( A1*a0[site] \
													+ A2*a1[site] + a2[site] \
												- self.primary.data[site][1]\
												 	)**2 / sig2
							
							if self.debug:
								print(site,' Fs1, Fs2, Fb:', a0[site], a1[site]\
										, a2[site])

				chi2_sum += np.sum(chi2_elements[site]) 
				# must be outside the repeat loop
				

	if not source or (self.spitzer_flux_ratio_reference_site in source):
		site = 'spitzer'
		
		if not (isinstance(self.limb_constant, float)) :  	# site specific 
															# limb darkening
			if (site in self.limb_constant.keys()) \
			and (self.spitzer_limb_constant != self.limb_constant[site]):
				print('\n\n')
				print('Replacing Spitzer limb constant (%f) with %f.' \
						%(self.spitzer_limb_constant,self.limb_constant[site]) )
				print('\n\n')
				self.spitzer_limb_constant = self.limb_constant[site]
		AS1 = self.primary.spitzer_magnification( \
											self.primary.spitzer_data[0], p=p1\
											, LD=self.spitzer_limb_constant)
		AS2 = self.secondary.spitzer_magnification( \
											self.primary.spitzer_data[0], p=p2\
											, LD=self.spitzer_limb_constant)

		if self.scale_error_bars_multiplicative:
			if (site in self.error_bar_scale_index.keys()):
				sig 	= p_in[self.error_bar_scale_index[site]] \
						* self.primary.spitzer_data[2]	
		elif self.scale_spitzer_error_bars_multiplicative:
			sig 	= p_in[self.spitzer_error_bar_scale_index] \
					* self.primary.spitzer_data[2]
		else:
			sig = self.primary.spitzer_data[2].copy()
		sig2 = (sig)**2

		if self.treat_flux_parameters_as_nonlinear:
			a0[site] = p_in[self.flux_index[site]]
			a1[site] = p_in[self.flux_index[site]+1]
			if self.use_zero_blending: # I'm not actually sure the code allows for these two things happening together
				a2[site] = 0
			else:
				a2[site] = p_in[self.flux_index[site]+2]

		else:
			#if self.use_zero_blending:
			#	a0[site] = np.sum(A*self.spitzer_data[1]/sig2) 
			#				/ np.sum(A*A/sig2)
			#	a1[site] = 0.0

			ma = 		np.sum( AS1*AS1/sig2 )  # matrix equaiton, element a
			mb = me	=	np.sum( AS1*AS2/sig2 )  # mb == me
			mc = mi = 	np.sum( AS1/sig2 )     # mc == mi
			mf = 		np.sum( AS2*AS2/sig2 )
			mg = mj = 	np.sum( AS2/sig2 )     # mg == mj
			mk = 		np.sum( 1.0/sig2 )
						
			md = np.sum( self.primary.spitzer_data[1]*AS1/sig2 )
			mh = np.sum( self.primary.spitzer_data[1]*AS2/sig2 )
			ml = np.sum( self.primary.spitzer_data[1]/sig2 )
			
			if self.use_zero_blending:
				M = M = np.array([ [ma,mb], [me,mf] ])
				V = np.array([md,mh])
			else:
				M = np.array([ [ma,mb,mc], [me,mf,mg], [mi,mj,mk] ])
				V = np.array([md,mh,ml])  
			
			# Solve the matrix equation
			try:
				x = np.linalg.solve(M,V)
			except:
				print('Warning: numpy.linalg.solve failed')
				print('a must be square and of full-rank, i.e., all rows (or, '\
					+'equivalently, columns) must be linearly independent; if '\
					+'either is not true, use lstsq for the least-squares best'\
					+'“solution” of the system/equation.')
				x = np.linalg.lstsq(M,V,rcond=None)[0]
			else:
				if self.use_zero_blending:
					a0[site], a1[site] = x 
					a2[site] = 0.
				else:
					a0[site], a1[site], a2[site] = x 
			
			if compute_uncertainties:
				M_inv = np.linalg.inv(M)
				sigma_a0[site] = np.sqrt(M_inv[0,0])
				sigma_a1[site] = np.sqrt(M_inv[1,1])
				if not self.use_zero_blending:
					sigma_a2[site] = np.sqrt(M_inv[2,2])
			
		if self.use_gaussian_process_model:
			N = self.primary.spitzer_data[0].shape[0]
			sys.exit(\
				'This path has not been rewriten for a binary source model')
			chi2_site = self.gaussian_process_spitzer_chi2(A, a0, a1, p=p_in)
			chi2_elements[site] 	= np.ones_like(self.spitzer_data[1]) \
									* chi2_site/N
			chi2_sum += chi2_site
		else:
			chi2_elements[site] 	= ( AS1*a0[site] + AS2*a1[site] + a0[site] \
									- self.spitzer_data[1] )**2 \
									/ sig2
			chi2_sum += np.sum(chi2_elements[site])
		
		chi2_extra = 0 	# because I'm adding it to the thigns that are always
						# returned
		if self.spitzer_has_colour_constraint and not use_spitzer_only:
			if (a0['spitzer'] <= 0) \
			or (a0[self.spitzer_flux_ratio_reference_site]<= 0) \
			or (a1['spitzer'] <= 0) \
			or (a1[self.spitzer_flux_ratio_reference_site]<= 0):
				print('Warning: source flux less than 0')
				print('site =',site)
				print('Fs1, Fs2, Fb:', a0[site], a1[site], a2[site])
				chi2_extra = 1e10  # big numbers are always floats
			else:
				#flux_ratio = ( a0[self.spitzer_flux_ratio_reference_site]
				#+a1[self.spitzer_flux_ratio_reference_site]) / (a0['spitzer'] 
				#+ a1['spitzer'])
				flux_ratio 	= a0[self.spitzer_flux_ratio_reference_site] \
							/ a0['spitzer']
				#print(a0, self.spitzer_flux_ratio_reference_site, flux_ratio, 
				#self.spitzer_flux_ratio)
				chi2_extra  = ( 2.5 * np.log10(flux_ratio \
							/ self.spitzer_flux_ratio) )**2 \
							/ self.spitzer_colour_uncertainty**2
			chi2_sum += chi2_extra
			
		if self.debug:
			print(site,' Fs1, Fs2, Fb:', a0[site], a1[site], a2[site])
				
		# if zero blending: a2 = 0.0 and sigma_a2 = {}
		
	return chi2_sum, a0, a1, a2, sigma_a0, sigma_a1, sigma_a2, chi2_extra, chi2_elements


def ln_prior_prob(self,p):
	if (self.range_log_s[0] < p[0] < self.range_log_s[1]) & \
			(self.range_log_q[0] < p[1] < self.range_log_q[1]) & \
			(self.range_log_rho[0] < p[2] < self.range_log_rho[1]) & \
			(np.abs(p[3]) < self.upper_limit_u0) & \
			(np.abs(p[4]) < self.upper_limit_alpha) & \
			(self.range_tE[0] < p[6] < self.range_tE[1]):
		return 0.0
	else:
		return -np.inf


def lnprob(self,q):

	#print('q', q)

	p = np.asarray(self.p.copy())
	p[np.where(1-self.freeze)[0]] = q
	#print(p)
	
	lp = 0.0
	
	for parameter in p:
		if np.isfinite(parameter)==False:
			lp = -np.inf
	if not np.isfinite(lp):
 		return lp
	
	else:

		lp = self.ln_prior_prob(p)
		
		if self.known_parameters is not None:
			lp += self.known_parameters_prior(p)
			
		#if self.murelgeo_prior is not None:
		#	if self.thetastar_prior is not None:
		#		lp += self.murelgeo_lnprior(p)
		#	else:
		#		sys.exit('thetastar prior is required to use a murelgeo prior')

		if self.use_lens_orbital_motion_energy_constraint:
			lp += self.lens_orbital_motion_energy_prior(p)

		if self.model_bad_data:
			lp += self.bad_data_ln_prior_prob(p)

		if self.scale_error_bars_multiplicative:
			lp += self.error_bar_scale_ln_prior_prob(p)
			
		if self.scale_spitzer_error_bars_multiplicative:
			lp += self.lnprior_scale_spitzer_multiplicative(p)
			
		if self.use_limb_darkening:
			lp += self.limb_darkening_ln_prior_prob(p)
			
		if self.use_galactic_prior:
			lp += self.galactic_ln_prior_prob(p)
					
		if self.use_gaussian_process_model:
			lp += self.gaussian_process_prior(p)

		chi2, f_source, _, _, _, chi2_elements = self.chi2_calc(p)
		chi2 /= self.data_variance_scale

		if self.model_bad_data:

			sys.exit('I broke this path')

		elif np.isfinite(lp):

			if self.scale_error_bars_multiplicative:
				for site in self.data:
					if not ((site in self.gaussian_process_sites) and self.chi2_method=='celerite'): #don't do this if using GP specifically on this site
						lp -= len(self.data[site][0]) * np.log(p[self.error_bar_scale_index[site]])
                        
			if self.scale_spitzer_error_bars_multiplicative and not ('spitzer' in self.gaussian_process_sites) and (self.chi2_method=='celerite'): #GP is being used but not for spitzer
    				lp -= len(self.spitzer_data[0])*np.log(p[self.spitzer_error_bar_scale_index])
    				#print('A')
                
			elif self.scale_spitzer_error_bars_multiplicative and self.chi2_method=='manual': #chi2_method default to 'manual' if not using GP
				lp -= len(self.spitzer_data[0])*np.log(p[self.spitzer_error_bar_scale_index])
				#print('B')
                
			lp += chi2/-2.0

			return lp		
				
		else:
			#print('p = ', p)
			return -np.inf


def neg_lnprob(self,q):

	return -self.lnprob(q)
	#return [-self.lnprob(qi) for qi in q] 




