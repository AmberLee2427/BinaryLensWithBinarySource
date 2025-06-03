import numpy as np
import sys

def chi2_calc(self, p_in=None, source=None, use_spitzer_only=False, compute_uncertainties=False):

	"""Compute chi^2 for all data sources or for a single source."""

	if source is None:
		source = []
	if p_in is None:
		p_in = self.p.copy()

	d		=	10.0**np.float64(p_in[0])
	q		=	10.0**np.float64(p_in[1])
	rho		=	10.0**np.float64(p_in[2])

	#if d <= 0.0 or q <= 0.0 or rho <= 1e-7 or d > 20:
	#	return np.inf, 0, 0, 0, 0, 0

	a0 = {}
	a1 = {}
	sigma_a0 = {}
	sigma_a1 = {}

	chi2_sum = 0.0

	chi2_elements = {}
		
	if not use_spitzer_only:

		if isinstance(self.limb_constant, float):
			A_all = self.magnification(self.ts,p=p_in)
		else:
			A_all = None

		for k, site in enumerate(self.data):
			
			if site in source or not source:

				if A_all is not None:
					A = A_all[self.data_range[site][0]:self.data_range[site][1]]
				else:
					LD = self.limb_constant[site]
					A = self.magnification(self.ts[self.data_range[site][0]:self.data_range[site][1]],p=p_in,LD=LD)

				if self.scale_error_bars_multiplicative:

					sig2 = (p_in[self.error_bar_scale_index[site]]*self.data[site][2])**2

				else:

					sig2 = self.data[site][2]**2

				if self.treat_flux_parameters_as_nonlinear:

					a0[site] = p_in[self.flux_index[site]]
					a1[site] = p_in[self.flux_index[site]+1]

					chi2_elements[site] = (A*a0[site] + a1[site] - self.data[site][1])**2 / sig2 

				else:

					a0[site] = 0.0
					a1[site] = 0.0

					accept = np.arange(len(A))

					for iteration in range(3):


						if self.use_zero_blending:

							a0[site] = np.sum(A[accept]*self.data[site][1][accept]/sig2[accept]) / np.sum(A[accept]**2/sig2[accept])

							delta = (A[accept]*a0[site] - self.data[site][1][accept])/np.sqrt(sig2[accept])

						else:

							ma = np.sum( A[accept]*A[accept]/sig2[accept] )
							mb = np.sum( A[accept]/sig2[accept] )
							md = np.sum( 1.0/sig2[accept] )
							me = np.sum( self.data[site][1][accept]*A[accept]/sig2[accept] )
							mf = np.sum( self.data[site][1][accept]/sig2[accept] )
							
							denom = ma*md - mb*mb
							if denom!=0:
								a0[site] = (md*me-mb*mf)/denom
								a1[site] = (ma*mf-mb*me)/denom

							delta = (A[accept]*a0[site] + a1[site] - self.data[site][1][accept])/np.sqrt(sig2[accept])

						
						sig_delta = np.std(delta)

						accept = accept[np.abs(delta/sig_delta) < self.data_outlier_threshold]

					if compute_uncertainties:

						if self.use_zero_blending:

							sigma_a0[site] = np.sqrt(1.0/np.sum(A[accept]**2/sig2[accept]))
							a1[site] = 0.0
							sigma_a1[site] = 1.0

						else:

							C = np.array([[ma,mb],[mb,md]])
							C_inv = np.linalg.inv(C)
							sigma_a0[site] = np.sqrt(C_inv[0,0])
							sigma_a1[site] = np.sqrt(C_inv[1,1])
				
					if a0[site] <= 0:
						chi2_elements[site] = A*1e6
					else:
						chi2_elements[site] = (A*a0[site] + a1[site] - self.data[site][1])**2 / sig2

				chi2_sum += np.sum(chi2_elements[site])
				

	if self.use_spitzer and (not source or (self.spitzer_flux_ratio_reference_site in source)):

		site = 'spitzer'
		
		if isinstance(self.limb_constant, float):
			A = self.spitzer_magnification(self.spitzer_data[0], p=p_in)
		else:
			if self.spitzer_limb_constant != self.limb_constant[site]:
				print('\n\n\n Replacing Spitzer limb constant (%f) with %f.\n\n\n' %(self.spitzer_limb_constant,self.limb_constant[site]))
				self.spitzer_limb_constant = self.limb_constant[site]
			A = self.spitzer_magnification(self.spitzer_data[0], p=p_in,LD=self.spitzer_limb_constant)

		if self.scale_error_bars_multiplicative:
			sig = p_in[self.error_bar_scale_index[site]]*self.spitzer_data[2]	
		elif self.scale_spitzer_error_bars_multiplicative:
			sig = p_in[self.spitzer_error_bar_scale_index]*self.spitzer_data[2]
		else:
			sig = self.spitzer_data[2].copy()
		sig2 = (sig)**2

		if self.treat_flux_parameters_as_nonlinear:

			a0[site] = p_in[self.flux_index[site]]
			a1[site] = p_in[self.flux_index[site]+1]

		else:

			if self.use_zero_blending:

				a0[site] = np.sum(A*self.spitzer_data[1]/sig2) / np.sum(A*A/sig2)
				a1[site] = 0.0

			else:

				ma = np.sum(A*A/sig2)
				mb = np.sum(A/sig2)
				md = np.sum(1.0/sig2)
				me = np.sum(self.spitzer_data[1]*A/sig2)
				mf = np.sum(self.spitzer_data[1]/sig2)
				
				denom = ma*md-mb*mb
				if denom!=0:
					a0[site] = (md*me-mb*mf)/denom
					a1[site] = (ma*mf-mb*me)/denom
				else:
					a0[site] = 0.
					a1[site] = 0.

			if compute_uncertainties:

				if self.use_zero_blending:

					sigma_a0[site] = np.sqrt(1.0/np.sum(A*A/sig2))
					a1[site] = 0.0
					sigma_a1[site] = 1.0

				else:

					C = np.array([[ma,mb],[mb,md]])
					C_inv = np.linalg.inv(C)
					sigma_a0[site] = np.sqrt(C_inv[0,0])
					sigma_a1[site] = np.sqrt(C_inv[1,1])

		if self.use_gaussian_process_model:
			N = self.spitzer_data[0].shape[0]
			chi2_site = self.gaussian_process_spitzer_chi2(A,a0,a1,p=p_in)
			chi2_elements[site] = np.ones_like(self.spitzer_data[1]) * chi2_site/N
			chi2_sum += chi2_site
		else:
			chi2_elements[site] = (A*a0[site] + a1[site] -  self.spitzer_data[1])**2 / sig2
			chi2_sum += np.sum(chi2_elements[site])

		if self.spitzer_has_colour_constraint and not use_spitzer_only:

			if (a0['spitzer'] <= 0) or (a0[self.spitzer_flux_ratio_reference_site]<= 0):
				#print('a0 spitzer' , a0['spitzer'])
				chi2_extra = 1e10
			else:
				flux_ratio = a0[self.spitzer_flux_ratio_reference_site]/a0['spitzer']
				#print(a0, self.spitzer_flux_ratio_reference_site, flux_ratio, self.spitzer_flux_ratio)
				chi2_extra = (2.5 * np.log10(flux_ratio/self.spitzer_flux_ratio))**2 / self.spitzer_colour_uncertainty**2
			chi2_sum += chi2_extra
		
	return chi2_sum, a0, a1, sigma_a0, sigma_a1, chi2_elements


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




