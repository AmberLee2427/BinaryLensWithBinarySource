import numpy as np
import sys
from scipy.stats import skewnorm
from scipy.optimize import minimize
from scipy.optimize import curve_fit

#def add_galactic_prior(self,site_data,galactic_l,VI_VK_data_file='besgiant.dat'):
#
#	""" Set up the initial conditions to use a galactic model prior on theta_star, rho, t_E.
#
#		site_data is a dictionary, indexed by site name, with each entry being a tuple of
#			(VmI_source, I_RC, VmI_RC)
#
#	"""
#
#	self.use_galactic_prior = True
#
#	self.galactic_prior_data = site_data
#
#	# constants
#	VmI_RC_0 = 1.06
#
#	# Interpolate I_RC_0 from Table 1 of Nataf et al, 2013, ApJ, 769, 88
#	from scipy.interpolate import PchipInterpolator
#	longitude = np.arange(21) - 9
#	I0 = np.array([14.662,14.624,14.620,14.619,14.616,14.605,14.589,14.554,14.503,14.443, \
#				14.396,14.373,14.350,14.329,14.303,14.277,14.245,14.210,14.177,14.147,14.121])
#	interp = PchipInterpolator(longitude,I0)
#
#	if galactic_l > 300:
#		galactic_l -= 360.0
#
#	if galactic_l < -9.0 or galactic_l > 11.0:
#		raise ValueError('galactic_l must be between -9 and +11')
#
#	self.I_RC_0 = interp(galactic_l)
#
#	# Average over sites
#
#	# Calculate VmI_source_0 = VmI_source + VmI_RC_0 - VmI_RC
#	VmI_source_0 = [site_data[site][0] + VmI_RC_0 - site_data[site][2] for site in site_data]
#
#	self.VmI_source_0 = np.mean(VmI_source_0)
#
#	self.VmK_source_0 = self.VItoVK(self.VmI_source_0,VI_VK_data_file)
#
#
#def galactic_ln_prior_prob(self,p,fs):
#
#	I_source_0 = np.mean([self.zp - 2.5*np.log10(fs[site]) + self.I_RC_0 - self.galactic_prior_data[site][1] \
#							for site in self.galactic_prior_data])
#
#	V_source_0 = self.VmI_source_0 + I_source_0
#
#	qlog2thetastar  = 0.5410+0.2667*self.VmK_source_0 - V_source_0/5.0
#	theta_star = 10.**qlog2thetastar/2.0
#	
#	# add some stuff to get the galaxy ln prior
#
#	return 0.0

		
def neg_skew_pdf(self,x):
	return -1.*skewnorm.pdf(x,
                            self.a, 
                            loc=self.loc, 
                            scale=self.scale)

def fit_skew_normal(self,posterior_values,percentiles):
        
	p0 = np.array([5., posterior_values.mean(), posterior_values.std()])
	print('D-bug #009:', p0)
        
	def skew_ppf(percentile, a, loc, scale):
		x = skewnorm.ppf(percentile, a, loc=loc, scale=scale)
		return x
        
	print('D-bug #010:', posterior_values, percentiles)
	popt, pcov = curve_fit(skew_ppf,percentiles,posterior_values,p0=p0)
	print('D-bug #011:',popt)
        
	return popt
    
def fit_log_skew_normal(self,posterior_values,percentiles):
                         
	p0 = np.array([5., posterior_values.mean(), posterior_values.std()])
        
	print('D-bug #004:', p0)
        
	def log_skew_ppf(percentile, a, loc, scale):
		log_x = np.log10(skewnorm.ppf(percentiles, a, loc=loc, scale=scale))
		print('D-bug #006:', a, loc, scale)
		return log_x
        
	log_x = np.log10(posterior_values)
	print('D-bug #005:',log_x, posterior_values, percentiles)
	popt, pcov = curve_fit(log_skew_ppf,percentiles,log_x,p0=p0)
	print('D-bug #002:',popt)
        
	return popt
        
def set_skew_parameters(self, a, loc, scale):
        
	print('Setting skew normal parameters')
        
	self.a, self.loc, self.scale = a, loc, scale
	print('a, loc, scale:', a, loc, scale)
	self.mean, self.var, self.skew, self.kurt = skewnorm.stats(self.a, 
                                                                   loc=self.loc, 
                                                                   scale=self.scale, 
                                                                   moments='mvsk')
	print('mean, var, skew, kurtoise:', self.mean, self.var, self.skew, self.kurt)        
	self.median = skewnorm.median(self.a, 
                                      loc=self.loc, 
                                      scale=self.scale)
	res = minimize(self.neg_skew_pdf,
                       self.median) # using negative function because we are after the max 
	print('res:', res)
	self.mode = float(res.x) # x coordinate of the optima is the same for negative or positive function
	print('median, mode:', self.median, self.mode)
	self.skewnormal_renorm = skewnorm.pdf(self.mode, 
                          self.a, 
                          loc=self.loc, 
                          scale=self.scale) # so that P(mu) = 1)
	print('renorm', self.skewnormal_renorm)  
        
def skewnormal_prior(self,x):
	lp = 0.
	P = skewnorm.pdf(x, 
                         self.a, 
                         loc=self.loc, 
                         scale=self.scale)
	print('P inside prior func:', P, P/self.skewnormal_renorm)
	lp = np.log(P/self.skewnormal_renorm)
	return lp
    
def add_galactic_prior(self,posterior_values,percentiles,parameter,colour=None):

	""" Set up the initial conditions to use a galactic model prior
	"""

	self.use_galactic_prior = True
	self.galactic_prior_parameter = parameter
	if colour is not None:
		self.colour_params_for_galactic_prior = colour
	

	if (parameter == 'host mass') or (parameter == 'companion mass'):
		a, loc, scale = self.fit_log_skew_normal(posterior_values,percentiles)
		print('D-bug #003:',posterior_values,percentiles)
		print('D-bug #001:',a,loc,scale)
	elif (parameter == 'lens distance') or (parameter == 'seperation'):
		a, loc, scale = self.fit_skew_normal(posterior_values,percentiles)
		print('D-bug #007:', a, loc, scale)
	else:
		sys.exit('invalid parameter option. '+
                     "Valid options are \'host mass\', \'companion mass\', '\'lens distance\', or \'seperation\'")
        
	print('D-bug #008:', a, loc, scale)
	self.set_skew_parameters(a,loc,scale)
    
def galactic_ln_prior_prob(self,p=None,sig=None):

	if p is not None:
		self.p = p.copy()
	if sig is not None:
		self.p_sig = sig.copy()

	VI_source_0, VI_source_0_err, I_source_0, I_source_0_err, D_clump, theta_star, theta_star_err = self.colour_params_for_galactic_prior
	
	self.compute_physical_properties(self.p, self.p_sig, VI_source_0, VI_source_0_err, I_source_0, I_source_0_err, D_clump, theta_star=theta_star, theta_star_err=theta_star_err,quiet=True)
	

	parameter = self.galactic_prior_parameter
	if parameter == 'host mass':
		x = self.lens_mass_1_solar.copy()
	elif parameter == 'companion mass': 
		x = self.lens_mass_2_solar.copy()
		#convert to earth masses
	elif parameter == 'lens distance': #pc
		x = self.lens_distance.copy()
		if x < 100:
			sys.exit('distance unit bug - fix me!')
	elif parameter == 'seperation': #AU
		x = self.lens_separation_AU.copy()
        
	return self.skewnormal_prior(x)

