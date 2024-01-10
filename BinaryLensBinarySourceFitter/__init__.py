import sys
import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import minimize
from scipy.optimize import curve_fit
Code = '../' # puppis python3
sys.path.append(Code)
from BinaryLensFitter import Binary_lens_mcmc


class Binary_lens_Binary_source_mcmc():

	"""

	A class that implements methods to fit binary gravitational lensing
	lightcurves.

	This is a front end to Alistair McDougall's GPU code, described in McDougall
	& Albrow (2016).
	
	(c) Michael Albrow

	inputs:

		data:			A dictionary with each key being a data set name string
						and each value being a tuple of (date, flux, flux_err).
						Each of these is a 1-D numpy array of the same length.
							
		spitzer_data:	A tuple with (date, flux, flux_err)
		
		spitzer_location_path:	...

		initial_parameters:	A numpy array of starting guess values for log s,
							log q, log rho, u_0, phi, t_0, t_E, pi_EE, pi_EN,
							log rho1, u_01, phi1, t_01, t_E1.
		
		RA:				...
		
		DEC:			...
		
		YEAR:			...
	
		initial_sigmas:	A numpy array of starting guess values for the
						uncertainties in log s, log q, log rho1, u_0, phi, t_0,
						t_E1, piEE, piEN, log rho2, delta u_0, delta phi, 
						delta t_0, t_E2.

		reference_source:  	One of the data dictionary keys to use as the
							fiducial flux or magnitude scale when plotting.

	"""
	
	from ._plotting import plot_caustic_and_trajectory, circles\
							, plot_lightcurve, plot_spitzer
	#, plot_chain, plot_chain_corner, plot_lightcurve, plot_GP, 
	from ._likelihood import chi2_calc

	def __init__(self, data, spitzer_data, spitzer_location_path, initial_params, RA, DEC, YEAR, initial_sigmas=None, reference_source=None):

		self.data = data

		# reference source
		if reference_source is None:
			self.reference_source = list(self.data.keys())[0]
			print('Using',self.reference_source,'as reference.')
		else:
			if reference_source in self.data:
				self.reference_source = reference_source
			else:
				self.reference_source = list(self.data.keys())[0]
				print('Warning:',reference_source,'is not a valid data source.')
				print('Using',self.reference_source,'as reference.')


		#self.reformat_data()  # !!!! probably need to work out what this does
		
		# Parameters and child set-up
		self.p0 = initial_params
		self.p = np.array(initial_params[:14])
		self.primary_indexes = [0,1,2,3,4,5,6]
		self.secondary_indexes = [0,1,9,10,11,12,13]
		p0_primary = [self.p[i] for i in self.primary_indexes]  # I can't find a nicer way of doing that selection
		p0_secondary = [self.p[i] for i in self.secondary_indexes]
		p0_secondary[3] += p0_primary[3]  # u02 from delta u0
		p0_secondary[4] += p0_primary[4]  # alpha 2 from delta alpha
		p0_secondary[5] += p0_primary[5]  # t02 from delta t0
		print('p0 primary = ',p0_primary)
		print('p0 secondary = ',p0_secondary)
		
		self.primary = Binary_lens_mcmc(data,p0_primary,reference_source=reference_source)
		self.secondary = Binary_lens_mcmc(data,p0_secondary,reference_source=reference_source)
		
		self.primary.p = p0_primary.copy()
		self.primary.p0 = self.primary.p.copy()
		
		self.secondary.p = p0_secondary.copy()
		self.secondary.p0 = self.secondary.p.copy()
		
		self.piEE_index = 7
		self.piEN_index = 8
		piEE, piEN = initial_params[self.piEE_index], initial_params[self.piEN_index]
		
		if initial_sigmas is None:

			self.p_sig = np.zeros_like(self.p)
			self.p_sig[0]			=	0.000025		# logs
			self.p_sig[1]			=	0.00001			# logq
			self.p_sig[2]			=	0.00001			# logrho1
			self.p_sig[3]			=	0.00002*abs(self.p[3])	# u01
			self.p_sig[4]			=	0.00002 * np.pi/180.0	# phi1
			self.p_sig[5]			=	0.0001			# t01
			self.p_sig[6]			=	0.0001			# tE1
			self.p_sig[self.piEE_index]	=	0.0001			# piEE
			self.p_sig[self.piEN_index]	=	0.0001			# piEN
			self.p_sig[9]			=	0.00001			# logrho2
			self.p_sig[10]			=	0.00002*abs(self.p[10])	# u02
			self.p_sig[11]			=	0.00002 * np.pi/180.0	# phi2
			self.p_sig[12]			=	0.0001			# t02
			self.p_sig[13]			=	0.0001			# tE2
			self.p_sig0			=	self.p_sig.copy()
			
			self.primary.p_sig0 = self.p_sig[self.primary_indexes]
			self.secondary.p_sig0 = self.p_sig[self.secondary_indexes]
			
			dpiEE, dpiEN = self.p_sig[self.piEE_index].copy(), self.p_sig[self.piEN_index].copy()
			
 
		else:
		
			self.primary.p_sig0 = initial_sigmas[self.primary_indexes]
			self.secondary.p_sig0 = initial_sigmas[self.secondary_indexes]
			
			dpiEE, dpiEN = initial_sigmas[self.piEE_index], initial_sigmas[self.piEN_index]
			
		
		self.primary.p_sig = self.primary.p_sig0.copy()
		self.secondary.p_sig = self.secondary.p_sig0.copy()
			
		self.freeze = np.zeros(14,dtype=int)

		# Photometric zeropoint
		self.zp = 28.0

		# Limb darkening parameter
		#
		# From Claret et al. 2012
		# http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/546/A14&-to=3
		#
		self.limb_constant = 0.6541  # for K2.5 4300K I-band
		self.spitzer_limb_constant = 0.2973

		self.parameter_labels 	= [r"$log_{10}s$", r"$log_{10}q$"\
								, r"$log_{10}\rho_1$", r"$u_{0}$"\
								, r"$\alpha$", r"$t_{0}$", r"$t_{E,1}$"\
								, r"$\pi_{E,E}$", r"$\pi_{E,N}"\
								, r"$log_{10}\rho_2$", r"$\Delta u_{0}$"\
								, r"$\Delta\alpha$", r"$\Delta t_{0}$"\
								, r"$t_{E,2}$"]
		self.plot_colours 	= ['#000000', '#008000', '#800080' \
							, '#FFA500', '#A52A2A', '#ff9999', '#999999'\
							, '#4c804c', '#804d80', '#ffdb98', '#a56262'\
							, '#CCFF66', '#CC9900', '#9966FF', '#FF3366']
							#'#FF0000', 
		self.count_limit = 4000
		self.dims = len(self.p)
		self.plot_chains = False
		self.plotprefix = 'emcee'
		self.plot_date_range = None
		self.primary_plotprefix = None
		self.secondary_plotprefix = None
		self.primary_plot_date_range = None
		self.secondary_plot_date_range = None
		
		# Misc
		self.data_outlier_threshold = 1000  # used in recursive LS of chi2_calc
		self.treat_flux_parameters_as_nonlinear = False  # i.e. Fs1, Fs2, Fb

		# Distances from the nearest caustic for using these approximations
		self.hexadecapole_magnification_threshold = 0.1
		self.point_source_approximation_threshold = 30.0
		self.hexadecapole_approximation_threshold = 2.0

		# Upper and lower limits for parameter priors 
		self.range_log_s = (-5.0,2.0)
		self.range_log_q = (-8.0,1.0)
		
		# Parameters for the emcee sampler
		self.known_sampler_packages = self.primary.known_sampler_packages
		self.sampler_package = 'emcee'
		self.state = None
		self.emcee_walkers = 100
		self.emcee_burnin_walkers = 50
		self.emcee_burnin_steps = 100
		self.emcee_burnin_discard_percent = 50
		self.emcee_min_burnin_steps= 1500
		self.emcee_max_burnin_steps= 5000
		self.emcee_max_optimize_steps = 5000
		self.emcee_relax_steps = 500
		self.emcee_production_steps = 5000
		self.emcee_lnp_convergence_threshold = 0.0
		self.emcee_mean_convergence_threshold = 0.3
		self.emcee_old_tau = np.inf
		self.data_variance_scale = 1.0

		# Parameters for the in-built markov chain monte carlo
		self.converge_chain_length = 100
		self.converge_burn_in_length = 100
		self.converge_max_steps = 5000
		self.mcmc_alpha = 1.0
		self.converge_plot_interval = 50
		self.chi2_convergence_threshold = 0.5
		self.param_convergence_threshold = 0.2
		self.proposal_distribution = 'gaussian'

		# Choice is 'flux', 'magnitudes', or 'logA'
		self.lightcurve_plot_type = 'magnitudes'
		# These are additions to the standard binary model
		self.use_limb_darkening = False
		self.use_lens_orbital_motion = False
		self.use_lens_orbital_motion_energy_constraint = False
		self.use_lens_orbital_acceleration = False
		self.plot_lens_rotation = False
		self.use_zero_blending = False
		
		# Parameters for parallax models 
		self.spitzer_delta_tau = None
		self.spitzer_delta_beta = None
		self.spitzer_u0_sign = 1
		self.spitzer_has_colour_constraint = False
		self.spitzer_flux_ratio = None
		self.spitzer_colour_uncertainty = None
		self.spitzer_flux_ratio_reference_site = None
		self.spitzer_plot_scaled = True
		self.v_Earth_perp = None
		self.parallax_t_ref = self.primary.parallax_t_ref \
		= self.secondary.parallax_t_ref = self.p0[5]

		# Parameter to flag error bar scaling
		self.scale_error_bars_multiplicative = False
		self.scale_spitzer_error_bars_multiplicative = False

		# Parameter for using Valerio Bozza's VBBL code for magnification
		self.use_VBBL = False
		self.VBBL = None
		self.VBBL_RelTol = 1e-03
		
		# Parallax
		self.primary.add_parallax(RA,DEC,YEAR)
		self.secondary.add_parallax(RA,DEC,YEAR)
		# add paralalx class variables
		self.right_ascension = RA
		self.declination = DEC
		self.vernal, self.peri = self.primary.vernal, self.primary.peri
		self.earth_ecc = self.primary.earth_ecc
		self.parallax_offset = self.primary.parallax_offset
		self.parallax_rad = self.primary.parallax_rad
	
		# East direction on lens plane
		self.parallax_east = self.primary.parallax_east
		# North direction on lens plane
		self.parallax_north = self.primary.parallax_north
	
		# Earth position at perihelion?
		self.parallax_xpos = self.primary.parallax_xpos
		self.parallax_ypos = self.primary.parallax_xpos

		# Galactic rotation
		self.galaxy_rotation_direction = self.primary.galaxy_rotation_direction
		self.galaxy_rotation_east = self.primary.galaxy_rotation_east
		self.galaxy_rotation_north = self.primary.galaxy_rotation_north
		self.galaxy_north_direction = self.primary.galaxy_north_direction
		self.solar_pecular_velocity_kms \
		= self.primary.solar_pecular_velocity_kms
		
		# make all parallax values the same
		# resolve indexing
		self.primary_indexes += [0,0]
		self.secondary_indexes += [0,0]
		self.primary_indexes[self.primary.Pi_EE_index] \
		= self.secondary_indexes[self.secondary.Pi_EE_index] \
		= self.piEE_index
		self.primary_indexes[self.primary.Pi_EN_index] \
		= self.secondary_indexes[self.secondary.Pi_EN_index] \
		= self.piEN_index
		#replace values
		self.primary.p[self.primary.Pi_EE_index] = piEE
		self.primary.p[self.primary.Pi_EN_index] = piEN
		self.secondary.p[self.secondary.Pi_EE_index] = piEE
		self.secondary.p[self.secondary.Pi_EN_index] = piEN
		# pi sigmas
		self.primary.p_sig[self.primary.Pi_EE_index] = dpiEE
		self.primary.p_sig[self.primary.Pi_EN_index] = dpiEN
		self.secondary.p_sig[self.secondary.Pi_EE_index] = dpiEE
		self.secondary.p_sig[self.secondary.Pi_EN_index] = dpiEN
		# add to initial arrays
		self.primary.p_sig0 = self.primary.p_sig
		self.primary.p0 = self.primary.p
		self.secondary.p_sig0 = self.secondary.p_sig
		self.secondary.p0 = self.secondary.p
		
		# add Spitzer to subclasses
		self.primary.add_spitzer(	spitzer_location_path, spitzer_data\
									, right_ascension=RA, declination=DEC)
		self.secondary.add_spitzer(	spitzer_location_path, spitzer_data\
									, right_ascension=RA, declination=DEC)
		self.spitzer_data = self.primary.spitzer_data
		
		# Debugging prints
		self.debug = self.primary.debug = self.secondary.debug = False
		
		# GP params
		self.primary.use_gaussian_process_model \
		= self.secondary.use_gaussian_process_model \
		= self.use_gaussian_process_model = False
		self.primary.gaussian_process_sites \
		= self.secondary.gaussian_process_sites = self.gaussian_process_sites \
		= []


	def macth_data(self):
		self.primary.data = self.secondary.data = self.data  	# is this a bad
																# idea? 
		self.primary.reference_source = self.secondary.reference_source \
		= self.reference_source	
		self.primary.spitzer_data = self.secondary.spitzer_data


	def match_class_variables(self):
		'''make sure the subclass defaults match the parent. There has got to be a better way to do this, but I don't know it.'''
	
		self.primary.reference_source =self.secondary.reference_source \
		= self.reference_source
		self.primary.freeze = self.freeze[self.primary_indexes]
		self.secondary.freeze = self.freeze[self.secondary_indexes]

		# Photometric zeropoint
		self.primary.zp = self.secondary.zp = self.zp

		# Limb darkening parameter
		self.primary.limb_constant = self.secondary.limb_constant \
		= self.limb_constant
		self.primary.spitzer_limb_constant \
		= self.secondary.spitzer_limb_constant = self.spitzer_limb_constant

		# Distances from the nearest caustic for using these approximations
		self.primary.hexadecapole_magnification_threshold \
		= self.secondary.hexadecapole_magnification_threshold \
		= self.hexadecapole_magnification_threshold
		self.primary.point_source_approximation_threshold \
		= self.secondary.point_source_approximation_threshold \
		= self.point_source_approximation_threshold
		self.primary.hexadecapole_approximation_threshold \
		= self.secondary.hexadecapole_approximation_threshold \
		= self.hexadecapole_approximation_threshold

		# Upper and lower limits for parameter priors 
		self.primary.range_log_s = self.secondary.range_log_s \
		= self.range_log_s
		self.primary.range_log_q = self.secondary.range_log_q \
		= self.range_log_q
		
		# Lightcurve parameters
		if self.primary_plotprefix is None:
			self.primary_plotprefix = self.plotprefix+'-primary'
		if self.secondary_plotprefix is None:
			self.secondary_plotprefix = self.plotprefix+'-secondary'
		if self.primary_plot_date_range is None:
			self.primary_plot_date_range = self.plot_date_range
		if self.secondary_plot_date_range is None:
			self.secondary_plot_date_range = self.plot_date_range
		self.primary.lightcurve_plot_type = self.secondary.lightcurve_plot_type\
		= self.lightcurve_plot_type
		
		# These are additions to the standard binary model
		self.primary.use_limb_darkening = self.secondary.use_limb_darkening \
		= self.use_limb_darkening
		self.primary.use_lens_orbital_motion \
		= self.secondary.use_lens_orbital_motion = self.use_lens_orbital_motion
		self.primary.use_lens_orbital_motion_energy_constraint \
		= self.secondary.use_lens_orbital_motion_energy_constraint \
		= self.use_lens_orbital_motion_energy_constraint
		self.primary.use_lens_orbital_acceleration \
		= self.secondary.use_lens_orbital_acceleration \
		= self.use_lens_orbital_acceleration
		self.primary.plot_lens_rotation = self.secondary.plot_lens_rotation \
		= self.plot_lens_rotation
		self.primary.use_zero_blending = self.secondary.use_zero_blending \
		= self.use_zero_blending
		
		# Parameters for parallax models 
		self.primary.spitzer_delta_tau = self.secondary.spitzer_delta_tau \
		= self.spitzer_delta_tau
		self.primary.spitzer_delta_beta = self.secondary.spitzer_delta_beta \
		= self.spitzer_delta_beta
		self.primary.spitzer_u0_sign = self.secondary.spitzer_u0_sign \
		= self.spitzer_u0_sign
		self.primary.spitzer_has_colour_constraint \
		= self.secondary.spitzer_has_colour_constraint \
		= self.spitzer_has_colour_constraint
		self.primary.spitzer_flux_ratio = self.secondary.spitzer_flux_ratio \
		= self.spitzer_flux_ratio
		self.primary.spitzer_colour_uncertainty \
		= self.secondary.spitzer_colour_uncertainty \
		= self.spitzer_colour_uncertainty
		self.primary.spitzer_flux_ratio_reference_site \
		= self.secondary.spitzer_flux_ratio_reference_site \
		= self.spitzer_flux_ratio_reference_site
		self.primary.spitzer_plot_scaled = self.secondary.spitzer_plot_scaled \
		= self.spitzer_plot_scaled
		self.primary.v_Earth_perp = self.secondary.v_Earth_perp \
		= self.v_Earth_perp
		self.primary.parallax_t_ref = self.secondary.parallax_t_ref \
		= self.parallax_t_ref
		self.primary.right_ascension = self.secondary.right_ascension \
		= self.right_ascension
		self.primary.declination = self.secondary.declination = self.declination

		# Parameter to flag error bar scaling
		self.primary.scale_error_bars_multiplicative \
		= self.secondary.scale_error_bars_multiplicative \
		= self.scale_error_bars_multiplicative
		self.primary.scale_spitzer_error_bars_multiplicative \
		= self.secondary.scale_spitzer_error_bars_multiplicative \
		= self.scale_spitzer_error_bars_multiplicative

		# Parameter for using Valerio Bozza's VBBL code for magnification
		self.primary.use_VBBL = self.secondary.use_VBBL = self.use_VBBL
		self.primary.VBBL = self.secondary.VBBL = self.VBBL
		self.primary.VBBL_RelTol = self.secondary.VBBL_RelTol = self.VBBL_RelTol
		
		# Debug prints
		self.primary.debug = self.secondary.debug = self.debug
		
		# GP params
		self.primary.use_gaussian_process_model \
		= self.secondary.use_gaussian_process_model \
		= self.use_gaussian_process_model
		self.primary.gaussian_process_sites \
		= self.secondary.gaussian_process_sites = self.gaussian_process_sites
		
	def match_p(self,p=None):
		''''
		Ensuring parameter values are universally consistant for 
		parent and child.'''
	
		if self.debug:	
			print(	'Ensuring parameter values are universally consistant for '\
					+'parent and child')
		
		if p is None:
			p = self.p.copy()
		else:
			self.p = p.copy()
		p_sig = self.p_sig.copy()
		
		self.primary.p = p[self.primary_indexes]
		self.primary.p_sig = p_sig[self.primary_indexes]
		self.secondary.p = p[self.secondary_indexes]
		#print(self.secondary.p, self.secondary_indexes)
		self.secondary.p[3] = p[3] + p[11]  # u02 from delta u0
		self.secondary.p[4] = p[4] + p[12]  # alpha 2 from delta alpha
		self.secondary.p[5] = p[5] + p[13]  # t02 from delta t0
		self.secondary.p_sig = p_sig[self.secondary_indexes]
		self.primary.freeze = self.freeze[self.primary_indexes]
		self.secondary.freeze = self.freeze[self.secondary_indexes]
		
		if self.debug:
			print(	'p:', self.p)
			print(	'p1:', self.primary.p, self.primary_indexes)
			print(	'p2:', self.secondary.p, self.secondary_indexes)
			print(	'freeze:', self.freeze, self.primary.freeze\
					 , self.secondary.freeze)
			
		
	def spitzer_delta_beta_tau_to_parallax(self,delta_beta, delta_tau, p=None, t0=None, debug=False):

		"""Convert satellite trajactory offset (delta_beta, delta_tau) to microlensing parallax (piE_E, piE_N)."""

		if t0 is None:

			if p is None:
				p = self.p.copy()

			self.primary.p = self.p[self.primary_indexes]
			self.secondary.p = self.p[self.secondary_indexes]
			logd, logq, logrho1, u01, phi1, t01, tE1 = self.primary.p[:7]
			_, _, logrho2, u02, phi2, t02, tE2 = self.secondary.p[:7]

		if self.parallax_t_ref is None:
			self.parallax_t_ref = t01 # I have no idea if this makes sense or if I need it
		
		pi_EN, pi_EE = self.primary.spitzer_delta_beta_tau_to_parallax(delta_beta, delta_tau, p=p, t0=t01)
			
		print()
		print('spitzer_delta_beta_tau_to_parallax')
		print('(pi_EN, pi_EE)', (pi_EN, pi_EE))
		print('(delta_tau, delta_beta)', (delta_tau, delta_beta))

		return pi_EN, pi_EE 
	
		
