import sys
import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import minimize
from scipy.optimize import curve_fit


class Binary_lens_mcmc():

	"""

	A class that implements methods to fit binary gravitational lensing lightcurves.

	This is a front end to Alistair McDougall's GPU code, described in McDougall & Albrow (2016).
	
	(c) Michael Albrow

	inputs:

		data:				A dictionary with each key being a data set name string and each 
							value being a tuple of (date, flux, flux_err). Each of these is a 1-D numpy
							array of the same length.

		initial_parameters:	A numpy array of starting guess values for log d, log q, log rho, u_0, phi, 
							t_0, t_E.
	
		initial_sigmas:		A numpy array of starting guess values for the uncertainties in log d, log q, 
							log rho, u_0, phi, t_0, t_E.

		reference_source:  	One of the data dictionary keys to use as the fiducial flux or magnitude scale when plotting.

	"""

	from ._parallax import add_parallax, get_vernal_peri, get_parallax_psi,compute_parallax_terms, \
				parallax_trajectory, search_parallax_trajectory, grid_search_parallax, grid_search_parallax2, \
				grid_search_parallax3, grid_search_parallax_simple, plot_map
	
	from ._spitzer import add_spitzer, spitzer_delta_beta_tau_to_parallax, spitzer_magnification, \
				compute_spitzer_parallax_terms, compute_spitzer_perpendicular_distance, compute_spitzer_parallax, \
				grid_search_spitzer, lnprob_spitzer, emcee_converge_spitzer

	from ._physical import compute_relative_proper_motion, VItoVK, compute_theta_E, compute_physical_properties

	from ._orbital import add_lens_orbital_motion, add_lens_orbital_acceleration, lens_orbital_motion_energy_prior, \
				grid_search_lens_orbital_motion

	from ._likelihood import chi2_calc, ln_prior_prob, lnprob, neg_lnprob

	from ._fitting import emcee_has_converged, converge, advance_mcmc

	from ._plotting import plot_chain, plot_chain_corner, plot_caustic_and_trajectory, plot_lightcurve, circles, plot_GP, plot_spitzer

	from ._magnification import caustic, magnification_map, point_source_magnification, magnification, trajectory

	from ._data import add_model_bad_data, bad_data_ln_prior_prob, add_error_bar_scaling_multiplicative, \
				error_bar_scale_ln_prior_prob, reformat_data, renormalise_data_uncertainties_simple, \
				renormalise_data_uncertainties, set_zero_blending, add_spitzer_error_bar_scaling_multiplicative, \
				lnprior_scale_spitzer_multiplicative, known_parameters_prior, murelgeo_lnprior
				
	from ._misc import add_limb_darkening, limb_darkening_ln_prior_prob, add_flux_parameters

	from ._galactic import add_galactic_prior, galactic_ln_prior_prob, neg_skew_pdf, fit_skew_normal, fit_log_skew_normal, set_skew_parameters, skewnormal_prior
	
	from ._gaussian_process import add_gaussian_process_model, gaussian_process_prior, gaussian_process_ground_chi2, \
				gaussian_process_spitzer_chi2, CeleriteModel, get_gaussian_process_bounds

	from ._VBBL import VBBL_magnification


	def __init__(self, data, initial_params, initial_sigmas=None, reference_source=None):

		self.data = data

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


		self.reformat_data()

		# Holds current parameter values
		self.p = np.array(initial_params[:7])
		self.freeze = np.zeros(7,dtype=int)

		if initial_sigmas is None:

			self.p_sig = np.zeros_like(initial_params)
			self.p_sig[0]	=	0.000025
			self.p_sig[1]	=	0.00001
			self.p_sig[2]	=	0.00001
			self.p_sig[3]	=	0.00002*abs(self.p[3])
			self.p_sig[4]	=	0.00002 * np.pi/180.0
			self.p_sig[5]	=	0.0001
			self.p_sig[6]	=	0.0001
			self.p_sig0 = self.p_sig.copy()
 
		else:

			self.p_sig = initial_sigmas[:7]
			self.p_sig0 = self.p_sig.copy()

		# Photometric zeropoint
		self.zp = 28.0

		# Limb darkening parameter
		#
		# From Claret et al. 2012
		# http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A+A/546/A14&-to=3
		#
		self.limb_constant = 0.6541  # for K2.5 4300K I-band
		self.spitzer_limb_constant = 0.2973

		self.parameter_labels = [r"$log_{10}s$",r"$log_{10}q$",r"$log_{10}\rho$",r"$u_0$",r"$\alpha$",r"$t_0$",r"$t_E$"]
		self.plot_colours = ['#FF0000', '#000000', '#008000', '#800080', '#FFA500', '#A52A2A', '#ff9999', '#999999', \
					'#4c804c', '#804d80', '#ffdb98', '#a56262', '#CCFF66', '#CC9900', '#9966FF', '#FF3366']

		self.count_limit = 4000
		self.dims = len(self.p)
		self.plot_chains = False
		self.plotprefix = 'emcee'
		self.plot_date_range = None

		# Distances from the nearest caustic for using these approximations
		self.hexadecapole_magnification_threshold = 0.1
		self.point_source_approximation_threshold = 30.0
		self.hexadecapole_approximation_threshold = 2.0

		# Upper and lower limits for parameter priors 
		self.range_log_s = (-5.0,2.0)
		self.range_log_q = (-8.0,1.0)
		self.range_log_rho = (-5.0,-1.0)
		self.upper_limit_u0 = 5.0
		self.upper_limit_alpha = 3.0*np.pi
		self.range_tE = (0.01,500.0)
		self.known_parameters = None #[(index,value,uncertainty),(index,value,uncertainty)]
		self.murelgeo_prior = None # [[N,E],[dN,dE]]
		self.thetastar_prior = None

		# Parameters for the emcee sampler
		self.known_sampler_packages = ['emcee','zeus']
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
		self.use_parallax = False
		self.use_spitzer = False
		self.use_lens_orbital_motion = False
		self.use_lens_orbital_motion_energy_constraint = False
		self.use_lens_orbital_acceleration = False
		self.plot_lens_rotation = False
		self.use_zero_blending = False
		
		# Galactic prior
		self.use_galactic_prior = False
		self.galactic_prior_parameter = None
		self.colour_params_for_galactic_prior = None
		self.skewnormal_renorm = 1./np.sqrt(2.*np.pi)
		self.a, self.loc, self.scale = (0., 0., 1.)
		self.mean, self.var, self.skew, self.kurt = (0., 1., 0., 0.)
		self.median = 1.
		self.mode = 1.

		# Parameters for parallax models
		self.spitzer_delta_tau = None
		self.spitzer_delta_beta = None
		self.spitzer_u0_sign = 1
		self.spitzer_has_colour_constraint = False
		self.spitzer_flux_ratio = None
		self.spitzer_colour_uncertainty = None
		self.spitzer_flux_ratio_reference_site = None
		self.spitzer_plot_scaled = True
		self.Pi_EE_index = None
		self.Pi_EN_index = None
		self.v_Earth_perp = None
		self.parallax_t_ref = None

		# Parameter to control treatment of the flux parameters
		self.treat_flux_parameters_as_nonlinear = False

		# Parameter to flag error bar scaling
		self.scale_error_bars_multiplicative = False
		self.scale_spitzer_error_bars_multiplicative = False

		# Parameter to flag modelling of bad data
		self.model_bad_data = False

		# Paameter to control rejection of outliers when fitting
		self.data_outlier_threshold = 1000

		# Parameter for using Valerio Bozza's VBBL code for magnification
		self.use_VBBL = False
		self.VBBL = None
		self.VBBL_RelTol = 1e-03
		
		# Parameters for using Gaussian process modelling
		self.chi2_method='manual'
		self.use_gaussian_process_model = False
		self.gaussian_process_sites = []
		self.GP_default_params = (-0.5,-2.0)
		self.GP_sig0 = (0.00001, 0.00001)
		# Real
		self.GP_model='Real'
		self.ln_a_limits = (-5.,15.)
		self.ln_c_limits = (-5.5, 7.0)
		self.ln_a = {}
		self.ln_c = {}
		# Matern32
		self.ln_sigma_limits = (-10., 10.)
		self.ln_rho_limits = (-10., 10.)
		self.ln_sigma = {}
		self.ln_rho = {}
		
		# Debugging
		self.debug=False
		


