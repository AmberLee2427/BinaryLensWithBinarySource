import numpy as np
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from pylab import subplots_adjust
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable, inset_locator

import corner

matplotlib.rcParams['axes.titlesize'] = 10

def plot_chain(self,chain,ln_prob=None,index=None,suffix='',parameter_labels=None):

	plot_lnprob = ln_prob is not None

	if index is None:
		index = list(range(chain.shape[2]))

	if parameter_labels is None:
		parameter_labels = [self.parameter_labels[i] for i in np.where(1-self.freeze)[0].tolist()]

	n_plots = len(index)

	plt.figure(figsize=(8,11))
	
	subplots_adjust(hspace=0.0001)

	for i in range(n_plots):

		if i == 0:
			plt.subplot(n_plots+plot_lnprob,1,i+1)
			ax1 = plt.gca()
		else:
			plt.subplot(n_plots+plot_lnprob,1,i+1,sharex=ax1)

		plt.plot(chain[:,:,index[i]].T, '-', color='k', alpha=0.3)

		plt.ylabel(parameter_labels[i])

		ax = plt.gca()

		if i < n_plots-1 + plot_lnprob:
			plt.setp(ax.get_xticklabels(), visible=False)
			ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
			ax.locator_params(axis='y',nbins=4)

	if plot_lnprob:
		plt.subplot(n_plots+plot_lnprob,1,n_plots+plot_lnprob,sharex=ax1)
		plt.plot(ln_prob.T, '-', color='r', alpha=0.3)
		plt.ylabel(r"$ln P$")
		ax = plt.gca()
		ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
		ax.locator_params(axis='y',nbins=4)

	plt.savefig(self.plotprefix+'-chain.png'+suffix)
	plt.close()


def plot_chain_corner(self,parameter_labels=None):

	if parameter_labels is None:
		parameter_labels = [self.parameter_labels[i] for i in np.where(1-self.freeze)[0].tolist()]

	figure = corner.corner(self.samples,
				labels=parameter_labels,
				quantiles=[0.16, 0.5, 0.84],
				show_titles=True, title_args={"fontsize": 12})
	figure.savefig(self.plotprefix+'-pdist.png')

def circles(self,x, y, s, c='b', vmin=None, vmax=None, **kwargs):
	"""
	Make a scatter of circles plot of x vs y, where x and y are sequence 
	like objects of the same lengths. The size of circles are in data scale.

	Parameters
	----------
	x,y : scalar or array_like, shape (n, )
		Input data
	s : scalar or array_like, shape (n, ) 
		Radius of circle in data unit.
	c : color or sequence of color, optional, default : 'b'
		`c` can be a single color format string, or a sequence of color
		specifications of length `N`, or a sequence of `N` numbers to be
		mapped to colors using the `cmap` and `norm` specified via kwargs.
		Note that `c` should not be a single numeric RGB or RGBA sequence 
		because that is indistinguishable from an array of values
		to be colormapped. (If you insist, use `color` instead.)  
		`c` can be a 2-D array in which the rows are RGB or RGBA, however. 
	vmin, vmax : scalar, optional, default: None
		`vmin` and `vmax` are used in conjunction with `norm` to normalize
		luminance data.  If either are `None`, the min and max of the
		color array is used.
	kwargs : `~matplotlib.collections.Collection` properties
		Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
		norm, cmap, transform, etc.

	Returns
	-------
	paths : `~matplotlib.collections.PathCollection`

	Examples
	--------
	a = np.arange(11)
	circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
	plt.colorbar()

	License
	--------
	This code is under [The BSD 3-Clause License]
	(http://opensource.org/licenses/BSD-3-Clause)
	"""
	
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.patches import Circle
	from matplotlib.collections import PatchCollection

	if np.isscalar(c):
		kwargs.setdefault('color', c)
		c = None
	if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
	if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
	if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
	if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

	patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
	collection = PatchCollection(patches, **kwargs)
	if c is not None:
		collection.set_array(np.asarray(c))
		collection.set_clim(vmin, vmax)

	ax = plt.gca()
	ax.add_collection(collection)
	ax.autoscale_view()
	if c is not None:
		plt.sci(collection)
	
	return collection


def plot_caustic_and_trajectory(	self, p=None, plot_caustic=True\
									, plot_map=False, plot_colourbar=False\
									, v_range=None, plot_data=True\
									, plot_trajectory=True\
									, trajectory_colour='k'\
									, plot_Spitzer_data=True\
									, plot_Spitzer_trajectory=True\
									, Spitzer_trajectory_colour='r'\
									, axis_location=None, drawNE=False\
									, plot_lens_position=False, grid=False\
									, labels=True, bounds=None\
									, parent_figure=None, figure_loc=None\
									, close_fig=True, fs=12\
									, NE_origin=(0.0,0.0), plot_thetaE=False):

	""" We use a coordinate system where m1 is at (-a,0), m2 at (a,0), 
		with m1 > m2.
		
		The source approaches at angle alpha from (-inf,0) with impact
		parameter u0 to the centre of mass (-b,0), which is to the right of the
		source if u0>0."""

	# getting the parameters for plotting
	if p is None:
		p = self.p.copy()
	else:
		self.p = p.copy()
	self.match_p() # makes sure p is universal for parent and child classes

	# Figure set-up
	if parent_figure is None:
		fig = plt.figure(figsize=(7,5))
	else:
		fig = parent_figure
		if figure_loc is None:
			fig.add_axes([0.15,0.55,0.4,0.3])
		elif figure_loc == 'parent':
			pass
		else:
			fig.add_axes(figure_loc)

	# unpacking 2S2L parameters
	logd, logq, logrho1, u01, phi1, t01, tE1, piEE1, piEN1, logrho2, deltau0, deltaphi, deltat0, tE2 = p[:14]
	#logd, logq, logrho1, u01, phi1, t01, tE1, piEE1, piEN1, logrho2, u02, t02 = p[:12]
	u02 = u01 + deltau0
	phi2 = phi1 + deltaphi
	t02 = t01 + deltat0
	#phi2 = phi1
	#tE2 = tE1
	d = 10.0**logd
	q = 10.0**logq
	rho1 = 10.0**logrho1
	rho2 = 10.0**logrho2
	piEN1 = np.float64(piEN1)
	piEE1 = np.float64(piEE1)

	# pi_tref is fixed once for a run. It changes the chi2 values you get when
	# using the same model parameters, so it may pay to pick one pi_tref for
	# all comparable models. It is the t at which all motion is considered to
	# have no parallax from the ground perspective. Changing pi_tref changes
	# relative trajectory parameters (t0, u0, and phi), but not lens parameters
	# (s, q). Therefore, it will not change the mass and distance outcomes. I
	# am unsure if it will effect proper motions directions.
	if self.parallax_t_ref is None:
		self.parallax_t_ref = t0

	# LOM & LOA
	if self.use_lens_orbital_motion:
		dphidt = np.float64(p[self.dphidt_index])
		dddt = np.float64(p[self.dddt_index])
		if self.use_lens_orbital_acceleration:
			d2phidt2 = np.float64(p[self.d2phidt2_index])
			d2ddt2 = np.float64(p[self.d2ddt2_index])

	# Conversion from centre-of-mass to mid-point coordinates
	a = 0.5*d
	b = a*(1.0-q)/(1.0+q)
	h1 = b*np.cos(phi1)
	h2 = b*np.cos(phi2)
	
	# Plot theta_E
	if plot_thetaE:
		self.circles(-b,0,1.,c='b',fc='none',alpha=0.4,ls=':')
		
	u0n1 = u01 - np.sin(phi1)*b
	t0n1 = t01 + tE1*b*np.cos(phi1)
	u0n2 = u02 - np.sin(phi2)*b
	t0n2 = t02 + tE2*b*np.cos(phi2)

	if parent_figure is None or plot_caustic == True:
	
		# Plot the magnification map
		if plot_map:
			if bounds is None:
				extent = [-1.0,1.0,-1.0,1.0]
			else:
				extent = bounds
			A = self.primary.magnification_map(	u1_range=(extent[0]\
												, extent[1])\
												, u2_range=(extent[2]\
												, extent[3]) )
			logA = np.log10(A)
			if v_range is None:
				v_range 	= ( np.min(logA), np.max(logA) )
			plt.imshow(	logA, extent=extent, vmin=v_range[0], vmax=v_range[1]\
						, origin='lower')
			if plot_colourbar:
				plt.colorbar()
				
		# Plot lens objects
		if plot_lens_position:
			self.primary.circles(-a,0.0,0.025,c='b')
			self.primary.circles(a,0.0,0.015,c='b')

			# Plot centre of mass
			plt.plot(-b,0,'k+',markersize=5)

		# Plot the caustic
		if not self.use_lens_orbital_motion:
			zeta_real, zeta_imag = self.primary.caustic(params=p)
			plt.plot(zeta_real, zeta_imag, '.', mec='b', markersize=1)

		else:  # LOM
			pp = p.copy()
			colours = ['c','m']

			#if self.lens_orbital_motion_plot_delta:
			#	delta_t = self.lens_orbital_motion_plot_delta
			#else:
			#	delta_t = tE
			#	delta_t = tE
			delta_t1 = tE1
			delta_t2 = tE2

			for i, tt in enumerate([-delta_t1,delta_t1,-delta_t2,delta_t2]):
				pp[0] = np.log10(10.0**p[0] + dddt*tt/365.2425)
				if self.use_lens_orbital_acceleration:
					pp[0] 	= np.log10( 10.0**p[0] + dddt*tt/365.2425 \
							+ 0.5*d2ddt2*(tt/365.2425)**2 )
				dphi = np.zeros_like(tt)
				if self.plot_lens_rotation:
					dphi = tt*dphidt/365.2425

					# LOA
					if self.use_lens_orbital_acceleration:
						dphi += 0.5 * d2phidt2 * (tt/365.2425)**2

				zeta_real, zeta_imag = self.primary.caustic(params=pp)
				plt.plot(zeta_real*np.cos(dphi)+zeta_imag*np.sin(dphi), -zeta_real*np.sin(dphi)+zeta_imag*np.cos(dphi), '.', mec=colours[i], markersize=1, alpha=0.5)
				#circles(-a*np.cos(dphi),-a*np.sin(dphi),0.025,c=colours[i])
				#circles(a*np.cos(dphi),a*np.sin(dphi),0.015,c=colours[i]

	# Plot the data epochs on the trajectory
	if not(self.data is None):
		t_min = t_max = t0n1
		
		# Looping over data sources
		for k, site in enumerate(sorted(self.data)):
		
			# t for this source 
			t_plot = self.data[site][0]
			t_min = np.min([t_min,np.min(t_plot)])  # extend tmin?
			t_max = np.max([t_max,np.max(t_plot)])  # extend tmax?
			
			# Array set-up
			delta_tau1 = delta_tau2 = np.zeros_like(t_plot)
			delta_beta1 = delta_beta2 = np.zeros_like(t_plot)
			delta_phi = np.zeros_like(t_plot)
			delta_d = np.zeros_like(t_plot)
			
			# parallax arrays
			q_n, q_e = self.primary.compute_parallax_terms(t_plot)
			# Do I need to put p into compute_parallax_terms? 
			# No they are dependent on pi_t_ref
			# Make sure the values are getting updated. 
			# Or maybe just do that at the begining of the function?
			#print('q_n:', q_n) # Checking
			#print('q_e:', q_e) # Checking
			delta_tau1 = q_n*piEN1 + q_e*piEE1
			delta_beta1 = -q_n*piEE1 + q_e*piEN1
			
			# LOM & LOA
			if self.use_lens_orbital_motion and not self.plot_lens_rotation:
				delta_phi = dphidt*(t_plot-t01)/365.2425
				delta_d = dddt*(t_plot-t01)/365.2425
				if self.use_lens_orbital_acceleration:
					delta_phi += 0.5 * d2phidt2 * ((t_plot-t01)/365.2425)**2
			
			# Primary-source, ground epochs
			u1 	= ( (t_plot - t0n1)/tE1 + delta_tau1 ) \
				* np.cos(phi1 + delta_phi) \
				+ (u0n1 + delta_beta1) * np.sin(phi1 + delta_phi)
			u2 	= -( (t_plot - t0n1)/tE1 + delta_tau1 ) \
				* np.sin(phi1 + delta_phi) \
				+ (u0n1 + delta_beta1) * np.cos(phi1 + delta_phi)
			if plot_data:
				self.circles(	u1, u2, rho1, c=self.plot_colours[k]\
								, fc='none', alpha=0.5)
			
			# Secondary-source, ground epochs
			u1 	= ( (t_plot - t0n2)/tE2 + delta_tau1 ) \
				* np.cos(phi2 + delta_phi) \
				+ (u0n2 + delta_beta1) * np.sin(phi2 + delta_phi)
			u2 	= -( (t_plot - t0n2)/tE2 + delta_tau1 ) \
				* np.sin(phi2 + delta_phi) \
				+ (u0n2 + delta_beta1) * np.cos(phi2 + delta_phi)
			#u1 = ((t_plot - t0n2)/tE2 + delta_tau2)*np.cos(phi2+delta_phi) 
			#+ (u0n2 + delta_beta2)*np.sin(phi2+delta_phi)
			#u2 = -((t_plot - t0n2)/tE2 + delta_tau2)*np.sin(phi2+delta_phi) 
			#+ (u0n2 + delta_beta2)*np.cos(phi2+delta_phi)
			if plot_data:
				self.circles(	u1, u2, rho2, c=self.plot_colours[k]\
								, fc='none', alpha=0.5)

		if plot_Spitzer_data:
		
			# t for Spitzer
			t_plot = self.spitzer_data[0]

			# Parallax arrays
			q_n = np.zeros_like(t_plot)
			q_e = np.zeros_like(t_plot)
			q_n, q_e = self.primary.compute_parallax_terms(t_plot)
			sq_n, sq_e, sq_r = \
							self.primary.compute_spitzer_parallax_terms(t_plot)	
			delta_tau1 = (q_n+sq_n)*piEN1 + (q_e+sq_e)*piEE1
			delta_beta1 = -(q_n+sq_n)*piEE1 + (q_e+sq_e)*piEN1
			delta_phi = np.zeros_like(t_plot)

			# LOM & LOA
			if self.use_lens_orbital_motion and not self.plot_lens_rotation:
				delta_phi = dphidt*(t_plot-t01)/365.2425
				delta_d = dddt*(t_plot-t01)/365.2425
				if self.use_lens_orbital_acceleration:
					delta_phi += 0.5 * d2phidt2 * ((t_plot-t01)/365.2425)**2
			#if self.use_lens_orbital_motion:
			#	delta_phi = dphidt*(t_plot-t0)/365.2425
			#	delta_d = dddt*(t_plot-t0)/365.2425

			# Model parameterisation stuff
			u0s1 = u01
			phis1 = phi1
			u0s2 = u02
			phis2 = phi2
			if self.primary.spitzer_u0_sign < 0:
				u0s1 *= -1.0
				phis1 = 2.0*np.pi - phi1
				u0s2 *= -1.0
				phis2 = 2.0*np.pi - phi2
			# COM -> mid-point ?
			u0ns1 = u0s1 - np.sin(phis1)*b
			t0ns1 = t01 + tE1*b*np.cos(phis1)
			u0ns2 = u0s2 - np.sin(phis2)*b
			t0ns2 = t02 + tE2*b*np.cos(phis2)
			
			# Primary-source, Spitzer epochs
			u1 	= ( (t_plot - t0ns1)/tE1 + delta_tau1 ) \
				* np.cos(phis1+delta_phi) \
				+ (u0ns1 + delta_beta1) * np.sin(phis1 + delta_phi)
			u2 	= -( (t_plot - t0ns1)/tE1 + delta_tau1 ) \
				* np.sin(phis1 + delta_phi) \
				+ (u0ns1 + delta_beta1) * np.cos(phis1 + delta_phi)
			self.circles(u1,u2,rho1,c='r',fc='r',alpha=0.5)
			
			# Secondary-source, Spitzer epochs
			u1 	= ( (t_plot - t0ns2)/tE2 + delta_tau1 ) \
				* np.cos(phis2 + delta_phi) \
				+ (u0ns2 + delta_beta1) * np.sin(phis2+delta_phi)
			u2 	= -( (t_plot - t0ns2)/tE2 + delta_tau1 ) \
				* np.sin(phis2 + delta_phi) \
				+ (u0ns2 + delta_beta1) * np.cos(phis2 + delta_phi)
			#u1 = ((t_plot - t0ns2)/tE2 + delta_tau2)*np.cos(phis2+delta_phi) 
			# + (u0ns2+delta_beta2)*np.sin(phis2+delta_phi)
			#u2 = -((t_plot - t0ns2)/tE2 + delta_tau2)*np.sin(phis2+delta_phi)
			# + (u0ns2+delta_beta2)*np.cos(phis2+delta_phi)
			self.circles(u1,u2,rho2,c='r',fc='r',alpha=0.25)

	# Trajectory
	if self.plot_date_range:
		t_plot = np.linspace(	self.plot_date_range[0]\
								, self.plot_date_range[1], 1001)
	else:
		t_len = np.max( np.abs([t0n1-t_min, t0n1-t_max]) )
		t_plot = np.linspace(t0n1-t_len, t0n1+t_len, 1001)

	# Array set-up
	delta_tau1 = delta_tau2 = np.zeros_like(t_plot)
	delta_beta1 = delta_beta2 = np.zeros_like(t_plot)
	delta_phi = np.zeros_like(t_plot)

	#if self.use_lens_orbital_motion:
	#	delta_phi = dphidt*(t_plot-t0)/365.2425
	#	delta_d = dddt*(t_plot-t0)/365.2425

	# Parallax arryays
	q_n, q_e = self.primary.compute_parallax_terms(t_plot)
	delta_tau1 = q_n*piEN1 + q_e*piEE1
	delta_beta1 = -q_n*piEE1 + q_e*piEN1
	#delta_tau2 = q_n2*piEN + q_e2*piEE
	#delta_beta2 = -q_n2*piEE + q_e2*piEN

	#if self.use_lens_orbital_motion and not self.plot_lens_rotation:
	#	delta_phi = dphidt*(t_plot-t0)/365.2425
	#	if self.use_lens_orbital_acceleration:
	#		delta_phi += 0.5 * d2phidt2 * ((t_plot-t0)/365.2425)**2

	# Primary-source, ground trajectory
	u11 	= ( (t_plot - t0n1)/tE1 + delta_tau1 ) * np.cos(phi1 + delta_phi) \
			+ (u0n1 + delta_beta1) * np.sin(phi1 + delta_phi) 
	u21 	= -( (t_plot - t0n1)/tE1 + delta_tau1 ) * np.sin(phi1 + delta_phi)\
			+ (u0n1 + delta_beta1) * np.cos(phi1 + delta_phi)
			
	# Secondary-source, ground trajectory
	u12 	= ( (t_plot - t0n2)/tE2 + delta_tau1 ) * np.cos(phi2 + delta_phi) \
			+ (u0n2 + delta_beta1) * np.sin(phi2 + delta_phi) 
	u22 	= -( (t_plot - t0n2)/tE2 + delta_tau1 ) * np.sin(phi2 + delta_phi)\
			+ (u0n2 + delta_beta1) * np.cos(phi2 + delta_phi) 
	#u12 = ((t_plot - t0n2)/tE2 + delta_tau2)*np.cos(phi2+delta_phi) 
	#+ (u0n2 + delta_beta2)*np.sin(phi2+delta_phi) 
	#u22 = -((t_plot - t0n2)/tE2 + delta_tau2)*np.sin(phi2+delta_phi) 
	#+ (u0n2 + delta_beta2)*np.cos(phi2+delta_phi) 

	# Plot the ground trajectory with parallax
	if plot_trajectory:
		plt.plot(u11,u21,'-',color=trajectory_colour,lw=0.5)
		plt.plot(u12,u22,'-',color=trajectory_colour,alpha=0.6,lw=0.5)

	# For figure bounds
	u1_max = np.max(u11)
	u2_max = np.max(u21)
	u1_min = np.min(u11)
	u2_min = np.min(u21)
	#u1_min = np.min([np.min(u11),np.min(u12)])
	#u1_max = np.max([np.max(u11),np.max(u12)])
	#u2_min = np.min([np.min(u21),np.min(u22)])
	#u2_max = np.max([np.max(u21),np.max(u22)])

	# Reset arrays for Spitzer trajectories
	q_n = np.zeros_like(t_plot)
	q_e = np.zeros_like(t_plot)

	# Parallax arrays
	q_n, q_e = self.primary.compute_parallax_terms(t_plot)
	sq_n, sq_e, sq_r = self.primary.compute_spitzer_parallax_terms(t_plot)
	delta_tau1 = (q_n+sq_n)*piEN1 + (q_e+sq_e)*piEE1
	delta_beta1 = -(q_n+sq_n)*piEE1 + (q_e+sq_e)*piEN1
	#delta_tau = (q_n+sq_n)*piEN + (q_e+sq_e)*piEE
	#delta_beta = -(q_n+sq_n)*piEE + (q_e+sq_e)*piEN
	
	# Model stuff
	if self.primary.spitzer_u0_sign < 0:
		u01 *= -1.0
		phi1 = 2.0*np.pi - phi1
		u02 *= -1.0
		phi2 = 2.0*np.pi - phi2
	# COM -> MP
	u0n1 = u01 - np.sin(phi1)*b
	t0n1 = t01 + tE1*b*np.cos(phi1)
	u0n2 = u02 - np.sin(phi2)*b
	t0n2 = t02 + tE2*b*np.cos(phi2)

	# Primary-source, Spitzer trajectory
	u1 	= ( (t_plot - t0n1)/tE1 + delta_tau1 ) * np.cos(phi1 + delta_phi) \
		+ (u0n1 + delta_beta1) * np.sin(phi1 + delta_phi) 
	u2 	= -( (t_plot - t0n1)/tE1 + delta_tau1 ) * np.sin(phi1 + delta_phi) \
		+ (u0n1 + delta_beta1) * np.cos(phi1 + delta_phi) 
	if plot_Spitzer_trajectory:
		plt.plot(u1,u2,'-',color=Spitzer_trajectory_colour,lw=0.5)

	# For figure bounds
	u1_min = np.min([u1_min,np.min(u1)])
	u1_max = np.max([u1_max,np.max(u1)])
	u2_min = np.min([u2_min,np.min(u2)])
	u2_max = np.max([u2_max,np.max(u2)])
	
	# Primary-source, Spitzer trajectory
	u1 	= ( (t_plot - t0n2)/tE2 + delta_tau1 ) * np.cos(phi2 + delta_phi) \
		+ (u0n2 + delta_beta1) * np.sin(phi2 + delta_phi) 
	u2 	= -( (t_plot - t0n2)/tE2 + delta_tau1 ) * np.sin(phi2 + delta_phi) \
		+ (u0n2 + delta_beta1) * np.cos(phi2 + delta_phi)	
	#u1 = ((t_plot - t0n)/tE + delta_tau)*np.cos(phi+delta_phi) 
	#+ (u0n + delta_beta)*np.sin(phi+delta_phi) 
	#u2 = -((t_plot - t0n)/tE + delta_tau)*np.sin(phi+delta_phi) 
	#+ (u0n + delta_beta)*np.cos(phi+delta_phi) 
	if plot_Spitzer_trajectory:
		plt.plot(u1,u2,'-',color=Spitzer_trajectory_colour,lw=0.5)

	# For figure bounds
	u1_min = np.min([u1_min,np.min(u1)])
	u1_max = np.max([u1_max,np.max(u1)])
	u2_min = np.min([u2_min,np.min(u2)])
	u2_max = np.max([u2_max,np.max(u2)])

	# Plot stuff
	if grid:
		plt.grid()
	if not labels:
		plt.gca().tick_params(labelbottom='off', labelleft='off')

	# Bounds for figure
	u1_range = np.abs(u1_max-u1_min)
	u2_range = np.abs(u2_max-u2_min)
	u_range = 0.5*np.max([u1_range,u2_range])
	u1_centre = np.mean([u1_min,u1_max])
	u2_centre = np.mean([u2_min,u2_max])
	u1_min = u1_centre - u_range
	u1_max = u1_centre + u_range
	u2_min = u2_centre - u_range
	u2_max = u2_centre + u_range
	if bounds is not None:
		plt.xlim((bounds[0],bounds[1]))
		plt.ylim((bounds[2],bounds[3]))
	else:
		plt.xlim((u1_min,u1_max))
		plt.ylim((u2_min,u2_max))

	# NE arrows
	if drawNE:
		q_n, q_e 	= self.primary.compute_parallax_terms(np.array([t01]))
		angle 		= ( np.arccos(q_e/np.sqrt(q_n**2 + q_e**2))[0] ) \
					% (2*np.pi) + phi1
		plt.arrow(	0.0, 0.0, np.cos(angle), np.sin(angle), head_width=0.05\
					, head_length=0.1, fc='k', ec='k')
		plt.arrow(	0.0, 0.0, -np.sin(angle), np.cos(angle), head_width=0.05\
					, head_length=0.1, fc='k', ec='k')
		plt.text(np.cos(angle),np.sin(angle),'E')
		plt.text(-np.sin(angle),np.cos(angle),'N')

	# More plot stuff
	ax = plt.gca()
	ax.set_aspect('equal', adjustable='box')
	if not plot_map and not figure_loc:
		ax.tick_params(axis="both", direction="in")
	plt.xlabel(r'$u_1$',fontsize=14)
	plt.ylabel(r'$u_2$',fontsize=14)
	if close_fig:
		plt.savefig(self.plotprefix+'-caustic.png',bbox_inches='tight')
		plt.close()

	return fig


def plot_spitzer(self, p=None, t_range=None, plot_title='', plot_type=None\
					, y_range=None, parent_figure=None, close_fig=True\
					, y_residual_range=None, draw_title=True\
					, axis_location=None, x_tick_step=None, plot_residual=True\
					, y_tick_step_residual=None, samples=None\
					, samples_to_plot=50, draw_grid=True, model_colour='k'\
					, Spitzer_model_colour='r', fs=12, pad=0.5\
					, legend_location=None, primary_model=False\
					, secondary_model=False):
	''' '''
	
	# Limb darkening stuff
	if isinstance(self.limb_constant, float):
		LD = self.limb_constant
		SLD = self.spitzer_limb_constant
	else:
		LD = self.limb_constant[self.reference_source]
		if self.spitzer_limb_constant != self.limb_constant['spitzer']:
			print('\n\n')
			print(	'Replacing Spitzer limb constant (%f) with %f.' \
					%( self.spitzer_limb_constant\
					, self.limb_constant['spitzer']) )
			print('\n\n')
			self.spitzer_limb_constant = self.limb_constant['spitzer']
		SLD = self.spitzer_limb_constant
	
	# changing default matplotlb font size
	matplotlib.rcParams.update({'font.size': fs})

	# Getting the parameters for plotting
	if p is None: #default
		p = self.p.copy()
	elif (len(p)!=len(self.freeze)) and (len(p)==len(p[self.freeze==0])):
		ps = p.copy() 	# params specified in function call. This will not
						# accept tuples
		p = self.p.copy()	# class stored, full set of parameters (including
							# frozen ones)
		p[self.freeze==0] = ps	# replacing non-frozen parameters with those
								# from the function call
		# now p is the full parameter set including frozen values
	self.match_p() # makes sure p is universal for parent and child classes
	p1 = self.primary.p.copy()
	p2 = self.secondary.p.copy()

	if plot_type is None:
		plot_type = self.lightcurve_plot_type

	# t stuff
	if t_range:
		t_min, t_max = t_range
	elif self.plot_date_range:
		t_min, t_max = self.plot_date_range
	else:
		t_min = np.min(self.ts) - 2.5
		t_max = np.max(self.ts) + 2.5

	# Unpacking p
	logd, logq, logrho1, u01, phi1, t01, tE1, piEE, piEN, logrho2, deltau0, deltaphi, deltat0, tE2 = p[:14]
	# logd, logq, logrho1, u01, phi1, t01, tE1, piEE1, piEN1, logrho2, u02, t02
	#     = p[:12]
	u02 = u01 + deltau0
	phi2 = phi1 + deltaphi
	t02 = t01 + deltat0
	d = 10.0**logd
	q = 10.0**logq
	rho1 = 10.0**logrho1
	rho2 = 10.0**logrho2

	# chi2, Fs1, Fs2, FB, colour constraint chi2 component, elementwise chi2
	chi2, a0, a1, a2, _, _, _, chi2_colour, chi2_elements = self.chi2_calc(p)
	
	# Debugging prints  
	print('p', p)
	print('chi2', chi2)
	print('a0', a0)  # Fs1
	print('a1', a1)  # Fs2
	print('a2', a2)  # Fs3
	print('cc', chi2_colour)

	# Figure set-up
	if parent_figure is None:
		fig = plt.figure(figsize=(7,5.3))
		if axis_location is None:
			ax0 = fig.add_axes([0.12,0.25,0.85,0.6])
		else:
			ax0 = fig.add_axes(axis_location)
	else:
		#ax0 = inset_locator.inset_axes(parent_axes, width="40%", height="20%"
		# , loc=1)
		fig = parent_figure
		if axis_location is None:
			ax0 = fig.add_axes([0.15,0.6,0.4,0.3])
		else:
			ax0 = fig.add_axes(axis_location)

	ratio = 1.0  # for scaling Spitzer vs Ground
	
	# grid = AxesGrid(fig, 111, nrows_ncols = (2,1), axes_pad = 0.0,
	# share_all=True, label_mode = "L")
	# gs = gridspec.GridSpec(2,1,height_ratios=(3,1))

	# Main lightcurve plot
	#ax0 = plt.subplot(gs[0])

	# Plot the model
	n_plot_points = np.min((1001,100*(t_max-t_min)//(np.abs(tE1)*rho1)))
	  # estimating a nnumber that will make the model look smooth
	n_plot_points = int(n_plot_points)  # so that the next line works
	t_plot = np.linspace(t_min,t_max,n_plot_points)

	AS1 = self.primary.spitzer_magnification(t_plot,p=p1,LD=SLD)
	AS2 = self.secondary.spitzer_magnification(t_plot,p=p2,LD=SLD)

	if plot_type == 'magnitudes':
		if self.spitzer_plot_scaled:
			ax0.plot(t_plot,\
			self.zp-2.5*np.log10(\
			a0[self.reference_source]*AS1 \
			+ a1[self.reference_source]*AS2\
			+ a2[self.reference_source])\
			,'-',color=Spitzer_model_colour)
		else:
			ax0.plot(	t_plot,	self.zp - 2.5 * np.log10( \
								a0['spitzer']*AS1 + a1['spitzer']*AS1 \
								+ a2['spitzer'] ) - 2.5*np.log10(ratio)\
						,'-',color=Spitzer_model_colour)

		ax0.invert_yaxis()  # because magnitudes are upsidedown

		if parent_figure is None:
			ax0.set_ylabel(	r'$I_{' + self.reference_source + r'}$'\
							, fontdict={'fontsize':14})

	elif plot_type == 'logA':
		r = 1.0 / (a0['spitzer'] + a1['spitzer'])
		Atot = a0['spitzer']*AS1*r + a1['spitzer']*AS2*r
		ax0.plot(t_plot, 2.5*np.log10(Atot), '-'\
					, color=Spitzer_model_colour)
		if parent_figure is None:
			ax0.set_ylabel(r'$2.5\, \log_{10}\, A$',fontdict={'fontsize':14})
		if primary_model:
			r = 1.0 / (a0['spitzer'] + a1['spitzer'])
			Atot = a0['spitzer']*AS1*r
			ax0.plot(t_plot, 2.5*np.log10(Atot), ':'\
					, color='g')
		if secondary_model:
			r = 1.0 / (a0['spitzer'] + a1['spitzer'])
			Atot = a1['spitzer']*AS2*r
			ax0.plot(t_plot, 2.5*np.log10(Atot), ':'\
					, color='m')

	elif plot_type == 'A':
		r = 1.0 / (a0['spitzer'] + a1['spitzer'])
		Atot = a0['spitzer']*AS1*r + a1['spitzer']*AS2*r
		ax0.plot(t_plot, Atot, '-'\
					, color=Spitzer_model_colour)
		if parent_figure is None:
			ax0.set_ylabel(r"$A'$",fontdict={'fontsize':14})
		if primary_model:
			r = 1.0 / (a0['spitzer'] + a1['spitzer'])
			Atot = a0['spitzer']*AS1*r
			ax0.plot(t_plot, Atot, ':', color='g', label='primary source')
			print(t_plot[np.where(AS1==max(AS1))[0]])
		if secondary_model:
			r = 1.0 / (a0['spitzer'] + a1['spitzer'])
			Atot = a1['spitzer']*AS2*r
			ax0.plot(t_plot, Atot, ':', color='m', label='secondary source')
			print(t_plot[np.where(AS2==max(AS2))[0]])

	else:
		if self.spitzer_plot_scaled:
			ax0.plot(t_plot,\
			a0[self.reference_source]*AS1 \
			+ a1[self.reference_source]*AS2 \
			+a2[self.reference_source]\
			,'-',Spitzer_color=model_colour)
		else:
			ax0.plot(t_plot,\
			(a0['spitzer']*AS1 \
			+ a1['spitzer']*AS2 \
			+ a2['spitzer'])*ratio,'-',color=Spitzer_model_colour)
		if parent_figure is None:
			ax0.set_ylabel(	r'$F_{' + self.reference_source + r'}$'\
							, fontdict={'fontsize':14} )

	# Overlay posterior predictive samples
	if samples is not None:
		ps = self.p.copy()

		for psample in samples[np.random.randint(len(samples), size=samples_to_plot)]:
			ps[np.where(1-self.freeze)[0]] = psample.copy()
			self.match_p(ps)
			# ps for sub classes
			ps1 = self.primary.p.copy()
			ps2 = self.secondary.p.copy()

			AS1 = self.primary.spitzer_magnification(t_plot,p=ps1,LD=SLD)
			AS2 = self.primary.spitzer_magnification(t_plot,p=ps2,LD=SLD)			

			if plot_type == 'magnitudes':
				chi2, a0, a1, a2, _, _, _, _, _ = self.chi2_calc(ps)
				if not('spitzer' in self.gaussian_process_sites):
					if self.spitzer_plot_scaled:
						ax0.plot(t_plot,\
						self.zp-2.5*np.log10(\
						a0[self.reference_source]*AS1 \
						+ a1[self.reference_source]*AS2\
						+ a2[self.reference_source])\
						,'r-',alpha=0.02)
					else:
						ax0.plot(t_plot,\
						self.zp-2.5*np.log10(\
						a0['spitzer']*AS1\
						+ a1['spitzer']*AS2\
						+ a2['spitzer'])\
						-2.5*np.log10(ratio)\
						,'r-',alpha=0.02)

			elif plot_type == 'logA':
				r = 1.0/(a0['spitzer']+a1['spitzer'])
				Atot = a0['spitzer']*AS1*r + a1['spitzer']*AS2*r  
				ax0.plot(t_plot, 2.5*np.log10(Atot),'r-',alpha=0.02)
					
			elif plot_type == 'A':
				r = 1.0/(a0['spitzer']+a1['spitzer'])
				Atot = a0['spitzer']*AS1*r + a1['spitzer']*AS2*r  
				ax0.plot(t_plot,(Atot),'r-',alpha=0.02)

			else:
				chi2, a0, a1, a2, _, _, _, _, _ = self.chi2_calc(ps)
				if not('spitzer' in self.gaussian_process_sites):
					if self.spitzer_plot_scaled:
						ax0.plot(t_plot,\
						a0[self.reference_source]*AS1\
						+ a1[self.reference_source]*AS2\
						+ a2[self.reference_source]\
						,'r-',alpha=0.02)
					else:
						ax0.plot(t_plot,\
						(a0['spitzer']*AS1\
						+ a1['spitzer']*AS2\
						+ a2['spitzer'])\
						*ratio\
						,'r-',alpha=0.02)


	# Plot the data
	self.match_p(p)
	# ps for sub classes
	p1 = self.primary.p.copy()
	p2 = self.secondary.p.copy()
	k = 1
	error_bar_scale = 1.0  # default
	
	if not ('spitzer' in self.gaussian_process_sites):

		k += 1
		site = 'spitzer'
		ratio = 1.0
		
		if self.scale_spitzer_error_bars_multiplicative:
			error_bar_scale = p[self.spitzer_error_bar_scale_index]
		elif self.scale_error_bars_multiplicative:
			error_bar_scale = p[self.error_bar_scale_index[site]]
		else:
			error_bar_scale = 1.0

		if self.spitzer_plot_scaled:
			scaled_dflux 		= a0[self.reference_source] \
								* ( (self.primary.spitzer_data[1] - a2[site])\
								/ a0[site] )\
								+ a2[self.reference_source]
			scaled_dflux_err 	= a0[self.reference_source]\
								* ( (self.primary.spitzer_data[1] + \
								self.primary.spitzer_data[2]*error_bar_scale \
								- a2[site]) / a0[site] ) + \
								a2[self.reference_source] - scaled_dflux
		else:
			ratio 				= (a0[self.reference_source] \
								+ a1[self.reference_source] \
								+ a2[self.reference_source])\
								/ (a0[site]+a1[site]+a2[site])
			scaled_dflux 		= self.primary.spitzer_data[1]*ratio
			scaled_dflux_err 	= self.primary.spitzer_data[2] \
								* error_bar_scale * ratio

		if plot_type == 'magnitudes':
			data_merge 	= self.zp - 2.5*np.log10(scaled_dflux)
			sigs_merge 	= np.abs(self.zp - 2.5*np.log10( \
						scaled_dflux+scaled_dflux_err ) - data_merge)
						
		elif plot_type == 'logA':
			Atot		= ( self.primary.spitzer_data[1] - a2[site] ) \
						/ (a0[site] + a1[site])
			data_merge 	= 2.5*np.log10(Atot)
			dAtot_dF 	= 1.0 / (a0[site] + a1[site])
			dAtot	 	= np.sqrt( dAtot_dF**2 * \
						(self.primary.spitzer_data[2]*error_bar_scale)**2)
			d_dAtot		= 1.08574/Atot
			sigs_merge	= np.sqrt( d_dAtot**2*dAtot**2 )
			
		elif plot_type == 'A':
			Atot		= ( self.primary.spitzer_data[1] - a2[site] ) \
						/ (a0[site] + a1[site])
			data_merge 	= Atot
			dAtot_dF 	= 1.0 / (a0[site] + a1[site])
			dAtot	 	= np.sqrt( dAtot_dF**2 * \
						(self.primary.spitzer_data[2]*error_bar_scale)**2)
			sigs_merge	= dAtot
			
		else:
			data_merge = scaled_dflux
			sigs_merge = scaled_dflux_err

		pts = np.where( (t_min < self.primary.spitzer_data[0]) \
						& (self.primary.spitzer_data[0] < t_max) )[0]
		
		if pts.any():
			ax0.errorbar(self.primary.spitzer_data[0][pts], data_merge[pts], \
						 sigs_merge[pts], fmt='.', ms=8, \
						 mec=Spitzer_model_colour, c=Spitzer_model_colour, \
						 label='Spitzer')

	if draw_grid:
		ax0.grid()

	if parent_figure is None:
		if legend_location is None:
			ax0.legend(loc='upper right')
		else:
			ax0.legend(loc=legend_location) 

	title_string 	= plot_title + '\n' + \
					r'$  s = %5.3f, \log q = %6.3f, \rho_1 = %7.5f, u_{0,1} = %6.3f, \alpha_1 = %5.3f$'\
					%(d,np.log10(q),rho1,u01,phi1) + '\n' + \
					r'$t_0 = %9.3f, t_E = %8.3f$'%(t01,tE1) + \
					r'$, \pi_{E,E} = %5.3f, \pi_{E,N} = %5.3f$'%(piEE,piEN) + '\n' + \
					r'$\rho_2 = %7.5f, \Delta u_0 = %6.3f, \Delta\alpha = %5.3f, \Delta t_0 = %9.3f, t_{E,2} = %8.3f$'\
					%(rho2,deltau0,deltaphi,deltat0,tE2)

	if self.use_limb_darkening:
		Gamma = p[self.limb_index]
		title_string += r'$, \Gamma = %5g$'%Gamma

	if self.use_lens_orbital_motion:
		dddt = p[self.dddt_index]
		dphidt = p[self.dphidt_index]
		title_string 	+= '\n' + \
						r'$\dot{s}\,(\rm{yr}^{-1}) = %5.3f, \dot{\alpha}\,(\rm{yr}^{-1})  = %5.3f$'\
						%(dddt,dphidt)

		if self.use_lens_orbital_acceleration:

			d2ddt2 = p[self.d2ddt2_index]
			d2phidt2 = p[self.d2phidt2_index]
			title_string 	+= r', $\ddot{s}\,(\rm{yr}^{-2}) = %3g, \ddot{\alpha}\,(\rm{yr}^{-2})  = %3g$'\
							%(d2ddt2,d2phidt2)
			
	if self.use_gaussian_process_model:
	
		a = p[self.gaussian_process_index]
		c = p[self.gaussian_process_index+1]
		if self.GP_model == 'Real':
			title_string += r'$,\ln{a} = %5.3f, \ln{c}  = %5.3f$'%(a,c)
		elif self.GP_model == 'Matern':
			title_string += r'$,\ln{\sigma} = %5.3f, \ln{\rho}  = %5.3f$'%(a,c)

	if self.scale_spitzer_error_bars_multiplicative:
	
		S = p[self.spitzer_error_bar_scale_index]
		title_string += r'$,\n S_{\rm Spitzer} = %5.3f$'%(S)

	if parent_figure is None and draw_title:
		ax0.set_title(title_string,fontdict={'fontsize':fs})

	if y_range is not None:
		ax0.set_ylim(y_range)

	if plot_residual:
		ax0.tick_params(labelbottom='off') 
		plt.setp(ax0.get_xticklabels(), visible=False)

	ax0.set_xlim((t_min,t_max))
	xlim = ax0.get_xlim()

	if x_tick_step is not None:
		start, end = ax0.get_xlim()
		ax0.xaxis.set_ticks(np.arange(start, end, x_tick_step))

	# Residuals plot
	#ax1 = plt.subplot(gs[1],sharex=ax0)
	if plot_residual:
		chi2, a0, a1, a2, _, _, _, _, _ = self.chi2_calc(p)

		if parent_figure is None:
			ax1 = fig.add_axes([0.12,0.1,0.85,0.15],sharex=ax0)
		else:
			if axis_location is None:
				ax1 = fig.add_axes([0.15,0.6-0.0643,0.4,0.0643],sharex=ax0)
			else:
				ax1 = fig.add_axes(	[axis_location[0], axis_location[1]-0.0643\
									, axis_location[2], 0.0643])

		#divider = make_axes_locatable(ax0)
		#ax1 = divider.append_axes("bottom", size="20%", pad=0.0, sharex=ax0)
								
		# Spitzer data residuals
		if not('spitzer' in self.gaussian_process_sites):
			k += 1
			site = 'spitzer'
			AS1 = self.primary.spitzer_magnification( \
												self.primary.spitzer_data[0], \
												p=p1, LD=SLD)
			AS2 = self.secondary.spitzer_magnification(\
												self.primary.spitzer_data[0], \
												p=p2, LD=SLD)
			ratio = 1.0

			if self.scale_error_bars_multiplicative:
				error_bar_scale = p[self.error_bar_scale_index[site]]
			elif self.scale_spitzer_error_bars_multiplicative:
				error_bar_scale = p[self.spitzer_error_bar_scale_index]

			if self.spitzer_plot_scaled:
				scaled_dflux 		= a0[self.reference_source] * \
									( (self.primary.spitzer_data[1] \
									- a2[site])/a0[site] ) \
									+ a2[self.reference_source]
				scaled_dflux_err 	= a0[self.reference_source] * \
									( (self.spitzer_data[1] \
									+ self.primary.spitzer_data[2] \
									* error_bar_scale - a2[site])\
									/ a0[site] ) \
									+ a2[self.reference_source] - scaled_dflux
			else:
				ratio 				= (a0[self.reference_source] \
									+ a1[self.reference_source]\
									+ a2[self.reference_source])\
									/(a0[site]+a1[site]+a2[site])
				scaled_dflux 		= self.primary.spitzer_data[1]*ratio
				scaled_dflux_err 	= self.primary.spitzer_data[2] \
									* ratio*error_bar_scale

			if plot_type == 'magnitudes':
				if self.spitzer_plot_scaled:
					scaled_model 	= self.zp - 2.5*np.log10(\
									a0[self.reference_source]*AS1 \
									+ a1[self.reference_source]*AS2 \
									+ a2[self.reference_source])
				else:
					scaled_model 	= self.zp - 2.5*np.log10(a0[site]*AS1 \
									+ a1[site]*AS2 + a2[site]) \
									- 2.5*np.log10(ratio)
				data_merge 		= self.zp - 2.5*np.log10(scaled_dflux)
				sigs_merge 		= np.abs( self.zp \
								- 2.5*np.log10(scaled_dflux+scaled_dflux_err) \
								- data_merge )
								
			elif plot_type == 'logA':
				r				= 1.0 / (a0[site] + a1[site])
				Atot			= a0[site]*AS1*r + a1[site]*AS2*r
				scaled_model 	= 2.5*np.log10(Atot)
				Atot			= (self.primary.spitzer_data[1] \
								- a2[site]) \
								/ (a0[site] + a1[site])
				data_merge 		= 2.5*np.log10(Atot)
				dAtot_dF 		= 1.0 / (a0[site] + a1[site])
				dAtot	 		= np.sqrt( dAtot_dF**2 * \
								(self.primary.spitzer_data[2]\
								*error_bar_scale)**2 )
				d_dAtot			= 1.08574/Atot
				sigs_merge		= np.sqrt( d_dAtot**2*dAtot**2 )
			
			elif plot_type == 'A':
				r				= 1.0 / (a0[site] + a1[site])
				Atot			= a0[site]*AS1*r + a1[site]*AS2*r
				scaled_model 	= Atot
				Atot			= (self.primary.spitzer_data[1] \
								- a2[site]) \
								/ (a0[site] + a1[site])
				data_merge 		= Atot
				dAtot_dF 		= 1.0 / (a0[site] + a1[site])
				dAtot	 		= np.sqrt( dAtot_dF**2 * \
								(self.primary.spitzer_data[2]\
								*error_bar_scale)**2 )
				sigs_merge		= dAtot
				
			else:
				if self.spitzer_plot_scaled:
					scaled_model 	= ( a0[self.reference_source]*AS1 \
									+ a1[self.reference_source]*AS2 \
									+ a2[self.reference_source])
				else:
					scaled_model 	= (a0[site]*AS1 + a1[site]*AS2 + a2[site])\
									* ratio
				data_merge = scaled_dflux
				sigs_merge = scaled_dflux_err
			data_merge -= scaled_model
			pts = np.where((t_min < self.primary.spitzer_data[0]) \
							& (self.primary.spitzer_data[0] < t_max))[0]
			if pts.any():
				ax1.errorbar(self.primary.spitzer_data[0][pts], \
							 data_merge[pts], sigs_merge[pts], fmt='.', \
							 ms=2, mec=Spitzer_model_colour, \
							 c=Spitzer_model_colour, label='Spitzer')

		ax1.set_xlim(xlim)
		if draw_grid:
			ax1.grid()
		if self.plot_date_range:
			ax1.set_xlim(self.plot_date_range)
		if y_residual_range is not None:
			ax1.set_ylim(y_residual_range)
		if plot_type == 'magnitudes':
			ax1.invert_yaxis()
		ax1.yaxis.set_major_locator(MaxNLocator(2,prune='upper'))

		if parent_figure is None:
			plt.xlabel('HJD-2450000',fontdict={'fontsize':fs})
			if plot_type == 'magnitudes':
				ax1.set_ylabel(r'$\Delta I_{'+self.reference_source+r'}$',\
								fontdict={'fontsize':fs})
			elif plot_type == 'logA':
				ax1.set_ylabel(r'$Residual$',fontdict={'fontsize':fs})
			elif plot_type == 'A':
				ax1.set_ylabel(r'$Residual$',fontdict={'fontsize':fs})
			else:
				ax1.set_ylabel(r'$\Delta F_{'+self.reference_source+r'}$',\
								fontdict={'fontsize':fs})

		if x_tick_step is not None:
			start, end = ax1.get_xlim()
			ax1.xaxis.set_ticks(np.arange(start, end, x_tick_step))

		if y_tick_step_residual is not None:
			start, end = ax1.get_ylim()
			ax1.yaxis.set_ticks(np.arange(start, end, y_tick_step_residual))

	if close_fig:
		plt.savefig(self.plotprefix+'-spitzer.png',pad_inches=pad)
		plt.close()

	return fig
	

def plot_lightcurve(self, p=None, t_range=None, plot_title='', plot_type=None\
					, y_range=None, parent_figure=None, close_fig=True\
					, y_residual_range=None, draw_title=True\
					, axis_location=None, x_tick_step=None, plot_residual=True\
					, y_tick_step_residual=None, samples=None\
					, samples_to_plot=50, draw_grid=True, model_colour='k'\
					, Spitzer_model_colour='r', fs=12, pad=0.5\
					, legend_location=None):
	''' '''
	
	# Limb darkening stuff
	if isinstance(self.limb_constant, float):
		LD = self.limb_constant
		SLD = self.spitzer_limb_constant
	else:
		LD = self.limb_constant[self.reference_source]
		if self.spitzer_limb_constant != self.limb_constant['spitzer']:
			print('\n\n')
			print(	'Replacing Spitzer limb constant (%f) with %f.' \
					%( self.spitzer_limb_constant\
					, self.limb_constant['spitzer']) )
			print('\n\n')
			self.spitzer_limb_constant = self.limb_constant['spitzer']
		SLD = self.spitzer_limb_constant
	
	# changing default matplotlb font size
	matplotlib.rcParams.update({'font.size': fs})

	# Getting the parameters for plotting
	if p is None: #default
		p = self.p.copy()
	elif (len(p)!=len(self.freeze)) and (len(p)==len(p[self.freeze==0])):
		ps = p.copy() 	# params specified in function call. This will not
						# accept tuples
		p = self.p.copy()	# class stored, full set of parameters (including
							# frozen ones)
		p[self.freeze==0] = ps	# replacing non-frozen parameters with those
								# from the function call
		# now p is the full parameter set including frozen values
	self.match_p() # makes sure p is universal for parent and child classes
	p1 = self.primary.p.copy()
	p2 = self.secondary.p.copy()

	if plot_type is None:
		plot_type = self.lightcurve_plot_type

	# t stuff
	if t_range:
		t_min, t_max = t_range
	elif self.plot_date_range:
		t_min, t_max = self.plot_date_range
	else:
		t_min = np.min(self.ts) - 2.5
		t_max = np.max(self.ts) + 2.5

	# Unpacking p
	logd, logq, logrho1, u01, phi1, t01, tE1, piEE, piEN, logrho2, deltau0, deltaphi, deltat0, tE2 = p[:14]
	# logd, logq, logrho1, u01, phi1, t01, tE1, piEE1, piEN1, logrho2, u02, t02
	#     = p[:12]
	u02 = u01 + deltau0
	phi2 = phi1 + deltaphi
	t02 = t01 + deltat0
	d = 10.0**logd
	q = 10.0**logq
	rho1 = 10.0**logrho1
	rho2 = 10.0**logrho2

	# chi2, Fs1, Fs2, FB, colour constraint chi2 component, elementwise chi2
	chi2, a0, a1, a2, _, _, _, chi2_colour, chi2_elements = self.chi2_calc(p)
	
	# Debugging prints  
	print('p', p)
	print('chi2', chi2)
	print('a0', a0)  # Fs1
	print('a1', a1)  # Fs2
	print('a2', a2)  # Fs3
	print('cc', chi2_colour)

	# Figure set-up
	if parent_figure is None:
		fig = plt.figure(figsize=(7,5.3))
		if axis_location is None:
			ax0 = fig.add_axes([0.12,0.25,0.85,0.6])
		else:
			ax0 = fig.add_axes(axis_location)
	else:
		#ax0 = inset_locator.inset_axes(parent_axes, width="40%", height="20%"
		# , loc=1)
		fig = parent_figure
		if axis_location is None:
			ax0 = fig.add_axes([0.15,0.6,0.4,0.3])
		else:
			ax0 = fig.add_axes(axis_location)

	ratio = 1.0  # for scaling Spitzer vs Ground
	
	# grid = AxesGrid(fig, 111, nrows_ncols = (2,1), axes_pad = 0.0,
	# share_all=True, label_mode = "L")
	# gs = gridspec.GridSpec(2,1,height_ratios=(3,1))

	# Main lightcurve plot
	#ax0 = plt.subplot(gs[0])

	# Plot the model
	n_plot_points = np.min((1001,100*(t_max-t_min)//(np.abs(tE1)*rho1)))
	  # estimating a nnumber that will make the model look smooth
	n_plot_points = int(n_plot_points)  # so that the next line works
	t_plot = np.linspace(t_min,t_max,n_plot_points)

	A1 = self.primary.magnification(t_plot,p=p1,LD=LD)
	A2 = self.secondary.magnification(t_plot,p=p2,LD=LD)
	AS1 = self.primary.spitzer_magnification(t_plot,p=p1,LD=SLD)
	AS2 = self.secondary.spitzer_magnification(t_plot,p=p2,LD=SLD)

	if plot_type == 'magnitudes':
		ax0.plot(t_plot,\
				self.zp-2.5*np.log10(\
				a0[self.reference_source]*A1 \
				+ a1[self.reference_source]*A2\
				+ a2[self.reference_source])\
					,'-',color=model_colour)

		if not('spitzer' in self.gaussian_process_sites):
			if self.spitzer_plot_scaled:
				ax0.plot(t_plot,\
				self.zp-2.5*np.log10(\
				a0[self.reference_source]*AS1 \
				+ a1[self.reference_source]*AS2\
				+ a2[self.reference_source])\
				,'-',color=Spitzer_model_colour)
			else:
				ax0.plot(	t_plot,	self.zp - 2.5 * np.log10( \
									a0['spitzer']*AS1 + a1['spitzer']*AS1 \
									+ a2['spitzer'] ) - 2.5*np.log10(ratio)\
							,'-',color=Spitzer_model_colour)

		ax0.invert_yaxis()  # because magnitudes are upsidedown

		if parent_figure is None:
			ax0.set_ylabel(	r'$I_{' + self.reference_source + r'}$'\
							, fontdict={'fontsize':14})

	elif plot_type == 'logA':
		r = 1.0 / (a0[self.reference_source] + a1[self.reference_source])
		Atot = a0[self.reference_source]*A1*r + a1[self.reference_source]*A2*r
		ax0.plot(t_plot, 2.5*np.log10(Atot), '-', color=model_colour)
		
		if not('spitzer' in self.gaussian_process_sites):
			r = 1.0 / (a0['spitzer'] + a1['spitzer'])
			Atot = a0['spitzer']*AS1*r + a1['spitzer']*AS2*r
			ax0.plot(t_plot, 2.5*np.log10(Atot), '-'\
						, color=Spitzer_model_colour)
		if parent_figure is None:
			ax0.set_ylabel(r'$2.5\, \log_{10}\, A$',fontdict={'fontsize':14})

	else:
		ax0.plot(	t_plot, a0[self.reference_source]*A1 \
							+ a1[self.reference_source]*A2 \
							+ a2[self.reference_source]\
					, '-', color=model_colour)
		if not('spitzer' in self.gaussian_process_sites):
			if self.spitzer_plot_scaled:
				ax0.plot(t_plot,\
				a0[self.reference_source]*AS1 \
				+ a1[self.reference_source]*AS2 \
				+a2[self.reference_source]\
				,'-',Spitzer_color=model_colour)
			else:
				ax0.plot(t_plot,\
				(a0['spitzer']*AS1 \
				+ a1['spitzer']*AS2 \
				+ a2['spitzer'])*ratio,'-',color=Spitzer_model_colour)
		if parent_figure is None:
			ax0.set_ylabel(	r'$F_{' + self.reference_source + r'}$'\
							, fontdict={'fontsize':14} )

	# Overlay posterior predictive samples
	if samples is not None:
		ps = self.p.copy()

		for psample in samples[np.random.randint(len(samples), size=samples_to_plot)]:
			ps[np.where(1-self.freeze)[0]] = psample.copy()
			self.match_p(ps)
			# ps for sub classes
			ps1 = self.primary.p.copy()
			ps2 = self.secondary.p.copy()

			A1 = self.primary.magnification(t_plot,p=ps1,LD=LD)
			A2 = self.secondary.magnification(t_plot,p=ps2,LD=LD)
			AS1 = self.primary.spitzer_magnification(t_plot,p=ps1,LD=SLD)
			AS2 = self.primary.spitzer_magnification(t_plot,p=ps2,LD=SLD)			

			if plot_type == 'magnitudes':
				chi2, a0, a1, a2, _, _, _, _, _ = self.chi2_calc(ps)
				ax0.plot(t_plot,\
				self.zp-2.5*np.log10(\
				a0[self.reference_source]*A1 \
				+ a1[self.reference_source]*A2\
				+ a2[self.reference_source])\
				,'b-',alpha=0.02)

				if not('spitzer' in self.gaussian_process_sites):
					if self.spitzer_plot_scaled:
						ax0.plot(t_plot,\
						self.zp-2.5*np.log10(\
						a0[self.reference_source]*AS1 \
						+ a1[self.reference_source]*AS2\
						+ a2[self.reference_source])\
						,'r-',alpha=0.02)
					else:
						ax0.plot(t_plot,\
						self.zp-2.5*np.log10(\
						a0['spitzer']*AS1\
						+ a1['spitzer']*AS2\
						+ a2['spitzer'])\
						-2.5*np.log10(ratio)\
						,'r-',alpha=0.02)

			elif plot_type == 'logA':
				r = 1.0/(a0[self.reference_source]+a1[self.reference_source])
				Atot 	= a0[self.reference_source]*A1*r \
						+ a1[self.reference_source]*A2*r 
				ax0.plot(t_plot,\
				2.5*np.log10(Atot)
				,'b-',alpha=0.02)

				if not('spitzer' in self.gaussian_process_sites):
					r = 1.0/(a0['spitzer']+a1['spitzer'])
					Atot = a0['spitzer']*AS1*r + a1['spitzer']*AS2*r  
					ax0.plot(t_plot,\
					2.5*np.log10(Atot)\
					,'r-',alpha=0.02)

			else:
				chi2, a0, a1, a2, _, _, _, _, _ = self.chi2_calc(ps)
				ax0.plot(t_plot,\
				a0[self.reference_source]*A1\
				+ a1[self.reference_source]*A2\
				+ a2[self.reference_source]\
				,'b-',alpha=0.02)

				if not('spitzer' in self.gaussian_process_sites):
					if self.spitzer_plot_scaled:
						ax0.plot(t_plot,\
						a0[self.reference_source]*AS1\
						+ a1[self.reference_source]*AS2\
						+ a2[self.reference_source]\
						,'r-',alpha=0.02)
					else:
						ax0.plot(t_plot,\
						(a0['spitzer']*AS1\
						+ a1['spitzer']*AS2\
						+ a2['spitzer'])\
						*ratio\
						,'r-',alpha=0.02)


	# Plot the data
	self.match_p(p)
	# ps for sub classes
	p1 = self.primary.p.copy()
	p2 = self.secondary.p.copy()
	error_bar_scale = 1.0  # default
	
	for k, site in enumerate(self.primary.data):
		if not(site in self.gaussian_process_sites):
			if self.scale_error_bars_multiplicative:
				error_bar_scale = p[self.error_bar_scale_index[site]]

			scaled_dflux 	= a0[self.reference_source] \
							* ( (self.primary.data[site][1] - a2[site])\
							/ a0[site] ) + a2[self.reference_source]
			# the above line asume Fs1/Fs2 = Fs1'/Fs2'
			# i.e. Fs2' = Fs1'xFs2/Fs1.
			# F = A1xFs1 + A2xFs2 + FB and F' = A1xFs1' + A2xFs2' + FB'
			# equate A1 and solve for F' (scaled change in flux; scaled_dflux):
			# F' = A2x(Fs2'-Fs1'xFs2/Fs1) + Fs1'x(F - FB)/Fs1 + FB' 
			#    = Fs1'x(F - FB)/Fs1 + FB'
			# because, with the assumption, *A2 term* = 0  
			scaled_dflux_err 	= a0[self.reference_source] * \
								( (self.primary.data[site][1] \
								+ self.primary.data[site][2]*error_bar_scale \
								- a2[site]) / a0[site] )\
								+ a2[self.reference_source] - scaled_dflux

			bad_points = ( scaled_dflux < 0 )

			if plot_type == 'magnitudes':
				data_merge 	= self.zp - 2.5*np.log10(scaled_dflux)
				sigs_merge 	= np.abs( self.zp \
							- 2.5*np.log10(scaled_dflux+scaled_dflux_err) \
							- data_merge )
			elif plot_type == 'logA':
				# Atot = Fs1xA1/(Fs1+Fs2) + Fs2xA2/(Fs1+Fs2).
				# Need Atot not in terms of A1 or A2.
				# (Fs1+Fs2)xAtot = Fs1xA1 + Fs2xA2, and
				# F = Fs1xA1 + Fs2xA2 + Fb = (Fs1+Fs2)xAtot + Fb, so
				#
				# Atot = (F-Fb)/(Fs1+Fs2).
				#
				# I believe this did not require an assumption that 
				# Fs1/Fs2 = Fs1'/Fs2'
				Atot		= (self.primary.data[site][1] - a2[site]) \
							/ (a0[site] + a1[site])
				data_merge 	= 2.5*np.log10(Atot)
				dAtot_dF 	= 1.0 / (a0[site] + a1[site])
				dAtot	 	= np.sqrt( dAtot_dF**2 * \
							(self.primary.data[site][2]*error_bar_scale)**2)
				# d/dA (2.5 log10(A)) = 1.08574/A
				d_dAtot		= 1.08574/Atot
				sigs_merge	= np.sqrt( d_dAtot**2*dAtot**2 )
			else:
				data_merge = scaled_dflux
				sigs_merge = scaled_dflux_err

			pts = np.where( (t_min < self.primary.data[site][0]) \
							& (self.primary.data[site][0] < t_max))[0]
			
			if pts.any():
				ax0.errorbar(self.primary.data[site][0][pts], data_merge[pts],\
							 sigs_merge[pts], fmt='.', ms=8, \
							 mec=self.plot_colours[k], c=self.plot_colours[k],\
							 label=site)
	
	if not ('spitzer' in self.gaussian_process_sites):

		k += 1
		site = 'spitzer'
		ratio = 1.0
		
		if self.scale_spitzer_error_bars_multiplicative:
			error_bar_scale = p[self.spitzer_error_bar_scale_index]
		elif self.scale_error_bars_multiplicative:
			error_bar_scale = p[self.error_bar_scale_index[site]]
		else:
			error_bar_scale = 1.0

		if self.spitzer_plot_scaled:
			scaled_dflux 		= a0[self.reference_source] \
								* ( (self.primary.spitzer_data[1] - a2[site])\
								/ a0[site] )\
								+ a2[self.reference_source]
			scaled_dflux_err 	= a0[self.reference_source]\
								* ( (self.primary.spitzer_data[1] + \
								self.primary.spitzer_data[2]*error_bar_scale \
								- a2[site]) / a0[site] ) + \
								a2[self.reference_source] - scaled_dflux
		else:
			ratio 				= (a0[self.reference_source] \
								+ a1[self.reference_source] \
								+ a2[self.reference_source])\
								/ (a0[site]+a1[site]+a2[site])
			scaled_dflux 		= self.primary.spitzer_data[1]*ratio
			scaled_dflux_err 	= self.primary.spitzer_data[2] \
								* error_bar_scale * ratio

		if plot_type == 'magnitudes':
			data_merge 	= self.zp - 2.5*np.log10(scaled_dflux)
			sigs_merge 	= np.abs(self.zp - 2.5*np.log10( \
						scaled_dflux+scaled_dflux_err ) - data_merge)
		elif plot_type == 'logA':
			Atot		= ( self.primary.spitzer_data[1] - a2[site] ) \
						/ (a0[site] + a1[site])
			data_merge 	= 2.5*np.log10(Atot)
			dAtot_dF 	= 1.0 / (a0[site] + a1[site])
			dAtot	 	= np.sqrt( dAtot_dF**2 * \
						(self.primary.spitzer_data[2]*error_bar_scale)**2)
			d_dAtot		= 1.08574/Atot
			sigs_merge	= np.sqrt( d_dAtot**2*dAtot**2 )
		else:
			data_merge = scaled_dflux
			sigs_merge = scaled_dflux_err

		pts = np.where( (t_min < self.primary.spitzer_data[0]) \
						& (self.primary.spitzer_data[0] < t_max) )[0]
		
		if pts.any():
			ax0.errorbar(self.primary.spitzer_data[0][pts], data_merge[pts], \
						 sigs_merge[pts], fmt='.', ms=8, \
						 mec=Spitzer_model_colour, c=Spitzer_model_colour, \
						 label='Spitzer')

	if draw_grid:
		ax0.grid()

	if parent_figure is None:
		if legend_location is None:
			ax0.legend(loc='upper right')
		else:
			ax0.legend(loc=legend_location) 

	title_string 	= plot_title + '\n' + \
					r'$  s = %5.3f, \log q = %6.3f, \rho_1 = %7.5f, u_{0,1} = %6.3f, \alpha_1 = %5.3f$'\
					%(d,np.log10(q),rho1,u01,phi1) + '\n' + \
					r'$t_0 = %9.3f, t_E = %8.3f$'%(t01,tE1) + \
					r'$, \pi_{E,E} = %5.3f, \pi_{E,N} = %5.3f$'%(piEE,piEN) + '\n' + \
					r'$\rho_2 = %7.5f, \Delta u_0 = %6.3f, \Delta\alpha = %5.3f, \Delta t_0 = %9.3f, t_{E,2} = %8.3f$'\
					%(rho2,deltau0,deltaphi,deltat0,tE2)

	if self.use_limb_darkening:
		Gamma = p[self.limb_index]
		title_string += r'$, \Gamma = %5g$'%Gamma

	if self.use_lens_orbital_motion:
		dddt = p[self.dddt_index]
		dphidt = p[self.dphidt_index]
		title_string 	+= '\n' + \
						r'$\dot{s}\,(\rm{yr}^{-1}) = %5.3f, \dot{\alpha}\,(\rm{yr}^{-1})  = %5.3f$'\
						%(dddt,dphidt)

		if self.use_lens_orbital_acceleration:

			d2ddt2 = p[self.d2ddt2_index]
			d2phidt2 = p[self.d2phidt2_index]
			title_string 	+= r', $\ddot{s}\,(\rm{yr}^{-2}) = %3g, \ddot{\alpha}\,(\rm{yr}^{-2})  = %3g$'\
							%(d2ddt2,d2phidt2)
			
	if self.use_gaussian_process_model:
	
		a = p[self.gaussian_process_index]
		c = p[self.gaussian_process_index+1]
		if self.GP_model == 'Real':
			title_string += r'$,\ln{a} = %5.3f, \ln{c}  = %5.3f$'%(a,c)
		elif self.GP_model == 'Matern':
			title_string += r'$,\ln{\sigma} = %5.3f, \ln{\rho}  = %5.3f$'%(a,c)

	if self.scale_spitzer_error_bars_multiplicative:
	
		S = p[self.spitzer_error_bar_scale_index]
		title_string += r'$,\n S_{\rm Spitzer} = %5.3f$'%(S)

	if parent_figure is None and draw_title:
		ax0.set_title(title_string,fontdict={'fontsize':fs})

	if y_range is not None:
		ax0.set_ylim(y_range)

	if plot_residual:
		ax0.tick_params(labelbottom='off') 
		plt.setp(ax0.get_xticklabels(), visible=False)

	ax0.set_xlim((t_min,t_max))
	xlim = ax0.get_xlim()

	if x_tick_step is not None:
		start, end = ax0.get_xlim()
		ax0.xaxis.set_ticks(np.arange(start, end, x_tick_step))

	# Residuals plot
	#ax1 = plt.subplot(gs[1],sharex=ax0)
	if plot_residual:
		chi2, a0, a1, a2, _, _, _, _, _ = self.chi2_calc(p)

		if parent_figure is None:
			ax1 = fig.add_axes([0.12,0.1,0.85,0.15],sharex=ax0)
		else:
			if axis_location is None:
				ax1 = fig.add_axes([0.15,0.6-0.0643,0.4,0.0643],sharex=ax0)
			else:
				ax1 = fig.add_axes(	[axis_location[0], axis_location[1]-0.0643\
									, axis_location[2], 0.0643])

		#divider = make_axes_locatable(ax0)
		#ax1 = divider.append_axes("bottom", size="20%", pad=0.0, sharex=ax0)
		'''
		# Ground data residuals
		if isinstance(self.limb_constant, float):
			A_all1 = self.primary.magnification(self.primary.ts)
			A_all2 = self.secondary.magnification(self.primary.ts)
		else:
			A_all1 = A_all2 = None

		for k, site in enumerate(self.primary.data):
			if not (site in self.gaussian_process_sites):
				if self.scale_error_bars_multiplicative:
					error_bar_scale = p[self.error_bar_scale_index[site]]
					
				if A_all1 is not None:
					A1 	= A_all1[self.primary.data_range[site][0]\
						:self.primary.data_range[site][1]]
					A2 	= A_all2[self.primary.data_range[site][0]\
						:self.primary.data_range[site][1]]
				else:
					LD 	= self.limb_constant[site]
					A1 	= self.primary.magnification(\
						self.primary.ts[self.primary.data_range[site][0]\
						:self.primary.data_range[site][1]],LD=LD)
					A2 	= self.secondary.magnification( \
						self.primary.ts[self.primary.data_range[site][0]\
						:self.primary.data_range[site][1]],LD=LD)

				scaled_dflux 		= a0[self.reference_source] \
									* ( (self.primary.data[site][1] \
									- a2[site])/a0[site] ) \
									+ a2[self.reference_source]
				scaled_dflux_err 	= a0[self.reference_source]*\
									( (self.primary.data[site][1] \
									+ self.data[site][2]*error_bar_scale \
									- a2[site])/a0[site] ) \
									+ a2[self.reference_source] - scaled_dflux
				bad_points = ( scaled_dflux < 0 )

				if plot_type == 'magnitudes':
					scaled_model 	= self.zp -2.5*np.log10( \
									a0[self.reference_source]*A1 \
									+ a1[self.reference_source]*A2 \
									+ a0[self.reference_source]) 
					data_merge 		= self.zp - 2.5*np.log10(scaled_dflux) 
					sigs_merge 		= np.abs(self.zp - 2.5*np.log10( \
									scaled_dflux+scaled_dflux_err) \
									- data_merge)
				elif plot_type == 'logA':
					r				= 1.0 / (a1[site] + a0[site])
					Atot 			= a1[site]*A1*r + a0[site]*A2*r
					scaled_model 	= 2.5*np.log10(Atot)
					Atot			= (self.primary.data[site][1] - a2[site]) \
									/ (a1[site] + a0[site])
					data_merge 		= 2.5*np.log10(Atot)
					dAtot_dF 		= 1.0 / (a1[site] + a0[site])
					dAtot	 		= np.sqrt( dAtot_dF**2 * \
									(self.primary.data[site][2]\
									* error_bar_scale)**2)
					d_dAtot			= 1.08574/Atot
					sigs_merge		= np.sqrt( d_dAtot**2*dAtot**2 )
				else:
					scaled_model 	= a0[self.reference_source]*A1 \
									+ a1[self.reference_source]*A2\
									+ a2[self.reference_source]
					data_merge 		= scaled_dflux
					sigs_merge 		= scaled_dflux_err
				data_merge -= scaled_model
				pts = np.where( (t_min < self.primary.data[site][0]) & \
								(self.primary.data[site][0] < t_max) )[0]
				if pts.any():
					ax1.errorbar(self.data[site][0][pts], data_merge[pts], \
								 sigs_merge[pts], fmt='.', ms=2, \
								 mec=self.plot_colours[k], \
								 c=self.plot_colours[k], label=site)'''
								
		# Spitzer data residuals
		if not('spitzer' in self.gaussian_process_sites):
			k += 1
			site = 'spitzer'
			AS1 = self.primary.spitzer_magnification( \
												self.primary.spitzer_data[0], \
												p=p1, LD=SLD)
			AS2 = self.secondary.spitzer_magnification(\
												self.primary.spitzer_data[0], \
												p=p2, LD=SLD)
			ratio = 1.0

			if self.scale_error_bars_multiplicative:
				error_bar_scale = p[self.error_bar_scale_index[site]]
			elif self.scale_spitzer_error_bars_multiplicative:
				error_bar_scale = p[self.spitzer_error_bar_scale_index]

			if self.spitzer_plot_scaled:
				scaled_dflux 		= a0[self.reference_source] * \
									( (self.primary.spitzer_data[1] \
									- a2[site])/a0[site] ) \
									+ a2[self.reference_source]
				scaled_dflux_err 	= a0[self.reference_source] * \
									( (self.spitzer_data[1] \
									+ self.primary.spitzer_data[2] \
									* error_bar_scale - a2[site])\
									/ a0[site] ) \
									+ a2[self.reference_source] - scaled_dflux
			else:
				ratio 				= (a0[self.reference_source] \
									+ a1[self.reference_source]\
									+ a2[self.reference_source])\
									/(a0[site]+a1[site]+a2[site])
				scaled_dflux 		= self.primary.spitzer_data[1]*ratio
				scaled_dflux_err 	= self.primary.spitzer_data[2] \
									* ratio*error_bar_scale

			if plot_type == 'magnitudes':
				if self.spitzer_plot_scaled:
					scaled_model 	= self.zp - 2.5*np.log10(\
									a0[self.reference_source]*AS1 \
									+ a1[self.reference_source]*AS2 \
									+ a2[self.reference_source])
				else:
					scaled_model 	= self.zp - 2.5*np.log10(a0[site]*AS1 \
									+ a1[site]*AS2 + a2[site]) \
									- 2.5*np.log10(ratio)
				data_merge 		= self.zp - 2.5*np.log10(scaled_dflux)
				sigs_merge 		= np.abs( self.zp \
								- 2.5*np.log10(scaled_dflux+scaled_dflux_err) \
								- data_merge )
			elif plot_type == 'logA':
				r				= 1.0 / (a0[site] + a1[site])
				Atot			= a0[site]*AS1*r + a1[site]*AS2*r
				scaled_model 	= 2.5*np.log10(Atot)
				Atot			= (self.primary.spitzer_data[1] \
								- a2[site]) \
								/ (a0[site] + a1[site])
				data_merge 		= 2.5*np.log10(Atot)
				dAtot_dF 		= 1.0 / (a0[site] + a1[site])
				dAtot	 		= np.sqrt( dAtot_dF**2 * \
								(self.primary.spitzer_data[2]\
								*error_bar_scale)**2 )
				d_dAtot			= 1.08574/Atot
				sigs_merge		= np.sqrt( d_dAtot**2*dAtot**2 )
				
			else:
				if self.spitzer_plot_scaled:
					scaled_model 	= ( a0[self.reference_source]*AS1 \
									+ a1[self.reference_source]*AS2 \
									+ a2[self.reference_source])
				else:
					scaled_model 	= (a0[site]*AS1 + a1[site]*AS2 + a2[site])\
									* ratio
				data_merge = scaled_dflux
				sigs_merge = scaled_dflux_err
			data_merge -= scaled_model
			pts = np.where((t_min < self.primary.spitzer_data[0]) \
							& (self.primary.spitzer_data[0] < t_max))[0]
			if pts.any():
				ax1.errorbar(self.primary.spitzer_data[0][pts], \
							 data_merge[pts], sigs_merge[pts], fmt='.', \
							 ms=2, mec=Spitzer_model_colour, \
							 c=Spitzer_model_colour, label='Spitzer')

		ax1.set_xlim(xlim)
		if draw_grid:
			ax1.grid()
		if self.plot_date_range:
			ax1.set_xlim(self.plot_date_range)
		if y_residual_range is not None:
			ax1.set_ylim(y_residual_range)
		if plot_type == 'magnitudes':
			ax1.invert_yaxis()
		ax1.yaxis.set_major_locator(MaxNLocator(2,prune='upper'))

		if parent_figure is None:
			plt.xlabel('HJD-2450000',fontdict={'fontsize':fs})
			if plot_type == 'magnitudes':
				ax1.set_ylabel(r'$\Delta I_{'+self.reference_source+r'}$',\
								fontdict={'fontsize':fs})
			elif plot_type == 'logA':
				ax1.set_ylabel(r'$Residual$',fontdict={'fontsize':fs})
			else:
				ax1.set_ylabel(r'$\Delta F_{'+self.reference_source+r'}$',\
								fontdict={'fontsize':fs})

		if x_tick_step is not None:
			start, end = ax1.get_xlim()
			ax1.xaxis.set_ticks(np.arange(start, end, x_tick_step))

		if y_tick_step_residual is not None:
			start, end = ax1.get_ylim()
			ax1.yaxis.set_ticks(np.arange(start, end, y_tick_step_residual))

	if close_fig:
		plt.savefig(self.plotprefix+'-lightcurve.png',pad_inches=pad)
		plt.close()

	return fig


'''def plot_GP(self, p=None, t_range=None, plot_title='', plot_type=None, \
			y_range=None, parent_figure=None, close_fig=True, \
			y_residual_range=None, draw_title=True, axis_location=None, \
			x_tick_step=None, plot_residual=True, samples=None, \
			samples_to_plot=None, fs=12, pad=0.5, legend_location=None, \
			dpi=300,model_colour='k',Spitzer_model_colour='r'):'''
#	'''Seperatre lightcurve plot for Gaussian process modelled sited'''
	
'''	sys.exit('Exiting: GP plots not functional')
	matplotlib.rcParams.update({'font.size': fs})		
	
	if isinstance(self.limb_constant, float):
		LD = self.limb_constant
		if self.use_spitzer:
			SLD = self.spitzer_limb_constant
	else:
		LD = self.limb_constant[self.reference_source]
		if self.use_spitzer:
			if self.spitzer_limb_constant != self.limb_constant['spitzer']:
				print( \
				'\n\n\n Replacing Spitzer limb constant (%f) with %f.\n\n\n'\
				%(self.spitzer_limb_constant, self.limb_constant['spitzer']) )
				self.spitzer_limb_constant = self.limb_constant['spitzer']
			SLD = self.spitzer_limb_constant
			
	ratio = 1.0
	
	from celerite import terms
	import celerite
		
	if (samples_to_plot is not None) and (samples is None):
		samples=self.samples
		
	if p is None:
		p = self.p.copy()
	elif (len(p)!=len(self.freeze)) and (len(p)==len(p[self.freeze==0])):
		ps = p.copy()  	# sample specified in function call. This will not
						# accept tuples
		p = self.p.copy()  	# class stored, full set of parameters (including
							# frozen ones)
		p[self.freeze==0] = ps  # replacing non-frozen parameters with those
								# from the function call

	if plot_type is None:
		plot_type = self.lightcurve_plot_type

	if t_range:
		t_min, t_max = t_range
	elif self.plot_date_range:
		t_min, t_max = self.plot_date_range
	else:
		t_min = np.zeros(len(self.gaussian_process_sites))
		t_max = np.zeros(len(self.gaussian_process_sites))
		for k, site in enumerate(self.gaussian_process_sites):
			if not site == 'spitzer':
				t_min[k] = np.min(self.data[site][0])
				t_max[k] = np.max(self.data[site][0])
			elif site == 'spitzer':
				t_min[k] = np.max(self.spitzer_data[0])
				t_max[k] = np.min(self.spitzer_data[0])
		
		t_min = np.min(t_min) - 2.5
		t_max = np.max(t_max) + 2.5
			
	logd, logq, logrho, u0, phi, t0, tE = p[:7]
	d = 10.0**logd
	q = 10.0**logq
	rho = 10.0**logrho

	chi2, a0, a1, _, _, _ = self.chi2_calc(p)
	print('p', p)
	print('chi2', chi2)
	print('a0', a0)
	print('a1',a1)
		
	if parent_figure is None:
		fig = plt.figure(figsize=(7,5),dpi=dpi)
		if axis_location is None:
			ax0 = fig.add_axes([0.12,0.25,0.85,0.6])
		else:
			ax0 = fig.add_axes(axis_location)
	else:
		#ax0 = inset_locator.inset_axes(parent_axes, width="40%", height="20%", loc=1)
		fig = parent_figure
		if axis_location is None:
			ax0 = fig.add_axes([0.15,0.6,0.4,0.3])
		else:
			ax0 = fig.add_axes(axis_location)
				
	# Residuals plot
	if plot_residual:
		if parent_figure is None:
			ax1 = fig.add_axes([0.12,0.1,0.85,0.15],sharex=ax0)
		else:
			if axis_location is None:
				ax1 = fig.add_axes([0.15,0.6-0.0643,0.4,0.0643], sharex=ax0)
			else:
				ax1 = fig.add_axes([axis_location[0], axis_location[1]-0.0643, axis_location[2], 0.0643])
	
			
	GP_dic={}	
	if self.gaussian_process_common:   #this does not belong here!!!
		if self.GP_model=='Real':		
			a = self.ln_a.copy()
			c = self.ln_c.copy()
			GP_dic[str(a)[-5:]+'_'+str(c)[-5:]] = self.CeleriteModel()		
			GP_dic[str(a)[-5:]+'_'+str(c)[-5:]].get_mc(self)
			GP_dic[str(a)[-5:]+'_'+str(c)[-5:]].p = p # I think this line might be super important
			kernel = terms.RealTerm(log_a=a, log_c=c)
		if self.GP_model=='Matern':
			sigma = self.ln_sigma.copy()
			rho = self.ln_rho.copy()
			GP_dic[str(sigma)[-5:]+'_'+str(rho)[-5:]] = self.CeleriteModel()		
			GP_dic[str(sigma)[-5:]+'_'+str(rho)[-5:]].get_mc(self)
			GP_dic[str(sigma)[-5:]+'_'+str(rho)[-5:]].p = p
			kernel = terms.Matern32Term(log_sigma=sigma, log_rho=rho)
			
		
	# Plot the model sans GP	
	n_plot_points = 1001 #np.min((1001,100*(t_max-t_min)/(tE*rho)))
	t_plot = np.linspace(t_min,t_max,n_plot_points)
		
	A = self.magnification(t_plot,p=p,LD=LD)

	if 'spitzer' in self.gaussian_process_sites:
		AS = self.spitzer_magnification(t_plot,p=p,LD=SLD)
			
	plot_ground_based_model = False
	for k, site in enumerate(self.data):
		if site in self.gaussian_process_sites:
			plot_ground_based_model = True

	if plot_type == 'magnitudes':
		if plot_ground_based_model:
			GBmodel = self.zp-2.5 * np.log10( a0[self.reference_source]*A + a1[self.reference_source])	
				
		if 'spitzer' in self.gaussian_process_sites:				
			if self.spitzer_plot_scaled:
				F = AS*a0[self.reference_source]+a1[self.reference_source]
				Smodel = self.zp-2.5*np.log10(F)
					
			else:
				F = AS*a0['spitzer']+a1['spitzer']
				Smodel = self.zp-2.5*np.log10(F)-2.5*np.log10(ratio)
		ax0.invert_yaxis()
		if parent_figure is None:
			ax0.set_ylabel(r'$I_{'+self.reference_source+r'}$',fontdict={'fontsize':fs})

	elif plot_type == 'logA':
		if plot_ground_based_model:
			GBmodel = 2.5*np.log10(A)
		if 'spitzer' in self.gaussian_process_sites:
			Smodel = 2.5*np.log10(AS)	
		if parent_figure is None:
			ax0.set_ylabel(r'$2.5\, \log_{10}\, A$',fontdict={'fontsize':fs})

	else:

		if plot_ground_based_model:
			GBmodel = a0[self.reference_source]*A+a1[self.reference_source]
		if 'spitzer' in self.gaussian_process_sites:
			if self.spitzer_plot_scaled:
				Smodel = a0[self.reference_source]*AS + a1[self.reference_source]
			else:
				Smodel = (a0['spitzer']*AS+a1['spitzer'])*ratio

		if parent_figure is None:
			plt.ylabel(r'$F_{'+self.reference_source+r'}$',fontdict={'fontsize':fs})
				
	if plot_ground_based_model:		
		ax0.plot(t_plot, GBmodel, 'k-')
	ax0.plot(t_plot, Smodel,'r-')	
		
				
	# Overlay posterior predictive samples with GP
	
	GBGPmodel= {}
	
	dim = np.count_nonzero(self.freeze==0) #len(self.freeze.tolist())
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	if samples is not None:
		
		if samples=='state':
			PS = self.state.copy()
		else:
			PS = np.zeros((samples_to_plot, dim))
			for k, ps in enumerate(samples[np.random.randint(len(samples), size=samples_to_plot)]):
				PS[k] = ps.copy()
		
	elif samples is None:
	
		PS = np.array([p])
		
	samples_to_plot = PS.shape[0]	
	
	ps =self.p.copy()
	
	for k in range(samples_to_plot):
		
		if len(ps) == len(PS[k]):
			ps = PS[k].copy()
		else:
			ps[np.where(1-self.freeze)[0]] = PS[k].copy()
			
		alpha = 5./float(samples_to_plot)
		if alpha > 1.:
			alpha=1.

		pi = self.gaussian_process_index
		for k, site in enumerate(self.data):
					
			if site in self.gaussian_process_sites:
				print('site GP params - ', site, ps[pi], ps[pi+1])
				if not self.gaussian_process_common:

					if self.GP_model=='Real':		
						kernel = terms.RealTerm(log_a=ps[pi], log_c=ps[pi+1])
					if self.GP_model=='Matern':
						kernel = terms.Matern32Term(log_sigma=ps[pi], log_rho=ps[pi+1])
					
					
				if not(site=='spitzer'): #this should always be true because self.data doesn't contain 'Spitzer')
				
					GP_dic[str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]] = self.CeleriteModel()		
					GP_dic[str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]].get_mc(self)
					GP_dic[str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]].p = ps
					GP_dic[str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]].site = site
					GP_dic['GP_'+str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]] = celerite.GP(kernel, mean=GP_dic[str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]], fit_mean=True)
					GP_dic['GP_'+str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]].compute(self.data[site][0], self.data[site][2])
					F, var = gp.predict(mc.spitzer_data[1], t_plot)
					A = (F-a1[site])/a0[site]
						
					if plot_type == 'magnitudes':
						GBGPmodel[site] = self.zp - 2.5*np.log10( a0[self.reference_source]*A + a1[self.reference_source])
					elif plot_type == 'logA':
						GBGPmodel[site] = 2.5*np.log10(A)
					else:
						GBGPmodel[site] = a0[self.reference_source]*A + a1[self.reference_source]
							
					ax0.plot(t_plot, GBGPmodel[site], 'b-', alpha=alpha, lw=1)
						
						
					# Residual GP samples
					if plot_residual:
						ax1.plot(t_plot, GBGPmodel-GBmodel, 'b-', alpha=alpha, lw=1)
						
				if not self.gaussian_process_common:
					pi += 2
						
		site = 'spitzer'			
		if 'spitzer' in self.gaussian_process_sites:
			print('site GP params - ', site, ps[pi], ps[pi+1])
			if not self.gaussian_process_common:

				if self.GP_model=='Real':		
					kernel = terms.RealTerm(log_a=ps[pi], log_c=ps[pi+1])
				if self.GP_model=='Matern':
					kernel = terms.Matern32Term(log_sigma=ps[pi], log_rho=ps[pi+1])
					
			GP_dic[str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]] = self.CeleriteModel()		
			GP_dic[str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]].get_mc(self)
			GP_dic[str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]].p = ps						
			GP_dic[str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]].site = 'spitzer'
			GP_dic['GP_'+str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]] = celerite.GP(kernel, mean=GP_dic[str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]], fit_mean=True)
			if self.scale_spitzer_error_bars_multiplicative:
				GP_dic['GP_'+str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]].compute(self.spitzer_data[0], self.spitzer_data[2]*ps[self.spitzer_error_bar_scale_index])
			else:
				GP_dic['GP_'+str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]].compute(self.spitzer_data[0],self.spitzer_data[2])
			F, var = GP_dic['GP_'+str(ps[pi])[-5:]+'_'+str(ps[pi+1])[-5:]].predict(self.spitzer_data[1], t_plot)
			AS = (F-a1['spitzer'])/a0['spitzer']
					
			if self.spitzer_plot_scaled:		
				if plot_type == 'magnitudes':
					SGPmodel = self.zp - 2.5*np.log10( a0[self.reference_source]*AS + a1[self.reference_source] )
				elif plot_type == 'logA':
					SGPmodel = 2.5*np.log10(AS)
				else:
					SGPmodel = a0[self.reference_source]*A+a1[self.reference_source]
			else:
				ratio = (a0[self.reference_source]+a1[self.reference_source])/(a0['spitzer']+a1['spitzer'])
				if plot_type == 'magnitudes':
					SGPmodel = self.zp - 2.5*np.log10( a0['spitzer']*AS + a1['spitzer'] ) - 2.5*np.log10( ratio )
				elif plot_type == 'logA': 
					SGPmodel = 2.5*np.log10(AS)
				else:
					SGPmodel = (a0['spitzer']*AS + a1['spitzer'])*ratio
			ax0.plot(t_plot,SGPmodel,'b-',alpha=alpha,lw=1)
									
			# Residual GP samples
			if plot_residual:
				ax1.plot(t_plot, SGPmodel-Smodel,'b-',alpha=alpha,lw=1)
				
			if not self.gaussian_process_common:
					pi += 2	
				
	# Plot the data
	for k, site in enumerate(self.data):
		if site in self.gaussian_process_sites:
	
			scaled_dflux = a0[self.reference_source]*((self.data[site][1] - a1[site])/a0[site]) + a1[self.reference_source]
			scaled_dflux_err = a0[self.reference_source]*((self.data[site][1] + self.data[site][2] - a1[site])/a0[site]) + \
								a1[self.reference_source] - scaled_dflux

			bad_points = ( scaled_dflux < 0 )
			
			if self.scale_error_bars_multiplicative:
				error_bar_scale = p[self.error_bar_scale_index[site]]
			else:
				error_bar_scale = 1.

			if plot_type == 'magnitudes':
				data_merge = self.zp - 2.5*np.log10(scaled_dflux)
				sigs_merge = np.abs(self.zp - 2.5*np.log10(scaled_dflux+scaled_dflux_err) - data_merge)
			elif plot_type == 'logA':
				data_merge = 2.5*np.log10((self.data[site][1] - a1[site])/a0[site])
				sigs_merge =  2.5/2.3026 * np.sqrt((self.data[site][2]*error_bar_scale)**2/(self.data[site][1] - a1[site])**2)
					#2.5 * self.data[site][2] / (2.3026*(self.data[site][1] - a1[site]))
			else:
				data_merge = scaled_dflux
				sigs_merge = scaled_dflux_err

			pts = np.where((t_min < self.data[site][0]) & (self.data[site][0] < t_max))[0]
			if pts.any():
				ax0.errorbar(self.data[site][0][pts], data_merge[pts], sigs_merge[pts], fmt='.', ms=2, mec=self.plot_colours[k], \
							c=self.plot_colours[k], label=site)
		
	if 'spitzer' in self.gaussian_process_sites:

		k += 1
		site = 'spitzer'

		ratio = 1.0
				
		if self.scale_spitzer_error_bars_multiplicative:
			spitzer_err = self.spitzer_data[2]*p[self.spitzer_error_bar_scale_index]
		elif self.scale_error_bars_multiplicative:
			spitzer_err = self.spitzer_data[2]*p[self.error_bar_scale_index[site]]
		else:
			spitzer_err = self.spitzer_data[2]

		if self.spitzer_plot_scaled:
			scaled_dflux = a0[self.reference_source]*((self.spitzer_data[1] - a1[site])/a0[site]) + a1[self.reference_source]
			scaled_dflux_err = a0[self.reference_source]*((self.spitzer_data[1] + spitzer_err - a1[site])/a0[site]) + \
									a1[self.reference_source] - scaled_dflux
		else:
			ratio = (a0[self.reference_source]+a1[self.reference_source])/(a0[site]+a1[site])
			scaled_dflux = self.spitzer_data[1]*ratio
			scaled_dflux_err = spitzer_err*ratio

		if plot_type == 'magnitudes':
			data_merge = self.zp - 2.5*np.log10(scaled_dflux)
			sigs_merge = np.abs(self.zp - 2.5*np.log10(scaled_dflux+scaled_dflux_err) - data_merge)
		elif plot_type == 'logA':
			data_merge = 2.5*np.log10((self.spitzer_data[1] - a1[site])/a0[site])
			sigs_merge = 2.5/2.3026 * np.sqrt(spitzer_err**2/(self.spitzer_data[1] - a1[site])**2)
				#2.5 * spitzer_err / (2.3026*(self.spitzer_data[1] - a1[site]))
		else:
			data_merge = scaled_dflux
			sigs_merge = scaled_dflux_err								

		pts = np.where((t_min < self.spitzer_data[0]) & (self.spitzer_data[0] < t_max))[0]
		if pts.any():
			ax0.errorbar(self.spitzer_data[0][pts], data_merge[pts], sigs_merge[pts], fmt='x', ms=5, mec=Spitzer_model_colour, \
							c=Spitzer_model_colour, label='Spitzer')


	ax0.grid()

	if parent_figure is None:
		if legend_location is None:
			ax0.legend(loc='upper right')
		else:
			ax0.legend(loc=legend_location)

	print(d, q, rho, u0, phi, t0, tE)
		
	title_string = plot_title + '\n' + r'$  s = %5.3f, \log q = %6.3f, \rho = %7.5f, u_0 = %6.3f, \alpha = %5.3f$'%(d,np.log10(q),rho,u0,phi) + \
						'\n' + r'$t_0 = %9.3f, t_E = %8.3f$'%(t0,tE)

	if self.use_limb_darkening:

		Gamma = p[self.limb_index]
		title_string += r'$, \Gamma = %5g$'%Gamma

	if self.use_parallax or self.use_spitzer:

		pi_EE = p[self.Pi_EE_index]
		pi_EN = p[self.Pi_EN_index]
		title_string += r'$, \pi_{E,E} = %5.3f, \pi_{E,N} = %5.3f$'%(pi_EE,pi_EN)

	if self.use_lens_orbital_motion:

		dddt = p[self.dddt_index]
		dphidt = p[self.dphidt_index]
		title_string += '\n' + r'$\dot{s}\,(\rm{yr}^{-1}) = %5.3f, \dot{\alpha}\,(\rm{yr}^{-1})  = %5.3f$'%(dddt,dphidt)

		if self.use_lens_orbital_acceleration:

			d2ddt2 = p[self.d2ddt2_index]
			d2phidt2 = p[self.d2phidt2_index]
			title_string += r', $\ddot{s}\,(\rm{yr}^{-2}) = %3g, \ddot{\alpha}\,(\rm{yr}^{-2})  = %3g$'%(d2ddt2,d2phidt2)
		
	if self.use_gaussian_process_model:
		
		a = p[self.gaussian_process_index]
		c = p[self.gaussian_process_index+1]
		if self.GP_model == 'Real':
			title_string += r'$,\ln{a} = %5.3f, \ln{c}  = %5.3f$'%(a,c)
		elif self.GP_model == 'Matern':
			title_string += r'$,\ln{\sigma} = %5.3f, \ln{\rho}  = %5.3f$'%(a,c)
		
	if self.scale_spitzer_error_bars_multiplicative:
	
		S = p[self.spitzer_error_bar_scale_index]
		title_string += r'$,S_{\rm Spitzer} = %5.3f$'%(S)


	if parent_figure is None and draw_title:
		ax0.set_title(title_string,fontdict={'fontsize':fs})

	if y_range is not None:
		ax0.set_ylim(y_range)

	ax0.set_xlim((t_min,t_max))
	xlim = ax0.get_xlim()

	if x_tick_step is not None:
		start, end = ax0.get_xlim()
		ax0.xaxis.set_ticks(np.arange(start, end, x_tick_step))

			
	# Residuals plot
	if plot_residual:

		chi2, a0, a1, _, _, _ = self.chi2_calc(p)

		for k, site in enumerate(self.data):
			
			if site in self.gaussian_process_sites:
		
				if isinstance(self.limb_constant, float):
		
					A = self.magnification(self.data[site][0],p=p)
					
				else:
				
					A = self.magnification(self.data[site][0],p=p,LD=self.limb_constant[site])

				scaled_dflux = a0[self.reference_source]*((self.data[site][1] - a1[site])/a0[site]) + a1[self.reference_source]
				scaled_dflux_err = a0[self.reference_source]*((self.data[site][1] + self.data[site][2] - a1[site])/a0[site]) + \
									a1[self.reference_source] - scaled_dflux
				bad_points = ( scaled_dflux < 0 )

				if plot_type == 'magnitudes':
					scaled_model = self.zp -2.5*np.log10(a0[self.reference_source]*A + a1[self.reference_source]) 
					data_merge = self.zp - 2.5*np.log10(scaled_dflux) 
					sigs_merge = np.abs(self.zp - 2.5*np.log10(scaled_dflux+scaled_dflux_err) - data_merge)
				elif plot_type == 'logA':
					scaled_model = 2.5*np.log10(A)
					data_merge = 2.5*np.log10((self.data[site][1] - a1[site])/a0[site])
					sigs_merge = 2.5/2.3026 * np.sqrt(self.data[site][2]**2/(self.data[site][1] - a1[site])**2)
				else:
					scaled_model = a0[self.reference_source]*A + a1[self.reference_source]
					data_merge = scaled_dflux
					sigs_merge = scaled_dflux_err

				data_merge -= scaled_model

				pts = np.where((t_min < self.data[site][0]) & (self.data[site][0] < t_max))[0]
				if pts.any():
					ax1.errorbar(self.data[site][0][pts], data_merge[pts], sigs_merge[pts], fmt='.', ms=2, mec=self.plot_colours[k], \
								c=self.plot_colours[k], label=site, alpha=0.5)

		if 'spitzer' in self.gaussian_process_sites:

			site = 'spitzer'
			AS = self.spitzer_magnification(self.spitzer_data[0],p=p)

			ratio = 1.0

			if self.spitzer_plot_scaled:
				scaled_dflux = a0[self.reference_source]*((self.spitzer_data[1] - a1[site])/a0[site]) + a1[self.reference_source]
				scaled_dflux_err = a0[self.reference_source]*((self.spitzer_data[1] + spitzer_err - a1[site])/a0[site]) + \
										a1[self.reference_source] - scaled_dflux
			else:
				ratio = (a0[self.reference_source]+a1[self.reference_source])/(a0[site]+a1[site])
				scaled_dflux = self.spitzer_data[1]*ratio
				scaled_dflux_err = spitzer_err*ratio

			if plot_type == 'magnitudes':

				if self.spitzer_plot_scaled:
					scaled_model = self.zp - 2.5*np.log10(a0[self.reference_source]*AS + a1[self.reference_source])
				else:
					scaled_model = self.zp - 2.5*np.log10(a0[site]*AS + a1[site]) -2.5*np.log10(ratio)
				data_merge = self.zp - 2.5*np.log10(scaled_dflux)
				sigs_merge = np.abs(self.zp - 2.5*np.log10(scaled_dflux+scaled_dflux_err) - data_merge)

			elif plot_type == 'logA':

				scaled_model = 2.5*np.log10(AS)
				data_merge = 2.5*np.log10((self.spitzer_data[1] - a1[site])/a0[site])
				sigs_merge = 2.5/2.3026 * np.sqrt(spitzer_err**2/(self.spitzer_data[1] - a1[site])**2)
				
			else:

				if self.spitzer_plot_scaled:
					scaled_model = (a0[self.reference_source]*AS + a1[self.reference_source])
				else:
					scaled_model = (a0[site]*AS + a1[site])*ratio
				data_merge = scaled_dflux
				sigs_merge = scaled_dflux_err

			data_merge -= scaled_model

			pts = np.where((t_min < self.spitzer_data[0]) & (self.spitzer_data[0] < t_max))[0]
			if pts.any():
				ax1.errorbar(self.spitzer_data[0][pts], data_merge[pts], sigs_merge[pts], fmt='x', ms=5, mec=Spitzer_model_colour, \
								c=Spitzer_model_colour, label='Spitzer')

		ax1.set_xlim(xlim)
		ax1.grid()

		if self.plot_date_range:
			ax1.set_xlim(self.plot_date_range)

		if y_range is not None:
			ymean = np.mean(y_range)
			y_range = (y_range[0]-ymean,y_range[1]-ymean)
			ax1.set_ylim(y_range)

		if y_residual_range is not None:
			ax1.set_ylim(y_residual_range)

		if plot_type == 'magnitudes':
			ax1.invert_yaxis()

		if parent_figure is None:
			plt.xlabel('HJD-2450000',fontdict={'fontsize':fs})
			if plot_type == 'magnitudes':
				ax1.set_ylabel(r'$\Delta I_{'+self.reference_source+r'}$',fontdict={'fontsize':fs})
			elif plot_type == 'logA':
				ax1.set_ylabel(r'$Residual$',fontdict={'fontsize':fs})
			else:
				ax1.set_ylabel(r'$\Delta F_{'+self.reference_source+r'}$',fontdict={'fontsize':fs})

		if x_tick_step is not None:
			start, end = ax1.get_xlim()
			ax1.xaxis.set_ticks(np.arange(start, end, x_tick_step))
			
	if plot_residual:
		ax0.tick_params(labelbottom=False,labeltop=False) 

	if close_fig:
		plt.savefig(self.plotprefix+'-GPlightcurve.png',pad_inches=pad,bbox_inches='tight')
		plt.close()

	return fig '''
	
'''def plot_spitzer(self,p=None,t_range=None,plot_title='',plot_type=None,y_range=None,parent_figure=None,close_fig=True,y_residual_range=None,
	draw_title=True,axis_location=None,x_tick_step=None,plot_residual=True,y_tick_step_residual=None,samples=None,samples_to_plot=50,draw_grid=True,
	model_colour='k',Spitzer_model_colour='r', fs=12, pad=0.5, legend_location=None):'''

'''	# Amber added fs to args
	
	matplotlib.rcParams.update({'font.size': fs})
	
	if not isinstance(self.limb_constant, float):
		if self.spitzer_limb_constant != self.limb_constant['spitzer']:
			print('\n\n\n Replacing Spitzer limb constant (%f) with %f.\n\n\n' %(self.spitzer_limb_constant,self.limb_constant['spitzer']))
			self.spitzer_limb_constant = self.limb_constant['spitzer']
		SLD = self.spitzer_limb_constant

	if p is None: #default
		p = self.p.copy()
	elif (len(p)!=len(self.freeze)) and (len(p)==len(p[self.freeze==0])):
		ps = p.copy() #sample specified in function call. This will not accept tuples
		p = self.p.copy() #class stored, full set of parameters (including frozen ones)
		p[self.freeze==0] = ps #replacing non-frozen parameters with those from the function call
	

	if plot_type is None:
		plot_type = self.lightcurve_plot_type

	if t_range:
		t_min, t_max = t_range
	elif self.plot_date_range:
		t_min, t_max = self.plot_date_range
	else:
		t_min = np.min(self.ts) - 2.5
		t_max = np.max(self.ts) + 2.5


	logd, logq, logrho, u0, phi, t0, tE = p[:7]
	d = 10.0**logd
	q = 10.0**logq
	rho = 10.0**logrho

	chi2, a0, a1, _, _, _ = self.chi2_calc(p)
	print('p', p)
	print('chi2', chi2)
	print('a0', a0)
	print('a1',a1)

	if parent_figure is None:
		fig = plt.figure(figsize=(7,5.3))
		if axis_location is None:
			ax0 = fig.add_axes([0.12,0.25,0.85,0.6])
		else:
			ax0 = fig.add_axes(axis_location)
	else:
		#ax0 = inset_locator.inset_axes(parent_axes, width="40%", height="20%", loc=1)
		fig = parent_figure
		if axis_location is None:
			ax0 = fig.add_axes([0.15,0.6,0.4,0.3])
		else:
			ax0 = fig.add_axes(axis_location)

	ratio = 1.0

	# Plot the model

	n_plot_points = np.min((1001,100*(t_max-t_min)//(np.abs(tE)*rho)))
	n_plot_points = int(n_plot_points)
	t_plot = np.linspace(t_min,t_max,n_plot_points)

	if self.use_spitzer:
		AS = self.spitzer_magnification(t_plot,p=p)

	if plot_type == 'magnitudes':

		if self.use_spitzer and not('spitzer' in self.gaussian_process_sites):
			if self.spitzer_plot_scaled:
				ax0.plot(t_plot,self.zp-2.5*np.log10(a0[self.reference_source]*AS+a1[self.reference_source]),'-',color=Spitzer_model_colour)
			else:
				ax0.plot(t_plot,self.zp-2.5*np.log10(a0['spitzer']*AS+a1['spitzer'])-2.5*np.log10(ratio),'-',color=Spitzer_model_colour)

		ax0.invert_yaxis()

		if parent_figure is None:
			ax0.set_ylabel(r'$I_{'+self.reference_source+r'}$',fontdict={'fontsize':14})

	elif plot_type == 'logA':

		if self.use_spitzer and not('spitzer' in self.gaussian_process_sites):
			ax0.plot(t_plot,2.5*np.log10(AS),'-',color=Spitzer_model_colour)

		if parent_figure is None:
			ax0.set_ylabel(r'$2.5\, \log_{10}\, A$',fontdict={'fontsize':14})


	else:

		if self.use_spitzer and not('spitzer' in self.gaussian_process_sites):
			if self.spitzer_plot_scaled:
				ax0.plot(t_plot,a0[self.reference_source]*AS+a1[self.reference_source],'-',Spitzer_color=model_colour)
			else:
				ax0.plot(t_plot,(a0['spitzer']*AS+a1['spitzer'])*ratio,'-',color=Spitzer_model_colour)

		if parent_figure is None:
			ax0.set_ylabel(r'$F_{'+self.reference_source+r'}$',fontdict={'fontsize':14})


	# Overlay posterior predictive samples

	if samples is not None:

		ps = self.p.copy()

		for psample in samples[np.random.randint(len(samples), size=samples_to_plot)]:

			ps[np.where(1-self.freeze)[0]] = psample.copy()

			if self.use_spitzer:
				AS = self.spitzer_magnification(t_plot,p=ps)

			if plot_type == 'magnitudes':

				chi2, a0, a1, _, _, _ = self.chi2_calc(ps)

				if self.use_spitzer and not('spitzer' in self.gaussian_process_sites):
					if self.spitzer_plot_scaled:
						ax0.plot(t_plot,self.zp-2.5*np.log10(a0[self.reference_source]*AS+a1[self.reference_source]),'r-',alpha=0.02)
					else:
						ax0.plot(t_plot,self.zp-2.5*np.log10(a0['spitzer']*AS+a1['spitzer'])-2.5*np.log10(ratio),'r-',alpha=0.02)

			elif plot_type == 'logA':

				if self.use_spitzer and not('spitzer' in self.gaussian_process_sites):
					ax0.plot(t_plot,2.5*np.log10(AS),'r-',alpha=0.02)

			else:

				chi2, a0, a1, _, _, _ = self.chi2_calc(ps)

				if self.use_spitzer and not('spitzer' in self.gaussian_process_sites):
					if self.spitzer_plot_scaled:
						ax0.plot(t_plot,a0[self.reference_source]*AS+a1[self.reference_source],'r-',alpha=0.02)
					else:
						ax0.plot(t_plot,(a0['spitzer']*AS+a1['spitzer'])*ratio,'r-',alpha=0.02)


	# Plot the data

	error_bar_scale = 1.0
	
	if self.use_spitzer and not('spitzer' in self.gaussian_process_sites):

		k = 0
		site = 'spitzer'

		ratio = 1.0
		
		if self.scale_spitzer_error_bars_multiplicative:
			error_bar_scale = p[self.spitzer_error_bar_scale_index]
		elif self.scale_error_bars_multiplicative:
			error_bar_scale = p[self.error_bar_scale_index[site]]
		else:
			error_bar_scale = 1.0

		if self.spitzer_plot_scaled:
			scaled_dflux = a0[self.reference_source]*((self.spitzer_data[1] - a1[site])/a0[site]) + a1[self.reference_source]
			scaled_dflux_err = a0[self.reference_source]*((self.spitzer_data[1] + self.spitzer_data[2]*error_bar_scale - a1[site])/a0[site]) + \
								a1[self.reference_source] - scaled_dflux
		else:
			ratio = (a0[self.reference_source]+a1[self.reference_source])/(a0[site]+a1[site])
			scaled_dflux = self.spitzer_data[1]*ratio
			scaled_dflux_err = self.spitzer_data[2]*error_bar_scale*ratio

		if plot_type == 'magnitudes':
			data_merge = self.zp - 2.5*np.log10(scaled_dflux)
			sigs_merge = np.abs(self.zp - 2.5*np.log10(scaled_dflux+scaled_dflux_err) - data_merge)
		elif plot_type == 'logA':
			data_merge = 2.5*np.log10((self.spitzer_data[1] - a1[site])/a0[site])
			sigs_merge = 2.5/2.3026 * np.sqrt((self.spitzer_data[2]*error_bar_scale)**2/(self.spitzer_data[1] - a1[site])**2)
				#2.5 * self.spitzer_data[2]*error_bar_scale / (2.3026*(self.spitzer_data[1] - a1[site]))
		else:
			data_merge = scaled_dflux
			sigs_merge = scaled_dflux_err

		pts = np.where((t_min < self.spitzer_data[0]) & (self.spitzer_data[0] < t_max))[0]
		if pts.any():
			ax0.errorbar(self.spitzer_data[0][pts], data_merge[pts], sigs_merge[pts], fmt='.', ms=8, mec=Spitzer_model_colour, \
						c=Spitzer_model_colour, label='Spitzer')


	if draw_grid:
		ax0.grid()

	if parent_figure is None:
		if legend_location is None:
			ax0.legend(loc='upper right')
		else:
			ax0.legend(loc=legend_location)

	print(d, q, rho, u0, phi, t0, tE) 

	title_string = plot_title + '\n' + r'$  s = %5.3f, \log q = %6.3f, \rho = %7.5f, u_0 = %6.3f, \alpha = %5.3f$'%(d,np.log10(q),rho,u0,phi) + \
					'\n' + r'$t_0 = %9.3f, t_E = %8.3f$'%(t0,tE)

	if self.use_limb_darkening:

		Gamma = p[self.limb_index]
		title_string += r'$, \Gamma = %5g$'%Gamma

	if self.use_parallax or self.use_spitzer:

		pi_EE = p[self.Pi_EE_index]
		pi_EN = p[self.Pi_EN_index]
		title_string += r'$, \pi_{E,E} = %5.3f, \pi_{E,N} = %5.3f$'%(pi_EE,pi_EN)

	if self.use_lens_orbital_motion:

		dddt = p[self.dddt_index]
		dphidt = p[self.dphidt_index]
		title_string += '\n' + r'$\dot{s}\,(\rm{yr}^{-1}) = %5.3f, \dot{\alpha}\,(\rm{yr}^{-1})  = %5.3f$'%(dddt,dphidt)

		if self.use_lens_orbital_acceleration:

			d2ddt2 = p[self.d2ddt2_index]
			d2phidt2 = p[self.d2phidt2_index]
			title_string += r', $\ddot{s}\,(\rm{yr}^{-2}) = %3g, \ddot{\alpha}\,(\rm{yr}^{-2})  = %3g$'%(d2ddt2,d2phidt2)
			
	if self.use_gaussian_process_model:
	
		a = p[self.gaussian_process_index]
		c = p[self.gaussian_process_index+1]
		if self.GP_model == 'Real':
			title_string += r'$,\ln{a} = %5.3f, \ln{c}  = %5.3f$'%(a,c)
		elif self.GP_model == 'Matern':
			title_string += r'$,\ln{\sigma} = %5.3f, \ln{\rho}  = %5.3f$'%(a,c)

	if self.scale_spitzer_error_bars_multiplicative:
	
		S = p[self.spitzer_error_bar_scale_index]
		title_string += r'$,S_{\rm Spitzer} = %5.3f$'%(S)

	if parent_figure is None and draw_title:
		ax0.set_title(title_string,fontdict={'fontsize':fs})

	if y_range is not None:
		ax0.set_ylim(y_range)

	if plot_residual:
		ax0.tick_params(labelbottom='off') 
		plt.setp(ax0.get_xticklabels(), visible=False)

	ax0.set_xlim((t_min,t_max))
	xlim = ax0.get_xlim()

	if x_tick_step is not None:
		start, end = ax0.get_xlim()
		ax0.xaxis.set_ticks(np.arange(start, end, x_tick_step))

	# Residuals plot

	#ax1 = plt.subplot(gs[1],sharex=ax0)

	if plot_residual:

		chi2, a0, a1, _, _, _ = self.chi2_calc(p)

		if parent_figure is None:
			ax1 = fig.add_axes([0.12,0.1,0.85,0.15],sharex=ax0)
		else:
			if axis_location is None:
				ax1 = fig.add_axes([0.15,0.6-0.0643,0.4,0.0643],sharex=ax0)
			else:
				ax1 = fig.add_axes([axis_location[0],axis_location[1]-0.0643,axis_location[2],0.0643])

		#divider = make_axes_locatable(ax0)
		#ax1 = divider.append_axes("bottom", size="20%", pad=0.0, sharex=ax0)

		if self.use_spitzer and not('spitzer' in self.gaussian_process_sites):

			k += 1
			site = 'spitzer'
			AS = self.spitzer_magnification(self.spitzer_data[0],p=p)

			ratio = 1.0

			if self.scale_error_bars_multiplicative:
				error_bar_scale = p[self.error_bar_scale_index[site]]
			elif self.scale_spitzer_error_bars_multiplicative:
				error_bar_scale = p[self.spitzer_error_bar_scale_index]

			if self.spitzer_plot_scaled:
				scaled_dflux = a0[self.reference_source]*((self.spitzer_data[1] - a1[site])/a0[site]) + a1[self.reference_source]
				scaled_dflux_err = a0[self.reference_source]*((self.spitzer_data[1] + self.spitzer_data[2]*error_bar_scale - a1[site])/a0[site]) + \
									a1[self.reference_source] - scaled_dflux
			else:
				ratio = (a0[self.reference_source]+a1[self.reference_source])/(a0[site]+a1[site])
				scaled_dflux = self.spitzer_data[1]*ratio
				scaled_dflux_err = self.spitzer_data[2]*ratio*error_bar_scale

			if plot_type == 'magnitudes':

				if self.spitzer_plot_scaled:
					scaled_model = self.zp - 2.5*np.log10(a0[self.reference_source]*AS + a1[self.reference_source])
				else:
					scaled_model = self.zp - 2.5*np.log10(a0[site]*AS + a1[site]) -2.5*np.log10(ratio)
				data_merge = self.zp - 2.5*np.log10(scaled_dflux)
				sigs_merge = np.abs(self.zp - 2.5*np.log10(scaled_dflux+scaled_dflux_err) - data_merge)

			elif plot_type == 'logA':

				scaled_model = 2.5*np.log10(AS)
				data_merge = 2.5*np.log10((self.spitzer_data[1] - a1[site])/a0[site])
				sigs_merge = 2.5 * self.spitzer_data[2]*error_bar_scale / (2.3026*(self.spitzer_data[1] - a1[site]))

			else:

				if self.spitzer_plot_scaled:
					scaled_model = (a0[self.reference_source]*AS + a1[self.reference_source])
				else:
					scaled_model = (a0[site]*AS + a1[site])*ratio
				data_merge = scaled_dflux
				sigs_merge = scaled_dflux_err

			data_merge -= scaled_model

			pts = np.where((t_min < self.spitzer_data[0]) & (self.spitzer_data[0] < t_max))[0]
			if pts.any():
				ax1.errorbar(self.spitzer_data[0][pts], data_merge[pts], sigs_merge[pts], fmt='.', ms=2, mec=self.plot_colours[k], \
							c=self.plot_colours[k], label='Spitzer')

		ax1.set_xlim(xlim)
		if draw_grid:
			ax1.grid()

		if self.plot_date_range:
			ax1.set_xlim(self.plot_date_range)

		if y_residual_range is not None:
			ax1.set_ylim(y_residual_range)

		if plot_type == 'magnitudes':
			ax1.invert_yaxis()

		ax1.yaxis.set_major_locator(MaxNLocator(2,prune='upper'))

		if parent_figure is None:
			plt.xlabel('HJD-2450000',fontdict={'fontsize':fs})
			if plot_type == 'magnitudes':
				ax1.set_ylabel(r'$\Delta I_{'+self.reference_source+r'}$',fontdict={'fontsize':fs})
			elif plot_type == 'logA':
				ax1.set_ylabel(r'$Residual$',fontdict={'fontsize':fs})
			else:
				ax1.set_ylabel(r'$\Delta F_{'+self.reference_source+r'}$',fontdict={'fontsize':fs})

		if x_tick_step is not None:
			start, end = ax1.get_xlim()
			ax1.xaxis.set_ticks(np.arange(start, end, x_tick_step))


		if y_tick_step_residual is not None:
			start, end = ax1.get_ylim()
			ax1.yaxis.set_ticks(np.arange(start, end, y_tick_step_residual))

	if close_fig:
		plt.savefig(self.plotprefix+'-lightcurve.png',pad_inches=pad)
		plt.close()

	return fig'''

