import numpy as np
import emcee

try:
	import zeus
except ModuleNotFoundError:
	print('Warning: zeus MCMC module not available')


def emcee_has_converged(self,all_samples,lnprobability,auto_corr_time=None,n_steps=100):

	# Emcee convergence testing is not easy. The method we will adopt
	# is to test whether the parameter means and standard deviations, and ln_p, have 
	# stabilised, comparing the last n steps, with the previous n steps.

	#std_threshold = 0.01
	#mean_threshold = 0.01

	n_test = all_samples.shape[0]*n_steps
	print('n_test', n_test)

	lnp = lnprobability.T.ravel()
	if len(lnp) < 2*n_test:
		return False

	converged = True

	steps = all_samples.shape[1]

	with open(self.plotprefix+'_lnp','a') as fid:

		fid.write("After %d steps, parameter means, standard deviations, convergence metrics and ln_P:\n"%steps)

		for k in range(all_samples.shape[2]):

			samples = all_samples[:,:,k].T.ravel()
			mean2 = np.mean(samples[-2*n_test:-n_test])
			mean1 = np.mean(samples[-n_test:])
			std1 = np.std(samples[-n_test:])
			std2 = np.std(samples[-2*n_test:-n_test])

			delta_param = np.abs(mean1 - mean2)/std1
			delta_std = np.abs(std2-std1)/std1

			fid.write("%g %g %g %g\n"%(mean1,std1,delta_param,delta_std))

			if  delta_param > self.emcee_mean_convergence_threshold:
				converged = False
			if delta_std > self.emcee_mean_convergence_threshold:
				converged = False

		max_lnp0 = np.max(lnp[-n_test:])
		max_lnp1 = np.max(lnp[-2*n_test:-n_test])
		lnp_delta = max_lnp0 - max_lnp1

		if lnp_delta > self.emcee_lnp_convergence_threshold:
			converged = False

		fid.write("lnp: %g %g %d\n"%(max_lnp0,lnp_delta,converged))

		if auto_corr_time is not None:
			try:
				tau = auto_corr_time
				delta_tau = np.abs(self.emcee_old_tau - tau) / tau
				ac_converged = np.all(tau * 100 < steps)
				ac_converged &= np.all(delta_tau < 0.01)
				self.emcee_old_tau = tau*1.0
				print('tau')
				print(tau)
				print('relative change in tau')
				print(delta_tau)
				print('ac_converged')
				print(ac_converged)
				fid.write("autocorr time:\n")
				for i in range(tau.shape[0]):
					fid.write("%f "%(tau[i]))
				fid.write("\n%d\n\n"%(ac_converged))
			except emcee.autocorr.AutocorrError:
				fid.write("autocorr time not computed\n\n")
				pass


	return converged


def advance_mcmc(self,sampler,state,steps):

	if not self.sampler_package in self.known_sampler_packages:
		raise ValueError(str('sampler_package must be one of',self.known_sampler_packages)) 

	if self.sampler_package == 'emcee':

		state, lnp , _ = sampler.run_mcmc(state, steps, skip_initial_state_check=True)
		chain = sampler.chain
		flatchain = sampler.flatchain
		lnprobability = sampler.lnprobability
		flatlnprobability = sampler.flatlnprobability
		tau = sampler.get_autocorr_time(tol=0)

	else:

		sampler.run_mcmc(self.state, steps)
		state = sampler.get_last_sample()
		lnp = sampler.get_last_log_prob()
		chain = sampler.chain
		chain = np.swapaxes(chain,0,1)
		flatchain = sampler.get_chain(flat=True)
		lnprobability = sampler.get_log_prob().T
		flatlnprobability = sampler.get_log_prob(flat=True)
		tau = sampler.act

	return state, lnp, chain, flatchain, lnprobability, flatlnprobability, tau


def converge(self, converge=True, covariance=None, state=None, accel=2.0, optimize=True):

	if not self.sampler_package in self.known_sampler_packages:
		raise ValueError(str('sampler_package must be one of',self.known_sampler_packages)) 

	if state is None:
		self.state = None
	elif state is not None:
		self.state = state.copy()
	self.emcee_mean_convergence_threshold = 1.0/np.sqrt(self.emcee_burnin_walkers)

	dims = self.dims - np.sum(self.freeze)

	q = self.p[np.where(1-self.freeze)[0]].copy()
	q_sig = self.p_sig[np.where(1-self.freeze)[0]].copy()
	print(q_sig)
	print(1 - self.freeze)
	print(self.p[np.where(1-self.freeze)[0]])
	print(q)

	if converge:

		# Set up the backend
		# Don't forget to clear it in case the file already exists
		if self.sampler_package == 'emcee':
			filename = self.plotprefix+'.h5'
			backend = emcee.backends.HDFBackend(filename)
			backend.reset(self.emcee_burnin_walkers, dims)


		print("Running burn-in ...")

		if self.sampler_package == 'emcee':
			sampler = emcee.EnsembleSampler(self.emcee_burnin_walkers, dims, self.lnprob,live_dangerously=True,a=accel,backend=backend)
		else:
			sampler =zeus.EnsembleSampler(self.emcee_burnin_walkers, dims, self.lnprob)


		if self.state is None:

			print(dims, q, self.p_sig, self.emcee_burnin_walkers)
			print(np.where(1-self.freeze)[0], type(np.where(1-self.freeze)[0]))
			#print(self.p_sig[np.where(1-self.freeze)][0])
			self.state = [q + q_sig * np.random.randn(dims) \
							for i in range(self.emcee_burnin_walkers)]
			# print "Initial walker state:"
			# for i in xrange(self.emcee_burnin_walkers):
			# 	w = self.state[i]
			# 	c2, _, _, _, _, _ = self.chi2_calc(p_in=w)
			# 	print w
			# 	print c2
			# 	print self.lnprob(w)


		if covariance is not None:

			S_var, T = np.linalg.eig(covariance)
			S_sig = np.sqrt(S_var)
			self.state = [q + np.dot(T,S_sig*np.random.randn(dims)) \
							for i in range(self.emcee_burnin_walkers)]


		if optimize:

			# Run a cold MCMC

			converged = False
			steps = 0
			self.data_variance_scale = 0.1

			print('Optimizing with scaled variances.')

			while not (converged and steps > self.emcee_min_burnin_steps) and steps < self.emcee_max_optimize_steps:

				print('converged, steps, emcee_max_optimize_steps', converged, steps, self.emcee_max_optimize_steps, not converged and steps < self.emcee_max_burnin_steps)

				self.state, lnp, chain, flatchain, lnprobability, flatlnprobability, tau = self.advance_mcmc(sampler,self.state,self.emcee_burnin_steps)

				steps = chain.shape[1]

				self.plot_chain(chain,ln_prob=lnprobability,suffix='-burnin1.png')
				np.save(self.plotprefix+'-state-burnin1',np.asarray(self.state))
				np.save(self.plotprefix+'-chain-burnin1',np.asarray(chain))
				np.save(self.plotprefix+'-lnp-burnin1',np.asarray(lnprobability))

				#self.p = np.percentile(sampler.flatchain[-self.emcee_burnin_steps*self.emcee_walkers:],50,axis=0)
				kmax = np.argmax(flatlnprobability)
				self.p[np.where(1-self.freeze)[0]] = flatchain[kmax,:].copy()
				print('self.p:', self.p)

				self.plot_caustic_and_trajectory()
				self.plot_lightcurve(samples=flatchain[-self.emcee_burnin_steps*self.emcee_burnin_walkers:,:])
				np.save(self.plotprefix+'-min_chi2-burnin1',np.asarray(self.p))
				if self.use_gaussian_process_model:
					self.plot_GP(samples='state')

				converged = self.emcee_has_converged(chain,lnprobability,n_steps=self.emcee_burnin_steps,auto_corr_time=tau)

				# Replace the worst walkers with random draws from the good-walker distribution
				ind = np.argsort(lnprobability[:,-1])
				#n_replace = int(self.emcee_burnin_walkers * self.emcee_burnin_discard_percent/100.0)
				n_replace = self.emcee_burnin_walkers//2
				if n_replace > 0:
					bad = ind[:n_replace]
					good = ind[n_replace:]
					latest_flatchain = chain[good,-self.emcee_burnin_steps:,:].reshape((self.emcee_burnin_steps*len(good),dims))
					#walker_mean = np.mean(latest_flatchain,axis=0)
					covariance = np.cov(latest_flatchain,rowvar=False)
					S_var, T = np.linalg.eig(covariance)
					S_sig = np.sqrt(S_var)
					#self.state[bad] = self.state[good].copy()
					self.state[bad] = [self.state[good][i] + 0.1*np.dot(T,S_sig*np.random.randn(dims)) \
										for i in range(n_replace)]
					#self.state[bad] = [walker_mean + np.dot(T,S_sig*np.random.randn(dims)) \
					#					for i in range(n_replace)]


			if steps >= self.emcee_max_optimize_steps:
				print('Maximum number of steps reached. Terminating optimize burn-in.')


			max_state = flatchain[np.argmax(flatlnprobability)]

			self.state = [max_state + 0.01*(self.state[i] - max_state) for i in range(self.emcee_burnin_walkers)]

			index = np.argmax(flatlnprobability)
			q = flatchain[index].tolist()
			self.p[np.where(1-self.freeze)[0]] = q.copy()
			print('lowest chi2 of',-2.0*flatlnprobability[index]*self.data_variance_scale,'at', self.p)

			self.data_variance_scale = 1.0
			sampler.reset()


		print('Converging with standard variances.')

		converged = False
		steps = 0

		while not (converged and steps > self.emcee_min_burnin_steps) and steps < self.emcee_max_burnin_steps:

			print('converged, steps, emcee_max_burnin_steps', converged, steps, self.emcee_max_burnin_steps, not converged and steps < self.emcee_max_burnin_steps)

			self.state, lnp, chain, flatchain, lnprobability, flatlnprobability, tau = self.advance_mcmc(sampler,self.state,self.emcee_burnin_steps)

			steps = chain.shape[1]

			self.plot_chain(chain,ln_prob=lnprobability,suffix='-burnin2.png')
			np.save(self.plotprefix+'-state-burnin2',np.asarray(self.state))
			np.save(self.plotprefix+'-chain-burnin2',np.asarray(chain))
			np.save(self.plotprefix+'-lnp-burnin2',np.asarray(lnprobability))

			#self.p = np.percentile(sampler.flatchain[-self.emcee_burnin_steps*self.emcee_walkers:],50,axis=0)
			kmax = np.argmax(flatlnprobability)
			self.p[np.where(1-self.freeze)[0]] = flatchain[kmax,:].copy()
			print('self.p:', self.p)

			self.plot_caustic_and_trajectory()
			self.plot_lightcurve(samples=flatchain[-self.emcee_burnin_steps*self.emcee_burnin_walkers:,:])
			np.save(self.plotprefix+'-min_chi2-burnin2',np.asarray(self.p))

			converged = self.emcee_has_converged(chain,lnprobability,n_steps=self.emcee_burnin_steps,auto_corr_time=tau)

		index = np.argmax(flatlnprobability)
		q = flatchain[index].tolist()
		self.p[np.where(1-self.freeze)[0]] = q.copy()
		print('lowest chi2 of',-2.0*flatlnprobability[index],'\n at', self.p)



		if self.emcee_walkers > self.emcee_burnin_walkers:


			print('Adding',self.emcee_walkers - self.emcee_burnin_walkers, 'new walkers.')
			for i in range(self.emcee_walkers - self.emcee_burnin_walkers):
				self.state = np.vstack((self.state,q + np.dot(T,S_sig*np.random.randn(dims))))


			if self.sampler_package == 'emcee':
				backend.reset(self.emcee_walkers, dims)


			print('Running relax.')


			self.data_variance_scale = 1.0

			if self.sampler_package == 'emcee':
				sampler = emcee.EnsembleSampler(self.emcee_walkers, dims, self.lnprob,live_dangerously=True,a=accel,backend=backend)
			else:
				sampler =zeus.EnsembleSampler(self.emcee_walkers, dims, self.lnprob)

			self.state, lnp, chain, flatchain, lnprobability, flatlnprobability, tau = self.advance_mcmc(sampler,self.state,self.emcee_relax_steps)

			index = np.argmax(flatlnprobability)
			q = flatchain[index].tolist()
			self.p[np.where(1-self.freeze)[0]] = q.copy()
			print('lowest chi2 of',-2.0*flatlnprobability[index],'at', self.p)

			self.plot_chain(chain,ln_prob=lnprobability,suffix='-relax.png')

		print("Running production...")

		sampler.reset()

		self.state, lnp, chain, flatchain, lnprobability, flatlnprobability, tau = self.advance_mcmc(sampler,self.state,self.emcee_production_steps)

		index = np.argmax(flatlnprobability)
		q = flatchain[index].tolist()
		self.p[np.where(1-self.freeze)[0]] = q.copy()
		print('lowest chi2 of',-2.0*flatlnprobability[index],'at', self.p)

		self.samples = flatchain.copy()
		self.samples_lnp = flatlnprobability.copy()

		params = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(self.samples, \
							[16, 50, 84], axis=0))]

		print('parameter means and uncertainties:')
		print(params)

		self.plot_chain(chain,ln_prob=lnprobability)
		self.plot_chain_corner()

		np.save(self.plotprefix+'-samples-production',np.asarray(self.samples))
		np.save(self.plotprefix+'-lnp-production',np.asarray(self.samples_lnp))
		np.save(self.plotprefix+'-state-production',np.asarray(self.state))
		np.save(self.plotprefix+'-min_chi2-production',np.asarray(flatchain[np.argmax(flatlnprobability)]))

		self.solution = flatchain[np.argmax(flatlnprobability)].copy()

	return params
