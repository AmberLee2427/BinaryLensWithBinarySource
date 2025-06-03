
import numpy as np
import celerite
from celerite import terms
from celerite.modeling import Model
import sys

def add_gaussian_process_model(self,common=True, sites=None, model=None, default_params=None, sig0=None,chi2_method='celerite'):

    self.use_gaussian_process_model = True
    self.gaussian_process_index = self.dims
    self.gaussian_process_common = common
    self.chi2_method = chi2_method
		
    if model is not None:
        self.GP_model = model
    if default_params is not None:
        self.GP_default_params = default_params
    if sig0 is not None:
        self.GP_sig0 = sig0
    if isinstance(sites, str):
        self.gaussian_process_sites = [sites]
    elif sites is not None:
        self.gaussian_process_sites = sites	
		
	# this option is untested
    if common: # same GP parameters for all sites
			
        self.p = np.hstack((self.p, self.GP_default_params))
        self.p_sig = np.hstack((self.p_sig, self.GP_sig0))
        self.freeze = np.hstack((self.freeze,np.zeros(2,dtype=int)))
        self.dims += len(self.GP_default_params)
			
        if self.GP_model=='Real':
            a, c = self.GP_default_params
            self.parameter_labels.append('ln_a')
            self.parameter_labels.append('ln_c')
            self.ln_a = a
            self.ln_c = c
				
        elif self.GP_model=='Matern':
            sigma, rho = self.GP_default_params
            self.parameter_labels.append(r'$ln_\sigma$')
            self.parameter_labels.append(r'$ln_\rho$')
            self.ln_sigma = sigma
            self.ln_rho = rho
				
        else:
            sys.exit('Gaussian Process model %s not recognised' %self.GP_model)

    else: # seperate model parameters for each site

        # this has only been tested for site='spitzer'
        for site in self.gaussian_process_sites:

            self.p = np.hstack((self.p, self.GP_default_params))
            self.p_sig = np.hstack((self.p_sig, self.GP_sig0))
            self.freeze = np.hstack((self.freeze,np.zeros(2,dtype=int)))
            self.dims += len(self.GP_default_params)
				
            if self.GP_model=='Real':
                a, c = self.GP_default_params
                self.parameter_labels.append(site+'_ln_a')
                self.parameter_labels.append(site+'_ln_c')
                self.ln_a[site] = a
                self.ln_c[site] = c
					
            if self.GP_model=='Matern':
                sigma, rho = self.GP_default_params
                self.parameter_labels.append(site+r'_$ln_\sigma$')
                self.parameter_labels.append(site+r'_$ln_\rho$')
                self.ln_sigma[site] = sigma
                self.ln_rho[site] = rho
	
    if self.chi2_method != 'manual' and self.chi2_method != 'celerite':
        sys.exit('Chi2 calculation method %s not recognised.\nTry \'manual\' or \'celerite\'.' %self.chi2_method)
        
					
def gaussian_process_prior(self, p):
	lp = 0.		
	pi = self.gaussian_process_index

	if self.GP_model=='Real':
		param_strings = ['ln_a','ln_c']
			
	if self.GP_model=='Matern':
		param_strings = ['ln_sigma', 'ln_rho']

	if self.gaussian_process_common:

		if self.GP_model == 'Real':
			self.ln_a = p[pi]
			self.ln_c = p[pi+1]
				
		if self.GP_model == 'Matern':
			self.ln_sigma = p[pi]
			self.ln_rho = p[pi+1]
		
		GP_params = p[pi:pi+2]
		
		for i, param_string in enumerate(param_strings):
			prange = eval('self.'+param_string+'_limits')  # set in __init__
			if prange:
				if GP_params[i] < prange[0] or GP_params[i] > prange[1]:
					lp += -np.inf
					print('GP param %s outside prior.'%param_string)
					print(prange[0],'<', GP_params[i],'<', prange[1], ' is not True.')
		
	else:

		for site in self.gaussian_process_sites:

			if self.GP_model == 'Real':
				self.ln_a[site] = p[pi]
				self.ln_c[site] = p[pi+1]
						
			if self.GP_model == 'Matern':	
				self.ln_sigma[site] = p[pi]
				self.ln_rho[site] = p[pi+1]
			
			GP_params = p[pi:pi+2]
			
			for i, param_string in enumerate(param_strings):
				prange = eval('self.'+param_string+'_limits')  # set in __init__
				if prange:
					if GP_params[i] < prange[0] or GP_params[i] > prange[1]:
						lp += -np.inf
						print('GP param %s %s outside prior.'%(site, param_string))
						print(prange[0],'<', GP_params[i],'<', prange[1], ' is not True.')
			
			pi += 2
				
	if np.isfinite(lp):
	    return lp
	else:
	    return -np.inf
	
	
def get_gaussian_process_bounds(self):

    bounds = []

    if self.GP_model=='Real':
        param_strings = ['ln_a','ln_c']
			
    if self.GP_model=='Matern':
        param_strings = ['ln_sigma', 'ln_rho']
		
    bound1 = eval('self.'+param_strings[0]+'_limits')  # set in __init__
    bound2 = eval('self.'+param_strings[1]+'_limits')  # set in __init__
	
    return bound1, bound2
	

def gaussian_process_ground_chi2(self):
	pass

def gaussian_process_spitzer_chi2(self,A,a0,a1,p=None):

    if p is None:
        p = self.p.copy()

    from celerite import terms

    site = 'spitzer'
    chi2_sum = 0.
	
    def nan_check(y):
        '''return true if nans are present'''
        return not (np.isfinite(y)).all()
	
    # Errors		
    if self.scale_spitzer_error_bars_multiplicative:	
        sig = p[self.spitzer_error_bar_scale_index]*self.spitzer_data[2]
    else:
        sig = self.spitzer_data[2].copy()
    sig2 = (sig)**2
    
    # GP parameter index
    if self.gaussian_process_common or ('spitzer' in self.gaussian_process_sites):
				
        pi = self.gaussian_process_index
        
        if (not self.gaussian_process_common) and ('spitzer' in self.gaussian_process_sites):
            stop = 0
            for site in self.gaussian_process_sites:
                while stop == 0:
                    if not(site=='spitzer'):
                        pi+=2
                    else:
                        stop = 1
                    
    else:
        sys.exit('Logic error in gaussian_process_spitzer_chi2')

    # get GP parameters and set kernel    
    if self.GP_model == 'Real':
        a = p[pi]
        c = p[pi+1]
        kernel = terms.RealTerm(log_a=a, log_c=c)
				
    if self.GP_model == 'Matern':
        sigma = p[pi]
        rho = p[pi+1]
        kernel = terms.Matern32Term(log_sigma=sigma, log_rho=rho)

    # build kernel and model object
    GP_dic={}
    GP_dic[str(p[pi])[-5:]+'_'+str(p[pi])[-5:]] = celerite.GP(kernel)			
    GP_dic[str(p[pi])[-5:]+'_'+str(p[pi])[-5:]].compute(self.spitzer_data[0],sig)
    model = A*a0[site] + a1[site]
    if nan_check(model):
        print('Warning: nan present in GP model build')
        print('model:', model)
        print('p:', p)
        print('A:', A)
        print('Fs:', a0[site])
        print('Fb:', a1[site])
    ll = GP_dic[str(p[pi])[-5:]+'_'+str(p[pi])[-5:]].log_likelihood(model-self.spitzer_data[1])
    chi2 = -2.*ll
		
    #print('Chi2 test:', chi2)

    if self.chi2_method == 'celerite':
        chi2_sum += chi2
    else:
        sys.exit('Chi2 method error: method %s unrecognised' %chi2_method)
		
    return chi2_sum


# used before 211021	
'''
def gaussian_process_spitzer_chi2(self,A,a0,a1):

    from celerite import terms

    site = 'spitzer'
    chi2_sum = 0.
	
    def nan_check(y):
        #return true if nans are present
        return not (np.isfinite(y)).all()
	
    # Errors
    if self.scale_error_bars_multiplicative:
        sig = self.p[self.error_bar_scale_index[site]]*self.spitzer_data[2]		
    elif self.scale_spitzer_error_bars_multiplicative:	
        sig = self.p[self.spitzer_error_bar_scale_index]*self.spitzer_data[2]
    else:
        sig = self.spitzer_data[2]
    sig2 = (sig)**2
	
    if self.gaussian_process_common or (site in self.gaussian_process_sites):
						
        if self.GP_model=='Real':		
            if self.gaussian_process_common:
								
                a = self.ln_a
                c = self.ln_c 	
										
            else:

                if site in self.gaussian_process_sites:
								
                    a = self.ln_a[site]
                    c = self.ln_c[site]							
				
            kernel = terms.RealTerm(log_a=a, log_c=c)
					
        elif self.GP_model=='Matern':
            if self.gaussian_process_common:
								
                sigma = self.ln_sigma
                rho = self.ln_rho
										
            else:

                if site in self.gaussian_process_sites:
								
                    sigma = self.ln_sigma[site]
                    rho = self.ln_rho[site]				
								
				
            kernel = terms.Matern32Term(log_sigma=sigma, log_rho=rho)
					
        gp = celerite.GP(kernel)
        gp.compute(self.spitzer_data[0],sig)
        model = A*a0[site] + a1[site]
        if nan_check(model):
            print('Warning: nan present in GP model build')
            print('model:', model)
            print('p:', mc.p)
            print('A:', A)
            print('Fs:', a0[site])
            print('Fb:', a1[site])
        chi2 = -2.*gp.log_likelihood(model-self.spitzer_data[1])
		
        #cov = gp.get_matrix(self.spitzer_data[0])
        I = np.identity(len(self.spitzer_data[0]))
        L = gp.solver.dot_L(I)
        K = L@L.T # as is the nature of L
        r = model - self.spitzer_data[1]
        Kinv = np.linalg.inv(K)
        N = len(self.spitzer_data[0])
        chi2_manual = r.T@Kinv@r + np.log(np.linalg.det(K)) + N*np.log(2.*np.pi)
        lnl_manual = -1./2.*chi2_manual
        print('Chi2 test:', chi2, chi2_manual)
		
        if self.chi2_method == 'manual':
            chi2_sum += chi2_manual
        elif self.chi2_method == 'celerite':
            chi2_sum += chi2
        else:
            sys.exit('Chi2 method error: method %s unrecognised' %chi2_method)
		
    return chi2_sum'''
		
import celerite
from celerite.modeling import Model
    
class CeleriteModel(Model):
    
    def get_mc(self, mc):
        self.mc = mc
        self.p = mc.p.copy() #otherwise changing a celerite p will change mc.p
        self.site = 'spitzer'
        #self.ref_site = mc.reference_source
    
    def get_value(self, t):
        ''' this function calculated some shit
		input are:'''

        _, a0, a1, _, _, _ = self.mc.chi2_calc(p_in=self.p)
        if self.site == 'spitzer':
            if not isinstance(self.mc.limb_constant, float):
                if self.mc.spitzer_limb_constant != self.mc.limb_constant['spitzer']:
                    print('\n\n\n Replacing Spitzer limb constant (%f) with %f.\n\n\n' %(self.mc.spitzer_limb_constant,self.mc.limb_constant['spitzer']))
                    self.mc.spitzer_limb_constant = self.mc.limb_constant['spitzer']
            A = self.mc.spitzer_magnification(t, p=self.p)
        elif not isinstance(self.mc.limb_constant, float):
            A = self.mc.magnification(t,p=self.p,LD=self.mc.limb_constant[self.site])
        else:
            A = self.mc.magnification(t,p=self.p)
        Fs = a0[self.site]
        Fb = a1[self.site]
        #Fs = a0[self.ref_site]
        #Fb = a1[self.ref_site]
        F = A*Fs + Fb
        return F		
	
