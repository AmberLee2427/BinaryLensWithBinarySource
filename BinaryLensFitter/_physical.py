import numpy as np

def compute_relative_proper_motion(self,theta_star,p=None,p_sig=None,quiet=False):

	if self.use_parallax:

		if p is None:
			p = self.p

		if p_sig is None:
			p_sig = self.p_sig

		if self.parallax_t_ref is None:
			self.parallax_t_ref = p[5] #t0

		days_per_year = 365.25

		pi_EN = np.float64(p[self.Pi_EN_index])
		pi_EE = np.float64(p[self.Pi_EE_index])
		pi_EN_err = np.float64(p_sig[self.Pi_EN_index])
		pi_EE_err = np.float64(p_sig[self.Pi_EE_index])
		rho = 10.0**p[2]
		t0 = p[5]
		tE = p[6]

		theta_E = theta_star/rho

		pi_E2 = pi_EN**2 + pi_EE**2
		pi_E = np.sqrt(pi_E2)

		mu_geo = theta_E/tE * days_per_year * np.array([pi_EN,pi_EE])/pi_E

		dEE = (1.0-pi_EE**2/pi_E2)/pi_E
		dEN = -pi_EE*pi_EN/(pi_E2*pi_E)
		dNN = (1.0-pi_EN**2/pi_E2)/pi_E
		dNE = dEN

		mu_geo_err = np.zeros_like(mu_geo)
		mu_geo_err[0] = theta_E/tE * days_per_year * np.sqrt(dNN**2 * pi_EN_err**2 + dNE**2 * pi_EE_err**2)
		mu_geo_err[1] = theta_E/tE * days_per_year * np.sqrt(dEE**2 * pi_EE_err**2 + dEN**2 * pi_EN_err**2)

		pi_rel = pi_E * theta_E

		self.compute_parallax_terms(np.array([t0]))
		mu_helio = mu_geo + pi_rel * self.v_Earth_perp * days_per_year

		galaxy_rot_angle = np.arctan2(self.galaxy_rotation_north,self.galaxy_rotation_east)*180.0/np.pi
		proper_motion_angle = np.arctan2(pi_EN,pi_EE)*180.0/np.pi

		if not quiet:
			print()
			print('compute_relative_proper_motion')
			print('pi_EN:', pi_EN, '+/-', pi_EN_err)
			print('pi_EE:', pi_EE, '+/-', pi_EE_err)
			print('theta_E, pi_rel:', theta_E, pi_rel)
			print('v_Earth_perp:', self.v_Earth_perp)
			print('mu_geo:', mu_geo)
			print('mu_geo_err:', mu_geo_err)
			print('mu_helio:', mu_helio)
			print('galaxy rotation:', self.galaxy_rotation_north, self.galaxy_rotation_east)
			print('galaxy rotation angle N of E (deg):', galaxy_rot_angle)
			print('relative lens source proper motion angle N of E (deg):', proper_motion_angle)
			print('proper motion angle - galaxy rotation angle:', proper_motion_angle - galaxy_rot_angle)
			print()

		return mu_geo, mu_helio, mu_geo_err

	return None


def VItoVK(self,VI,datafile):

	try:
		with open(datafile,'r') as fid:
			d = fid.read()
	except IOError:
		print(datafile, 'not found. Exiting.')
		sys.exit(0)

	data = np.loadtxt(datafile,skiprows=1,usecols=(1,2))

	if VI < data[0,0] or VI > data[-1,0]:
		VK = np.nan
	else:
		VK = np.interp(VI,data[:,0],data[:,1])

	return VK


def compute_theta_E(self,rho,rho_err,VI_source_0,VI_source_0_err,I_source_0,I_source_0_err,VI_VK_data_file):

	V_source_0 = VI_source_0 + I_source_0
	V_source_0_err = np.sqrt(VI_source_0_err**2 + I_source_0_err**2)

	VK_source_0 = self.VItoVK(VI_source_0,VI_VK_data_file)
	VK_source_0_err = np.abs(self.VItoVK(VI_source_0+VI_source_0_err,VI_VK_data_file) - VK_source_0)

	print('Intrinsic (V-K), V:', VK_source_0, V_source_0)
	print('Intrinsic (V-K), V uncertainty:', VK_source_0_err, V_source_0_err)

	qlog2thetastar  = 0.5410+0.2667*VK_source_0 - V_source_0/5.0
	theta_star = 10.**qlog2thetastar/2.0

	log2thetastar_variance = (0.2667/2)**2 * VK_source_0_err**2 + (0.1)**2 * V_source_0_err**2
	theta_star_err = 2.3026 * theta_star * np.sqrt(log2thetastar_variance)

	print()
	print('theta_star:', theta_star, 'pm', theta_star_err, 'mas')

	theta_E = theta_star/rho
	theta_E_err = np.sqrt((theta_star_err/rho)**2 + (rho_err*theta_star/rho**2)**2)

	print()
	print('theta_E:', theta_E, 'pm', theta_E_err, 'mas')

	return theta_E, theta_E_err


def compute_physical_properties(self,p,p_sig,VI_source_0,VI_source_0_err,I_source_0,I_source_0_err,
								D_clump,theta_star=None, theta_star_err=None,VI_VK_data_file=None,relation='old',pi_E_err=None,quiet=False):


	if self.use_parallax:

		pi_EN = np.float64(p[self.Pi_EN_index])
		pi_EE = np.float64(p[self.Pi_EE_index])
		s = 10.0**p[0]
		q = 10.0**p[1]
		rho = 10.0**p[2]
		t0 = p[5]
		tE = p[6]

		pi_EN_err = np.float64(p_sig[self.Pi_EN_index])
		pi_EE_err = np.float64(p_sig[self.Pi_EE_index])
		s_err = np.log(10.0) * s * p_sig[0]
		q_err = np.log(10.0) * q * p_sig[1]
		rho_err = np.log(10.0) * rho * p_sig[2]
		t0_err = p_sig[5]
		tE_err = p_sig[6]

		pi_E = np.sqrt(pi_EE**2 + pi_EN**2)
		if pi_E_err==None:
		    pi_E_err = np.sqrt(pi_EE**2 * pi_EE_err**2 + pi_EN**2 * pi_EN_err**2) / pi_E
		# partial derivative of pi_E by pi_EE = pi_EE/pi_E 

		if not quiet:
			print()
			print('piEE, piEN, piE:', pi_EE, pi_EN, pi_E)


		V_source_0 = VI_source_0 + I_source_0
		V_source_0_err = np.sqrt(VI_source_0_err**2 + I_source_0_err**2)
		
		if VI_VK_data_file is not None:

			VK_source_0 = self.VItoVK(VI_source_0,VI_VK_data_file)
			VK_source_0_err = np.abs(self.VItoVK(VI_source_0+VI_source_0_err,VI_VK_data_file) - VK_source_0)

			if not quiet:
				print()
				print('Intrinsic (V-K), V:', VK_source_0, V_source_0)
				print('Intrinsic (V-K), V uncertainty:', VK_source_0_err, V_source_0_err)
		
		def SB(params, V, dV, VK, dVK):
			a, b = params[:2]
			logtheta = a + b*VK - V/5.0
			thetastar = (10.0**logtheta)/2.0 #/2 to go from diameter to radius
			if len(params)==4:
				da, db = params[2:4]
				squared_logtheta_variance = da**2+(VK*db)**2+(b*dVK)**2+(dV/5.0)**2
			else:
				#logtheta_variance = np.sqrt( (b/2.0)**2 * dVK**2 + (0.1)**2 * dV**2 )
				#                = 1/2 * np.sqrt( b**2 * dVK**2 + (0.2)**2 * dV**2 )
				squared_logtheta_variance = b**2 * dVK**2 + (0.2)**2 * dV**2
			dthetastar = 2.3026 * thetastar * np.sqrt(squared_logtheta_variance) # 2.3026 = log(10)
			return thetastar, dthetastar
            
		def compute_theta_star(V, dV, VK, dVK, relation):

			SB_relations = {} #surface brightness relation coefficients
			SB_relations['old'] = [0.5410, 0.2667] # old is very similar to 
			SB_relations['vanBelle'] = [0.669, 0.223]
			SB_relations['Nordgren'] = [(4.2207-3.934)/0.5, 0.123/0.5,2.0*0.005,2.0*0.002]
			SB_relations['KervellaC'] = [(4.2207-3.9530)/0.5, 0.1336/0.5]

			if not (relation in SB_relations.keys()):
				print('Error, unknown surface bnrightness realtion: %s. Exiting.' %relation)
				sys.exit()

			thetastar, dthetastar = SB(SB_relations[relation],V,dV,VK,dVK)
			if not quiet:
				print('\n', relation, ':')
				print(SB_relations[relation])
				print('theta_star = ', thetastar, '+/-', dthetastar, ' mas')

			return thetastar, dthetastar
            
		if theta_star is None:
			qlog2thetastar  = 0.5410+0.2667*VK_source_0 - V_source_0/5.0
			theta_star = 10.**qlog2thetastar/2.0

			log2thetastar_variance = (0.2667/2)**2 * VK_source_0_err**2 + (0.1)**2 * V_source_0_err**2
			theta_star_err = 2.3026 * theta_star * np.sqrt(log2thetastar_variance)

		if not quiet:
			print()
			print('theta_star (Michael\'s code):', theta_star, 'pm', theta_star_err, 'mas')
            
		#theta_star, theta_star_err = compute_theta_star(V_source_0,V_source_0_err,VK_source_0, VK_source_0_err,relation)
        
		if not quiet:
			print()
			print('theta_star:', theta_star, 'pm', theta_star_err, 'mas')

		theta_E = theta_star/rho
		theta_E_err = np.sqrt((theta_star_err/rho)**2 + (rho_err*theta_star/rho**2)**2)

		if not quiet:
			print()
			print('theta_E:', theta_E, 'pm', theta_E_err, 'mas')

		s_angle = s * theta_E
		s_angle_err = np.sqrt(s_err**2 * theta_E**2 + theta_E_err**2 * s**2)

		if not quiet:
			print()
			print('s_angle:', s_angle, 'pm', s_angle_err, 'mas')


		pi_rel = theta_E * pi_E
		pi_rel_err = np.sqrt(pi_E**2 * theta_E_err**2 + theta_E**2 * pi_E_err**2)

		if not quiet:
			print()
			print('pi_rel:', pi_rel, 'pm', pi_rel_err, 'mas')

		D_lens = 1.0/((pi_rel/1000)+1.0/D_clump)
		D_lens_err = (1.0/((pi_rel/1000)+1.0/D_clump)**2) * (pi_rel_err/1000)

		if not quiet:
			print()
			print('D_lens:', D_lens, 'pm', D_lens_err)

		Rich_factor = D_lens/pi_E
		Rich_factor_err = np.sqrt((D_lens_err/pi_E)**2 + (pi_E_err*D_lens/pi_E**2)**2)

		if not quiet:
			print()
			print('Rich factor (D_lens/pi_E):', Rich_factor, 'pm', Rich_factor_err)

		s_angle_rad = (np.pi/180.0)*(s_angle/1000)/3600
		s_angle_rad_err = (np.pi/180.0)*(s_angle_err/1000)/3600 # = s_angle_err*s_angle_rad/s_angle
		s_real = D_lens * s_angle_rad
		s_real_err = np.sqrt(D_lens**2 * s_angle_rad_err**2 + s_angle_rad**2 * D_lens_err**2)

		s_real_AU = s_real * 206265
		s_real_AU_err = s_real_err * 206265 # = s_real_err*s_real_AU/s_real

		if not quiet:
			print()
			print('s_real:',s_real,'pm',s_real_err, 'pc')
			print('s_real:',s_real_AU,'pm',s_real_AU_err, 'AU')

		kappa = 8.144

		mass = theta_E/(kappa*pi_E)
		mass_err = np.sqrt((theta_E_err/(kappa*pi_E))**2 + (theta_E*pi_E_err/(kappa*pi_E**2))**2) # same as dtheta_E calc 

		if not quiet:
			print()
			print('mass:', mass, 'pm', mass_err, 'solar masses')

		m1 = mass/(1.0+q)
		m2 = mass*q/(1.0+q)

		m1_err = np.sqrt((mass_err/(1.0+q))**2 + (mass*q_err/(1.0+q))**2)
		m2_err = m1_err*q/(1.0+q)

		if not quiet:
			print('m1:',m1, 'pm', m1_err, 'solar masses')
			print('m1:',m1*1047.3, 'pm', m1_err*1047.3, 'jupiter masses')
			print('m2:',m2, 'pm', m2_err, 'solar masses')
			print('m2:',m2*1047.3, 'pm', m2_err*1047.3, 'jupiter masses')

		self.pi_E = pi_E
		self.theta_star = theta_star
		self.theta_star_err = theta_star_err
		self.theta_E = theta_E
		self.theta_E_err = theta_E_err
		self.pi_rel = pi_rel
		self.pi_rel_err = pi_rel_err
		self.lens_distance = D_lens
		self.lens_distance_err = D_lens_err
		self.lens_separation_AU = s_real_AU 
		self.lens_separation_AU_err = s_real_AU_err
		self.lens_mass_solar = mass
		self.lens_mass_solar_err = mass_err
		self.lens_mass_1_solar = m1
		self.lens_mass_1_solar_err = m1_err
		self.lens_mass_2_solar = m2
		self.lens_mass_2_solar_err = m2_err
		self.lens_mass_1_jupiter = m1*1047.3
		self.lens_mass_1_jupiter_err = m1_err*1047.3
		self.lens_mass_2_jupiter = m2*1047.3
		self.lens_mass_2_jupiter_err = m2_err*1047.3

		self.mu_geo, self.mu_helio, self.mu_err = self.compute_relative_proper_motion(theta_star,p=p,p_sig=p_sig,quiet=quiet)

		mu_sun_pec = -210.8 * self.solar_pecular_velocity_kms / D_lens
		self.mu_lsr = self.mu_helio - mu_sun_pec
		if not quiet:
			print('mu_sun_pec:', mu_sun_pec)
			print('mu_lsr:', self.mu_lsr)


