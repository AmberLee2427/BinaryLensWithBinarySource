import numpy as np


def caustic(self,n_angles=256,params=None):

	"""Compute the coordinates of the caustics."""

	if params is None:
		params = self.p
	
	if self.debug:	
		print('ca')

	logd, logq = params[:2]

	d = 10.0**logd
	q = 10.0**logq

	p = np.zeros([4], complex)
	angle = np.zeros([n_angles],np.float64)
	solutions = np.zeros((4,n_angles),complex)
	zeta = np.zeros((4,n_angles),complex)
	zeta_all_real = np.zeros([4*n_angles],np.float64)
	zeta_all_imag = np.zeros([4*n_angles],np.float64)
	
	e1 = q/(1.0+q)
	e2 = 1.0/(1.0+q)
	a = 0.5*d
	b = -(a*(q-1.0)) / (1.0+q)
	
	if self.debug:
		print('cb')

	for k in range (n_angles):

		angle = 2*np.pi / n_angles * k
		
		c4 = -np.exp((1j*1)*angle)
		c3 = 0.0;
		c2 = 1.0+2.0*(a**2)*np.exp((1j*1)*angle);
		c1 = -2.0*a + 4.0*e1*a;
		c0 = -(a**4)*np.exp((1j*1)*angle) + a**2;
		
		p = [c4,c3,c2,c1,c0]
		
		solutions[:,k] = np.roots(p)	
	
	if self.debug:
		print('cc')

	zeta = solutions.conjugate() - e1/(solutions-a) - (1-e1)/(solutions+a)
	
	for k in range (4):
		zeta_all_real[0+k*n_angles:n_angles+k*n_angles] = zeta[k,:].real
		zeta_all_imag[0+k*n_angles:n_angles+k*n_angles] = zeta[k,:].imag
	
	return zeta_all_real, zeta_all_imag


def magnification_map(self,p=None,u1_range=(-1.0,1.0),u2_range=(-1.0,1.0),n_u1=1000,n_u2=1000):

	import pycuda.autoinit
	import pycuda.driver as drv
	from ._gpu_mag_maps_cuda import gpu_mag_maps
	gpu_magnification_map = gpu_mag_maps.get_function("magnification")

	if p is None:
		p = self.p
	
	if self.debug:
		print('mma')

	d = 10.0**p[0]
	q = 10.0**p[1]

	e2 = 1.0/(1.0+q)
	e1 = q/(1.0+q)
	a = 0.5*d

	x0 = u1_range[0]
	dx = (u1_range[1] - x0)/(n_u1-1)
	y0 = u2_range[0]
	dy = (u2_range[1] - y0)/(n_u2-1)

	A = np.zeros((n_u1,n_u2),dtype=np.float64)

	blockshape = (256, 1, 1)
	gridshape = (n_u1//2, n_u2//256+1)
	
	if self.debug:
		print('mmb')

	gpu_magnification_map(	np.float64(e1), np.float64(e2), np.float64(a)\
							, np.float64(dx), np.float64(x0), np.float64(dy)\
							, np.float64(y0), drv.Out(A), block=blockshape\
							, grid=gridshape) 
	
	if self.debug:
		print('mmc')

	return A


def point_source_magnification(self,u1,u2,delta_d,p=None):

	import pycuda.autoinit
	import pycuda.driver as drv
	from ._gpu_mag_maps_cuda import gpu_mag_maps
	gpu_magnification = gpu_mag_maps.get_function("point_source_magnification")

	if p is None:
		p = self.p

	logd, logq = p[:2]
	d = 10.0**logd
	q = 10.0**logq
	e2 = 1.0/(1.0+q)
	e1 = q/(1.0+q)

	a = np.atleast_1d(np.float64(0.5*(d+delta_d)))

	n_points = len(u1)

	n_blocks = (n_points-1)//256 + 1
	blockshape = (256,1,1)
	gridshape = (n_blocks,1)

	A = np.atleast_1d(np.float64(np.zeros(n_points)))
	u1_f = np.atleast_1d(np.float64(u1))
	u2_f = np.atleast_1d(np.float64(u2))

	gpu_magnification(np.float64(e1), np.float64(e2), np.int32(n_points), drv.In(a), drv.In(u1_f), drv.In(u2_f), \
					drv.Out(A), block=blockshape, grid=gridshape)

	return A


def trajectory(self,t, p=None):

	if self.debug:
		print('ta')

	if p is None:
		p = self.p

	logd, logq, logrho, u0, phi, t0, tE = p[:7]
	d = 10.0**logd
	q = 10.0**logq
	rho = 10.0**logrho
	a = 0.5*d
	b = a*(1.0-q)/(1.0+q)

	#print 'Magnification for',d,q,rho,u0,phi,t0,tE

	delta_tau = np.zeros_like(t)
	delta_beta = np.zeros_like(t)
	delta_phi = np.zeros_like(t)
	delta_d = np.zeros_like(t)
	
	if self.debug:
		print('tb')

	if self.use_parallax:
		if self.parallax_t_ref is None:
			self.parallax_t_ref = t0
		pi_EN = np.float64(p[self.Pi_EN_index])
		pi_EE = np.float64(p[self.Pi_EE_index])
		q_n, q_e = self.compute_parallax_terms(t)
		delta_tau = q_n*pi_EN + q_e*pi_EE
		delta_beta = -q_n*pi_EE + q_e*pi_EN

	if self.use_lens_orbital_motion:
		if self.lens_orbital_motion_reference_date is None:
			t_ref = t0
		else:
			t_ref = self.lens_orbital_motion_reference_date
		dphidt = np.float64(p[self.dphidt_index])
		dddt = np.float64(p[self.dddt_index])
		delta_phi = dphidt*(t-t_ref)/365.2425
		delta_d = dddt*(t-t_ref)/365.2425

		if self.use_lens_orbital_acceleration:
			d2phidt2 = np.float64(p[self.d2phidt2_index])
			d2ddt2 = np.float64(p[self.d2ddt2_index])
			delta_phi += 0.5 * d2phidt2 * ((t-t_ref)/365.2425)**2
			delta_d += 0.5 * d2ddt2 *  ((t-t_ref)/365.2425)**2
	
	# Trajectory in centre-of-mass coordinates
	#self.u1 = ((t - t0)/tE + delta_tau)*np.cos(phi+delta_phi) + (u0+delta_beta)*np.sin(phi+delta_phi)
	#self.u2 = -((t - t0)/tE + delta_tau)*np.sin(phi+delta_phi) + (u0+delta_beta)*np.cos(phi+delta_phi)

	if self.use_VBBL:
		if self.debug:
			print('tva')
		# tau = (t - t0)/tE
		# u1 = -u0*np.sin(phi+delta_phi) + (tau+delta_tau)*np.cos(phi+delta_phi)
		# u2 = u0*np.cos(phi+delta_phi) + (tau+delta_tau)*np.sin(phi+delta_phi)
		phi = - phi
		u1 	= ( (t - t0)/tE + delta_tau ) * np.cos(phi - delta_phi) \
			- (u0 + delta_beta) * np.sin(phi - delta_phi)
		u2 	= -( (t - t0)/tE + delta_tau ) * np.sin(phi - delta_phi) \
			- (u0 + delta_beta) * np.cos(phi - delta_phi)
		if self.debug:
			print('tvc')
	else:
		# Trajectory in mid-point coordinates
		u0n = u0 - b*np.sin(phi)
		t0n = t0 + tE*b*np.cos(phi)

		u1 	= ( (t - t0n)/tE + delta_tau ) * np.cos(phi+delta_phi) \
			+ (u0n + delta_beta) * np.sin(phi + delta_phi)
		u2 	= -( (t - t0n)/tE + delta_tau ) * np.sin(phi + delta_phi) \
			+ (u0n + delta_beta) * np.cos(phi + delta_phi)
		if self.debug:
			print('tc')

	self.u1 = u1
	self.u2 = u2
	if self.debug:
		print('td')

	return u1, u2, delta_phi, delta_d



def magnification(self, t, p=None, LD=None):

	if self.debug:
		print('ma')

	if LD==None:
		LD = self.limb_constant

	if self.use_VBBL:
		if self.debug:
			print('mv')
		return self.VBBL_magnification(t,p,LD=LD)

	import pycuda.autoinit
	import pycuda.driver as drv

	from ._gpu_image_centred import gpu_image_centred
	image_centred = gpu_image_centred.get_function("data_magnifications")

	from scipy.spatial import cKDTree
	
	if self.debug:
		print('mb')

	if p is None:
		p = self.p

	logd, logq, logrho, u0, phi, t0, tE = p[:7]
	d = 10.0**logd
	q = 10.0**logq
	rho = 10.0**logrho

	u1, u2, delta_phi, delta_d = self.trajectory(t,p)

	if self.debug:
		print('mc')

	# First compute point-source mag for all epochs
	A = self.point_source_magnification(u1,u2,delta_d,p=p)

	high_points = np.where(A > self.hexadecapole_magnification_threshold)[0]
	if len(high_points) > 0:

		n_points = len(high_points)
		blockshape = (int(128),1, 1)
		gridshape = (n_points, 1)
		
		A_high = np.atleast_1d(np.zeros(n_points,np.float64))
		u1_f = np.atleast_1d(np.float64(u1[high_points]))
		u2_f = np.atleast_1d(np.float64(u2[high_points]))
		d_f = np.atleast_1d(np.float64(d+delta_d[high_points]))
		
		Gamma = np.float64(LD)

		if self.use_limb_darkening:
			Gamma = np.float64(p[self.limb_index])

		zeta_real, zeta_imag = self.caustic(params=p)
		zeta_real_f = np.float64(zeta_real)
		zeta_imag_f = np.float64(zeta_imag)

		image_centred(np.float64(q), np.float64(rho), Gamma, np.int32(1), np.float64(self.hexadecapole_approximation_threshold), 
						np.int32(len(zeta_real)), drv.In(zeta_real_f), drv.In(zeta_imag_f),
						drv.In(d_f), drv.In(u1_f), drv.In(u2_f), drv.Out(A_high), block=blockshape, grid=gridshape)

		A[high_points] = A_high


	# First compute point-source mag for all epochs
	#A = self.point_source_magnification(u1,u2,delta_d,p=p)
	#A = self.hexadecapole_magnification(u1,u2,delta_d,p=p,LD=LD)

	# Now consider points above a magnification threshold for hexadecaople or
	# image-centred ray shooting
	#high_points = np.where(A > self.hexadecapole_magnification_threshold)[0]
	#if len(high_points) > 0:
#		A[high_points] = self.finite_source_magnification(u1[high_points],u2[high_points],delta_d[high_points],p=p,LD=LD)
		# First compute hexadecapole mag for these points
		#A[high_points] = self.hexadecapole_magnification(u1[high_points],u2[high_points],delta_d[high_points],p=p,LD=LD)

	# Distance to nearest caustic
	# There is a potential problem here with lens orbital motion because we are using the caustic at t=t0.
	#zeta_real, zeta_imag = self.caustic(params=p)
	#caustic_tree = cKDTree(np.stack((zeta_real,zeta_imag),axis=-1),balanced_tree=False)
	#dist, ind = caustic_tree.query(np.stack((u1,u2),axis=-1),distance_upper_bound=rho*self.hexadecapole_approximation_threshold,eps=0.05*rho)

	#	points = np.where((dist > rho*self.hexadecapole_approximation_threshold))[0]
	#	if len(points) > 0:
	#		A[high_points[points]] = self.finite_source_magnification(u1[high_points[points]],u2[high_points[points]],delta_d[high_points[points]],p=p,LD=LD)

		# points = np.where((dist > rho*self.hexadecapole_approximation_threshold))[0]
		# if len(points)>0:
		# 	A[high_points[points]] = self.hexadecapole_magnification(u1[high_points[points]],u2[high_points[points]],delta_d[high_points[points]],p=p,LD=LD)

	#points = np.where(dist <= rho*self.hexadecapole_approximation_threshold)[0]
	#points = np.where(A>2.0)[0]
	#if len(points)>0:
		#print len(points), np.min(A),np.max(A),np.min(A[points]),np.max(A[points])
	#	A[points] = self.finite_source_magnification(u1[points],u2[points],delta_d[points],p=p,LD=LD)

	return A




def hexadecapole_magnification(self,u1,u2,delta_d,p=None, LD=None):

	if LD==None:
		LD = self.limb_constant

	import pycuda.autoinit
	import pycuda.driver as drv
	from ._gpu_mag_maps_cuda import gpu_mag_maps
	gpu_magnification = gpu_mag_maps.get_function("point_source_magnification")

	if p is None:
		p = self.p

	Gamma = LD

	if self.use_limb_darkening:

		Gamma = p[self.limb_index]

	logd, logq, logrho = p[:3]
	d = 10.0**logd
	q = 10.0**logq
	rho = 10.0**logrho
	e2 = 1.0/(1.0+q)
	e1 = q/(1.0+q)

	nu = len(u1)
	n_points = nu*13
	u1h = np.zeros(n_points)
	u2h = np.zeros(n_points)
	ah = np.zeros(n_points)

	u1h[:nu] = u1
	u2h[:nu] = u2
	ah[:nu] = 0.5*(d+delta_d)

	for j in range(1,5):
		u1h[j*nu:(j+1)*nu] = u1 + rho*np.cos((j-1)*np.pi*0.5)
		u2h[j*nu:(j+1)*nu] = u2 + rho*np.sin((j-1)*np.pi*0.5)
		ah[j*nu:(j+1)*nu] = 0.5*(d+delta_d)

	for j in range(5,9):
		u1h[j*nu:(j+1)*nu] = u1 + rho*np.cos((j-5)*np.pi*0.5+np.pi*0.25)
		u2h[j*nu:(j+1)*nu] = u2 + rho*np.sin((j-5)*np.pi*0.5+np.pi*0.25)
		ah[j*nu:(j+1)*nu] = 0.5*(d+delta_d)

	for j in range(9,13):
		u1h[j*nu:(j+1)*nu] = u1 + 0.5*rho*np.cos((j-9)*np.pi*0.5)
		u2h[j*nu:(j+1)*nu] = u2 + 0.5*rho*np.sin((j-9)*np.pi*0.5)
		ah[j*nu:(j+1)*nu] = 0.5*(d+delta_d)

	n_blocks = (n_points-1)//256 + 1
	blockshape = (256,1,1)
	gridshape = (n_blocks,1)

	A = np.zeros(n_points).astype(np.float64)
	u1_f = u1h.astype(np.float64)
	u2_f = u2h.astype(np.float64)
	a_f = ah.astype(np.float64)

	gpu_magnification(np.float64(e1), np.float64(e2), np.int32(n_points), drv.In(a_f), drv.In(u1_f), drv.In(u2_f), \
					drv.Out(A), block=blockshape, grid=gridshape)

	A1p = np.zeros(nu)
	for j in range(1,5):
		A1p += A[j*nu:(j+1)*nu]
	A1p *= 0.25
	A1p -= A[:nu]

	A1x = np.zeros(nu)
	for j in range(5,9):
		A1x += A[j*nu:(j+1)*nu]
	A1x *= 0.25
	A1x -= A[:nu]

	A2p = np.zeros(nu)
	for j in range(9,13):
		A2p += A[j*nu:(j+1)*nu]
	A2p *= 0.25
	A2p -= A[:nu]

	A2rho2 = (16.0*A2p - A1p)/3.0
	A4rho4 = (A1p + A1x)/2.0 - A2rho2

	mag = A[:nu] + A2rho2*(1.0-Gamma/5.0)/2.0 + A4rho4*(1.0-11.0*Gamma/35.0)/3.0

	return mag



def finite_source_magnification(self,u1,u2,delta_d,p=None,n_angles=64, LD=None):

	if LD==None:
		LD = self.limb_constant

	import pycuda.autoinit
	import pycuda.driver as drv
	from ._gpu_image_centred import gpu_image_centred
	image_centred = gpu_image_centred.get_function("data_magnifications")

	if p is None:
		p = self.p

	logd, logq, logrho = p[:3]
	d = np.float64(10.0**logd)
	q = np.float64(10.0**logq)
	rho = np.float64(10.0**logrho)

	# Magnification along trajectory
	n_points = len(u1)
	blockshape = (int(128),1, 1)
	gridshape = (n_points, 1)
	
	A = np.atleast_1d(np.zeros(n_points,np.float64))
	u1_f = np.atleast_1d(np.float64(u1))
	u2_f = np.atleast_1d(np.float64(u2))
	d_f = np.atleast_1d(np.float64(d+delta_d))
	
	Gamma = np.float64(LD)

	if self.use_limb_darkening:
		Gamma = np.float64(p[self.limb_index])

	image_centred(q, rho, Gamma, np.int32(1), drv.In(d_f), drv.In(u1_f), drv.In(u2_f), drv.Out(A), block=blockshape, \
					grid=gridshape)

	return A


