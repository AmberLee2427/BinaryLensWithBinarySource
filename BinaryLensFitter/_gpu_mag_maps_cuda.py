import pycuda.driver as drv
import pycuda.compiler
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule



gpu_mag_maps= SourceModule("""
	#include <pycuda-complex.hpp>
	#include <stdio.h>
	#include <math.h>
	#include <float.h>
	#define	COMPLEXTYPE	pycuda::complex<double>
	
	#include "texture_fetch_functions.h"
	#include "texture_types.h"
	
	// Used by Numerical Recipies (NR)
	#define NRANSI
	#include "/home/scratch/users/alm167/BinaryLensFitter/Code/nrutil.h"
	
	// For zroots
	#define	FALSE	0
	#define	TRUE	1
	#define	LARGE_VAL	1e100
	#define	PI			3.141592653589793
	#define TOL_ERR		2e-15
	
	// For NR Amoeba
	#define NDIM 2			// This value needs to be included in the #define GET_SUM " for (...j<=NDIM...)" and in p[(i-1)*NDIM+j]
	#define MP 3
	#define FTOL 1.0e-5
	#define NMAX 5000
	#define GET_PSUM \
						for (j=1;j<=2;j++) {\
						for (sum=0.0,i=1;i<=mpts;i++) sum += p[(i-1)*2+j];\
						psum[j]=sum;}
	#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}
	
	// For NR zroots
	#define	NUM_POLYS	6
	#define	MAX_THREADS 512			// Must be a value of 2^n
	#define	MAX_DATA_POINTS 8192
	#define SUB_PIXELS 10000
	#define SUB_PIX_DIM 100 		// sqrt(SUB_PIXELS)
	#define INV_SUB_PIX_DIM 0.01 	// = 1/(sqrt(SUB_PIXELS))
	#define SUB_PIXELS_DIM_X 500
	
	#define	MAX_DATA_SITES	15
	
	texture<float, 2> texref;
	
//////////////////////////////////////////////////////////////	Faster zoroots - J. Skowran & A. Gould

	/*
	 *-------------------------------------------------------------------*
	 * _14_                    MULTIPLY_POLY_1                           *
	 *-------------------------------------------------------------------*
	 * Subroutine will multiply polynomial 'polyin' by (x-p)
	 * results will be returned in polynomial 'polyout' of degree+1
	 *
	 * You can provide same array as 'polyin' and 'polyout' - this
	 * routine will work fine.
	 * 
	 * poly - is an array of polynomial cooefs, length = degree+1, poly(1) is constant 
	 *             1              2             3
	 *        poly(1) x^0 + poly(2) x^1 + poly(3) x^2 + ...
	 */
	__device__ void multiply_poly_1(COMPLEXTYPE *polyout, COMPLEXTYPE p, COMPLEXTYPE *polyin, int degree)
	{
		int i;
		
		for(i=1; i<=degree+1; i++)	// Copy
		{
			polyout[i-1] = polyin[i-1];
		}
		
		polyout[degree+1] = polyout[degree];
		
		for(i=degree+1; i>=2; i--)
		{
			polyout[i-1] = polyout[i-2] - polyout[i-1]*p;
		}
		polyout[0] *= -p;
		
		return;
	}


	/*
	 *-------------------------------------------------------------------*
	 * _13_                    CREATE_POLY_FROM_ROOTS                    *
	 *-------------------------------------------------------------------*
	 * Routine will build polynomial from a set of points given in 
	 * the array 'roots'. These points will be zeros of the resulting 
	 * polynomial.
	 *
	 * poly - is an array of polynomial cooefs, length = degree+1, poly(1) is constant 
	 *             1              2             3
	 *        poly(1) x^0 + poly(2) x^1 + poly(3) x^2 + ...
	 * degree - is and integer denoting size of the 'roots' array
	 * a - gives the leading coefficient of the resulting polynomial
	 * roots - input array of points, size=degree
	 *
	 * This subroutine works, but it is not optimal - it will work  
	 * up to a polynomial of degree~50, if you would like to have 
	 * more robust routine, up to a degree ~ 2000, you should 
	 * split your factors into a binary tree, and then multiply 
	 * leafs level by level from down up with subroutine like: 
	 * multiply_poly for arbitrary polynomial multiplications
	 * not multiply_poly_1.
	*/
	__device__ void create_poly_from_roots(COMPLEXTYPE *poly, int degree, COMPLEXTYPE a, COMPLEXTYPE *roots)
	{
		int i;
		COMPLEXTYPE zero;
		zero = COMPLEXTYPE((double)0.0, (double)0.0);
		
		
		for(i=1; i<=degree+1; i++)
		{
			poly[i-1] = zero;
		}
		poly[0] = a;	// Leading coeff of the polynomial
		
		for(i=1; i<=degree; i++)
		{
			multiply_poly_1(poly, roots[i-1], poly, i-1);
		}
		
		
		return;
	}


	/*
	 *-------------------------------------------------------------------*
	 * _12_                    EVAL_POLY                                 *
	 *-------------------------------------------------------------------*
	 * Evaluation of the complex polynomial 'poly' of a given degree 
	 * at the point 'x'. This routine calculates also the simplified
	 * Adams' (1967) stopping criterion. ('errk' should be multiplied 
	 * by 2d-15 for double precision,real*8, arithmetic)
	 * 
	 * poly - is an array of polynomial cooefs, length = degree+1, poly(1) is constant 
	 *             1              2             3
	 *        poly(1) x^0 + poly(2) x^1 + poly(3) x^2 + ...
	 */
	__device__ COMPLEXTYPE eval_poly(COMPLEXTYPE x, COMPLEXTYPE *poly, int degree, double *errk)
	{
		COMPLEXTYPE val;
		double absx;
		int i;
		
		// Prepare stoping criterion.
		*errk = abs(poly[degree]);
		absx = abs(x);
		for(i=degree; i>=1; i--)	// Horner Scheme, see for eg.  Numerical Recipes Sec. 5.3 how to evaluate polynomials and derivatives
		{
			val = val*x + poly[i-1];
			/*
			 * Adams, Duane A., 1967, "A stopping criterion for polynomial root finding",
			 * Communications of the ACM, Volume 10 Issue 10, Oct. 1967, p. 655
			 * ftp://reports.stanford.edu/pub/cstr/reports/cs/tr/67/55/CS-TR-67-55.pdf
			 * Eq 8.
			 */
			*errk = (*errk)*absx + abs(val);
		}
		
		// if(abs(val) < *errk) return;	// (simplified a little Eq. 10 of Adams 1967)
		
		return val;
	}


	/*
	 *-------------------------------------------------------------------*
	 * _11_                    DIVIDE_POLY_1                             *
	 *-------------------------------------------------------------------*
	 * Subroutine will divide complex polynomial 'polyin' by (x-p)
	 * results will be returned in polynomial 'polyout' of degree-1
	 * The remainder of the division will be returned in 'remainder'
	 *
	 * You can provide same array as 'polyin' and 'polyout' - this
	 * routine will work fine, though it will not set to zero the 
	 * unused, highest coefficient in the output array. You just have
	 * remember the proper degree of a polynomial.
	 *
	 * poly - is an array of polynomial cooefs, length = degree+1, poly(1) is constant 
	 *             1              2             3
	 *        poly(1) x^0 + poly(2) x^1 + poly(3) x^2 + ...
	 */
	__device__ void divide_poly_1(COMPLEXTYPE *polyout, COMPLEXTYPE *remainder, COMPLEXTYPE p, COMPLEXTYPE *polyin, int degree)
	{
		COMPLEXTYPE coef, prev;
		int i;
		
		coef = polyin[degree];
		for(i=1; i<=degree; i++)
		{
			polyout[i-1] = polyin[i-1];
		}
		for(i=degree; i>=1; i--)
		{
			prev = polyout[i-1];
			polyout[i-1] = coef;
			coef = prev + p*coef;
		}
		*remainder = coef;
		
		return;
	}


	/*
	 *-------------------------------------------------------------------*
	 * _10_                    SOLVE_CUBIC_EQ                            *
	 *-------------------------------------------------------------------*
	 * Cubic equation solver for complex polynomial (degree=3)
	 * http://en.wikipedia.org/wiki/Cubic_function   Lagrange's method
	 * 
	 * poly - is an array of polynomial cooefs, length = degree+1, poly(1) is constant 
	 *              1              2             3             4
	 *         poly(1) x^0 + poly(2) x^1 + poly(3) x^2 + poly(4) x^3
	 */
	__device__ void solve_cubic_eq(COMPLEXTYPE *x0, COMPLEXTYPE *x1, COMPLEXTYPE *x2, COMPLEXTYPE *poly)
	{
		COMPLEXTYPE zeta, zeta2, zero, s0, s1, s2, A, B, a_1, E12, delta, A2, val, x;
		COMPLEXTYPE E1;	// x0+x1+x2
		COMPLEXTYPE E2;	// x0x1+x1x2+x2x0
		COMPLEXTYPE E3;	// x0x1x2
		
		double third;
		
		int i;
		
		zero	= COMPLEXTYPE((double)0.0,(double) 0.0);
		zeta	= COMPLEXTYPE((double)-0.5, (double)0.8660254037844386);	// sqrt3(1)
		zeta2	= COMPLEXTYPE((double)-0.5, (double)-0.8660254037844386);	// sqrt3(1)^2
		third	= 0.3333333333333333;				// 1/3
		
		a_1	=	 1.0 / poly[3];
		E1	=	-poly[2]*a_1;
		E2	=	 poly[1]*a_1;
		E3	=	-poly[0]*a_1;
		
		s0	=	E1;
		E12	=	E1*E1;
		A	=	2.0*E1*E12 - 9.0*E1*E2 + 27.0*E3;	// =s1^3 + s2^3
		B	=	E12 - 3.0*E2;	// =s1 s2
		
		//	Quadratic equation: z^2-Az+B^3=0  where roots are equal to s1^3 and s2^3
		A2		= A*A;
		delta	= sqrt(A2 - 4.0*(B*B*B));
		
		if(real(conj(A)*delta) >= 0.0)	// Scallar product to decide the sign yielding bigger magnitude
		{
			//	Don't lose precision in double precision when taking the real part.
			s1 = pow(0.5*(A+delta), third);
		}
		else
		{
			s1 = pow(0.5*(A-delta), third);
		}
		
		if(s1 == zero)
		{
			s2 = zero;
		}
		else
		{
			s2 = B/s1;
		}
		
		*x0 = third*(s0 + s1 + s2);
		*x1 = third*(s0 + s1*zeta2 + s2*zeta );
		*x2 = third*(s0 + s1*zeta  + s2*zeta2);
		
		if(FALSE)	// Print the results
		{
			x = *x0;
			val = poly[3];
			for(i=3; i>=1; i--)
			{
				val = val*x+poly[i-1];
			}
			// write(*,'(2f19.15,a3,2f19.15)') x,' ->',val
			
			x = *x1;
			val = poly[3];
			for(i=3; i>=1; i--)
			{
				val = val*x+poly[i-1];
			}
			// write(*,'(2f19.15,a3,2f19.15)') x,' ->',val

			
			x = *x2;
			val = poly[3];
			for(i=3; i>=1; i--)
			{
				val = val*x+poly[i-1];
			}
			// write(*,'(2f19.15,a3,2f19.15)') x,' ->',val
		}
		
		return;
	}


	/*
	 *-------------------------------------------------------------------*
	 * _9_                     SOLVE_QUADRATIC_EQ                        *
	 *-------------------------------------------------------------------*
	 * Quadratic equation solver for complex polynomial (degree=2)
	 *
	 * poly - is an array of polynomial cooefs, length = degree+1, poly(1) is constant 
	 *              1              2             3             4
	 *         poly(1) x^0 + poly(2) x^1 + poly(3) x^2 + poly(4) x^3
	 */
	__device__ void solve_quadratic_eq(COMPLEXTYPE *x0, COMPLEXTYPE *x1, COMPLEXTYPE *poly)
	{
		COMPLEXTYPE a, b, c, b2, delta, val, x, zero;
		int i;
		
		zero = COMPLEXTYPE((double)0.0, (double)0.0);
		
		a = poly[2];
		b = poly[1];
		c = poly[0];
		//	quadratic equation: a z^2 + b z + c = 0;
		
		b2 = b*b;
		delta = sqrt(b2 - 4.0*(a*c));
		
		
		if( real(conj(b)*delta) >= 0.0 )	// Scallar product to decide the sign yielding bigger magnitude
		{
			//	Don't lose precision in double precision when taking the real part.
			*x0 = -0.5*(b+delta);
		}
		else
		{
			*x0 = -0.5*(b-delta);
		}
		
		if( (real(*x0) == real(zero)) && (imag(*x0) == imag(zero)) )
		{
			*x1 = zero;
		}
		else	// Viete's formula
		{
			*x1 = c/(*x0);
			*x0 = (*x0)/a;
		}
		
		if(FALSE)	// Print the results
		{
			x = *x0;
			val = poly[2];
			for(i=2; i>=1; i--)
			{
				val = val*x+poly[i-1];
			}
			//	write(*,'(2f19.15,a3,2f19.15)') x,' ->',val
			
			x = *x1;
			val = poly[2];
			for(i=2; i>=1; i--)
			{
				val = val*x + poly[i-1];
			}
			//	write(*,'(2f19.15,a3,2f19.15)') x,' ->',val
		}
		
		
		
		return;
	}


	/*
	 *-------------------------------------------------------------------*
	 * _8_                     CMPLX_LAGUERRE2NEWTON                     *
	 *-------------------------------------------------------------------*
	 * Subroutine finds one root of a complex polynomial using 
	 * Laguerre's method, Second-order General method and Newton's
	 * method - depending on the value of function F, which is a 
	 * combination of second derivative, first derivative and
	 * value of polynomial [F=-(p"*p)/(p'p')].
	 * 
	 * Subroutine has 3 modes of operation. It starts with mode=2
	 * which is the Laguerre's method, and continues until F 
	 * becames F<0.50, at which point, it switches to mode=1, 
	 * i.e., SG method (see paper). While in the first two
	 * modes, routine calculates stopping criterion once per every
	 * iteration. Switch to the last mode, Newton's method, (mode=0) 
	 * happens when becomes F<0.05. In this mode, routine calculates
	 * stopping criterion only once, at the beginning, under an 
	 * assumption that we are already very close to the root.
	 * If there are more than 10 iterations in Newton's mode, 
	 * it means that in fact we were far from the root, and
	 * routine goes back to Laguerre's method (mode=2).
	 *
	 * Uses 'root' value as a starting point (!!!!!)
	 * Remember to initialize 'root' to some initial guess or to 
	 * point (0,0) if you have no prior knowledge.
	 *
	 * poly - is an array of polynomial cooefs
	 *        length = degree+1, poly(1) is constant 
	 *               1              2             3
	 *          poly(1) x^0 + poly(2) x^1 + poly(3) x^2 + ...
	 * degree - a degree of the polynomial
	 * root - input: guess for the value of a root
	 *        output: a root of the polynomial
	 * iter - number of iterations performed (the number of polynomial
	 *        evaluations and stopping criterion evaluation)
	 * success - is false if routine reaches maximum number of iterations
	 * starting_mode - this should be by default = 2. However if you  
	 *                 choose to start with SG method put 1 instead. 
	 *                 Zero will cause the routine to 
	 *                 start with Newton for first 10 iterations, and 
	 *                 then go back to mode 2.
	 *                 
	 *
	 * For a summary of the method see the paper: Skowron & Gould (2012)
	 *
	 */
	__device__ void cmplx_laguerre2newton(COMPLEXTYPE *poly, int degree, COMPLEXTYPE *root, int *iter, int *success, int starting_mode)
	{
		int MAX_ITERS = 50;
		// Constants needed to break cycles in the scheme
		int FRAC_JUMP_EVERY=10;
		int FRAC_JUMP_LEN=10;
		
		double FRAC_JUMPS[10] = {0.64109297, 0.91577881, 0.25921289, 0.50487203, 0.08177045, 0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137};
		
		double faq;	// Jump length
		double FRAC_ERR = TOL_ERR;	// 2x10^-15
		
		COMPLEXTYPE p;	// vale of polynomial
		COMPLEXTYPE dp;	// vale of 1st derivative
		COMPLEXTYPE d2p_half;	// vale of 2nd derivative
		
		
		COMPLEXTYPE denom, denom_sqrt, dx, newroot, zero, c_one, fac_netwon, fac_extra, F_half, c_one_nth;
		double ek, absroot, abs2p, abs2_F_half, one_nth, n_1_nth, two_n_div_n_1, stopping_crit2;
		int i, j, k, mode;
		int good_to_go;
		
		zero = COMPLEXTYPE((double)0.0, (double)0.0);
		c_one = COMPLEXTYPE((double)1.0, (double)0.0);
		
		*iter = 0;
		*success = TRUE;
		
		//	Next if-endif block is an EXTREME failsafe, not usually needed, and thus turned off in this version.
		if(FALSE)	// Change false-->true if you would like to use caution about having first coefficient == 0
		{
			if(degree < 0)
			{
				printf("Error: cmplx_laguerre2newton: degree<0");
				return;
			}
			
			if(poly[degree] == zero)
			{
				if(degree == 0) return;
				cmplx_laguerre2newton(poly, degree-1, root, iter, success, starting_mode);
				return;
			}
			
			if(degree <= 1)
			{
				if(degree == 0)	// We know from previous check than poly[0] not equal zero
				{
					success = FALSE;
					printf("Warning: cmplx_laguerre2newton: degree = 0 and poly[0] != 0, no roots");
					return;
				}
				else
				{
					*root = -poly[0]/poly[1];
					return;
				}
			}
		}
		//	end EXTERME failsafe
		
		
		
		
		j = 1;
		good_to_go = FALSE;
		
		mode = starting_mode;	// mode=2 Full laguerre, mode=1 SG, mode=0 Newton
		
		while(TRUE)	// infinite loop, just to be able to come back from newton, if more than 10 iteration there
		{
			//------------------------------------------------------------- mode 2
			if(mode >= 2)	// LAGUERRE'S METHOD
			{
				one_nth = 1.0 / (double)degree;
				n_1_nth = (degree-1.0)*one_nth;
				two_n_div_n_1 = 2.0/n_1_nth;
				c_one_nth = COMPLEXTYPE((double)one_nth,(double) 0.0);
				
				for(i=1; i<=MAX_ITERS; i++)
				{
					
					faq = 1.0;
					
					//	Prepare stoping criterion
					ek = abs(poly[degree]);
					absroot = abs(*root);
					
					// Calculate value of polynomial and its first two derivatives
					p = poly[degree];
					dp = zero;
					d2p_half = zero;
					for(k=degree; k>=1; k--)	// Horner Scheme, see for eg.  Numerical Recipes Sec. 5.3 how to evaluate polynomials and derivatives
					{
						d2p_half = dp + d2p_half*(*root);
						dp = p + dp*(*root);
						p = poly[k-1] + p*(*root);	// b_k
						/*
						 * Adams, Duane A., 1967, "A stopping criterion for polynomial root finding",
						 * Communications of the ACM, Volume 10 Issue 10, Oct. 1967, p. 655ftp://reports.stanford.edu/pub/cstr/reports/cs/tr/67/55/CS-TR-67-55.pdf
						 * Eq 8.
						 */
						ek = absroot*ek + abs(p);
					}
					abs2p = real(conj(p)*p);	// abs(p)^2
					*iter += 1;
					
					if(abs2p == 0.0) return;
					
					stopping_crit2 = (FRAC_ERR*ek)*(FRAC_ERR*ek);
					if(abs2p < stopping_crit2)	// (simplified a little Eq. 10 of Adams 1967) 
					{
						// Do additional iteration if we are less than 10x from stopping criterion
						if(abs2p < 0.01*stopping_crit2)	// ten times better than stopping criterion
						{
							return;	// Return immediately, because we are at very good place
						}
						else
						{
							good_to_go = TRUE;	// Do one iteration more
						}
					}
					else
					{
						good_to_go = FALSE;	// Reset if we are outside the zone of the root
					}
					
					fac_netwon = p / dp;
					fac_extra = d2p_half / dp;
					F_half = fac_netwon*fac_extra;
					
					abs2_F_half = real(conj(F_half)*F_half);
					if(abs2_F_half <= 0.0625)	// F<0.50,	F/2<0.25
					{
						//	Go to SG method
						if(abs2_F_half <= 0.000625)	// F<0.05,	F/2<0.025
						{
							mode = 0;	// Go to Newton's
						}
						else
						{
							mode = 1;	// Go to SG
						}
					}
					
					denom_sqrt = sqrt(c_one - two_n_div_n_1*F_half);
					
					// NEXT LINE PROBABLY CAN BE COMMENTED OUT
					if(real(denom_sqrt) >= 0.0)	// Use realpart
					{
						/*
						 * Real part of a square root is positive for probably all compilers. You can 
						 * test this on your compiler and if so, you can omit this check
						 */
						denom = c_one_nth + n_1_nth*denom_sqrt;
					}
					else
					{
						denom = c_one_nth - n_1_nth*denom_sqrt;
					}
					
					if(denom == zero)	// Test if demoninators are > 0.0 not to divide by zero
					{
						dx = (abs(*root) + 1.0) * exp( COMPLEXTYPE((double)0.0, (double)FRAC_JUMPS[i%FRAC_JUMP_LEN]*2.0*PI) );	// Make some random jump
					}
					else
					{
						dx = fac_netwon / denom;
					}
					
					newroot = *root - dx;
					
					if(newroot == *root) return;	// Nothing changes -> return;
					if(good_to_go == TRUE)	// This was jump already after stopping criterion was met
					{
						*root = newroot;
						return;
					}
					
					if(mode != 2)
					{
						*root = newroot;
						j = i + 1;
						break;	// go to Newton's or SG
					}
					
					if(i%FRAC_JUMP_EVERY == 0)	// Decide whether to do a jump of modified length (to break cycles)
					{
						faq = FRAC_JUMPS[(i/FRAC_JUMP_EVERY-1)%(FRAC_JUMP_LEN)];
						newroot = *root - faq*dx;	// Do jump of some semi-random length (0<faq<1)
					}
					*root = newroot;
				}	// End for loop of mode 2
				
				if(i >= MAX_ITERS)
				{
					*success = FALSE;
					return;
				}
				
			}	// End if mode 2
			
			
			
			//------------------------------------------------------------- mode 1
			if(mode == 1)	// SECOND-ORDER GENERAL METHOD (SG)
			{
				for(i=j; i<= MAX_ITERS; i++)
				{
					faq = 1.0;
					
					// Calculate value of polynomial and its first two derivatives
					p = poly[degree];
					dp = zero;
					d2p_half = zero;
					if(((i-j)%10) == 0)
					{
						// Prepare stopping criterion
						ek = abs(poly[degree]);
						absroot = abs(*root);
						for(k=degree; k>=1; k--)	// Horner Scheme, see for eg.  Numerical Recipes Sec. 5.3 how to evaluate polynomials and derivatives
						{
							d2p_half = dp + d2p_half*(*root);
							dp = p + dp*(*root);
							p = poly[k-1] + p*(*root);	// b_k
							/*
							 * Adams, Duane A., 1967, "A stopping criterion for polynomial root finding",
							 * Communications of the ACM, Volume 10 Issue 10, Oct. 1967, p. 655
							 * ftp://reports.stanford.edu/pub/cstr/reports/cs/tr/67/55/CS-TR-67-55.pdf
							 * Eq 8. 
							 */
							ek = absroot*ek + abs(p);
						}
						stopping_crit2 = (FRAC_ERR*ek)*(FRAC_ERR*ek);
					}
					else
					{
						for(k=degree; k>=1; k--)	// Horner Scheme, see for eg.  Numerical Recipes Sec. 5.3 how to evaluate polynomials and derivatives
						{
							d2p_half = dp + d2p_half*(*root);
							dp = p + dp*(*root);
							p = poly[k-1] + p*(*root);	// b_k
						}
					}
					
					abs2p = real(conj(p) * p);	// abs(p)^2
					*iter += 1;
					if(abs2p == 0.0) return;
					
					if(abs2p < stopping_crit2)	// (simplified a little Eq. 10 of Adams 1967) 
					{
						if(dp == zero) return;
						// Do additional iteration if we are less than 10x from stopping criterion
						if(abs2p <= 0.01*stopping_crit2)	// Ten times better than stopping criterion
						{
							return;	// Return immediately, because we are at very good place
						}
						else
						{
							good_to_go = TRUE;	// Do one iteration more
						}
					}
					else
					{
						good_to_go = FALSE;	// Reset if we are outside the zone of the root
					}
					
					if(dp == zero)	// Test if demoninators are > 0.0 not to divide by zero
					{
						dx = (abs(*root) + 1.0) * exp( COMPLEXTYPE((double)0.0, (double)FRAC_JUMPS[i%FRAC_JUMP_LEN]*2.0*PI) );	// Make some random jump
					}
					else
					{
						fac_netwon = p/dp;
						fac_extra = d2p_half/dp;
						F_half = fac_netwon*fac_extra;
						
						abs2_F_half = real(conj(F_half)*F_half);
						if(abs2_F_half <= 0.000625)	// F<0.05,	F/2<0.025
						{
							mode = 0;	// Set Newton's, go there after jump
						}
						
						dx = fac_netwon*(c_one+F_half);	// SG
					}
					
					newroot = *root - dx;
					if(newroot == *root) return;	// nothing changes -> return
					if(good_to_go == TRUE)	// This was jump already after stopping criterion was met
					{
						*root = newroot;
						return;
					}
					
					if(mode != 1)
					{
						*root = newroot;
						j = i + 1;	// remember iteration number
						break;		// Go to Newton's
					}
					
					if(i%FRAC_JUMP_EVERY == 0)	// Decide whether to do a jump of modified length (to break cycles)
					{
						faq = FRAC_JUMPS[(i/FRAC_JUMP_EVERY - 1)%(FRAC_JUMP_LEN)];
						newroot = *root - faq*dx;	// Do jump of some semi-random length (0<faq<1)
					}
					*root = newroot;
				}	// End for loop of mode 1
				
				
				if(i >= MAX_ITERS)
				{
					*success = FALSE;
					return;
				}
				
			}	// End if mode 1
			
			//------------------------------------------------------------- mode 0
			if(mode == 0)	// Newton
			{
				for(i=j; i<= j+10; i++)	// Do only 10 iterations the most, then go back to full Laguerre's
				{
					faq = 1.0;
					
					// Calculate value of polynomial and its first two derivatives
					p = poly[degree];
					dp = zero;
					if(i == j)	// Calculate stopping crit only once at the begining
					{
						//	Prepare stopping criterion
						ek = abs(poly[degree]);
						absroot = abs(*root);
						for(k=degree; k>=1; k--)	// Horner Scheme, see for eg.  Numerical Recipes Sec. 5.3 how to evaluate polynomials and derivatives
						{
							dp = p + dp*(*root);
							p = poly[k-1] + p*(*root);	// b_k
							/*
							 * Adams, Duane A., 1967, "A stopping criterion for polynomial root finding",
							 * Communications of the ACM, Volume 10 Issue 10, Oct. 1967, p. 655
							 * ftp://reports.stanford.edu/pub/cstr/reports/cs/tr/67/55/CS-TR-67-55.pdf
							 * Eq 8. 
							 */
							ek = absroot*ek + abs(p);
						}
						stopping_crit2 = (FRAC_ERR*ek)*(FRAC_ERR*ek);
					}
					else
					{
						for(k=degree; k>=1; k--)	// Horner Scheme, see for eg.  Numerical Recipes Sec. 5.3 how to evaluate polynomials and derivatives
						{
							dp = p + dp*(*root);
							p = poly[k-1] + p*(*root);	// b_k
						}
					}
					
					abs2p = real(conj(p)*p);	// abs(p)^2
					*iter += 1;
					if(abs2p == 0.0) return;
					
					if(abs2p < stopping_crit2)	// (simplified a little Eq. 10 of Adams 1967)
					{
						if(dp == zero) return;
						//	Do additional iteration if we are less than 10x from stopping criterion
						if(abs2p < 0.01*stopping_crit2)	// Ten times better than stopping criterion
						{
							return;	// Return immediately, because we are at very good place
						}
						else
						{
							good_to_go = TRUE;	// Do one iteration more
						}
					}
					else
					{
						good_to_go = FALSE;	// Reset if we are outside the zone of the root
					}
					
					if(dp == zero) // Test if demoninators are > 0.0 not to divide by zero
					{
						dx = (abs(*root) + 1.0) * exp( COMPLEXTYPE((double)0.0, (double)FRAC_JUMPS[i%FRAC_JUMP_LEN]*2.0*PI) );	// Make some random jump
					}
					else
					{
						dx = p/dp;
					}
					
					newroot = *root - dx;
					
					if(newroot == *root) return;	// nothing changes -> return
					if(good_to_go == TRUE)
					{
						*root = newroot;
						return;
					}
					
					/*
					 * this loop is done only 10 times. So skip this check
					 * if(mod(i,FRAC_JUMP_EVERY).eq.0) then ! decide whether to do a jump of modified length (to break cycles)
					 * {
					 * 	faq=FRAC_JUMPS[(i/FRAC_JUMP_EVERY-1)%FRAC_JUMP_LEN]
					 * 	newroot=*root-faq*dx ! do jump of some semi-random length (0<faq<1)
					 * }
					 */
					*root = newroot;
					
				}	// end mode 0 for loop (10 times)
				
				if(*iter >= MAX_ITERS)
				{
					// Too many iterations here
					*success = FALSE;
					return;
				}
				mode = 2;	// Go back to Laguerre's. This happens when we were unable to converge in 10 iterations with Newton's
				
			}	// End if mode 0
			
		}	// End of infinite loop
		
		*success = FALSE;
		
		return;
	}

	/*
	 *-------------------------------------------------------------------*
	 * _7_                     CMPLX_NEWTON_SPEC                         *
	 *-------------------------------------------------------------------*
	 * Subroutine finds one root of a complex polynomial using 
	 * Newton's method. It calculates simplified Adams' stopping 
	 * criterion for the value of the polynomial once per 10 iterations,
	 * after initial iteration. This is done to speed up calculations
	 * when polishing roots that are known pretty well, and stopping
	 * criterion does not significantly change in their neighbourhood.
	 * 
	 * Uses 'root' value as a starting point (!!!!!)
	 * Remember to initialize 'root' to some initial guess.
	 * Do not initilize 'root' to point (0,0) if the polynomial 
	 * coefficients are strictly real, because it will make going 
	 * to imaginary roots impossible.
	 * 
	 * poly - is an array of polynomial coefs
	 *        length = degree+1, poly(1) is constant 
	 *               1              2             3
	 *          poly(1) x^0 + poly(2) x^1 + poly(3) x^2 + ...
	 * degree - a degree of the polynomial
	 * root - input: guess for the value of a root
	 *        output: a root of the polynomial
	 * iter - number of iterations performed (the number of polynomial
	 *        evaluations)
	 * success - is false if routine reaches maximum number of iterations
	 * 
	 * For a summary of the method go to: 
	 * http://en.wikipedia.org/wiki/Newton's_method
	 * 
	 */
	__device__ void cmplx_newton_spec(COMPLEXTYPE *poly, int degree, COMPLEXTYPE *root, int *iter, int *success)
	{
		int MAX_ITERS = 50;
		// Constants needed to break cycles in the scheme
		int FRAC_JUMP_EVERY=10;
		int FRAC_JUMP_LEN=10;
		
		double FRAC_JUMPS[10] = {0.64109297, 0.91577881, 0.25921289, 0.50487203, 0.08177045, 0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137};
		
		double faq;	// Jump length
		double FRAC_ERR = TOL_ERR;	// 2x10^-15
		
		COMPLEXTYPE p;	// vale of polynomial
		COMPLEXTYPE dp;	// vale of 1st derivative
		
		COMPLEXTYPE dx, newroot, zero;
		double ek, absroot, abs2p, stopping_crit2;
		
		int i, k;
		
		int good_to_go;
		
		zero = COMPLEXTYPE((double)0.0, (double)0.0);
		*iter = 0;
		*success = TRUE;
		
		// Next if-endif block is an EXTREME failsafe, not usually needed, and thus turned off in this version.
		if(FALSE)	// Change false-->true if you would like to use caution about having first coefficient == 0
		{
			if(degree < 0)
			{
				printf("Error: cmplx_newton_spec: degree<0");
				return;
			}
			if(poly[degree] == zero)
			{
				if(degree == 0) return;
				cmplx_newton_spec(poly, degree-1, root, iter, success);
				return;
			}
			if(degree <= 1)
			{
				if(degree == 0)
				{
					*success = FALSE;
					printf("Warning: cmplx_newton_spec: degree = 0 and poly[0] != 0, no roots");
					return;
				}
				else
				{
					*root = -poly[0]/poly[1];
					return;
				}
			}
		}
		//	End EXTERME failsafe
		
		
		good_to_go = FALSE;
		
		for(i=1; i<=MAX_ITERS; i++)
		{
			faq=1.0;
			
			/*
			 * Prepare stoping criterion
			 * calculate value of polynomial and its first two derivatives
			 */
			p = poly[degree];
			dp = zero;
			
			if(i%10 == 1)	// Calculate stopping criterion every tenth iteration
			{
				ek = abs(poly[degree]);
				absroot = abs(*root);
				for(k=degree; k>=1; k--)	// Horner Scheme, see for eg.  Numerical Recipes Sec. 5.3 how to evaluate polynomials and derivatives
				{
					dp = p + dp*(*root);
					p = poly[k-1] + p*(*root);	// b_k
					/*
					 * Adams, Duane A., 1967, "A stopping criterion for polynomial root finding",
					 * Communications of the ACM, Volume 10 Issue 10, Oct. 1967, p. 655
					 * ftp://reports.stanford.edu/pub/cstr/reports/cs/tr/67/55/CS-TR-67-55.pdf
					 * Eq 8.
					 */
					ek = absroot*ek + abs(p);
				}
				stopping_crit2 = (FRAC_ERR*ek)*(FRAC_ERR*ek);
			}
			else
			{
				for(k=degree; k>=1; k--)
				{
					dp = p + *root;
					p = poly[k-1] + p*(*root);	// b_k;
				}
			}
			*iter += 1;
			
			abs2p = real(conj(p) * p);	// abs(p)^2
			if(abs2p == 0.0) return;
			
			if(abs2p < stopping_crit2)	// (simplified a little Eq. 10 of Adams 1967)
			{
				if(dp == zero)	return;	// If we have problem with zero, but we are close to the root, just accept
				
				// Do additional iteration if we are less than 10x from stopping criterion
				if(abs2p < 0.01*stopping_crit2)
				{
					return;
				}
				else
				{
					good_to_go = TRUE;
				}
			}
			else
			{
				good_to_go = FALSE;
			}
			
			if(dp == zero)
			{
				//	Problem with zero
				dx = (abs(*root)+1.0) * exp( 0.0 + FRAC_JUMPS[(i%FRAC_JUMP_LEN)]*2.0*PI );	// make some random jump
			}
			else
			{
				dx = p / dp;	// Newton method, see http://en.wikipedia.org/wiki/Newton's_method
			}
			
			newroot = *root-dx;
			if(newroot == *root) return;	//nothing changes -> return
			if(good_to_go)	// This was jump already after stopping criterion was met
			{
				*root = newroot;
				return;
			}
			
			if((i%FRAC_JUMP_EVERY) == 0)
			{
				faq = FRAC_JUMPS[(i/FRAC_JUMP_EVERY-1)%(FRAC_JUMP_LEN)];
				newroot = *root - faq*dx;	// Do jump of some semi-random length (0<faq<1)
			}
			*root = newroot;
		}
		*success = FALSE;
		
		return;
	}

	/*
	 *-------------------------------------------------------------------*
	 * _6_                     CMPLX_LAGUERRE                            *
	 *-------------------------------------------------------------------*
	 *
	 * Subroutine finds one root of a complex polynomial using 
	 * Laguerre's method. In every loop it calculates simplified
	 * Adams' stopping criterion for the value of the polynomial.
	 *
	 * Uses 'root' value as a starting point (!!!!!)
	 * Remember to initialize 'root' to some initial guess or to 
	 * point (0,0) if you have no prior knowledge.
	 *
	 * poly - is an array of polynomial cooefs
	 *        length = degree+1, poly(1) is constant 
	 *               1              2             3
	 *          poly(1) x^0 + poly(2) x^1 + poly(3) x^2 + ...
	 * degree - a degree of the polynomial
	 * root - input: guess for the value of a root
	 *        output: a root of the polynomial
	 * iter - number of iterations performed (the number of polynomial
	 *        evaluations and stopping criterion evaluation)
	 * success - is false if routine reaches maximum number of iterations
	 *
	 * For a summary of the method go to: 
	 * http://en.wikipedia.org/wiki/Laguerre's_method
	 *
	 */
	__device__ void cmplx_laguerre(COMPLEXTYPE *poly, int degree, COMPLEXTYPE *root, int *iter, int *success)
	{
		int MAX_ITERS = 200;	// Laguerre is used as a failsafe 
		
		// Constants needed to break cycles in the scheme
		int FRAC_JUMP_EVERY = 10;
		int FRAC_JUMP_LEN = 10;
		
		double FRAC_JUMPS[10] = {0.64109297, 0.91577881, 0.25921289, 0.50487203, 0.08177045, 0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137};
		
		double faq;	// Jump length
		double FRAC_ERR = TOL_ERR;	// 2x10^-15
		
		COMPLEXTYPE p;			// value of polynomial
		COMPLEXTYPE dp;			// vale of 1st derivative
		COMPLEXTYPE d2p_half;	// vale of 2nd derivative
		
		COMPLEXTYPE denom, denom_sqrt, dx, newroot, fac_netwon, fac_extra, F_half, c_one_nth, zero, c_one;
		double ek, absroot, abs2p, one_nth, n_1_nth, two_n_div_n_1, stopping_crit2;
		int i, k;
		int good_to_go;
		
		zero = COMPLEXTYPE((double)0.0, (double)0.0);
		c_one = COMPLEXTYPE((double)1.0, (double)0.0);
		
		*iter = 0;
		*success = TRUE;
		
		// Next if-endif block is an EXTREME failsafe, not usually needed, and thus turned off in this version.
		if(FALSE)	// Change FALSE-->TRUE if you would like to use caution about having first coefficient == 0
		{
			if(degree < 0)
			{
				printf("Error: cmplx_laguerre: degree<0");
				return;
			}
			
			if(poly[degree] == zero)
			{
				if(degree == 0) return;
				cmplx_laguerre(poly, degree-1, root, iter, success);
				return;
			}
			
			if(degree <= 1)
			{
				if(degree == 0)
				{
					*success = FALSE;
					printf("Warning: cmplx_laguerre: degree == 0 and poly[0] != 0, no roots");
					return;
				}
				else
				{
					*root = -poly[0] / poly[1];
					return;
				}
			}
		}
		// end EXTREME failsafe    
		
		good_to_go = FALSE;
		one_nth = 1.0 / degree;
		n_1_nth = (degree-1.0)*one_nth;
		two_n_div_n_1=2.0/n_1_nth;
		c_one_nth = COMPLEXTYPE((double)one_nth, (double)0.0);
		
		
		for(i=1; i<=MAX_ITERS; i++)
		{
			// Prepare stoping criterion
			ek = abs(poly[degree]);
			absroot = abs(*root);
			
			// Calculate value of polynomial and its first two derivatives
			p = poly[degree];
			dp = zero;
			d2p_half = zero;
			for(k=degree; k>=1; k--)	// Horner Scheme, see for eg.  Numerical Recipes Sec. 5.3 how to evaluate polynomials and derivatives
			{
				d2p_half = dp + d2p_half*(*root);
				dp = p + dp*(*root);
				p = poly[k-1] + p*(*root);	// b_k
				
				/*
				 * Adams, Duane A., 1967, "A stopping criterion for polynomial root finding",
				 * Communications of the ACM, Volume 10 Issue 10, Oct. 1967, p. 655
				 * ftp://reports.stanford.edu/pub/cstr/reports/cs/tr/67/55/CS-TR-67-55.pdf
				 * Eq 8.
				 */
				ek = absroot*ek + abs(p);
			}
			*iter += 1;
			
			abs2p = real(conj(p) * p);	// abs(p)^2;
			if(abs2p == 0.0) return;
			
			stopping_crit2 = (FRAC_ERR*ek)*(FRAC_ERR*ek);
			if(abs2p < stopping_crit2)	// (simplified a little Eq. 10 of Adams 1967)
			{
				// Do additional iteration if we are less than 10x from stopping criterion
				if(abs2p < 0.01)
				{
					return;
				}
				else
				{
					good_to_go = TRUE;	// Do one iteration more
				}
			}
			else
			{
				good_to_go = FALSE;	// Reset if we are outside the zone of the root
			}
			
			faq = 1.0;
			
			fac_netwon = p / dp;
			fac_extra = d2p_half/dp;
			F_half = fac_netwon*fac_extra;
			
			denom_sqrt = sqrt(c_one - two_n_div_n_1*F_half);
			
			/*
			 * G=dp/p  ! gradient of ln(p)
			 * G2=G*G
			 * H=G2-2d0*d2p_half/p  ! second derivative of ln(p)
			 * denom_sqrt=sqrt( (degree-1)*(degree*H-G2) )
			 */
			
			//	NEXT LINE PROBABLY CAN BE COMMENTED OUT 
			if(real(denom_sqrt) >= 0.0)	// Use realpart
			{
				/*
				 * Real part of a square root is positive for probably all compilers. You can 
				 * test this on your compiler and if so, you can omit this check
				 */
				denom = c_one_nth + n_1_nth*denom_sqrt;
			}
			else
			{
				denom = c_one_nth - n_1_nth*denom_sqrt;
			}
			
			if(denom == zero)	// Test if demoninators are > 0.0 not to divide by zero
			{
				dx = (absroot + 1.0) * exp( COMPLEXTYPE((double)0.0, (double)FRAC_JUMPS[(i%FRAC_JUMP_LEN)]*2.0*PI) );	// Make some random jump
			}
			else
			{
				dx = fac_netwon / denom;
				//	dx = degree/denom;
			}
			
			newroot = *root-dx;
			if(newroot == *root) return;	// Nothing changes -> return
			if(good_to_go == TRUE)
			{
				*root = newroot;
				return;
			}
			
			if((i%FRAC_JUMP_EVERY) == 0)	// Decide whether to do a jump of modified length (to break cycles)
			{
				faq = FRAC_JUMPS[(i/FRAC_JUMP_EVERY-1)%(FRAC_JUMP_LEN)];
				//	write(*,'(3i5,f11.6)') i, i/FRAC_JUMP_EVERY, (i/FRAC_JUMP_EVERY-1)%FRAC_JUMP_LEN, faq			-------------------------------------------------------------------------------------------------------------- UNSURE?
				newroot = *root - faq*dx;	// Do jump of some semi-random length (0<faq<1)
			}
			*root = newroot;
		}
		*success = FALSE;
		//	Too many interations here
		
		return;
	}
	
	
	/*
	 *-------------------------------------------------------------------*
	 * _5_                     FIND_2_CLOSEST_FROM_5                     *
	 *-------------------------------------------------------------------*
	 *
	 * Returns indices of the two closest points out of array of 5
	 */
	__device__ void find_2_closest_from_5(int *i1, int *i2, double *d2min, COMPLEXTYPE *points)
	{
		int n = 5;
		
		COMPLEXTYPE p;
		double d2min1, d2;
		int i,j;
		
		d2min1 = LARGE_VAL;
		
		for(j=1; j<=n; j++)
		{
			for(i=1; i<=j-1; i++)
			{
				p = points[i-1] - points[j-1];
				d2 = real(conj(p)*p);
				if(d2 <= d2min1)
				{
					*i1 = i;
					*i2 = j;
					*d2min = d2;
				}
			}
		}
		*d2min = d2min1;
		
		return;
	}


	/*
	 *-------------------------------------------------------------------*
	 * _4_             SORT_5_POINTS_BY_SEPARATION_I                     *
	 *-------------------------------------------------------------------*
	 *
	 * Return index array that sorts array of five points 
	 * Index of the most isolated point will appear on the first place 
	 * of the output array.
	 * The indices of the closest 2 points will be at the last two 
	 * places in the 'sorted_points' array
	 * 
	 * Algorithm works well for all dimensions. We put n=5 as 
	 * a hardcoded value just for optimization purposes.

	*/
	__device__ void sort_5_points_by_separation_i(int *sorted_points, COMPLEXTYPE *points)
	{
		int n = 5;
		
		COMPLEXTYPE p;
		
		double distances2[25];	// A 5x5 array flattened into 1D
		double neigh1st[5], neigh2nd[5];
		
		double d1, d2, d;
		int ki, kj, ind2, put;
		
		for(kj=1; kj<=n; kj++)
		{
			distances2[ kj + 5*(kj-1) -1 ] = LARGE_VAL;
		}
		
		for(kj=1; kj<=n; kj++)
		{
			for(ki=1; ki<=kj-1; ki++)
			{
				p = points[ki-1] - points[kj-1];
				d = real(conj(p) * p);
				distances2[ ki + 5*(kj-1) -1 ] = d;
				distances2[ kj + 5*(ki-1) -1 ] = d;
			}
		}
		
		
		// Find neighbours
		for(kj=1; kj<=n; kj++)
		{
			neigh1st[kj-1] = LARGE_VAL;
			neigh2nd[kj-1] = LARGE_VAL;
		}
		
		for(kj=1; kj<=n; kj++)
		{
			for(ki=1; ki<=n; ki++)
			{
				d = distances2[ kj + 5*(ki-1) -1 ];
				
				if(d < neigh2nd[kj-1])
				{
					if(d < neigh1st[kj-1])
					{
						neigh2nd[kj-1] = neigh1st[kj-1];
						neigh1st[kj-1] = d;
					}
					else
					{
						neigh2nd[kj-1] = d;
					}
				}
			}
		}
		
		// Initialize sorted_points
		for(ki=1; ki<=n; ki++)
		{
			sorted_points[ki-1] = ki;
		}
		
		// Sort the rest 1..n-2
		for(kj=2; kj<=n; kj++)
		{
			d1 = neigh1st[kj-1];
			d2 = neigh2nd[kj-1];
			put = 1;
			for(ki=kj-1; ki>=1; ki--)
			{
				ind2 = sorted_points[ki-1];
				d = neigh1st[ind2-1];
				if(d >= d1)
				{
					if(d == d1)
					{
						if(neigh2nd[ind2-1] > d2)
						{
							put = ki+1;
							break;
						}
					}
					else
					{
						put = ki+1;
						break;
					}
				}
				sorted_points[ki] = sorted_points[ki-1];
			}
			sorted_points[put-1] = kj;
		}
		
		
		return;
	}


	/*
	 *-------------------------------------------------------------------*
	 * _3_                 SORT_5_POINTS_BY_SEPARATION                   *
	 *-------------------------------------------------------------------*
	 * Sort array of five points 
	 * Most isolated point will become the first point in the array
	 * The closest points will be the last two points in the array
	 * 
	 * Algorithm works well for all dimensions. We put n=5 as 
	 * a hardcoded value just for optimization purposes.
	 */
	__device__ void sort_5_points_by_separation(COMPLEXTYPE *points)
	{
		int n=5;
		
		int sorted_points[5];
		COMPLEXTYPE savepoints[5];
		int i;
		
		sort_5_points_by_separation_i(sorted_points, points);
		
		for(i=1; i<=n; i++)
		{
			savepoints[i-1] = points[i-1];
		}
		for(i=1; i<=n; i++)
		{
			points[i-1] = savepoints[sorted_points[i-1]-1];
		}
		
		
		return;
	}


	/*
	 *-------------------------------------------------------------------*
	 * _2_                     CMPLX_ROOTS_5                             *
	 *-------------------------------------------------------------------*
	 * Subroutine finds or polishes roots of a complex polynomial 
	 * (degree=5)
	 * This routine is especially tailored for solving binary lens 
	 * equation in form of 5th order polynomial. 
	 * 
	 * Use of this routine, in comparison to 'cmplx_roots_gen' can yield
	 * considerably faster code, because it makes polishing of the roots 
	 * (that come in as a guess from previous solutions) secure by
	 * implementing additional checks on the result of polishing. 
	 * If those checks are not satisfied then routine reverts to the 
	 * robust algorithm. These checks are designed to work for 5th order 
	 * polynomial originated from binary lens equation.
	 * 
	 * Usage:
	 * 
	 * polish_only == false - I do not know the roots, routine should  
	 *                find them from scratch. At the end it
	 *                sorts roots from the most distant to closest.
	 *                Two last roots are the closest (in no particular
	 *                order).
	 * polish_only = true - I do know the roots pretty well, for example
	 *                I have changed the coefficients of the polynomial 
	 *                only a bit, so the two closest roots are 
	 *                most likely still the closest ones.
	 *                If the output flag 'first_3_roots_order_changed'
	 *                is returned as 'false', then first 3 returned roots
	 *                are in the same order as initially given to the 
	 *                routine. The last two roots are the closest ones, 
	 *                but in no specific order (!).
	 *                If 'first_3_roots_order_changed' is 'true' then
	 *                it means that all roots had been resorted.
	 *                Two last roots are the closest ones. First is most 
	 *                isolated one.
	 * 
	 * 
	 * If you do not know the position of the roots just use flag
	 * polish_only=.false. In this case routine will find the roots by
	 * itself.
		  
	 * Returns all five roots in the 'roots' array.
	 * 
	 * poly  - is an array of polynomial cooefs, length = degree+1 
	 *       poly(1) x^0 + poly(2) x^1 + poly(3) x^2 + poly(4) x^3 + ...
	 * roots - roots of the polynomial ('out' and optionally 'in')
	 * 
	 * 
	 * Jan Skowron 2011
	 *
	 */
	__device__ void cmplx_roots_5(COMPLEXTYPE *roots, int *first_3_roots_order_changed, COMPLEXTYPE *poly, int polish_only)
	{
		int degree = 5;
		
		COMPLEXTYPE remainder, zero, roots_robust[5], poly2[6];
		double d2min;
		int iter, m, root4, root5, kk, go_to_robust, i, i2, loops;
		int succ;
		int Goto_flag;
		
		zero = COMPLEXTYPE((double)0.0, (double)0.0);
		
		go_to_robust = 0;
		if(polish_only == FALSE)
		{
			for(kk=1; kk<=degree; kk++)
			{
				roots[kk-1] = zero;
			}
			go_to_robust = 1;
		}
		*first_3_roots_order_changed = FALSE;
		
		loops = 1;
		while(loops <= 3)
		{
			Goto_flag = FALSE;
			/*
			 * ROBUST
			 * (we do not know the roots)
			 */
			if(go_to_robust > 0)
			{
				if(go_to_robust > 2)	// Something is wrong
				{
					for(kk=1; kk<=degree; kk++)
					{
						roots[kk-1] = roots_robust[kk-1];
					}
					return;
				}
				
				for(kk=1; kk<=degree+1; kk++)
				{
					poly2[kk-1] = poly[kk-1];	// Copy coeffs
				}
				
				for(m=degree; m>=4; m--)	// Find the roots one-by-one (until 3 are left to be found)
				{
					cmplx_laguerre2newton(poly2, m, &roots[m-1], &iter, &succ, 2);
					if(succ == FALSE)
					{
						roots[m-1] = zero;
						cmplx_laguerre(poly2, m, &roots[m-1], &iter, &succ);
					}
					
					// Divide polynomial by this root
					divide_poly_1(poly2, &remainder, roots[m-1], poly2, m);
				}
					
				// Find last 3 roots with cubic euqation solver (Lagrange's method)
				solve_cubic_eq(&roots[0], &roots[1], &roots[2], poly2);
				// All roots found
				
				// Sort roots - first will be most isolated, last two will be the closest
				sort_5_points_by_separation(roots);
				
				// Copy roots in case something goes wrong during polishing
				for(kk=1; kk<=degree; kk++)
				{
					roots_robust[kk-1] = roots[kk-1];
				}
				
				// Set flag, that roots have been resorted
				*first_3_roots_order_changed = TRUE;
			}
			
			/*
			 * POLISH 
			 * (we know the roots approximately, and we guess that last two are closest)
			 *---------------------
			 */
			
			for(kk=1; kk<=degree+1; kk++)
			{
				poly2[kk-1] = poly[kk-1];	// Copy coeffs
			}
			
			for(m=1; m<=degree-2; m++)
			{
				// for(m=1; m<=degree; m++)                      // POWN - polish only with Newton (option)

				// polish roots with full polynomial
				cmplx_newton_spec(poly2, degree, &roots[m-1], &iter, &succ);
				
				if(succ == FALSE)
				{
					// go back to robust
					go_to_robust += 1;
					for(kk=1; kk<=degree; kk++)
					{
						roots[kk-1] = zero;
					}
					// Go back
					Goto_flag = TRUE;
					break;
				}
			}
			if(Goto_flag == TRUE)
			{
				continue;
			}
			
			// Comment out division and quadratic if you (POWN) polish with Newton only
			for(m=1; m<=degree-2; m++)
			{
				divide_poly_1(poly2, &remainder, roots[m-1], poly2, degree-m+1);
			}
			
			// Last two roots are found with quadratic equation solver
			// (this is faster and more robust, although little less accurate)
			solve_quadratic_eq(&roots[degree-2], &roots[degree-1], poly2);
			
			// All roots found and polished
			
			/*
			 * TEST ORDER
			 * test closest roots if they are the same pair as given to polish
			 */
			
			find_2_closest_from_5(&root4, &root5, &d2min, roots);
			
			/*
			 * check if the closest roots are not too close, this could happen
			 * when using polishing with Newton only, when two roots erroneously 
			 * colapsed to the same root. This check is not needed for polishing
			 * 3 roots by Newton and using quadratic for the remaining two.
			 * If the real roots are so close indeed (very low probability), this will just 
			 * take more time and the unpolished result be returned at the end
			 * if(d2min < 1d-18)		// POWN - polish only with Newton 
			 * {						// POWN - polish only with Newton
			 * 	go_to_robust += 1		// POWN - polish only with Newton
			 * }						// POWN - polish only with Newton
			 * else 					// POWN - polish only with Newton
			 * {						// POWN - polish only with Newton
			 */
			
			if( (root4 < degree-1) || (root5 < degree-1) )
			{
				// After polishing some of the 3 far roots become one of the 2 closest ones
				// go back to robust
				if(go_to_robust > 0)
				{
					// If came from robust 
					// copy two most isolated roots as starting points for new robust
					for(i=1; i<=degree-3; i++)
					{
						roots[degree-i] = roots_robust[i-1];
					}
				}
				else
				{
					// Came from users initial guess
					// copy first 2 roots (except the closest ones)
					i2 = degree;
					for(i=1; i<=degree; i++)
					{
						if( (i != root4) && (i != root5) )
						{
							roots[i2-1] = roots[i-1];
							i2 -= 1;
						}
						if(i2 < 3) break;
					}
					
				}
				go_to_robust += 1;
			}
			else
			{
				// root4 and root5 comes from the initial last pair
				// most common case
				return;
			}
			
			/*
			 * }						// POWN - polish only with Newton
			 */
			
			loops += 1;
		}
		
		return;
	}


//////////////////////// Faster zoroots - end



//////////////////////////////////////////////////////////////	Sort comparisons
	__device__ void insertion_sort(double *a, COMPLEXTYPE *b, int n)
	{
		int k;
		for (k = 1; k < n; ++k)
		{
			double key = a[k];
			COMPLEXTYPE key2 = b[k];
			int i = k - 1;
			while ((i >= 0) && (key < a[i]))
			{
				a[i + 1] = a[i];
				b[i + 1] = b[i];
				--i;
			}
			a[i + 1] = key;
			b[i + 1] = key2;
		}
	}
//////////////////////// Sort comparisons - end

//////////////////////////////////////////////////////////////	Magnification Calculations
// 

	__global__ void point_source_magnification(double e1, double e2, int n_points, double *a_in, double *u1, double *u2, double *A)
	{

		double xpos, ypos, a;
		int first_3_roots_order_changed = TRUE;
		
		int index = threadIdx.x + blockIdx.x * blockDim.x; 

		if (index < n_points) {

			xpos = u1[index];
			ypos = u2[index];
			a = a_in[index];
			
		// only calcuate half of the u2 coordiantes, as the map is symetrical it can be mirrored.
			int n, j, polish;
			
		// Number of coefficients in the polynomial to solve.
			int m = NUM_POLYS;
			double detj[NUM_POLYS], test_ans[NUM_POLYS-1];
			
		// COMPLEXTYPE is a special data type from pycuda to help utilise complex numbers in CUDA
			COMPLEXTYPE coef5, coef4, coef3, coef2, coef1, coef0, zetabar, zeta;
			COMPLEXTYPE  z[NUM_POLYS], zeta_test[NUM_POLYS-1], zbar[NUM_POLYS-1], as[NUM_POLYS];
			
		//	Determine the complex conjugate of the source position
			zeta = COMPLEXTYPE(xpos, ypos);
			zetabar = COMPLEXTYPE(real(zeta), -imag(zeta));
						
		//	Determine the coefficients of the 5th order polynomial
			as[5] = zetabar*zetabar - a*a;
			as[4] = -zeta*pow(zetabar,2) - (double)2*e1*a + zetabar + zeta*pow(a,2) + a;
			as[3] = (double)4*e1*a*zetabar - (double)2*pow(zetabar,2)*pow(a,2) - (double)2*a*zetabar - (double)2*zeta*zetabar + (double)2*pow(a,4);
			as[2] = -(double)2*zeta*pow(a,4) - (double)4*zeta*e1*a*zetabar + (double)2*e1*a + 4*pow(a,3)*e1 + (double)2*zeta*pow(zetabar,2)*pow(a,2) - zeta - a - (double)2*pow(a,3) + (double)2*zeta*a*zetabar;
			as[1] = (double)2*zeta*zetabar*pow(a,2) - (double)4*zeta*e1*a + pow(zetabar,2)*pow(a,4) - (double)4*e1*pow(a,2) - pow(a,6) + (double)2*pow(a,2) + (double)4*pow(e1,2)*pow(a,2) + (double)2*zeta*a + (double)2*zetabar*pow(a,3) - (double)4*e1*pow(a,3)*zetabar;
			as[0] = -zeta*pow(a,2) + zeta*pow(a,6) + pow(a,5) - (double)4*zeta*pow(e1,2)*pow(a,2) - pow(a,3) + (double)4*zeta*zetabar*pow(a,3)*e1 + (double)4*zeta*e1*pow(a,2) - zeta*pow(zetabar,2)*pow(a,4) - pow(a,4)*zetabar - (double)2*zeta*pow(a,3)*zetabar + (double)2*pow(a,3)*e1 - (double)2*e1*pow(a,5);
				
			cmplx_roots_5(z, &first_3_roots_order_changed, as, FALSE);
				
		//	Determine the complex conjugate of the images positions
			n = 0;
			for (j=0;j<m-1;j++)
			{	
				zbar[j] = COMPLEXTYPE(real(z[j]), -imag(z[j]));
				
			//	Test the possible solutions
				zeta_test[j] = z[j] - (e1 / (zbar[j]-a)) - (e2 / (zbar[j]+a));
				test_ans[j] = abs(zeta_test[j] - zeta);
			}
				
			insertion_sort(test_ans, zbar, m-1);
				
			if (test_ans[m-2] < 0.0001)
			{
				n = 5;
			}
			else
			{
				n = 3;
			}
				
			// Initialise the magnification
			A[index] = 0;
				
			for (j=0;j<n;j++)
			{	
			//	Determine the magnification contribution that the real image solutions cause.
				detj[j] = 1 - abs( pow((e1 / pow(zbar[j] - a , 2)) +  (e2 / pow(zbar[j] + a , 2)) , 2) );
					
				//	Sum the magnification contributions from all real image solutions of a given source position.
				A[index] += 1/abs(detj[j]);
			}
		
		}
		__syncthreads();

		return;
	}
	
//////////////////////// Magnification - end



//////////////////////////////////////////////////////////////	Magnification Calculations
// Produce a magnification map

	__global__ void magnification(double e1, double e2, double a, double dx, double x0, double dy, double y0, double *A)
	{
		int first_3_roots_order_changed = TRUE;
		
	// using thread ids, determine the x-position (the u1 coordinate)	
		int xpos = threadIdx.x + (blockIdx.y * blockDim.x);
		
	// using block ids, determine the y-position (the u2 coordinate)
		int ypos = blockIdx.x;
		
	// only calcuate half of the u2 coordiantes, as the map is symetrical it can be mirrored.
		int n, j, polish;
		
	// Number of coefficients in the polynomial to solve.
		int m = NUM_POLYS;
		double detj[NUM_POLYS], test_ans[NUM_POLYS-1];
		
	// COMPLEXTYPE is a special data type from pycuda to help utilise complex numbers in CUDA
		COMPLEXTYPE coef5, coef4, coef3, coef2, coef1, coef0, zetabar, zeta;
		COMPLEXTYPE  z[NUM_POLYS], zeta_test[NUM_POLYS-1], zbar[NUM_POLYS-1], as[NUM_POLYS];
		
	//	If xpos outside of dimensions then skip
		if (xpos < 2*gridDim.x)
		{
		//	Determine the complex conjugate of the source position
			zeta = COMPLEXTYPE(x0 + dx*xpos, y0 + dy*ypos);
			zetabar = COMPLEXTYPE(real(zeta), -imag(zeta));
					
		//	Determine the coefficients of the 5th order polynomial
			as[5] = zetabar*zetabar - a*a;
			as[4] = -zeta*pow(zetabar,2) - (double)2*e1*a + zetabar + zeta*pow(a,2) + a;
			as[3] = (double)4*e1*a*zetabar - (double)2*pow(zetabar,2)*pow(a,2) - (double)2*a*zetabar - (double)2*zeta*zetabar + (double)2*pow(a,4);
			as[2] = -(double)2*zeta*pow(a,4) - (double)4*zeta*e1*a*zetabar + (double)2*e1*a + 4*pow(a,3)*e1 + (double)2*zeta*pow(zetabar,2)*pow(a,2) - zeta - a - (double)2*pow(a,3) + (double)2*zeta*a*zetabar;
			as[1] = (double)2*zeta*zetabar*pow(a,2) - (double)4*zeta*e1*a + pow(zetabar,2)*pow(a,4) - (double)4*e1*pow(a,2) - pow(a,6) + (double)2*pow(a,2) + (double)4*pow(e1,2)*pow(a,2) + (double)2*zeta*a + (double)2*zetabar*pow(a,3) - (double)4*e1*pow(a,3)*zetabar;
			as[0] = -zeta*pow(a,2) + zeta*pow(a,6) + pow(a,5) - (double)4*zeta*pow(e1,2)*pow(a,2) - pow(a,3) + (double)4*zeta*zetabar*pow(a,3)*e1 + (double)4*zeta*e1*pow(a,2) - zeta*pow(zetabar,2)*pow(a,4) - pow(a,4)*zetabar - (double)2*zeta*pow(a,3)*zetabar + (double)2*pow(a,3)*e1 - (double)2*e1*pow(a,5);
			
			cmplx_roots_5(z, &first_3_roots_order_changed, as, FALSE);
			
		//	Determine the complex conjugate of the images positions
			n = 0;
			for (j=0;j<m-1;j++)
			{	
				zbar[j] = COMPLEXTYPE(real(z[j]), -imag(z[j]));
				
			//	Test the possible solutions
				zeta_test[j] = z[j] - (e1 / (zbar[j]-a)) - (e2 / (zbar[j]+a));
				test_ans[j] = abs(zeta_test[j] - zeta);
			}
			
			insertion_sort(test_ans, zbar, m-1);
			
			if (test_ans[m-2] < 0.0001)
			{
				n = 5;
			}
			else
			{
				n = 3;
			}
			
		//	Initialise the mag. at the maps pixel to be zero
			A[xpos + ypos*2*(gridDim.x)] = 0;
			
		//	Mirror this operation across the symmetrical axis
			A[(2*gridDim.x)*(2*gridDim.x) - (ypos+1)*(2*gridDim.x) + xpos] = 0;
						
			for (j=0;j<n;j++)
			{	
			//	Determine the magnification contribution that the real image solutions cause.
				detj[j] = 1 - abs( pow((e1 / pow(zbar[j] - a , 2)) +  (e2 / pow(zbar[j] + a , 2)) , 2) );
				
			//	Sum the magnification contributions from all real image solutions of a given source position.
				A[xpos + ypos*2*(gridDim.x)] += 1/abs(detj[j]);
			}
			
		//	Mirror this mag. value to the corresponding symmetrical pixel
			A[(2*gridDim.x)*(2*gridDim.x) - (ypos+1)*(2*gridDim.x) + xpos] =  A[xpos + ypos*2*(gridDim.x)];
		}
		
		__syncthreads();

		return;
	}
	
	//////////////////////// Magnification - end
		
	#undef EPS
	#undef MAXM
	#undef EPSS
	#undef MR
	#undef MT
	#undef MAXIT
	#undef NRANSI
	
	""")




