
import pycuda.driver as drv
import pycuda.compiler
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule


gpu_image_centred=SourceModule("""
	#include <pycuda-complex.hpp>
	#include <stdio.h>
	#include <math.h>
	#include <float.h>
	#define	COMPLEXTYPE	pycuda::complex<double>
	
	#include "texture_fetch_functions.h"
	#include "texture_types.h"
	
	#define NRANSI																																																																									// Used by Numerical Recipies (NR)
	#include "/home/scratch/users/alm167/BinaryLensFitter/Code/nrutil.h"																																																								// 
	#define EPSS 1.0e-7	
	#define H_ERR 1.0e-7																																																																							// For NR ZRoots
	#define MR 8																																																																											// v
	#define MT 10																																																																										// v
	#define MAXIT (MT*MR)																																																																				// v
	#define EPS 2.0e-6																																																																								// v
	#define MAXM 100																																																																								// v
	#define	DELTA_C	0.15
	#define	MAX_GRID_DIM	512																																																																																	// For NR Zroots
	#define	NUM_POLYS	6																																																																					// v
	#define	MAX_THREADS 512				// Must be a value of 2^n																																																																// v
	#define	MAX_DATA_POINTS 8192																																																																				// v
	
	#define	MAX_DATA_SITES	15
	
	// For zroots
	#define	FALSE	0
	#define	TRUE	1
	#define	LARGE_VAL	1e100
	#define	PI			3.141592653589793
	#define TOL_ERR		2e-15

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
					 * 	newroot=root-faq*dx ! do jump of some semi-random length (0<faq<1)
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


	//////////////////////////////////////////////////////////////////////	Sort comparisons
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

	//////////////////////////////////////////////////////////////////////	Parallel summation

	__device__ void parallel_sum(double *A_local)
	{
				
		if ((int)(blockDim.x) >= 512)
		{ 
			if (threadIdx.x < 256)
			{
				A_local[threadIdx.x] += A_local[threadIdx.x + 256];
			}
		}

		__syncthreads();
		
		if ((int)(blockDim.x) >= 256)
		{ 
			if (threadIdx.x < 128)
			{
				A_local[threadIdx.x] += A_local[threadIdx.x + 128];
			}
		}

		__syncthreads();
		
		if ((int)(blockDim.x) >= 128)
		{ 
			if (threadIdx.x < 64)
			{
				A_local[threadIdx.x] += A_local[threadIdx.x + 64];
			}
		}

		__syncthreads();
		
		if (threadIdx.x < 32)
		{
			if (blockDim.x >= 64) A_local[threadIdx.x] += A_local[threadIdx.x + 32];
			if (blockDim.x >= 32) A_local[threadIdx.x] += A_local[threadIdx.x + 16];
			if (blockDim.x >= 16) A_local[threadIdx.x] += A_local[threadIdx.x + 8];
			if (blockDim.x >= 8)  A_local[threadIdx.x] += A_local[threadIdx.x + 4];
			if (blockDim.x >= 4)  A_local[threadIdx.x] += A_local[threadIdx.x + 2];
			if (blockDim.x >= 2)  A_local[threadIdx.x] += A_local[threadIdx.x + 1];
		}
		
		__syncthreads();

	}


///////////  Map polar coordinates from (b1,0) to (b2,0)

	__device__ void map_coords(double *r,double *theta,double b,double *r1,double *theta1,double b1)
	{
		double x = *r * cos(*theta) + b;
		double y = *r * sin(*theta);
		x -= b1;
		*r1 = sqrt(x*x + y*y);
		*theta1 = atan2(y,x);
	}

///////////  Return 1.0 if ray at (r,theta) is inside source at (u1,u2) with radius rho.
///////////  Otherwise return 0.0

	__device__ double check_ray_inside_source(double r, double theta, double e1, double e2, double a, double b,
												double u1, double u2, double rho)
	{

		COMPLEXTYPE zeta, zetabar, omega;

		zeta = COMPLEXTYPE( r*cos(theta), r*sin(theta) );
		zetabar = COMPLEXTYPE(real(zeta), -imag(zeta));
					
		//	Determine the inverse ray position
		omega = zeta - ( (e1) / (zetabar-(a-b)) )   -  ( (e2) / (zetabar+(a+b)) );
					
		//	Calculate the distance between the ray and the source centre
		//	If inside source return 1.0, otherwise 0.0
		if (  (u1-real(omega))*(u1-real(omega)) + (u2-imag(omega))*(u2-imag(omega))  <  rho*rho  )
		{
			return 1.0;
		}
		return 0.0;
	}

//////////  Print a double array of length blockDim.x

	__device__ void print_array(double *A)
	{
		for (int k=0; k<blockDim.x; k++)
		{
			printf("%f ",A[k]);
		}
		printf("\\n");		
	}


/////////  Point source magnification

	__device__ double point_source(double u1,double u2,double e1,double e2,double a)
	{

		int 		 first_3_roots_order_changed = TRUE;
		int          j, n;
		double       test_ans[5], mag, det;
		COMPLEXTYPE  zeta, zetabar, z[6], zbar[5], zeta_test[5], as[6];

		zeta = COMPLEXTYPE(u1, u2);
		zetabar = COMPLEXTYPE(real(zeta), -imag(zeta));
		
	//	Determine the coefficients of the 5th order polynomial
		as[5] = zetabar*zetabar - a*a;
		as[4] = -zeta*pow(zetabar,2) - (double)2*e1*a + zetabar + zeta*pow(a,2) + a;
		as[3] = (double)4*e1*a*zetabar - (double)2*pow(zetabar,2)*pow(a,2) - (double)2*a*zetabar - (double)2*zeta*zetabar + (double)2*pow(a,4);
		as[2] = -(double)2*zeta*pow(a,4) - (double)4*zeta*e1*a*zetabar + (double)2*e1*a + 4*pow(a,3)*e1 + (double)2*zeta*pow(zetabar,2)*pow(a,2) - zeta - a - (double)2*pow(a,3) + (double)2*zeta*a*zetabar;
		as[1] = (double)2*zeta*zetabar*pow(a,2) - (double)4*zeta*e1*a + pow(zetabar,2)*pow(a,4) - (double)4*e1*pow(a,2) - pow(a,6) + (double)2*pow(a,2) + (double)4*pow(e1,2)*pow(a,2) + (double)2*zeta*a + (double)2*zetabar*pow(a,3) - (double)4*e1*pow(a,3)*zetabar;
		as[0] = -zeta*pow(a,2) + zeta*pow(a,6) + pow(a,5) - (double)4*zeta*pow(e1,2)*pow(a,2) - pow(a,3) + (double)4*zeta*zetabar*pow(a,3)*e1 + (double)4*zeta*e1*pow(a,2) - zeta*pow(zetabar,2)*pow(a,4) - pow(a,4)*zetabar - (double)2*zeta*pow(a,3)*zetabar + (double)2*pow(a,3)*e1 - (double)2*e1*pow(a,5);
		
		cmplx_roots_5(z, &first_3_roots_order_changed, as, FALSE);

		for (j=0; j<5; j++)
		{	
			zbar[j] = COMPLEXTYPE(real(z[j]), -imag(z[j]));
			zeta_test[j] = z[j] - (e1 / (zbar[j]-a)) - (e2 / (zbar[j]+a));
			test_ans[j] = abs(zeta_test[j] - zeta);
		}

		insertion_sort(test_ans, zbar, 5);

	//	Determine how many real images there are
		if (test_ans[4] < 0.0001)
		{
			n = 5;
		}
		else
		{
			n = 3;
		}

		mag = 0.0;

		for (j=0;j<n;j++)
		{	
			det = 1 - abs( pow((e1 / pow(zbar[j] - a , 2)) +  (e2 / pow(zbar[j] + a , 2)) , 2) );
			mag += 1/abs(det);
		}
		
		return mag;

	}


/////////   Hexadecapole approximation

	__device__ void hexadecapole(double u1, double u2, double rho, double s, double q, double Gamma, double *mag)
	{

		double a, e1, e2, A1x, A1p, A2x, A2p, A2rho2, A4rho4;
		int    j;
		__shared__ double A[13];

		e2 = 1.0/(1.0+q);
		e1 = q/(1.0+q);
		a = 0.5*s;

		mag[threadIdx.x] = 0.0;

		if ((threadIdx.x > 0) && (threadIdx.x < 5))
		{
			u1 += rho*cos((threadIdx.x-1)*PI*0.5);
			u2 += rho*sin((threadIdx.x-1)*PI*0.5);
		}

		if ((threadIdx.x >= 5) && (threadIdx.x < 9))
		{
			u1 += rho*cos((threadIdx.x-5)*PI*0.5+PI*0.25);
			u2 += rho*sin((threadIdx.x-5)*PI*0.5+PI*0.25);
		}

		if ((threadIdx.x >= 9) && (threadIdx.x < 13))
		{
			u1 += 0.5*rho*cos((threadIdx.x-9)*PI*0.5);
			u2 += 0.5*rho*sin((threadIdx.x-9)*PI*0.5);
		}

		if (threadIdx.x < 13)
		{
			A[threadIdx.x] = point_source(u1, u2, e1, e2, a);
		}
		__syncthreads();

		if (threadIdx.x == 0)
		{

			A1p = 0.0;
			for (j=1; j<5; j++)
			{
				A1p += A[j];
			}
			A1p *= 0.25;
			A1p -= A[0];

			A1x = 0.0;
			for (j=5; j<9; j++)
			{
				A1x += A[j];
			}
			A1x *= 0.25;
			A1x -= A[0];

			A2p = 0.0;
			for (j=9; j<13; j++)
			{
				A2p += A[j];
			}
			A2p *= 0.25;
			A2p -= A[0];

			A2rho2 = (16.0*A2p - A1p)/3.0;
			A4rho4 = (A1p + A1x)/2.0 - A2rho2;

			mag[threadIdx.x] = A[0] + A2rho2*(1.0-Gamma/5.0)/2.0 + A4rho4*(1.0-11.0*Gamma/35.0)/3.0;
		}
		__syncthreads();

		return;

	}

	//////////////////////////////////////////////////////////////////////	Magnification at the data points

	__device__ void error(int j, double d, double q, double rho, double  u1, double u2)
	{
		printf("Trap %d triggered \\n",j);
		printf("block, thread, d, q, rho, u1, u2: %d %d %g %g %g %g %g\\n", blockIdx.x, threadIdx.x, d, q, rho, u1, u2);
		__syncthreads();
		return;
	}


	//////////////////////////////////////////////////////////////////////	Magnification at the data points

	
    __global__ void data_magnifications(double q, double rho, double limb_const, int ratio, double hexadecapole_threshold, int n_caustic, double *caustic_x, double *caustic_y, double *d, double *u1_in, double* u2_in, double *A)
	{
		__shared__ double A_local[5*MAX_GRID_DIM+10], A_row[5*MAX_GRID_DIM+10], A_col[MAX_GRID_DIM+2], img_grid_dims[20], A_img[5], r_imgs[5], 
							theta_imgs[5], u1_source, u2_source, ang_loc[2], rad_loc[2], boundary[2], outer_boundary[2], boundary_inside_edge[2], boundary_limb_edge[2], 
							A_limb[2], inner_loc[2], outer_loc[2], left_loc[2], right_loc[2], slope, range, mid, prev_mid, prev_theta_local, prev_r_local,
							max_range, left_mid, right_mid, ang_loc_0_save, b[5], b_com, b_old, max_theta_step;
		__shared__ int INVERSE_SOLVE_FLAG, GRID_DIM_FLAG, CAUSTIC_CROSSED, RING_FLAG, RING_FLAG_a, RING_FLAG_b, n_all, boundary_inside_edge_loc[2], 
						COLOUMN_FLAG, index_max, iteration1, iteration2, repeat_loop, image_found, iterate_coordinates,CHANGED_COORDS_FLAG,STOP;
		int n, j, k, l, l1, polish, index, closest_pair, start_point, loop_until, parity[5],loop_index;
		int m = NUM_POLYS;
		int first_3_roots_order_changed = TRUE;
		int found;
		double zr[NUM_POLYS], zi[NUM_POLYS], detj[NUM_POLYS], test_ans[NUM_POLYS-1], closest_sols[11];
		double e1, e2, a, u0n, t0n, u1, u2, r1, b1;
		double r_local, r_inner, r_outer, theta_local, ang_shift, rad_shift, prev_shift, h, eta_gs, grid_size, oversample_factor;
		double x_left, x_right, x_centre, y_left, y_right, y_centre;
		COMPLEXTYPE coef5, coef4, coef3, coef2, coef1, coef0, zetabar, zeta, omega;
		COMPLEXTYPE  z[NUM_POLYS], zeta_test[NUM_POLYS-1], zbar[NUM_POLYS-1], as[NUM_POLYS];
		

	//	Determine constants which describe the magnificat map
		e2 = 1.0/(1.0+q);
		e1 = q/(1.0+q);
		a = 0.5*d[blockIdx.x];
		//b = -(a*(q-1.0)) / (1.0+q);
		if (threadIdx.x == 0)
		{
			b[0] = b[1] = b[2] = b[3] = b[4] = -(a*(1.0-q)) / (1.0+q);   // note b is -ve in this definition
			b_com = b[0];
		}
		__syncthreads();

	//	Converthe the u0 and t0 to equal between the masses co-ordinate system
		//u0n = (double)( u0 - sin(phi)*(b) );																																																																// Calculate the new u0 for the middle of mass coordiante system
		//t0n = (double)( t0 - tE*((u0)/tan(phi) - u0n/tan(phi)) );	
		
	//	Determine the source position in the above co-ordiante system
		//u1 = (double)( ((ts[blockIdx.x] - t0n)/tE)*cos(phi) - u0n*sin(phi) );																																																	// Determine the u1 coordinates on the mag. map
		//u2 = (double)( ((ts[blockIdx.x] - t0n)/tE)*sin(phi) + u0n*cos(phi) );
		
		u1 = u1_in[blockIdx.x];
		u2 = u2_in[blockIdx.x];
			
		if (rho < 0.001)
		{
			r_local = 0.001;
		}
		else
		{
			r_local = rho;
		}
		
		
		n_all = 3;
		INVERSE_SOLVE_FLAG = 0;
		GRID_DIM_FLAG = 1;
		CAUSTIC_CROSSED = 0;

	//	Shift the co-ordinates for each thread to put them on the circumference (except the first)
		if (threadIdx.x > 0)
		{
			u1 += rho*cos((threadIdx.x) * 2.0*PI/(blockDim.x - 1));
			u2 += rho*sin((threadIdx.x) * 2.0*PI/(blockDim.x - 1));
		}

		__syncthreads();

	// Test if we require image-centred ray shooting, otherwise return hexadecapole approximation.

		if (threadIdx.x == 0)
		{
			STOP = 1;
		}
		__syncthreads();

		for (j=0; j<n_caustic; j++)
		{
			if ((caustic_x[j]-u1)*(caustic_x[j]-u1) + (caustic_y[j]-u2)*(caustic_y[j]-u2) < rho*rho*hexadecapole_threshold*hexadecapole_threshold)
			{
				STOP = 0;
			}
		}
		__syncthreads();

		if (STOP)
		{
			hexadecapole(u1_in[blockIdx.x], u2_in[blockIdx.x], rho, d[blockIdx.x], q, limb_const, A_local);

			if (threadIdx.x == 0)
			{
				A[blockIdx.x] = A_local[0];
			}
			__syncthreads();
			return;
		}


	//	Determine the image positions for each threads source position
	//	Determine the complex conjugate of the source position
		zeta = COMPLEXTYPE(u1, u2);
		zetabar = COMPLEXTYPE(real(zeta), -imag(zeta));
		
	//	Determine the coefficients of the 5th order polynomial
		as[5] = zetabar*zetabar - a*a;
		as[4] = -zeta*pow(zetabar,2) - (double)2*e1*a + zetabar + zeta*pow(a,2) + a;
		as[3] = (double)4*e1*a*zetabar - (double)2*pow(zetabar,2)*pow(a,2) - (double)2*a*zetabar - (double)2*zeta*zetabar + (double)2*pow(a,4);
		as[2] = -(double)2*zeta*pow(a,4) - (double)4*zeta*e1*a*zetabar + (double)2*e1*a + 4*pow(a,3)*e1 + (double)2*zeta*pow(zetabar,2)*pow(a,2) - zeta - a - (double)2*pow(a,3) + (double)2*zeta*a*zetabar;
		as[1] = (double)2*zeta*zetabar*pow(a,2) - (double)4*zeta*e1*a + pow(zetabar,2)*pow(a,4) - (double)4*e1*pow(a,2) - pow(a,6) + (double)2*pow(a,2) + (double)4*pow(e1,2)*pow(a,2) + (double)2*zeta*a + (double)2*zetabar*pow(a,3) - (double)4*e1*pow(a,3)*zetabar;
		as[0] = -zeta*pow(a,2) + zeta*pow(a,6) + pow(a,5) - (double)4*zeta*pow(e1,2)*pow(a,2) - pow(a,3) + (double)4*zeta*zetabar*pow(a,3)*e1 + (double)4*zeta*e1*pow(a,2) - zeta*pow(zetabar,2)*pow(a,4) - pow(a,4)*zetabar - (double)2*zeta*pow(a,3)*zetabar + (double)2*pow(a,3)*e1 - (double)2*e1*pow(a,5);
		
		cmplx_roots_5(z, &first_3_roots_order_changed, as, FALSE);
		__syncthreads();
		
		//printf("%f %f %f %f\\n",real(z[4]),imag(z[4]),sqrt(real(z[4])*real(z[4])+imag(z[4])*imag(z[4])),atan2(imag(z[4]),real(z[4])));


	//	Determine the photocentre and the complex conjugate of the image positions
		n = 0;
		A_row[threadIdx.x] = 0.0;
		A_col[threadIdx.x] = 0.0;
		A_local[threadIdx.x] = 0.0;
		for (j=0;j<m-1;j++)
		{	
			zbar[j] = COMPLEXTYPE(real(z[j]), -imag(z[j]));

		//	Test the possible solutions
			zeta_test[j] = z[j] - (e1 / (zbar[j]-a)) - (e2 / (zbar[j]+a));
			test_ans[j] = abs(zeta_test[j] - zeta);
		}
		__syncthreads();

		insertion_sort(test_ans, zbar, m-1);
		__syncthreads();


	//	Determine how many real images there are
		if (test_ans[m-2] < 0.0001)
		{
			n = 5;
			CAUSTIC_CROSSED = 1;
		}
		else
		{
			n = 3;
		}
		__syncthreads();
		
		A_local[threadIdx.x] = 0.0;
		for (j=0;j<n;j++)
		{	
		//	Store the image positions for use if merged images exist (re-use variables names)
		
		//	Determine the magnification contribution that the real image solutions cause.
			detj[j] = 1 - abs( pow((e1 / pow(zbar[j] - a , 2)) +  (e2 / pow(zbar[j] + a , 2)) , 2) );
			
		//	Sum the magnification contributions from all real image solutions of a given source position.
			A_local[threadIdx.x] += 1/abs(detj[j]);
		}
		__syncthreads();
	

	///////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	//	Otherwise continue and perform inverse ray mag. calc.
		
	//	Clear variables that are re-used
		A_col[threadIdx.x] = 0.0;
		A_row[threadIdx.x] = A_row[threadIdx.x + blockDim.x] = 0.0;
		
		
		if (threadIdx.x == 0)
		{
			n_all = n;
		}		
		__syncthreads();

		//	If a high mag threshold, but no contour point inside the source, then test to ensure no segment is inside the caustic
		if (CAUSTIC_CROSSED == 0)
		{
		//	Locate the highest magnification point, should be the one closest to a caustic
			if (threadIdx.x == 0)
			{
				A_local[0] = 0.0;
				index_max = 0;
				for (k=1; k<blockDim.x; k++)
				{
					if (A_local[k] > A_local[0])
					{
						A_local[0] = A_local[k];
						index_max = k;
					}
				}
			}
			__syncthreads();

			if (threadIdx.x > 0)
			{
			//	Determine the source position in the above co-ordiante system
				//u1 = (double)( ((ts[blockIdx.x] - t0n)/tE)*cos(phi) - u0n*sin(phi) );																																																	// Determine the u1 coordinates on the mag. map
				//u2 = (double)( ((ts[blockIdx.x] - t0n)/tE)*sin(phi) + u0n*cos(phi) );
				u1 = u1_in[blockIdx.x];
				u2 = u2_in[blockIdx.x];
				
			//	Determine the boundary of the area to explore, the segment of the source either side of the highest mag point
			//	Set up the threads to test this area at a higher resolution
				u1 += rho*cos( index_max * 2.0*PI/(blockDim.x - 1)  -  2.0*PI/(blockDim.x-1) + (threadIdx.x/(blockDim.x-1)) * 2.0*2.0*PI/(blockDim.x-1) );
				u2 += rho*sin( index_max * 2.0*PI/(blockDim.x - 1)  -  2.0*PI/(blockDim.x-1) + (threadIdx.x/(blockDim.x-1)) * 2.0*2.0*PI/(blockDim.x-1) );
				
			//	Perform a 5th order polynomial solution on these points to determine the new images.
			//	Determine the image positions for each threads source position
			//	Determine the complex conjugate of the source position
				zeta = COMPLEXTYPE(u1, u2);
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
				
			//	Determine how many real images there are
				if (test_ans[m-2] < 0.0001)
				{
					n = 5;
					CAUSTIC_CROSSED = 1;
				}
				else
				{
					n = 3;
				}
				
			}

			__syncthreads();
			
		}
		__syncthreads();

		if ( (n_all == 3) && (CAUSTIC_CROSSED == 1) )
		{
		//	Reset any array that will be used.
			A_col[threadIdx.x] = 0.0;
			A_row[threadIdx.x] = A_row[blockDim.x + threadIdx.x] = A_row[blockDim.x*2 + threadIdx.x] = A_row[blockDim.x*3 + threadIdx.x] = A_row[blockDim.x*4 + threadIdx.x] = 0.0;
			A_local[threadIdx.x] = A_local[blockDim.x + threadIdx.x] = A_local[blockDim.x*2 + threadIdx.x] = A_local[blockDim.x*3 + threadIdx.x] = A_local[blockDim.x*4 + threadIdx.x] = 0.0;
			
		//	Record in a shared array how many images per point on the source edge
			A_col[threadIdx.x] = (double)n;
			
			__syncthreads();

			if(threadIdx.x == 0)
			{
				A_col[0] = A_col[blockDim.x-1];
				A_col[blockDim.x] = A_col[1];
			}
			__syncthreads();
			
			if (threadIdx.x > 0)
			{
			//	Record image positions at the boundries
				if ( (fabs(A_col[threadIdx.x] -5.0) < 0.01) && (fabs(A_col[threadIdx.x -1] - 3.0) < 0.01) )
				{
					for (j=0;j<5;j++)
					{	
					//	Write image positions to a shared location so new image locations can be determined
						A_local[j*(blockDim.x) + threadIdx.x] = real(z[j]);
						A_row[j*(blockDim.x) + threadIdx.x] = imag(z[j]);
					}
				}
				
				if ( (fabs(A_col[threadIdx.x] -5.0) < 0.01) && (fabs(A_col[threadIdx.x +1] - 3.0) < 0.01) )
				{
					for (j=0;j<5;j++)
					{	
					//	Write image positions to a shared location so new image locations can be determined
						A_local[j*(blockDim.x) + threadIdx.x] = real(z[j]);
						A_row[j*(blockDim.x) + threadIdx.x] = imag(z[j]);
					}
				}
			}
		
			__syncthreads();
			
		//	Start at 0 degrees and work around until all pairs are found and averaged.
			if (threadIdx.x == 0)
			{
			//	flag to see if looking for entry or exit, [0]=entry, [1]=exit. (re-use INVERSE_SOLVE_FLAG)
				
				INVERSE_SOLVE_FLAG = 0;
				
				start_point = blockDim.x;
				zr[0] = zr[1] = zr[2] = zr[3] = zr[4] = 0;
				zi[0] = zi[1] = zi[2] = zi[3] = zi[4] = 0;
				
			//	Locate the point on the source boundary that enters the caustic.
				loop_index = 0;
				while(INVERSE_SOLVE_FLAG == 0)
				{

					loop_index++;
					if (loop_index > 100)
					{
						break;
					}

					loop_until = start_point;
					for (j=1; j<loop_until; j++)
					{
						if ( (INVERSE_SOLVE_FLAG == 0) && (fabs(A_col[j] - 5.0) < 0.01) && (fabs(A_col[j-1] - 3.0) < 0.01) )
						{
						//	identify the first time a caustic entry is located so that it can loop around back to this point if required.
							if (start_point == blockDim.x);
							{
								start_point = j;
							}
							
						//	Determine the distances between all real images.
							closest_sols[0] = (A_local[0*(blockDim.x) + j] - A_local[1*(blockDim.x) + j])*(A_local[0*(blockDim.x) + j] - A_local[1*(blockDim.x) + j])   +   (A_row[0*(blockDim.x) + j] - A_row[1*(blockDim.x) + j])*(A_row[0*(blockDim.x) + j] - A_row[1*(blockDim.x) + j]);
							closest_sols[1] = (A_local[0*(blockDim.x) + j] - A_local[2*(blockDim.x) + j])*(A_local[0*(blockDim.x) + j] - A_local[2*(blockDim.x) + j])   +   (A_row[0*(blockDim.x) + j] - A_row[2*(blockDim.x) + j])*(A_row[0*(blockDim.x) + j] - A_row[2*(blockDim.x) + j]);
							closest_sols[2] = (A_local[0*(blockDim.x) + j] - A_local[3*(blockDim.x) + j])*(A_local[0*(blockDim.x) + j] - A_local[3*(blockDim.x) + j])   +   (A_row[0*(blockDim.x) + j] - A_row[3*(blockDim.x) + j])*(A_row[0*(blockDim.x) + j] - A_row[3*(blockDim.x) + j]);
							closest_sols[3] = (A_local[0*(blockDim.x) + j] - A_local[4*(blockDim.x) + j])*(A_local[0*(blockDim.x) + j] - A_local[4*(blockDim.x) + j])   +   (A_row[0*(blockDim.x) + j] - A_row[4*(blockDim.x) + j])*(A_row[0*(blockDim.x) + j] - A_row[4*(blockDim.x) + j]);
							
							closest_sols[4] = (A_local[1*(blockDim.x) + j] - A_local[2*(blockDim.x) + j])*(A_local[1*(blockDim.x) + j] - A_local[2*(blockDim.x) + j])   +   (A_row[1*(blockDim.x) + j] - A_row[2*(blockDim.x) + j])*(A_row[1*(blockDim.x) + j] - A_row[2*(blockDim.x) + j]);
							closest_sols[5] = (A_local[1*(blockDim.x) + j] - A_local[3*(blockDim.x) + j])*(A_local[1*(blockDim.x) + j] - A_local[3*(blockDim.x) + j])   +   (A_row[1*(blockDim.x) + j] - A_row[3*(blockDim.x) + j])*(A_row[1*(blockDim.x) + j] - A_row[3*(blockDim.x) + j]);
							closest_sols[6] = (A_local[1*(blockDim.x) + j] - A_local[4*(blockDim.x) + j])*(A_local[1*(blockDim.x) + j] - A_local[4*(blockDim.x) + j])   +   (A_row[1*(blockDim.x) + j] - A_row[4*(blockDim.x) + j])*(A_row[1*(blockDim.x) + j] - A_row[4*(blockDim.x) + j]);
							
							closest_sols[7] = (A_local[2*(blockDim.x) + j] - A_local[3*(blockDim.x) + j])*(A_local[2*(blockDim.x) + j] - A_local[3*(blockDim.x) + j])   +   (A_row[2*(blockDim.x) + j] - A_row[3*(blockDim.x) + j])*(A_row[2*(blockDim.x) + j] - A_row[3*(blockDim.x) + j]);
							closest_sols[8] = (A_local[2*(blockDim.x) + j] - A_local[4*(blockDim.x) + j])*(A_local[2*(blockDim.x) + j] - A_local[4*(blockDim.x) + j])   +   (A_row[2*(blockDim.x) + j] - A_row[4*(blockDim.x) + j])*(A_row[2*(blockDim.x) + j] - A_row[4*(blockDim.x) + j]);
							
							closest_sols[9] = (A_local[3*(blockDim.x) + j] - A_local[4*(blockDim.x) + j])*(A_local[3*(blockDim.x) + j] - A_local[4*(blockDim.x) + j])   +   (A_row[3*(blockDim.x) + j] - A_row[4*(blockDim.x) + j])*(A_row[3*(blockDim.x) + j] - A_row[4*(blockDim.x) + j]);
							
							parity[0] = parity[1] = parity[2] = parity[3] = parity[4] = 0;
							for (l=0; l<5; l++)
							{
								if ( 1 / (1 - abs( pow((e1 / pow(COMPLEXTYPE(A_local[l*(blockDim.x) + j], -A_row[l*(blockDim.x) + j]) - a , 2)) +  (e2 / pow(COMPLEXTYPE(A_local[l*(blockDim.x) + j], -A_row[l*(blockDim.x) + j]) + a , 2)) , 2) ))  > 0)
								{
									parity[l] = 1;
								}
								else
								{
									parity[l] = -1;
								}
							}
							
							if (parity[0] == parity[1]) {closest_sols[0] = 999999;}
							if (parity[0] == parity[2]) {closest_sols[1] = 999999;}
							if (parity[0] == parity[3]) {closest_sols[2] = 999999;}
							if (parity[0] == parity[4]) {closest_sols[3] = 999999;}
							
							if (parity[1] == parity[2]) {closest_sols[4] = 999999;}
							if (parity[1] == parity[3]) {closest_sols[5] = 999999;}
							if (parity[1] == parity[4]) {closest_sols[6] = 999999;}
							
							if (parity[2] == parity[3]) {closest_sols[7] = 999999;}
							if (parity[2] == parity[4]) {closest_sols[8] = 999999;}
							
							if (parity[3] == parity[4]) {closest_sols[9] = 999999;}
							
							
						//	Locate which image pair are closest together
							closest_sols[10] = 9999999999;
							for (l=0; l<10; l++)
							{
								if (closest_sols[l] < closest_sols[10])
								{
									closest_sols[10] = closest_sols[l];
									closest_pair = l;
								}
							}
							
							switch(closest_pair)
							{
								case 0:
									zr[0] = A_local[0*(blockDim.x) + j];
									zi[0] = A_row[0*(blockDim.x) + j];
									break;
								case 1:
									zr[0] = A_local[0*(blockDim.x) + j];
									zi[0] = A_row[0*(blockDim.x) + j];
									break;
								case 2:
									zr[0] = A_local[0*(blockDim.x) + j];
									zi[0] = A_row[0*(blockDim.x) + j];
									break;
								case 3:
									zr[0] = A_local[0*(blockDim.x) + j];
									zi[0] = A_row[0*(blockDim.x) + j];
									break;
								case 4:
									zr[0] = A_local[1*(blockDim.x) + j];
									zi[0] = A_row[1*(blockDim.x) + j];
									break;
								case 5:
									zr[0] = A_local[1*(blockDim.x) + j];
									zi[0] = A_row[1*(blockDim.x) + j];
									break;
								case 6:
									zr[0] = A_local[1*(blockDim.x) + j];
									zi[0] = A_row[1*(blockDim.x) + j];
									break;
								case 7:
									zr[0] = A_local[2*(blockDim.x) + j];
									zi[0] = A_row[2*(blockDim.x) + j];
									break;
								case 8:
									zr[0] = A_local[2*(blockDim.x) + j];
									zi[0] = A_row[2*(blockDim.x) + j];
									break;
								case 9:
									zr[0] = A_local[3*(blockDim.x) + j];
									zi[0] = A_row[3*(blockDim.x) + j];
									break;
							}
							
							
						//	Change the flag to make it look for a caustic exit.
							INVERSE_SOLVE_FLAG = 1;
							break;
						}
						
					}
				}
							
				A_row[0] = zr[0];
				A_col[0] = zi[0];
				
				u1_source = u1-b[0];    // now in centre of mass coordinates
				u2_source = u2;
			}
			__syncthreads();	
			
		//	Take point and set up an array of image points around this to move the current image location further into the image.
		//	Determine a new image point on each thread and perform inverse ray to determine it's location from the source centre
			if (threadIdx.x < (int)blockDim.x/4 )
			{
				r_local		= A_row[0] - 2.5*rho * log10( (double)((int)blockDim.x/4) / ((double)((int)blockDim.x/4) - (double)(threadIdx.x) ) ) / log10((double)(int)blockDim.x/4);
				theta_local	= A_col[0];
			}
			else if ( (threadIdx.x >= (int)blockDim.x/4) && (threadIdx.x < (int)(blockDim.x/2)) )
			{
				r_local		= A_row[0] + 2.5*rho * log10( (double)((int)blockDim.x/4.0) / ((double)((int)blockDim.x/4.0) - (double)(threadIdx.x-(int)(blockDim.x/4.0)) ) ) / log10((double)(int)blockDim.x/4.0);
				theta_local	= A_col[0];
			}
			else if ( (threadIdx.x >= (int)(blockDim.x/2)) && (threadIdx.x < (int)(3*blockDim.x/4)) )
			{
				r_local		= A_row[0];
				theta_local	= A_col[0] - 2.5*rho * log10( (double)((int)blockDim.x/4.0) / ((double)((int)blockDim.x/4.0) - (double)(threadIdx.x-(int)(blockDim.x/2.0)) ) ) / log10((double)(int)blockDim.x/4.0);
			}
			else if (threadIdx.x >= (int)(3*blockDim.x/4) )
			{
				r_local		= A_row[0];
				theta_local	= A_col[0] + 2.5*rho * log10( (double)((int)blockDim.x/4.0) / ((double)((int)blockDim.x/4.0) - (double)(threadIdx.x-(int)(3.0*blockDim.x/4.0)) ) ) / log10((double)(int)blockDim.x/4.0);
			}
			
			__syncthreads();
			
			coef1 = COMPLEXTYPE( r_local, theta_local );
			coef2 = COMPLEXTYPE(real(coef1), -imag(coef1));
			
		//	Determine the inverse ray position
			omega = coef1 -  e1 / (coef2-a)   - e2 / (coef2+a) ;
			

		//	Calculate the distance between the ray and the source centre
			A_local[threadIdx.x] = (u1_source+b[0]-real(omega))*(u1_source+b[0]-real(omega)) + (u2_source-imag(omega))*(u2_source-imag(omega));

			__syncthreads();

		//	Make a single thread loop over the grid points to determine the ray that is closest to the source centre
		//	Count outwards from the centre and break if the point falls to zero (has left the merged image)
			if (threadIdx.x == 0)
			{
				A_row[0] = 9999.0;
				A_row[1] = 0.0;
				for (j=0+1; j<(int)blockDim.x/4; j++)
				{
					if (A_local[j] >= rho*rho)
					{
						break;
					}
					if (A_local[j] < A_row[0])
					{
						A_row[0] = A_local[j];
						A_row[1] = (double)j;
					}
				}
				
				for (j=(int)blockDim.x/4+1; j<(int)blockDim.x/2; j++)
				{
					if (A_local[j] >= rho*rho)
					{
						break;
					}
					if (A_local[j] < A_row[0])
					{
						A_row[0] = A_local[j];
						A_row[1] = (double)j;
					}
				}
				
				for (j=(int)blockDim.x/2+1; j<(int)3*blockDim.x/4; j++)
				{
					if (A_local[j] >= rho*rho)
					{
						break;
					}
					if (A_local[j] < A_row[0])
					{
						A_row[0] = A_local[j];
						A_row[1] = (double)j;
					}
				}
				
				for (j=(int)3*blockDim.x/4+1; j<blockDim.x; j++)
				{
					if (A_local[j] >= rho*rho)
					{
						break;
					}
					if (A_local[j] < A_row[0])
					{
						A_row[0] = A_local[j];
						A_row[1] = (double)j;
					}
				}
			}
			__syncthreads();
			
		//	Make a single thread store the new merged image position in replace of 
			if (A_row[1] > 0.1) // Prevents the new point being placed on an image edge.
			{
				if (threadIdx.x == A_row[1])
				{
					A_row[0] = r_local;
					A_col[0] = theta_local;
				}
				__syncthreads();
				
				if (threadIdx.x == 0)
				{
					for (j=0; j<5 ;j++)
					{
					//	Test the this is a true solution
					//	skip this image if not a solution
						if (abs(zeta_test[j] - zeta) > 0.0001)
						{
							z[j] = COMPLEXTYPE(A_row[0], A_col[0]);
							zbar[j] = COMPLEXTYPE(real(z[j]), -imag(z[j]));

						//	Test the possible solutions
							zeta_test[j] = z[j] - (e1 / (zbar[j]-a)) - (e2 / (zbar[j]+a));
							break;
						}
					}
				}
				__syncthreads();
			}
			
		}
		__syncthreads();


		if (threadIdx.x == 0)
		{
			u1_source = u1-b[0];
			u2_source = u2;
		
		//	Put the image centre co-ordinates back into COM frame of ref. and in polar coordinates
			for (j=0; j<5 ;j++)
			{

			//	Test if this image position lands in the source when ray shot
				
				zeta = COMPLEXTYPE( real(z[j]), imag(z[j]) );
				zetabar = COMPLEXTYPE(real(zeta), -imag(zeta));
				
			//	Determine the inverse ray position
				omega = zeta - e1 / (zetabar-a)    -  e2 / (zetabar+a);
				
			//	Calculate the distance between the ray and the source centre
			//	If inside source record as 1, otherwise 0
				A_local[threadIdx.x] = 0.0;
				if (   (u1_source+b[0]-real(omega))*(u1_source+b[0]-real(omega)) + (u2_source-imag(omega))*(u2_source-imag(omega))  >=  rho*rho  )
				{
					theta_imgs[j] = -99999.0;
					continue;
				}
				

				r_imgs[j]		= sqrt( (real(z[j]) - b[0])*(real(z[j]) - b[0]) + (imag(z[j]))*(imag(z[j])) );
				if (real(z[j])-b[0] >= 0)
				{
					if (imag(z[j]) >= 0)
					{
						theta_imgs[j]	= atan( imag(z[j]) / (real(z[j]) - b[0]) );
					}
					else
					{
						theta_imgs[j]	= 2*PI - atan( -imag(z[j]) / (real(z[j]) - b[0]) );
					}
				}
				else
				{
					if (imag(z[j]) >= 0)
					{
						theta_imgs[j]	= PI - atan( imag(z[j]) / -(real(z[j]) - b[0]) );
					}
					else
					{
						theta_imgs[j]	= PI + atan( -imag(z[j]) / -(real(z[j]) - b[0]) );
					}
				}
				
			}

		//	Set all image areas initially to 0
			A_img[0] = A_img[1] = A_img[2] = A_img[3] = A_img[4] = 0.0;
		}
		__syncthreads();
		
		img_grid_dims[0] = img_grid_dims[1] = img_grid_dims[2] = img_grid_dims[3] = img_grid_dims[4] = img_grid_dims[5] = img_grid_dims[6] = img_grid_dims[7] = img_grid_dims[8] = img_grid_dims[9] = img_grid_dims[10] = img_grid_dims[11] = img_grid_dims[12] = img_grid_dims[13] = img_grid_dims[14] = img_grid_dims[15] = img_grid_dims[16] = img_grid_dims[17] = img_grid_dims[18] = img_grid_dims[19] = 0.0;

	//	Loop over the images and use an inverse ray shooting method around images to determine the magnification.
		for (j=0; j<5; j++)
		{
			__syncthreads();


			if (theta_imgs[j] < -9000)
			{
				__syncthreads();
				continue;
			}

			// Check if image centre is within the integration bounds of any previous images
			if (threadIdx.x == 0)
			{

				A_img[j] = 0.0;
				b[j] = b_com;
				CHANGED_COORDS_FLAG = 0;
				u1_source = u1 - b[j];

				STOP = 0;

				for (k=0; k<j; k++)
				{
					map_coords(&r_imgs[j],&theta_imgs[j],b[j],&r_local,&theta_local,b[k]);

					if ( (img_grid_dims[4*k+0] < r_local)  &&  (r_local < img_grid_dims[4*k+1])  && ( ( (theta_local < img_grid_dims[4*k+2])  && (img_grid_dims[4*k+3] < theta_local) )  || ( (theta_local+2.0*PI < img_grid_dims[4*k+2])  && (img_grid_dims[4*k+3] < theta_local+2.0*PI) ) || ( (theta_local-2.0*PI < img_grid_dims[4*k+2])  && (img_grid_dims[4*k+3] < theta_local-2.0*PI) ) ))
					{
						STOP = 1;
						break;
					}

				}

				// Offset x coordinate origin to centre of image
				//b1 = r_imgs[j] * cos(theta_imgs[j]);
				b1 = 0.0;
				map_coords(&r_imgs[j],&theta_imgs[j],b[j],&r_local,&theta_local,b[j]+b1);
				b[j] += b1;
				u1_source = u1-b[j];
				u2_source = u2;
				r_imgs[j] = r_local;
				theta_imgs[j] = theta_local;

			}
			__syncthreads();


			if ((STOP) || (theta_imgs[j] < -99998))
			{
				__syncthreads();
				continue;
			}

			// Loop for cases when we change the coordinate origin
			if (threadIdx.x == 0)
			{
				iterate_coordinates = 0;
			}
			__syncthreads();
			
			while (iterate_coordinates <= 2) 
			{

						//	Clear variables that are re-used
				A_col[threadIdx.x] = 0.0;
				A_row[threadIdx.x] = A_row[threadIdx.x + blockDim.x] = 0.0;
				A_local[threadIdx.x] = A_local[threadIdx.x + blockDim.x] = 0.0;
		
				if (threadIdx.x == 0)
				{
					RING_FLAG = 0;
					RING_FLAG_a = 0;
					RING_FLAG_b = 0;
					
					A_limb[0] = r_imgs[j];
					A_limb[1] = r_imgs[j];
				}
				__syncthreads();
				
			////////////////////////////////////////////////////////////////
			//	Sample in a cross shape origin about the image centres to determine the approx edge locations 
				
			//	First determine angular dimensions using a log spaced line covering +-Pi about image centre
			//	Use even numbers as positive, and odd numbers as negative.
				
			//	Determine odd numbers by "Two's complement"


				theta_local = theta_imgs[j];
				r_local = r_imgs[j];

				if (iterate_coordinates == 1)
				{


					// if (rad_loc[0] < 2.e-8)
					// {
					// 	b1 = -0.5*(rad_loc[0]+rad_loc[1]);
					// 	if (b[j] < 0.0)
					// 	{
					// 		b1 *= -1.0;
					// 	}

					// } else {

					// 	// Set new coordinate origin to be the point where a straight line perpendicular
					// 	// to the line passing through the ends of the image intersects the x-axis
					// 	x_left = left_mid * cos(ang_loc[0]);
					// 	y_left = left_mid * sin(ang_loc[0]);
					// 	x_right = right_mid * cos(ang_loc[1]);
					// 	y_right = right_mid * sin(ang_loc[1]);
					// 	theta_local = atan2(x_right-x_left,y_left-y_right);
					// 	y_centre = 0.5*(y_left+y_right);
					// 	b1 = -y_centre/sin(theta_local);

					// }

					if (threadIdx.x == 0)
					{

						if (fabs(left_mid - right_mid) < 0.01)
						{
							left_mid -= 0.5*(left_mid-rad_loc[0]);
							right_mid += 0.5*(rad_loc[1]-right_mid);
						}
						x_left = b[j] + left_mid * cos(ang_loc[0]);
						y_left = left_mid * sin(ang_loc[0]);
						x_right = b[j] + right_mid * cos(ang_loc[1]);
						y_right = right_mid * sin(ang_loc[1]);
						x_centre = 0.5*(x_left+x_right);
						y_centre = 0.5*(y_left+y_right);

						// theta_local = atan2(x_right-x_left,y_left-y_right);
						// if (theta_local < 0.0)
						// {
						// 	theta_local += PI;
						// }
						// y_centre = 0.5*(y_left+y_right);
						// b1 = -y_centre/sin(theta_local);
						// if (b1 > 1.0)
						// {
						// 	b1 = 1.0;
						// }
						// if (b1 < -1.0)
						// {
						// 	b1 = -1.0;
						// }
						// b_old = b[j];
						// b[j] += b1;

						b1 = y_centre*(y_right-y_left)/(x_right-x_left);
						if (b1 > 1.0)
						{
							b1 = 1.0;
						}
						if (b1 < -1.0)
						{
							b1 = -1.0;
						}
						b_old = b[j];
						b[j] = x_centre + b1;

						u1_source = u1-b[j];
						u2_source = u2;
					}
					__syncthreads();  

					map_coords(&r_imgs[j],&theta_imgs[j],b_old,&r_local,&theta_local,b[j]);

					if (threadIdx.x == 0)
					{

						A_limb[0] = r_local;
						A_limb[1] = r_local;
						r_imgs[j] = r_local;
						theta_imgs[j] = theta_local;
						CHANGED_COORDS_FLAG = 1;

					}
					__syncthreads();

				}

				__syncthreads();

				if (threadIdx.x > 1)
				{
					if (threadIdx.x & 1)
					{
					//	If ODD count down from image middle
						//theta_local -= PI * log10( (double)(blockDim.x / 2.0) / ((double)(blockDim.x / 2.0) - (double)(int)(threadIdx.x/2) ) ) / log10((double)(blockDim.x / 2.0)) ;
						theta_local -= PI * pow(10.0,10.0*threadIdx.x/(blockDim.x-1.0) - 10.0);
					}
					else
					{
					//	if EVEN count up from the image middle
						//theta_local += PI * log10( (double)(blockDim.x / 2.0) / ((double)(blockDim.x / 2.0) - (double)(int)(threadIdx.x/2) ) ) / log10((double)(blockDim.x / 2.0)) ;
						theta_local += PI * pow(10.0,10.0*threadIdx.x/(blockDim.x-1.0) - 10.0);

					}
				}
				__syncthreads();
				

				A_local[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
												u1_source, u2_source, rho);
				__syncthreads();


			//	Loop through threads and locate angular limits
			//	Thread 0 looks at even numbers, and thread 1 looks at odd numbers
			//	Store the thread id of the angular upper limit (ang_loc[0]) and lower limit (ang_loc[1])
				if (threadIdx.x < 2)
				{
					k = threadIdx.x;
					while(k< blockDim.x)
					{
						ang_loc[threadIdx.x] = 99999.0;
						if (A_local[k] < 0.1)
						{
							ang_loc[threadIdx.x] = k-2.0;
							break;
						}
						
						k += 2;
					}
				}
				__syncthreads();

			//	If the boundary limit is less than 3 threads away from the start
			//	Repeat process to find the theta limit to greater accuracy.
				loop_index = 0;
				while (((int)(ang_loc[0]/2.0) < 2.0) || ((int)(ang_loc[1]/2.0) < 2.0))
				{

					__syncthreads();


					loop_index++;
					if (loop_index > 100)
					{
						break;
					}




				//	Store the actual angle of these limits.
					if (threadIdx.x == (int)ang_loc[0]+2)
					{
						ang_loc[0] = theta_local;
					}
					if (threadIdx.x == (int)ang_loc[1]+2)
					{
						ang_loc[1] = theta_local;
					}
					__syncthreads();	
					
				//	First determine angular dimensions using a log spaced line covering +-Pi about image centre
				//	Use even numbers as positive, and odd numbers as negative.
					
				//	Determine odd numbers by "Two's complement"
					theta_local = theta_imgs[j];
					if (threadIdx.x > 1)
					{
						if (threadIdx.x & 1)
						{
						//	If ODD count down from image middle
							//theta_local -= (theta_imgs[j] - ang_loc[1]) * log10( (double)(blockDim.x / 2.0) / ((double)(blockDim.x / 2.0) - (double)(int)(threadIdx.x/2) ) ) / log10((double)(blockDim.x / 2.0)) ;
							theta_local -= 2.0*(theta_imgs[j] - ang_loc[1]) * pow(10.0,10.0*threadIdx.x/(blockDim.x-1.0) - 10.0);
						}
						else
						{
						//	if EVEN count up from the image middle
							//theta_local += (ang_loc[0] - theta_imgs[j]) * log10( (double)(blockDim.x / 2.0) / ((double)(blockDim.x / 2.0) - (double)(int)(threadIdx.x/2) ) ) / log10((double)(blockDim.x / 2.0)) ;
							theta_local += 2.0*(ang_loc[0] - theta_imgs[j]) * pow(10.0,10.0*threadIdx.x/(blockDim.x-1.0) - 10.0);
						}
					}
					__syncthreads();
					
					r_local = r_imgs[j];
					A_local[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
													u1_source, u2_source, rho);
					__syncthreads();



				//	Loop through threads and locate angular limits
				//	Thread 0 looks at even numbers, and thread 1 looks at odd numbers
				//	Store the thread id of the angular upper limit (ang_loc[0]) and lower limit (ang_loc[1])

					if (threadIdx.x < 2)
					{
						k = threadIdx.x;
						while(k< blockDim.x)
						{
							if (A_local[k] < 0.1)
							{
								ang_loc[threadIdx.x] = k-2.0;
								break;
							}
							
							
							k += 2;
						}
					}
					__syncthreads();
					
				}

				__syncthreads();
				
			//	Store the angles of the edges, unless grid dimensions were not large enough in which case make it is large as possible. It will grow later.
				if (ang_loc[0] > 99998)
				{
					if (threadIdx.x == 0)
					{
						ang_loc[0] = theta_imgs[j] + PI;
					}
					__syncthreads();
				}
				else
				{
				//	Store the actual angle of these limits.
				//	[0]=anticlockwise
					if (threadIdx.x == (int)ang_loc[0])
					{
						ang_loc[0] = theta_local;
					}
					__syncthreads();

				}	

				
				if (ang_loc[1] > 99998)
				{
					if (threadIdx.x == 0)
					{
						ang_loc[1] = theta_imgs[j] - PI;
					}
					__syncthreads();

				} 
				else
				{
				//	Store the actual angle of these limits.
				//	[1]=clockwise
					if (threadIdx.x == (int)ang_loc[1])
					{
						ang_loc[1] = theta_local;
					}
					__syncthreads();

				}

				
			///////////////////////////////////////////
			//	Repeat process in the radial direction.
				
			//	First determine radial dimensions using a linear line covering +-0.5 R_E
			//	Use even numbers as positive, and odd numbers as negative.
				
			//	Determine odd numbers by "Two's complement"
				r_local = r_imgs[j];
				if (threadIdx.x > 1)
				{
					if (threadIdx.x & 1)
					{
					//	If ODD count up from image middle
						//r_local += 2.5*rho * log10( (double)(blockDim.x / 2.0) / ((double)(blockDim.x / 2.0) - (double)(int)(threadIdx.x/2) ) ) / log10((double)(blockDim.x / 2.0)) ;
						r_local += 2.0*rho * pow(10.0,10.0*threadIdx.x/(blockDim.x-1.0) - 10.0);
					}
					else
					{
					//	if EVEN down up from the image middle
						//r_local -= 2.5*rho * log10( (double)(blockDim.x / 2.0) / ((double)(blockDim.x / 2.0) - (double)(int)(threadIdx.x/2) ) ) / log10((double)(blockDim.x / 2.0)) ;
						r_local -= 2.0*rho * pow(10.0,10.0*threadIdx.x/(blockDim.x-1.0) - 10.0);
					}
				}
				__syncthreads();
				
				theta_local = theta_imgs[j];
				A_local[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
												u1_source, u2_source, rho);
				__syncthreads();


			//	Loop through threads and locate radial limits
			//	Thread 0 looks at even numbers, and thread 1 looks at odd numbers
			//	Store the thread id of the radial upper limit (rad_loc[0]) and lower limit (rad_loc[1])
				if (threadIdx.x < 2)
				{
					
					rad_loc[threadIdx.x] = blockDim.x - 2.0 + threadIdx.x;
					
					k = threadIdx.x;
					while(k < blockDim.x)
					{
						if (A_local[k] < 0.1)
						{
							rad_loc[threadIdx.x] = (double)k;
							break;
						}
						
						
						k += 2;
					}

				}
				__syncthreads();
				
			//	If the boundary limit is less than 3 threads away from the start
			//	Repeat process to find the theta limit to greater accuracy.
				loop_index = 0;
				while ( (rad_loc[0]/2.0 < 2.0) || (rad_loc[1]/2.0 < 2.0) )
				{

					loop_index++;
					if (loop_index > 100)
					{
						break;
					}

					
				//	Store the actual angle of these limits.
					if (threadIdx.x == (int)rad_loc[0]+2)
					{
						rad_loc[0] = r_local;
					}
					if (threadIdx.x == (int)rad_loc[1]+2)
					{
						rad_loc[1] = r_local;
					}
					__syncthreads();	
					
				//	Determine odd numbers by "Two's complement"
					r_local = r_imgs[j];
					if (threadIdx.x > 1)
					{
						if (threadIdx.x & 1)
						{
						//	If ODD count down from image middle
							//r_local += (rad_loc[1] - r_imgs[j]) * log10( (double)(blockDim.x / 2.0) / ((double)(blockDim.x / 2.0) - (double)(int)(threadIdx.x/2) ) ) / log10((double)(blockDim.x / 2.0)) ;
							r_local += 2.0* (rad_loc[1] - r_imgs[j])* pow(10.0,12.0*threadIdx.x/(blockDim.x-1.0) - 12.0);
						}
						else
						{
						//	if EVEN count up from the image middle
							//r_local -= (r_imgs[j] - rad_loc[0]) * log10( (double)(blockDim.x / 2.0) / ((double)(blockDim.x / 2.0) - (double)(int)(threadIdx.x/2) ) ) / log10((double)(blockDim.x / 2.0)) ;
							r_local -= 2.0* (r_imgs[j] - rad_loc[0])* pow(10.0,12.0*threadIdx.x/(blockDim.x-1.0) - 12.0);
						}
					}
					__syncthreads();
				
					theta_local = theta_imgs[j];
					A_local[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
														u1_source, u2_source, rho);
					__syncthreads();

				
				//	Loop through threads and locate radial limits
				//	Thread 0 looks at even numbers, and thread 1 looks at odd numbers
				//	Store the thread id of the radial upper limit (rad_loc[0]) and lower limit (rad_loc[1])

					if (threadIdx.x < 2)
					{
						rad_loc[threadIdx.x] = blockDim.x/2.0;
						
						k = threadIdx.x;
						while(k< blockDim.x)
						{
							if (A_local[k] < 0.1)
							{
								rad_loc[threadIdx.x] = (double)k;
								break;
							}
						
						
							k += 2;
						}
					}
					__syncthreads();

				}
				
				__syncthreads();

			//	Store the actual radius of these limits.
			//	[0]=inner radius(smaller), [1]=outer radius(larger)
				if (threadIdx.x == (int)rad_loc[0])
				{
					rad_loc[0] = r_local;
				}
				if (threadIdx.x == (int)rad_loc[1])
				{
					rad_loc[1] = r_local;
				}
				__syncthreads();
				
				if (threadIdx.x == 0)
				{
					if (rad_loc[0] > rad_loc[1])
					{
						rad_shift = rad_loc[0];
						rad_loc[0] = rad_loc[1];
						rad_loc[1] = rad_shift;
					}
					if (ang_loc[1] > ang_loc[0])
					{
						ang_shift = ang_loc[0];
						ang_loc[0] = ang_loc[1];
						ang_loc[1] = ang_shift;
					}
					
				//	Ensure the inner radius is not less than zero.
					if (rad_loc[0] < 0.0)
					{
						rad_loc[0] = 0.000001;
					}

					if (rad_loc[1] - rad_loc[0] < 2.e-4)
					{
						rad_loc[0] -= 1.e-4;
						rad_loc[1] += 1.e-4;
					}
					
				}
				__syncthreads();


			////////////////////////////////////////////////////////////////////////
			////////////////////////////////////////////////////////////////////////
			//	Create a rectangle to encompase the whole image area
			//	Check each edge to ensure the image is fully enclosed
			
				if (threadIdx.x == 0)
				{
					GRID_DIM_FLAG = 1;
					max_range = rad_loc[1]-rad_loc[0];
					left_loc[0] = rad_loc[0] - 0.1*max_range;
					left_loc[1] = rad_loc[1] + 0.1*max_range;
					right_loc[0] = rad_loc[0] - 0.1*max_range;
					right_loc[1] = rad_loc[1] + 0.1*max_range;
					max_range = ang_loc[0]-ang_loc[1];
					inner_loc[0] = ang_loc[0] + 0.1*max_range;
					inner_loc[1] = ang_loc[1] - 0.1*max_range;
					outer_loc[0] = ang_loc[0] + 0.1*max_range;
					outer_loc[1] = ang_loc[1] - 0.1*max_range;
					iteration1 = 1;
					max_range = left_loc[1] - left_loc[0];
					ang_loc_0_save = ang_loc[0];
					max_theta_step = 0.1*(ang_loc[0]-ang_loc[1]);
					if (max_theta_step > 0.008)
					{
						max_theta_step = 0.008;
					}
					//printf("%g\\n",max_theta_step);
				}
				__syncthreads();



			//	Loop over all 4 edges until never need to increase an edge size
			//	GRID_DIM_FLAG is triggered any time an edge is increased in size, loop over the 4 edges to confirm the whole grid is ok
			//	INVERSE_SOLVE_FLAG is used to indicate the edge needs to be expanded, loop until this edge no longer needs expanding


				
				grid_size = sqrt((rad_loc[0]-rad_loc[1])*(rad_loc[0]-rad_loc[1]) + (ang_loc[0]-ang_loc[1])*(ang_loc[0]-ang_loc[1]));
				oversample_factor = grid_size*5000 + 1.0;

				if (threadIdx.x == 0)
				{
					GRID_DIM_FLAG = 0;
					INVERSE_SOLVE_FLAG = 1;
					prev_mid = -1.0;
					if (iteration1 == 1)
					{
						iteration2 = 1;
					}			
				}
				__syncthreads();



				
			//////////////////////////////
			//	Check the anticlockwise edge
				prev_shift = ang_shift = 0.0;
				
				eta_gs = 0.0;

				loop_index = 0;
				while( (INVERSE_SOLVE_FLAG)	&&	(RING_FLAG_a == 0) )
				{

					__syncthreads();


					loop_index++;
					if (loop_index > 9999)
					{
						break;
					}


					theta_local = ang_loc[0] + ang_shift;

					h = (left_loc[1] - left_loc[0]) / (double)(blockDim.x-1);

					if (threadIdx.x == 0)
					{
						INVERSE_SOLVE_FLAG = 0;
					}
					__syncthreads();
					

					if (iteration2 == 1)
					{

						A_local[threadIdx.x] = 0.0;
						for (k=0; k<oversample_factor; k++)
						{

							r_local = left_loc[0] + ((k/oversample_factor)+threadIdx.x)*h;
							
							A_local[threadIdx.x] += check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
														u1_source, u2_source, rho);
						}

					}
					else
					{

						r_local = left_loc[0] + (double)threadIdx.x*h;

						A_local[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
													u1_source, u2_source, rho);
					}

					if (A_local[threadIdx.x] >= 0.5)
					{
						INVERSE_SOLVE_FLAG = 1;
						GRID_DIM_FLAG = 1;								
					}
					__syncthreads();


					if (threadIdx.x == 0)
					{
						repeat_loop = 0;
					}
					__syncthreads();

					// Locate the intercepts
					if ((threadIdx.x == 0) && (INVERSE_SOLVE_FLAG))
					{
						k = blockDim.x - 1;
						found = 0;
						while ((k >= 0) && (found == 0))
						{
							if (A_local[k] > 0.5)
							{
								found = 1;
								left_loc[1] = left_loc[0] + k*h + h;
								if (k == (blockDim.x-1))
								{
									left_loc[1] += (blockDim.x/5) * h;
									repeat_loop = 1;
								}
							}
							k--;
						}
						k = 0;
						found = 0;
						while ((k < blockDim.x) && (found == 0))
						{
							if (A_local[k] > 0.5)
							{
								found = 1;
								left_loc[0] = left_loc[0] + k*h - h;
								if (k == 0)
								{
									left_loc[0] += h;
									if (left_loc[0] > 1.e-9)
									{
										left_loc[0] -= (blockDim.x/5) * h;
										repeat_loop = 1;
									}
								}
							}
							k++;
						}
						if (left_loc[0] < 0.0)
						{
							left_loc[0] = 0.0;
						}
						mid = 0.5*(left_loc[1]+left_loc[0]);
						if ( (left_loc[1]-left_loc[0]) > max_range)
						{
							max_range = left_loc[1]-left_loc[0];
						}
						if (left_loc[0] < rad_loc[0])
						{
							rad_loc[0] = left_loc[0];
						}
						if (left_loc[1] > rad_loc[1])
						{
							rad_loc[1] = left_loc[1];
						}
					}
					__syncthreads();



				//	Identify if the rays still fall inside the image  even when greater then a complete circle.
					if ( (theta_local > (theta_imgs[j] +2.0*PI)) && (INVERSE_SOLVE_FLAG == 1) )
					{
						RING_FLAG_a = 1;
						repeat_loop = 0;
					}
					__syncthreads();


					if ((repeat_loop == 0) && (RING_FLAG_a ==0))
					{
						prev_shift = ang_shift;
						if (0.3 * (ang_loc[0] + ang_shift - ang_loc[1]) > max_theta_step)
						{
							ang_shift += max_theta_step;
						}
						else
						{
							ang_shift += 0.3 * (ang_loc[0] + ang_shift - ang_loc[1]);
						}

						if (threadIdx.x == 0) 
						{
							range = left_loc[1]-left_loc[0];
							if ((iteration2 == 1) || (prev_mid < 0.0))
							{
								left_loc[0] -= range;
								if (left_loc[0] < 0.0)
								{
									left_loc[0] = 0.0;
								}
								left_loc[1] += range;
								prev_mid = mid;
								prev_theta_local = theta_local;
							} 
							else
							{
								slope = (mid - prev_mid)/(theta_local - prev_theta_local);
								prev_mid = mid;
								prev_theta_local = theta_local;
								theta_local = ang_loc[0] + ang_shift;
								mid += (theta_local - prev_theta_local)*slope;
								left_loc[0] = mid - range;
								if (left_loc[0] < 0.0)
								{
									left_loc[0] = 0.0;
								}
								left_loc[1] = mid + range;
								if (left_loc[0] < rad_loc[0])
								{
									rad_loc[0] = left_loc[0];
								}
								if (left_loc[1] > rad_loc[1])
								{
									rad_loc[1] = left_loc[1];
								}
							}
						}
						__syncthreads();
					}
					else
					{
						if (threadIdx.x == 0)
						{
							iteration2++;
						}
						__syncthreads();
					}

				}
				
			//	Update the new edge location for the grid
				if (threadIdx.x == 0)
				{

					if (RING_FLAG_a == 1)
					{
						ang_loc[0] = theta_imgs[j] + PI;
						ang_loc[1] = theta_imgs[j] - PI;
					}
					else
					{
						ang_loc[0] += prev_shift;
					}
					INVERSE_SOLVE_FLAG = 1;
					if (iteration1 == 1)
					{
						iteration2 = 1;
						left_mid = prev_mid;
					}
				}
				__syncthreads();
				
				
			//////////////////////////////
			//	Check the clockwise edge

				ang_shift = 0.0;


				loop_index = 0;
				while( (INVERSE_SOLVE_FLAG)	&&	(RING_FLAG_b == 0) &&	(RING_FLAG_a == 0) )
				{

					loop_index++;
					if (loop_index > 9999)
					{
						break;
					}


					theta_local = ang_loc[1] - ang_shift;
					
					h = (right_loc[1] - right_loc[0]) / (double)(blockDim.x-1);

					if (threadIdx.x == 0)
					{
						INVERSE_SOLVE_FLAG = 0;
					}
					__syncthreads();

					if (iteration2 == 1)
					{

						A_local[threadIdx.x] = 0.0;
						for (k=0; k<oversample_factor; k++)
						{

							r_local = right_loc[0] + ((k/oversample_factor)+threadIdx.x)*h;
							
							A_local[threadIdx.x] += check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
													u1_source, u2_source, rho);

						}

					}
					else
					{

						r_local = right_loc[0] + (double)threadIdx.x*h;
						
						A_local[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
													u1_source, u2_source, rho);
					}

					if (A_local[threadIdx.x] >= 0.5)
					{
						INVERSE_SOLVE_FLAG = 1;
						GRID_DIM_FLAG = 1;
					}
					__syncthreads();

					if (threadIdx.x == 0)
					{
						repeat_loop = 0;
					}
					__syncthreads();

					// Locate the intercepts
					if ((threadIdx.x == 0) && (INVERSE_SOLVE_FLAG))
					{
						k = blockDim.x - 1;
						found = 0;
						while ((k >= 0) && (found == 0))
						{
							if (A_local[k] > 0.5)
							{
								found = 1;
								right_loc[1] = right_loc[0] + k*h + h;
								if (k == (blockDim.x-1))
								{
									right_loc[1] += (blockDim.x/5) * h;
									repeat_loop = 1;
								}
							}
							k--;
						}
						k = 0;
						found = 0;
						while ((k < blockDim.x) && (found == 0))
						{
							if (A_local[k] > 0.5)
							{
								found = 1;
								right_loc[0] = right_loc[0] + k*h - h;
								if (k == 0)
								{
									right_loc[0] += h;
									if (right_loc[0] > 1.e-9)
									{
										right_loc[0] -= (blockDim.x/5) * h;
										repeat_loop = 1;
									}
								}
							}
							k++;
						}
						if (right_loc[0] < 0.0)
						{
							right_loc[0] = 0.0;
						}
						mid = 0.5*(right_loc[1]+right_loc[0]);
						if ( (right_loc[1]-right_loc[0]) > max_range)
						{
							max_range = right_loc[1]-right_loc[0];
						}
						if (right_loc[0] < rad_loc[0])
						{
							rad_loc[0] = right_loc[0];
						}
						if (right_loc[1] > rad_loc[1])
						{
							rad_loc[1] = right_loc[1];
						}
					}
					__syncthreads();

					//	Identify if the rays still fall inside the image  even when greater then a complete circle.
					if ( (theta_local < theta_imgs[j] - 2.0*PI) && (INVERSE_SOLVE_FLAG == 1) )
					{
						RING_FLAG_b = 1;
						//break;
					}
					__syncthreads();
					
					if ((repeat_loop == 0) && (RING_FLAG_b ==0))
					{
						prev_shift = ang_shift;
						if (0.3 * (ang_loc_0_save + ang_shift - ang_loc[1]) > max_theta_step)
						{
							ang_shift += max_theta_step;
						}
						else
						{
							ang_shift += 0.3 * (ang_loc_0_save + ang_shift - ang_loc[1]);
						}

						if (threadIdx.x == 0) 
						{
							range = right_loc[1]-right_loc[0];
							if ((iteration2 == 1) || (prev_mid < 0.0))
							{
								right_loc[0] -= range;
								if (right_loc[0] < 0.0)
								{
									right_loc[0] = 0.0;
								}
								right_loc[1] += range;
								prev_theta_local = theta_local;
								prev_mid = mid;
							} 
							else
							{
								slope = (mid - prev_mid)/(theta_local - prev_theta_local);
								prev_mid = mid;
								prev_theta_local = theta_local;
								theta_local = ang_loc[1] - ang_shift;
								mid += (theta_local - prev_theta_local)*slope;
								right_loc[0] = mid - range;
								if (right_loc[0] < 0.0)
								{
									right_loc[0] = 0.0;
								}
								right_loc[1] = mid + range;
								if (right_loc[0] < rad_loc[0])
								{
									rad_loc[0] = right_loc[0];
								}
								if (right_loc[1] > rad_loc[1])
								{
									rad_loc[1] = right_loc[1];
								}
							}
						}
						__syncthreads();

					}
					else
					{
						if (threadIdx.x == 0)
						{
							iteration2++;
						}
						__syncthreads();
					}

				}

				
			//	Update the new edge location for the grid
				if (threadIdx.x == 0)
				{

					if ((left_loc[0] < 1.5e-8) || (right_loc[0] < 1.5e-8))
					{
						RING_FLAG_a = 1;
						left_loc[0] = 0.0;
						right_loc[0] = 0.0;
						rad_loc[0] = 0.0;
						left_mid = 0.5*left_loc[1];
						right_mid = 0.5*right_loc[1];
					}

					if ((RING_FLAG_b == 1) || (RING_FLAG_a == 1))
					{
						ang_loc[0] = theta_imgs[j] + PI;
						ang_loc[1] = theta_imgs[j] - PI;
						left_loc[0] = 0.0;
						right_loc[0] = 0.0;
					}
					else
					{
						ang_loc[1] -= prev_shift;
						right_mid = prev_mid;
					}
					INVERSE_SOLVE_FLAG = 1;
					if (iteration1 == 1)
					{
						iteration2 = 1;
					}
				}
				__syncthreads();
				
				//if ((threadIdx.x==0) && (j==0)) printf("%d %d %d %g %g %g %g %g %g %g %g %g %g %g\\n",blockIdx.x,iterate_coordinates,RING_FLAG_a,rad_loc[0],rad_loc[1],ang_loc[0],ang_loc[1],right_loc[0],left_loc[0],right_loc[1],left_loc[1],right_mid,left_mid,b[j]);
				//__syncthreads();

			//	Write the currently searched grid area to shared memory so double counting can be avoided.
				if (threadIdx.x == 0)
				{
					if ( (RING_FLAG_a == 1) || (RING_FLAG_b == 1) )
					{
						RING_FLAG = 1;
					}
					
					if (rad_loc[0] < 0.0)
					{
						rad_loc[0] = 0.0;
					}

					img_grid_dims[4*j + 0] = rad_loc[0];
					img_grid_dims[4*j + 1] = rad_loc[1];
					img_grid_dims[4*j + 2] = ang_loc[0];
					img_grid_dims[4*j + 3] = ang_loc[1];



					if ( ((rad_loc[1]-rad_loc[0]) > 0.1) && (iterate_coordinates < 2))
					{
						iterate_coordinates += 1;
					}
					else
					{
						iterate_coordinates += 5;
					}



				}
				__syncthreads();
			
			}  // end coordinate_iteration

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//	Now perform the integration of the coloumns/rows
		
		//	Reset variables for row calculations
			index = threadIdx.x;
			while (index < MAX_GRID_DIM)
			{
				A_row[index] = 0.0;
				index += blockDim.x;
			}
			__syncthreads();

			
		//	Flag to make the first  and last coloumn store the location of the central grid point inside the image. (for later row intergration)
			if (threadIdx.x == 0)
			{
				COLOUMN_FLAG = 0;
				slope = (right_mid - left_mid) / (ang_loc[1] - ang_loc[0]);
				prev_mid = -1.0;
				image_found = 0;
				boundary[0] = -1.0;
			}
			__syncthreads();
			
		//	Integrate radially first as this is the axis with a greater number of sampled points.
		//	Start on the inner edge coloumn and integrate along the length.
			for (l=0; l<(int)(blockDim.x/ratio); ++l)
			{


			//	Reset variables for coloumn/row calculations
				index = threadIdx.x;
				while (index < MAX_GRID_DIM)
				{
					A_col[index] = 0.0;
					index += blockDim.x;
				}
				__syncthreads();
				
				
			//	Assign a single theta value and loop over in l larger to smaller angular, 0=anticlock(larger), and end=clockwise(smaller)
				theta_local	=	ang_loc[1] + (double)l*(ang_loc[0] - ang_loc[1])/(double)((int)(blockDim.x/ratio)-1);


				A_col[threadIdx.x] = 0.0;

				h = 2.0*max_range/(double)(blockDim.x-1);

				if (image_found < 2)
				{
					oversample_factor = 5*(rad_loc[1]-rad_loc[0])/max_range;
					r_outer = left_mid + (theta_local-ang_loc[1])*slope + oversample_factor*max_range;
					r_inner = r_outer - 2*oversample_factor*max_range;
					r_inner = r_inner > rad_loc[0] ? r_inner : rad_loc[0];
					r_outer = r_outer < rad_loc[1] ? r_outer : rad_loc[1];
					h = (r_outer - r_inner)/(double)(blockDim.x-1);
				}
				else
				{
					oversample_factor = 1;
					if (boundary[0] > 0.0)
					{
						r_inner = 0.5*(boundary[0]+boundary[1]) + (theta_local - prev_theta_local)*slope - max_range;
					}
				}

				if (r_inner < 0.0)
				{
					r_inner = 0.0;
				} 

				A_col[threadIdx.x] = 0.0;
				for (k=0; k<oversample_factor; k++)
				{

					r_local = r_inner + (threadIdx.x + ((double)k/(double)oversample_factor)) * h;

					A_col[threadIdx.x] += check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
													u1_source, u2_source, rho);

				}
				__syncthreads();

			////////////////////////////////	
			//	Make a single thread determine the inner and outer (closest) edge locations

				if (threadIdx.x == 0)
				{
				//	Set a flag value to determine if any values inside this coloumn are inside the source. (edges shouldn't be)
					boundary[0] = -1.0;
					outer_boundary[0] = -1.0;
					A_row[l] = 0.0;
					
				//	Determine the inner radial edge of the img.
					if (r_inner > 0.0)
					{
					
						index = 0;
						while (index < blockDim.x)
						{
							if (A_col[index] > 0.1)
							{
								boundary[1]	= r_inner + (index + 2 + (A_col[index]-1)/(double)oversample_factor) * h;
								boundary[0]	= boundary[1] - 3* h; 
								break;
							}
							index += 1;
						}

					}

					//	Determine the outer radial edge of the img.
					index = blockDim.x-1;
					while(index >= 0)
					{
						if (A_col[index] > 0.1)
						{
							boundary_inside_edge_loc[1] = (int)index;
							outer_boundary[0] = r_inner + (index - 1 + (A_col[index]-1)/(double)oversample_factor) * h;
							outer_boundary[1] = outer_boundary[0] + 3* h; 
							break;
						}
						index -= 1;
					}

				}
				__syncthreads();
				

				if ((boundary[0] < 0.0) && (outer_boundary[0] < 0.0))
				{
					__syncthreads();
					continue;
				}

					
				// First time through we do an extra scan
				if ((image_found < 2) && (boundary[0] > 0) && (outer_boundary[0] > 0))
				{
					h = (boundary[1] - boundary[0])/(blockDim.x-1.0);
					r_inner = boundary[0];
					r_local = r_inner + threadIdx.x * h;

					A_col[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
												u1_source, u2_source, rho);
					__syncthreads();

					if (threadIdx.x == 0)
					{
					//	Set a flag value to determine if any values inside this coloumn are inside the source. (edges shouldn't be)
						boundary[0] = -1.0;
						A_row[l] = 0.0;
						
					//	Determine the inner radial edge of the img.
						index = 0;
						while (index < blockDim.x)
						{
							if (A_col[index] > 0.1)
							{
								boundary[1]	= r_inner + (index + 2) * h;
								boundary[0]	= boundary[1] - 3* h; 
								break;
							}
							index += 1;
						}
					}
					__syncthreads();

				}


				//while ( (h > 0.1*((boundary[1] - boundary[0])/(double)(blockDim.x-1))*((boundary[1] - boundary[0])/(double)(blockDim.x-1))) 	&& (h > 1.0e-08) )
				h = 1.0;
				loop_index = 0;
				while ( h > H_ERR )
				{

					loop_index++;
					if (loop_index > 100)
					{
						break;
					}

					h = (boundary[1] - boundary[0])/(double)(blockDim.x-1);

				//	Assign a new sub-divided range of rad values, 0=inner, and end = outer
					r_local	=	boundary[0] + threadIdx.x*h;

					A_local[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
													u1_source, u2_source, rho);
					__syncthreads();
					

				//	Test if the tread is inside the image, and then check if the previous thread is outside. If so this is the boundary
				//	Use Parrallel summing techniques to determine the boundary location.

					parallel_sum(A_local);
					__syncthreads();
					
					if (threadIdx.x == 0)
					{
//						boundary[0] = A_local[0] -1.0;
//						boundary[1] = A_local[0];
						boundary[1]	= boundary[1] - (A_local[0]-1) * h;
						boundary[0]	= boundary[1] - 2 * h; 
						//boundary[0] = boundary[0] + A_local[0]*h;
						//boundary[1] = boundary[0] + h;
					}
					__syncthreads();
					
				}

			//	Record the inside image outer radial edge location
				if (threadIdx.x == 0)
				{
					boundary_limb_edge[0] = boundary[1];
				}
				__syncthreads();
					
				
			////////////////////////////////////////////////////////////	
			//	Now repeat the process for the outer edge of the coloumn
				
			//	Determine the outer edge location


				boundary[0] = outer_boundary[0];
				boundary[1] = outer_boundary[1];

				// First time through we do an extra scan
				if ((image_found < 2) && (boundary[0] > -0.5))
				{
					h = (outer_boundary[1] - outer_boundary[0])/(blockDim.x-1.0);
					r_inner = outer_boundary[0];
					r_local = r_inner + threadIdx.x * h;

					A_col[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
													u1_source, u2_source, rho);
					__syncthreads();

					if (threadIdx.x == 0)
					{
					//	Set a flag value to determine if any values inside this coloumn are inside the source. (edges shouldn't be)
						boundary[0] = -1.0;
						A_row[l] = 0.0;
						
					//	Determine the inner radial edge of the img.
						index = blockDim.x - 1;
						while (index >= 0)
						{
							if (A_col[index] > 0.1)
							{
								boundary[1]	= r_inner + (index + 1) * h;
								boundary[0]	= boundary[1] - 2 * h; 
								break;
							}
							index -= 1;
						}
					}
					__syncthreads();

				}

			//	Keep sub-dividing boundary location to determine the limb edge to the desired accuracy (0.1h^2).

				__syncthreads();

				h = 1.0;
				loop_index = 0;
				while ( h > H_ERR )
				{

					loop_index++;
					if (loop_index > 100)
					{
						break;
					}

					h = (boundary[1] - boundary[0])/(double)(blockDim.x-1);

				//	Assign a new sub-divided range of rad values, 0=inner, and end = outer
					r_local	=	boundary[0] + threadIdx.x*h;

					A_local[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
													u1_source, u2_source, rho);
					__syncthreads();
					
				//	Test if the tread is inside the image, and then check if the previous thread is outside. If so this is the boundary
				//	Use Parrallel summing techniques to determine the boundary location.

					parallel_sum(A_local);
					__syncthreads();
					
					if (threadIdx.x == 0)
					{
						boundary[1]	= boundary[0] + (A_local[0]+1) * h;
						boundary[0]	= boundary[1] - 2 * h; 
					}
					__syncthreads();
					
				}
			
			//	Record the inside image outer radial edge location
				if (threadIdx.x == 0)
				{
					boundary_limb_edge[1] = boundary[0];
					if (boundary_limb_edge[0] < 0.0)
					{
						boundary_limb_edge[0] = 0.0;
					}
				}
				__syncthreads();



				
			////////////////////////////////////////////////////////////
			//	Increase the resolution of the search area of the image to ensure accuracy is achieved if only a few points fell inside the image
			//	Take the boundary limits, and determine a new search resolution
			
				if (threadIdx.x == 0)
				{
					boundary[0] = boundary_limb_edge[0];
					boundary[1] = boundary_limb_edge[1];
					if (boundary[0] < rad_loc[0])
					{
						rad_loc[0] = boundary[0];
					}
					if (boundary[1] > rad_loc[1])
					{
						rad_loc[1] = boundary[1];
					}
					if (boundary[0] < img_grid_dims[4*j + 0])
					{
						img_grid_dims[4*j + 0] = boundary[0];
					}
					if (boundary[1] > img_grid_dims[4*j + 1])
					{
						img_grid_dims[4*j + 1] = boundary[1];
					}
				}
				__syncthreads();

				h = (boundary[1] - boundary[0]) / (blockDim.x - 1.0);

			//	Calculate the new  rad co-ordinates of this new integration coloumn
				r_local = boundary[0] + threadIdx.x * h;
				
			//	Perform inverse ray shooting to determine the contribution to the image area.
				zeta = COMPLEXTYPE( r_local*cos(theta_local), r_local*sin(theta_local));
				zetabar = COMPLEXTYPE(real(zeta), -imag(zeta));
				
			//	Determine the inverse ray position
				omega = zeta - ( (e1) / (zetabar-(a-b[j])) )   -  ( (e2) / (zetabar+(a+b[j])) );
				
			//	Calculate the distance between the ray and the source centre
				A_col[threadIdx.x] = 0.0;
				if (   (u1_source-real(omega))*(u1_source-real(omega)) + (u2_source-imag(omega))*(u2_source-imag(omega))  <  rho*rho  )
				{
					A_col[threadIdx.x] = 1.0 - limb_const*(1.0 - sqrt(1.0-((u1_source-real(omega))*(u1_source-real(omega)) + (u2_source-imag(omega))*(u2_source-imag(omega)))/(rho*rho) ) );
				}
				A_col[threadIdx.x] *= r_local;
				__syncthreads();
		
			////////////////////////////////////////////////////////////	
			//	Make the first two threads deal with the edge boundary special conditions.
				if (((threadIdx.x == 0) && (boundary_limb_edge[0] > 1.e-9)) || (threadIdx.x == blockDim.x-1))
				{
					A_col[threadIdx.x] *= 0.5;
				}
				__syncthreads();
				
			//	Perform a parrallel sum over the coloumn and store it in a new array, multiplied by it's angle due to polar coordinates.
			//	Determine the smallest value of 2^n greater or equal to the number of pixels in the convolved map for parrallel summing.

				parallel_sum(A_col);
				__syncthreads();
				
				
				if (threadIdx.x == 0)
				{
					A_row[l] = h * A_col[0];

					if (A_row[l] > 1.e-10)
					{
						if (COLOUMN_FLAG == 0)
						{
							A_limb[0] = boundary_limb_edge[0];
							COLOUMN_FLAG = 1;
						} else {
							A_limb[1] = boundary_limb_edge[1];
						}

						mid = 0.5*(boundary[0]+boundary[1]);
						if (prev_mid > 0.0)
						{
							slope = (mid - prev_mid)/(theta_local - prev_theta_local);
							image_found += 1;
						}
						prev_mid = mid;
						prev_theta_local = theta_local;

					}

				}
				__syncthreads();

				
			} // coloumn integration loop end
			

			__syncthreads();


			if (1 < 0)
			{


		////////////////////////////////////////////////////////////////	
		//	Now integrate over the length of the row (angular)
			
			//r_local = right_mid - 0.01*(right_mid - left_mid);
			r_local = right_mid;
			
		//	If the image is a complete ring, ignore the boundary condition for the rows
			if (RING_FLAG == 0)
			{
				
			//	Determine the clockwise (smaller theta) edge of the image in the angular direction
				if (threadIdx.x == 0)
				{
					h = (ang_loc[0] - ang_loc[1]) / (blockDim.x/ratio-1.0);
					k = 0;
					while (k < blockDim.x)
					{
						if (A_row[k] > 1.e-10) 
						{
							boundary[0] = ang_loc[1] + (k+1) * h;
							boundary[1] = boundary[0] - 4*h;
							boundary_inside_edge[0] = boundary[0];
							boundary_inside_edge_loc[0] = (double)k;
							break;
						}
						k++;
					}

				}
				__syncthreads();


			//	Keep sub-dividing boundary location to determine the limb edge to the desired accuracy (0.1h^2).

				h = (boundary[0]-boundary[1])/(blockDim.x-1);
				//while ( (h > 0.1 * ((boundary[0] - boundary[1]) / (double)(int)((blockDim.x/ratio)-1 )) * ((boundary[0] - boundary[1]) / (double)(int)((blockDim.x/ratio)-1 )))	&& (h > 1.0e-08) )
				loop_index = 0;
				while (h > H_ERR)
				{

					loop_index++;
					if (loop_index > 100)
					{
						break;
					}

				//	Assign a new sub-divided range of ang values, 0=largest(anticlock), and end = smallest(clockwise)
					theta_local	=	boundary[1] + threadIdx.x*h;
					
					A_local[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
													u1_source, u2_source, rho);
					__syncthreads();
					
					if (threadIdx.x == 0)
					{

						k = 0;
						while (k < blockDim.x)
						{
							if (A_local[k] > 1.e-10) 
							{
								boundary[0] = boundary[1] + k * h;
								boundary[1] = boundary[0] - h;
								boundary_inside_edge[0] = boundary[0];
								boundary_limb_edge[0] = boundary[0];
								break;
							}
							k++;
						}

					}
					__syncthreads();


				//	Determine the new resolution of the grid spacing
					h = (boundary[0]-boundary[1])/(blockDim.x-1);

					__syncthreads();
				}

				__syncthreads();

				
			////////////////////////////////////////////////////////////////
			//	Now repeat the process for the outer angular edge
				
				//r_local = left_mid + 0.01*(right_mid - left_mid);
				r_local = left_mid;
				
			//	Determine the outer most anticlockwise edge of the image in the angular direction
				if (threadIdx.x == 0)
				{
					h = (ang_loc[0] - ang_loc[1]) / (blockDim.x/ratio-1.0);
					k = blockDim.x;
					while (k >= 0)
					{
						if (A_row[k] > 1.e-10) 
						{
							boundary[1] = ang_loc[1] + (k-1) * h;
							boundary[0] = boundary[1] + 4*h;
							boundary_inside_edge[1] = boundary[1];
							boundary_inside_edge_loc[1] = (double)k;
							break;
						}
						k--;
					}
				}
				__syncthreads();
				
				
			//	Keep sub-dividing boundary location to determine the limb edge to the desired accuracy (0.1h^2).
				

				h = (boundary[0]-boundary[1])/(blockDim.x-1);
				//while ( (h > 0.1 * ((boundary[0] - boundary[1]) / (double)(int)((blockDim.x/ratio)-1 )) * ((boundary[0] - boundary[1]) / (double)(int)((blockDim.x/ratio)-1 ))) 	&& (h > 1.0e-08) )
				loop_index = 0;
				while (h > H_ERR)
				{

					loop_index++;
					if (loop_index > 100)
					{
						break;
					}

				//	Assign a new sub-divided range of angle values, 0=inner, and end = outer
					theta_local	=	boundary[1] + threadIdx.x*h;

					A_local[threadIdx.x] = check_ray_inside_source(r_local, theta_local, e1, e2, a, b[j],
													u1_source, u2_source, rho);
					__syncthreads();

					if (threadIdx.x == 0)
					{

						k = blockDim.x;
						while (k >= 0)
						{
							if (A_row[k] > 1.e-10) 
							{
								boundary[1] = boundary[1] + k * h;
								boundary[0] = boundary[1] + h;
								boundary_inside_edge[1] = boundary[1];
								boundary_limb_edge[1] = boundary[1];
								break;
							}
							k--;
						}

					}
					__syncthreads();


				//	Determine the new resolution of the grid spacing
					h = (boundary[0]-boundary[1])/(blockDim.x-1);
				}
				
				__syncthreads();

				
			////////////////////////////////////////////////////////////////
			//	Perform modifications to the components of the row values which need summing


				if (threadIdx.x == 0)
				{
					eta_gs	=	fabs(boundary_inside_edge[0] - boundary_limb_edge[0]) / (double)((ang_loc[0] - ang_loc[1]) / (double)(int)((blockDim.x/ratio)-1 ));
					A_row[boundary_inside_edge_loc[0]-1] *= ( 0.375 + eta_gs + 0.5*eta_gs*eta_gs );
					A_row[boundary_inside_edge_loc[0]] *= ( 1.125 - eta_gs*eta_gs*0.5 );

					eta_gs	=	fabs(boundary_inside_edge[1] - boundary_limb_edge[1]) / (double)((ang_loc[0] - ang_loc[1]) / (double)(int)((blockDim.x/ratio)-1 ));
					A_row[boundary_inside_edge_loc[1]+1] *= ( 0.375 + eta_gs + 0.5*eta_gs*eta_gs );
					A_row[boundary_inside_edge_loc[1]] *= ( 1.125 - eta_gs*eta_gs*0.5 );

				}

				__syncthreads();
				
			} // end condition that will skip this step if the image is a complete ring
			else
			{
				if ((threadIdx.x == 0) || (threadIdx.x == blockDim.x-1))
				{
					A_row[threadIdx.x] *= 0.5;
				}
				__syncthreads();

			}

			}

		//	Perform a parrallel sum over the coloumn and store it in a new array, multiplied by it's radius due to polar coordinates.
			
			__syncthreads();

			parallel_sum(A_row);
			__syncthreads();

			
			if (threadIdx.x == 0)
			{
				A_img[j] =  ((ang_loc[0] - ang_loc[1]) / (double)(int)((blockDim.x/ratio)-1 )) * A_row[0];
				h = (ang_loc[0] - ang_loc[1]) / (double)(int)((blockDim.x/ratio)-1 );
			}
			__syncthreads();
			
			if (threadIdx.x == 0)
			{

				//	Test if this new grid included the centre of a previous image. 
				//  If so clear the previous image calculation.
				//

				for (k=0; k<j; k++)
				{

					map_coords(&r_imgs[k],&theta_imgs[k],b[k],&r_local,&theta_local,b[j]);

					if (theta_local < 0.0) 
					{
						theta_local += 6.2831853;
					}

					//if ( (rad_loc[0] < r_imgs[k])  &&  (r_imgs[k] < rad_loc[1])  && ( ( (theta_imgs[k] < ang_loc[0])  && (ang_loc[1] < theta_imgs[k]) ) || ( (theta_imgs[k]+2.0*PI < ang_loc[0])  && (ang_loc[1] < theta_imgs[k]+2.0*PI) ) || ( (theta_imgs[k]-2.0*PI < ang_loc[0])  && (ang_loc[1] < theta_imgs[k]-2.0*PI) ) ) )
					if ( (rad_loc[0] < r_local)  &&  (r_local < rad_loc[1])  && ( ( (theta_local < ang_loc[0])  && (ang_loc[1] < theta_local) ) || ( (theta_local+2.0*PI < ang_loc[0])  && (ang_loc[1] < theta_local+2.0*PI) ) || ( (theta_local-2.0*PI < ang_loc[0])  && (ang_loc[1] < theta_local-2.0*PI) ) ) )

					{
						A_img[k] = 0.0;
					}
				}
				
			}
			__syncthreads();



			
		}	// image loop end

		__syncthreads();

		if (threadIdx.x == 0)
		{
		//	Sum all image areas together.
			A_img[0] += A_img[1] + A_img[2] + A_img[3] + A_img[4];
			
		//	take area of images divided by the source area as the magnification.
			A[blockIdx.x] = A_img[0] / (2.0*PI*rho*rho  *((3.0-limb_const)/6.0) );
		}
		
		__syncthreads();

		return;
	}
	
	//////////////////////////////// - Magnification at the data points
	
	""")


