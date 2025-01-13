#include <cstdio>

#define TRUE true;
#define FALSE false;

// Basic matrix/vector multiplication
__global__ void mv_kernel (int n, bool compact_a, int * row_starts, double * non_zeros,
			double * soln, double * soln2, unsigned int * cols,
			double * dist, int dist_mask, int dist_shift){

		int i,j,k, l=0, h;
		
		double d;
		
		i =  (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
		
		for (j = 0; j < 8; ++j) {			
			if (i < n) {

				d = 0;

				l = row_starts[i];
				h = row_starts[i+1];

				// "row major" version
				if (!compact_a) {
					for (k = l; k < h; k++) {
						d += (non_zeros[k]) * soln[cols[k]];
					}
				}
				// "compact msr" version
				else {
					for (k = l; k < h; k++) {
						d += (dist[(int)(cols[k] & dist_mask)]) * soln[(int)(cols[k] >> dist_shift)];
					}
				}
			}
			//__syncthreads();
			if (i < n) {
				soln2[i] = d;
			}
			i += blockDim.x;
		}
}

// Matrix/vector multiplication for non det bounded until
__global__ void mv_nd_kernel (int n, bool compact_a, int * row_starts, double * non_zeros,
			double * soln, double * soln2, unsigned int * cols,
			double * dist, int dist_mask, int dist_shift, int * choice_starts, double * yes_vec,
			bool min, bool instreward) {

		int i,j,k1, k2, l1=0, h1, l2, h2;
		double d1, d2;
		bool first = FALSE;
		
		i =  (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
		
		for (j = 0; j < 8; ++j) {			
			if (i < n) {
				if (instreward) {
					d1 = 0;
					first = TRUE;
				}
				else {
					d1 = min ? 2 : -1;
				}
				l1 = row_starts[i];
				h1 = row_starts[i+1];
				for (k1 = l1; k1 < h1; k1++) {
					d2 = 0.0;
					l2 = choice_starts[k1];
					h2 = choice_starts[k1+1];
					// "row major" version
					if (!compact_a) {
						for (k2 = l2; k2 < h2; k2++) {
							d2 += (non_zeros[k2]) * soln[cols[k2]];
						}
					}
					// "compact msr" version
					else {
						for (k2 = l2; k2 < h2; k2++) {
							d2 += (dist[(int)(cols[k2] & dist_mask)]) * soln[(int)(cols[k2] >> dist_shift)];
						}
					}
					if (min) {
						if (first | d2 < d1) d1 = d2;
					}
					else {
						if (first | d2 > d1) d1 = d2;
					}
				}
				// set vector element
				// (if no choices, use value of yes, or zero if instreward)
				d2 = (h1 > l1) ? d1 : (instreward ? 0 : yes_vec[i]);
			}
			//__syncthreads();
			if (i < n) {
				soln2[i] = d2;
			}
			i += blockDim.x;
		}
}

// Matrix/vector multiplication for non det reach reward
__global__ void mv_ndr_kernel (int n, int * row_starts, double * non_zeros,
			double * soln, double * soln2, unsigned int * cols,
			double * dist, int dist_mask, int dist_shift,
			int	*choice_starts,	int *choice_starts_r, double * non_zeros_r, unsigned int * cols_r,
			double * sr_vec, double *inf_vec, int export_adv, int * adv, bool min, double huge) {

		int i,j,k1, k2, k_r, l1=0, h1, l2, h2, l2_r, h2_r;
		double d1, d2;
		bool first = FALSE;
		
		i =  (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
		
		for (j = 0; j < 8; ++j) {			
			if (i < n) {
				d1 = 0.0;
				first = TRUE;
				
				l1 = row_starts[i];
				h1 = row_starts[i+1];
				for (k1 = l1; k1 < h1; k1++) {
					d2 = sr_vec[i];
					l2 = choice_starts[k1];
					h2 = choice_starts[k1+1];
					l2_r = choice_starts_r[k1];
					h2_r = choice_starts_r[k1+1];
					
					for (k2 = l2; k2 < h2; k2++) {
						k_r = l2_r; while (k_r < h2_r && cols_r[k_r] != cols[k2]) k_r++;
						if (k_r < h2_r) {soln2[i] += non_zeros_r[k_r] * non_zeros[k2]; k_r++; }
						d2 += (non_zeros[k2]) * soln[cols[k2]];
					}
					if (first || (min&&(d2<d1)) || (!min&&(d2>d1))) {
						d1 = d2;
						if (export_adv) {
							if (!min) {
								if (adv[i] == -1 || d1>soln[i]) {
									adv[i] = k1;
								}
							}
							else {
								adv[i] = k1;
							}
						}
					}
					first = FALSE;
				}
				// set vector element
				// (if there were no choices from this state, reward is zero/infinity)
				d2 = (h1 > l1) ? d1 : inf_vec[i] > 0 ? huge : 0;
			}
			//__syncthreads();
			if (i < n) {
				soln2[i] = d2;
			}
			i += blockDim.x;
		}
}

// Matrix/vector multiplication for non det until
__global__ void mv_ndu_kernel (int n, int * row_starts, double * non_zeros,
			double * soln, double * soln2, unsigned int * cols,
			double * dist, int dist_mask, int dist_shift,
			int	*choice_starts,
			double * yes_vec, int export_adv, int * adv, bool min) {

		int i,j,k1, k2, l1=0, h1, l2, h2;
		double d1, d2;
		bool first = FALSE;
		
		i =  (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
		
		for (j = 0; j < 8; ++j) {			
			if (i < n) {
				d1 = 0.0;
				first = TRUE;
				
				l1 = row_starts[i];
				h1 = row_starts[i+1];
				for (k1 = l1; k1 < h1; k1++) {
					d2 = 0.0;
					l2 = choice_starts[k1];
					h2 = choice_starts[k1+1];
					
					for (k2 = l2; k2 < h2; k2++) {
						d2 += (non_zeros[k2]) * soln[cols[k2]];
					}
					if (first || (min&&(soln2[i]<d1)) || (!min&&(soln2[i]>d1))) {
						d1 = d2;
						if (export_adv) {
							if (!min) {
								if (adv[i] == -1 || d1>d2) {
									adv[i] = k1;
								}
							}
							else {
								adv[i] = k1;
							}
						}
					}
					first = FALSE;
				}
				// set vector element
				// (if there were no choices from this state, reward is zero/infinity)
				d2 = (h1 > l1) ? d1 : yes_vec[i];
			}
			if (i < n) {
				soln2[i] = d2;
			}
			i += blockDim.x;
		}
}

/** 
  * @brief Modify the diags values for stochastic bounded until checks
  *
  * This __global__ device function takes an element from the vector diags_vec or
  * diags_dist, depending on the value of compact_d, 
  * and divides it by unif, and adding 1 to the result 
  *
  */
__global__ 
void sparseMatrixVectorModifyDiags_stbu(
								bool             compact_d,
                                double			 *diags_vec,
								double			 *diags_dist,
								int				 n,
								int				 num_dist,
								double			 unif
                                )
{
    unsigned int i = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;

    for (unsigned int j = 0; j < 8; ++j)
    {
		if (!compact_d) {
			if (i < n) {
				diags_vec[i] = diags_vec[i] / unif + 1;
			}
		}
		else if (i < num_dist) {
			diags_dist[i] = diags_dist[i] / unif + 1;
		}

        i += blockDim.x;
    }
    
    __syncthreads();
}


/** 
  * @brief Uniformization of matrix for stochastic bounded until checks
  *
  * This __global__ device function takes an element from the non_zeros or dist
  * depending on the value of compact_tr, 
  * and divides it by unif 
  *
  */
__global__ 
void sparseMatrixUniformization_stbu(
								bool             compact_tr,
                                double			 *non_zeros,
								double			 *dist,
								long			 nnz,
								double			 unif
                                )
{
    unsigned int i = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;

    for (unsigned int j = 0; j < 8; ++j)
    {
		if (i < nnz) {
			if (!compact_tr) {
				non_zeros[i] /= unif;
			}
			else {
				dist[i] /= unif;
			}
		}

        i += blockDim.x;
    }
    
    __syncthreads();
}


/** 
  * @brief Set sum vector to zero
  */
__global__ 
void sparseMatrixVectorSetSumToZero(
                                double					*sum,
                                int						 n
                                )
{
    unsigned int i = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;

    for (unsigned int j = 0; j < 8; ++j)
    {
        if (i < n) {
				sum[i] = 0.0;
		}

        i += blockDim.x;
    }
    
    __syncthreads();
}


/** 
  * @brief Summation of sum elements with a given weight for stochastic
  * bounded until checks
  *
  * This __global__ device function takes an element from soln
  * and multiplies it by weight. The result is added to sum. 
  *
  */
__global__ 
void sparseMatrixVectorSummation_stbu(
								double           *sum,
                                double			 *soln,
								int			     n,
								double			 weight
                                )
{
    unsigned int i = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;

    for (unsigned int j = 0; j < 8; ++j)
    {
		if (i < n) {
			sum[i] = (sum[i] + (weight * soln[i]));
		}

        i += blockDim.x;
    }
    
    __syncthreads();
}

// Division by unif for Stochastic Cumul Reward
__global__ 
void sparseMatrixVectorDivUnif_stbu(
								double           *sum,
                                double			 *soln,
								int			     n,
								double			 unif
                                )
{
    unsigned int i = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;

    for (unsigned int j = 0; j < 8; ++j)
    {
		if (i < n) {
			sum[i] = (sum[i] + (soln[i] / unif));
		}

        i += blockDim.x;
    }
    
    __syncthreads();
}

/** 
  * @brief Gather final y values kernel and compute final results for Jacobi iteration
  *
  * This __global__ device function takes an element from the vector d_rowFindx,
  * which for each row gives the index of the last element of that row, reads the
  * corresponding position in d_prod and write it in d_y
  *
  * Template parameter \a T is the datatype of the matrix A and x.
  *
  * @param[out] d_y The output result array
  * @param[in] d_prod The input products array (which now contains sums for each row)
  * @param[in] d_rowFindx The starting index of each row in the "flattened" version
  *                       of matrix A
  * @param[in] numRows The number of rows in matrix A
  */
__global__ 
void gather(
             double                 *soln2,
             unsigned int			numRows,
			 double					*soln,
			 void					*b,
			 bool					compact_b,
			 bool					compact_d,
			 double					*b_vec,
			 unsigned short			*b_dist_ptrs,
			 double					*b_dist_dist,
			 double					*diags_vec,
			 unsigned short			*diags_dist_ptrs,
			 double					*diags_dist_dist,
			 double					omega,
			 int					term_crit,
			 double					term_crit_param,
			 char					*done
             )
{
    
    unsigned int i = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
    
	double d = 0;
	
    for (unsigned int j=0; j < 8; ++j)
    {
		if (i < numRows) {
			d = (b == NULL) ? 0.0 : ((!compact_b) ? b_vec[i] : b_dist_dist[b_dist_ptrs[i]]);
			d -= soln2[i];
			// divide by diagonal (multiply by inverted diagonal)
			if (!compact_d) d *= diags_vec[i]; else d *= diags_dist_dist[diags_dist_ptrs[i]];
			// over-relaxation
			if (omega != 1.0) {
				d = ((1-omega) * soln[i]) + (omega * d);
			}
			soln2[i] = d;

			// Termination check
			switch (term_crit) {
				case 1:
					if (fabs(soln2[i] - soln[i]) > term_crit_param) {
						*done = 0;
					}
					break;
				case 2:
					if (fabs(soln2[i] - soln[i])/soln2[i] > term_crit_param) {
						*done = 0;
					}
					break;
			}
        }
		i += blockDim.x;
    }

    __syncthreads();
}

// Gather for Power method
__global__ 
void gather_power(
             double                 *soln2,
             unsigned int			numRows,
			 double					*soln,
			 void					*b,
			 bool					compact_b,
			 double					*b_vec,
			 unsigned short			*b_dist_ptrs,
			 double					*b_dist_dist,
			 int					term_crit,
			 double					term_crit_param,
			 char					*done
             )
{
    
    unsigned int i = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
    
	double d = 0;
	
    for (unsigned int j=0; j < 8; ++j)
    {
		if (i < numRows) {
			d = (b == NULL) ? 0.0 : ((!compact_b) ? b_vec[i] : b_dist_dist[b_dist_ptrs[i]]);
			d += soln2[i];
			
			// Termination check
			switch (term_crit) {
				case 1:
					if (fabs(soln2[i] - soln[i]) > term_crit_param) {
						*done = 0;
					}
					break;
				case 2:
					if (fabs(soln2[i] - soln[i])/soln2[i] > term_crit_param) {
						*done = 0;
					}
					break;
			}
        }
		i += blockDim.x;
    }

    __syncthreads();
}

/** 
  * @brief Gather final y values kernel and compute final results for matrix/vector multiplication
  * for bounded until checks
  *
  * This __global__ device function takes an element from the vector d_rowFindx,
  * which for each row gives the index of the last element of that row, reads the
  * corresponding position in d_prod and write it in d_y
  *
  * Template parameter \a T is the datatype of the matrix A and x.
  *
  * @param[out] d_y The output result array
  * @param[in] d_prod The input products array (which now contains sums for each row)
  * @param[in] d_rowFindx The starting index of each row in the "flattened" version
  *                       of matrix A
  * @param[in] numRows The number of rows in matrix A
  */
__global__ 
void gather_bu(
             double                 *soln2, 
             unsigned int			numRows,
			 bool					compact_y,
			 double					*yes_vec,
			 unsigned short			*yes_dist_ptrs,
			 double					*yes_dist_dist
             )
{
    unsigned int i = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
    
    for (unsigned int j=0; j < 8; ++j)
    {
		if (i < numRows) {
			// Set yes states to 1
			if (!compact_y) { if (yes_vec[i]) soln2[i] = 1.0; } else { if (yes_dist_dist[yes_dist_ptrs[i]]) soln2[i] = 1.0; }
		}
		i += blockDim.x;
    }

    __syncthreads();
}

// Gather for prob cumul reward
__global__ 
void gather_pcr(
             double                 *soln2, 
             unsigned int			numRows,
			 bool					compact_r,
			 double					*rew_vec,
			 unsigned short			*rew_dist_ptrs,
			 double					*rew_dist_dist
             )
{
    unsigned int i = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
    
	double d;
	
    for (unsigned int j=0; j < 8; ++j)
    {
		if (i < numRows) {
			d = (!compact_r) ? rew_vec[i] : rew_dist_dist[rew_dist_ptrs[i]];
			soln2[i] = d + soln2[i];
		}
		i += blockDim.x;
    }

    __syncthreads();
}

// Gather for ProbTransient
__global__ 
void gather_pt (
             double             *soln2, 
             unsigned int       numRows,
			 double				*soln,
			 int				term_crit,
			 double				term_crit_param,
			 char				*done
             )
{
    
    unsigned int i = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
    
    for (unsigned int j=0; j < 8; ++j) {
		if (i < numRows) {
			switch (term_crit) {
				case 1:
					if (fabs(soln2[i] - soln[i]) > term_crit_param) {
						*done = 0;
					}
					break;
				case 2:
					if (fabs(soln2[i] - soln[i])/soln2[i] > term_crit_param) {
						*done = 0;
					}
					break;
			}
        }
		i += blockDim.x;
    }

    __syncthreads();

}

/** 
  * @brief Gather final y values kernel and compute final results for stochastic
  * bounded until checks
  *
  * This __global__ device function takes an element from the vector d_rowFindx,
  * which for each row gives the index of the last element of that row, reads the
  * corresponding position in d_prod and write it in d_y
  */
__global__ 
void gather_stbu (
             double             *soln2, 
             unsigned int       numRows,
			 double				*soln,
			 bool				compact_d,
			 double				*diags_vec,
			 unsigned short		*diags_dist_ptrs,
			 double				*diags_dist_dist,
			 int				term_crit,
			 double				term_crit_param,
			 char				*done,
			 bool				do_ss_detect
             )
{
    
    unsigned int i = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
    
	double d = 0;
	
    for (unsigned int j=0; j < 8; ++j) {
		if (i < numRows) {
			d = (!compact_d) ? (diags_vec[i] * soln[i]) : (diags_dist_dist[diags_dist_ptrs[i]] * soln[i]);
			d += soln2[i];
			soln2[i] = d;
		
			if (do_ss_detect) {
				switch (term_crit) {
					case 1:
						if (fabs(soln2[i] - soln[i]) > term_crit_param) {
							*done = 0;
						}
						break;
					case 2:
						if (fabs(soln2[i] - soln[i])/soln2[i] > term_crit_param) {
							*done = 0;
						}
						break;
				}
			}
        }
		i += blockDim.x;
    }

    __syncthreads();

}

