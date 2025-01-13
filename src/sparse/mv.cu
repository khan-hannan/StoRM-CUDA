const int CTA_SIZE = 512;
const int ELTS_PER_THREAD = 8;

#include "mv_kernel.cu"
#include "cuda.h"
#include "cutil.h"
#include <dv.h>
#include "jnipointer.h"

extern "C" {
		double * b_vec_dev ;
		unsigned short * ptrs_dev;
		double * b_dist_dev;
		int * row_starts_dev;
		double * non_zeros_dev;
		unsigned int * cols_dev;
		double * soln_dev;
		double * soln2_dev;
		double * dist_dev;
		double * diags_vec_dev;
		unsigned short * diags_ptrs_dev;
		double * diags_dist_dev;
		char * done_dev;
		int vmem = 0;
		
		// Additional variables for matrix/vector multiplications for bounded until checks
		double * yes_vec_dev;
		unsigned short * yes_ptrs_dev;
		double * yes_dist_dev;
		
		
		// Additional variables for matrix/vector multiplications for Prob Cumul Reward
		double * rew_vec_dev;
		unsigned short * rew_ptrs_dev;
		double * rew_dist_dev;
		
		// Additional variables for stochastic bounded until checks
		double * sum_dev;
		
		// Additional variables for non-det bounded until checks
		int * choice_starts_dev;
		
		// Additional variables for non-det reach reward
		int * choice_starts_r_dev;
		double * non_zeros_r_dev;
		unsigned int * cols_r_dev;
		int * adv_dev;
		double *sr_vec_dev;
		double *inf_vec_dev;

		//wrapper around cudaMalloc to count allocated memory and check for error while allocating
		int cudaMallocCount ( void ** ptr,int size) {
			cudaError_t err = cudaSuccess;
			vmem += size;
			err = cudaMalloc(ptr,size);
			if (err) printf("%s \n", cudaGetErrorString(err));
			fprintf (stdout, "allocated %d\n", size);
			return size;
		}

		void  datacpy (int n, void * b,bool compact_a, bool compact_b, bool compact_d, double * b_vec, DistVector * b_dist,
			int *row_starts, double *non_zeros, double *soln, unsigned int *cols,double * dist,
			int dist_mask, int dist_shift, double *diags_vec, DistVector * diags_dist) {

			cudaError_t err = cudaSuccess;

			printf("copying data \n");

			if (b != NULL) {
				if (!compact_b) {
					fprintf (stdout, "b:\n");
					cudaMallocCount((void **) &b_vec_dev,n*sizeof(double));
					err = cudaMemcpy(b_vec_dev,b_vec,n*sizeof(double),cudaMemcpyHostToDevice);
					if (err) printf("%s \n", cudaGetErrorString(err));
				}
				else {
					fprintf (stdout, "b:\n");
					cudaMallocCount((void **) &ptrs_dev,n*sizeof(unsigned short));
					err = cudaMemcpy(ptrs_dev,b_dist->ptrs,n*sizeof(unsigned short),cudaMemcpyHostToDevice);
					if (err) printf("%s \n", cudaGetErrorString(err));
					cudaMallocCount((void **) &b_dist_dev,b_dist->num_dist*sizeof(double));
					err = cudaMemcpy(b_dist_dev,b_dist->dist,b_dist->num_dist*sizeof(double),cudaMemcpyHostToDevice);
					if (err) printf("%s \n", cudaGetErrorString(err));
				}
			}
			int cols_size=0;
		 
			fprintf (stdout, "row starts:\n");
			cudaMallocCount((void **) &row_starts_dev,(n+1)*sizeof(int));
			err = cudaMemcpy(row_starts_dev,row_starts,(n+1)*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			cols_size = row_starts[n]; 
			if (!compact_a) {
				fprintf (stdout, "non_zeros:\n");
				cudaMallocCount((void **) &non_zeros_dev,cols_size*sizeof(double));
				err = cudaMemcpy(non_zeros_dev,non_zeros,cols_size*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
			else {
				fprintf (stdout, "dist_dev:\n");
				cudaMallocCount((void **) &dist_dev,dist_mask*sizeof(double));
				err = cudaMemcpy(dist_dev,dist,dist_mask*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
			fprintf (stdout, "cols_dev:\n");
			cudaMallocCount((void **) &cols_dev,cols_size*sizeof(unsigned int));
			err = cudaMemcpy(cols_dev,cols,cols_size*sizeof(unsigned int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln_dev:\n");
			cudaMallocCount((void **) &soln_dev,n*sizeof(double));
			err = cudaMemcpy(soln_dev,soln,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln2_dev:\n");
			cudaMallocCount((void **) &soln2_dev,n*sizeof(double));
		
			if (!compact_d) {
				fprintf (stdout, "d:\n");
				cudaMallocCount((void **) &diags_vec_dev,n*sizeof(double));
				err = cudaMemcpy(diags_vec_dev,diags_vec,n*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
			else{
				fprintf (stdout, "d:\n");
				cudaMallocCount((void **) &diags_ptrs_dev,n*sizeof(unsigned short));
				err = cudaMemcpy(diags_ptrs_dev,diags_dist->ptrs,n*sizeof(unsigned short),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
				cudaMallocCount((void **) &diags_dist_dev,diags_dist->num_dist*sizeof(double));
				err = cudaMemcpy(diags_dist_dev,diags_dist->dist,diags_dist->num_dist*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}

			fprintf (stdout, "done:\n");
			cudaMallocCount((void **) &done_dev, sizeof(char));
		
			printf("copied %i KB to GPU \n",vmem/1024);
			printf("copied %i MB to GPU \n",vmem/1024/1024);
		}

		// datacpy for Power
		void  datacpy_power (int n, void * b, bool compact_a, bool compact_b, double * b_vec, DistVector * b_dist, int *row_starts,
			double *non_zeros, double *soln, unsigned int *cols,double * dist, int dist_mask, int dist_shift) {

			cudaError_t err = cudaSuccess;

			printf("copying data \n");

			if (b != NULL) {
				if (!compact_b) {
					fprintf (stdout, "b:\n");
					cudaMallocCount((void **) &b_vec_dev,n*sizeof(double));
					err = cudaMemcpy(b_vec_dev,b_vec,n*sizeof(double),cudaMemcpyHostToDevice);
					if (err) printf("%s \n", cudaGetErrorString(err));
				}
				else {
					fprintf (stdout, "b:\n");
					cudaMallocCount((void **) &ptrs_dev,n*sizeof(unsigned short));
					err = cudaMemcpy(ptrs_dev,b_dist->ptrs,n*sizeof(unsigned short),cudaMemcpyHostToDevice);
					if (err) printf("%s \n", cudaGetErrorString(err));
					cudaMallocCount((void **) &b_dist_dev,b_dist->num_dist*sizeof(double));
					err = cudaMemcpy(b_dist_dev,b_dist->dist,b_dist->num_dist*sizeof(double),cudaMemcpyHostToDevice);
					if (err) printf("%s \n", cudaGetErrorString(err));
				}
			}
			int cols_size=0;
		 
			fprintf (stdout, "row starts:\n");
			cudaMallocCount((void **) &row_starts_dev,(n+1)*sizeof(int));
			err = cudaMemcpy(row_starts_dev,row_starts,(n+1)*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			cols_size = row_starts[n]; 
			if (!compact_a) {
				fprintf (stdout, "non_zeros:\n");
				cudaMallocCount((void **) &non_zeros_dev,cols_size*sizeof(double));
				err = cudaMemcpy(non_zeros_dev,non_zeros,cols_size*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
			else {
				fprintf (stdout, "dist_dev:\n");
				cudaMallocCount((void **) &dist_dev,dist_mask*sizeof(double));
				err = cudaMemcpy(dist_dev,dist,dist_mask*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
		
			fprintf (stdout, "cols_dev:\n");
			cudaMallocCount((void **) &cols_dev,cols_size*sizeof(unsigned int));
			err = cudaMemcpy(cols_dev,cols,cols_size*sizeof(unsigned int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln_dev:\n");
			cudaMallocCount((void **) &soln_dev,n*sizeof(double));
			err = cudaMemcpy(soln_dev,soln,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln2_dev:\n");
			cudaMallocCount((void **) &soln2_dev,n*sizeof(double));
		
			fprintf (stdout, "done:\n");
			cudaMallocCount((void **) &done_dev, sizeof(char));
		
			printf("copied %i KB to GPU \n",vmem/1024);
			printf("copied %i MB to GPU \n",vmem/1024/1024);
		}
	
		// datacopy for Prob bounded until
		void  datacpy_pbu (int n, bool compact_tr, bool compact_y, double * yes_vec, DistVector * yes_dist,
			int *row_starts, double *non_zeros, double *soln, unsigned int *cols,double * dist,
			int dist_mask, int dist_shift) {

			cudaError_t err = cudaSuccess;

			printf("copying data \n");

			if (!compact_y) {
				fprintf (stdout, "y:\n");
				cudaMallocCount((void **) &yes_vec_dev,n*sizeof(double));
				err = cudaMemcpy(yes_vec_dev,yes_vec,n*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
			else {
				fprintf (stdout, "y:\n");
				cudaMallocCount((void **) &yes_ptrs_dev,n*sizeof(unsigned short));
				err = cudaMemcpy(yes_ptrs_dev,yes_dist->ptrs,n*sizeof(unsigned short),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
				cudaMallocCount((void **) &yes_dist_dev,yes_dist->num_dist*sizeof(double));
				err = cudaMemcpy(yes_dist_dev,yes_dist->dist,yes_dist->num_dist*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
		
			int cols_size=0;
		 
			fprintf (stdout, "row starts:\n");
			cudaMallocCount((void **) &row_starts_dev,(n+1)*sizeof(int));
			err = cudaMemcpy(row_starts_dev,row_starts,(n+1)*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			cols_size = row_starts[n]; 
			if (!compact_tr) {
				fprintf (stdout, "non_zeros:\n");
				cudaMallocCount((void **) &non_zeros_dev,cols_size*sizeof(double));
				err = cudaMemcpy(non_zeros_dev,non_zeros,cols_size*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
			else {
				fprintf (stdout, "dist_dev:\n");
				cudaMallocCount((void **) &dist_dev,dist_mask*sizeof(double));
				err = cudaMemcpy(dist_dev,dist,dist_mask*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
		
			fprintf (stdout, "cols_dev:\n");
			cudaMallocCount((void **) &cols_dev,cols_size*sizeof(unsigned int));
			err = cudaMemcpy(cols_dev,cols,cols_size*sizeof(unsigned int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln_dev:\n");
			cudaMallocCount((void **) &soln_dev,n*sizeof(double));
			err = cudaMemcpy(soln_dev,soln,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln2_dev:\n");
			cudaMallocCount((void **) &soln2_dev,n*sizeof(double));
		
			printf("copied %i KB to GPU \n",vmem/1024);
			printf("copied %i MB to GPU \n",vmem/1024/1024);
		}

		// datacopy for Prob Cumul Reward
		void  datacpy_pcr (int n, bool compact_tr, bool compact_r, double * rew_vec, DistVector * rew_dist,
			int *row_starts, double *non_zeros, double *soln, unsigned int *cols,double * dist,
			int dist_mask, int dist_shift) {

			cudaError_t err = cudaSuccess;

			printf("copying data \n");

			if (!compact_r) {
				fprintf (stdout, "y:\n");
				cudaMallocCount((void **) &rew_vec_dev,n*sizeof(double));
				err = cudaMemcpy(rew_vec_dev,rew_vec,n*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
			else {
				fprintf (stdout, "y:\n");
				cudaMallocCount((void **) &rew_ptrs_dev,n*sizeof(unsigned short));
				err = cudaMemcpy(rew_ptrs_dev,rew_dist->ptrs,n*sizeof(unsigned short),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
				cudaMallocCount((void **) &rew_dist_dev,rew_dist->num_dist*sizeof(double));
				err = cudaMemcpy(rew_dist_dev,rew_dist->dist,rew_dist->num_dist*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
		
			int cols_size=0;
		 
			fprintf (stdout, "row starts:\n");
			cudaMallocCount((void **) &row_starts_dev,(n+1)*sizeof(int));
			err = cudaMemcpy(row_starts_dev,row_starts,(n+1)*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			cols_size = row_starts[n]; 
			if (!compact_tr) {
				fprintf (stdout, "non_zeros:\n");
				cudaMallocCount((void **) &non_zeros_dev,cols_size*sizeof(double));
				err = cudaMemcpy(non_zeros_dev,non_zeros,cols_size*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
			else {
				fprintf (stdout, "dist_dev:\n");
				cudaMallocCount((void **) &dist_dev,dist_mask*sizeof(double));
				err = cudaMemcpy(dist_dev,dist,dist_mask*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
		
			fprintf (stdout, "cols_dev:\n");
			cudaMallocCount((void **) &cols_dev,cols_size*sizeof(unsigned int));
			err = cudaMemcpy(cols_dev,cols,cols_size*sizeof(unsigned int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln_dev:\n");
			cudaMallocCount((void **) &soln_dev,n*sizeof(double));
			err = cudaMemcpy(soln_dev,soln,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln2_dev:\n");
			cudaMallocCount((void **) &soln2_dev,n*sizeof(double));
		
			printf("copied %i KB to GPU \n",vmem/1024);
			printf("copied %i MB to GPU \n",vmem/1024/1024);
		}

		// datacopy for Prob bounded until / ProbTransient
		void  datacpy_pir (int n, bool compact_tr,
			int *row_starts, double *non_zeros, double *soln, unsigned int *cols,double * dist,
			int dist_mask, int dist_shift) {

			cudaError_t err = cudaSuccess;

			printf("copying data \n");

			int cols_size=0;
		 
			fprintf (stdout, "row starts:\n");
			cudaMallocCount((void **) &row_starts_dev,(n+1)*sizeof(int));
			err = cudaMemcpy(row_starts_dev,row_starts,(n+1)*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			cols_size = row_starts[n]; 
			if (!compact_tr) {
				fprintf (stdout, "non_zeros:\n");
				cudaMallocCount((void **) &non_zeros_dev,cols_size*sizeof(double));
				err = cudaMemcpy(non_zeros_dev,non_zeros,cols_size*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
			else {
				fprintf (stdout, "dist_dev:\n");
				cudaMallocCount((void **) &dist_dev,dist_mask*sizeof(double));
				err = cudaMemcpy(dist_dev,dist,dist_mask*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
		
			fprintf (stdout, "cols_dev:\n");
			cudaMallocCount((void **) &cols_dev,cols_size*sizeof(unsigned int));
			err = cudaMemcpy(cols_dev,cols,cols_size*sizeof(unsigned int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln_dev:\n");
			cudaMallocCount((void **) &soln_dev,n*sizeof(double));
			err = cudaMemcpy(soln_dev,soln,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln2_dev:\n");
			cudaMallocCount((void **) &soln2_dev,n*sizeof(double));
		
			printf("copied %i KB to GPU \n",vmem/1024);
			printf("copied %i MB to GPU \n",vmem/1024/1024);
		}
			
		// datacpy for stochastic bounded until
		void  datacpy_stbu (int n, bool compact_tr, bool compact_d,
			int *row_starts, double *non_zeros, double *soln, unsigned int *cols,double * dist,
			int dist_mask, int dist_shift, double *diags_vec, DistVector * diags_dist) {

			cudaError_t err = cudaSuccess;

			printf("copying data \n");

			int cols_size=0;
		 
			fprintf (stdout, "row starts:\n");
			cudaMallocCount((void **) &row_starts_dev,(n+1)*sizeof(int));
			err = cudaMemcpy(row_starts_dev,row_starts,(n+1)*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			cols_size = row_starts[n]; 
			if (!compact_tr) {
				fprintf (stdout, "non_zeros:\n");
				cudaMallocCount((void **) &non_zeros_dev,cols_size*sizeof(double));
				err = cudaMemcpy(non_zeros_dev,non_zeros,cols_size*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
			else {
				fprintf (stdout, "dist_dev:\n");
				cudaMallocCount((void **) &dist_dev,dist_mask*sizeof(double));
				err = cudaMemcpy(dist_dev,dist,dist_mask*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
		
			fprintf (stdout, "cols_dev:\n");
			cudaMallocCount((void **) &cols_dev,cols_size*sizeof(unsigned int));
			err = cudaMemcpy(cols_dev,cols,cols_size*sizeof(unsigned int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln_dev:\n");
			cudaMallocCount((void **) &soln_dev,n*sizeof(double));
			err = cudaMemcpy(soln_dev,soln,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln2_dev:\n");
			cudaMallocCount((void **) &soln2_dev,n*sizeof(double));

			if (!compact_d) {
				fprintf (stdout, "d:\n");
				cudaMallocCount((void **) &diags_vec_dev,n*sizeof(double));
				err = cudaMemcpy(diags_vec_dev,diags_vec,n*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
			else {
				fprintf (stdout, "d:\n");
				cudaMallocCount((void **) &diags_ptrs_dev,n*sizeof(unsigned short));
				err = cudaMemcpy(diags_ptrs_dev,diags_dist->ptrs,n*sizeof(unsigned short),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
				cudaMallocCount((void **) &diags_dist_dev,diags_dist->num_dist*sizeof(double));
				err = cudaMemcpy(diags_dist_dev,diags_dist->dist,diags_dist->num_dist*sizeof(double),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
		
			cudaMallocCount((void **) &sum_dev,n*sizeof(double));

			fprintf (stdout, "done:\n");
			cudaMallocCount((void **) &done_dev, sizeof(char));
		
			printf("copied %i KB to GPU \n",vmem/1024);
			printf("copied %i MB to GPU \n",vmem/1024/1024);
		}
	
	
		// datacpy for non-det bounded until
		void  datacpy_nd (int n,
			int *row_starts, int *choice_starts, double *non_zeros, int nc, double *soln,
			unsigned int *cols, double *yes_vec) {

			cudaError_t err = cudaSuccess;

			printf("copying data \n");

			int cols_size=0;
		 
			fprintf (stdout, "row starts:\n");
			cudaMallocCount((void **) &row_starts_dev,(n+1)*sizeof(int));
			err = cudaMemcpy(row_starts_dev,row_starts,(n+1)*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			cols_size = row_starts[n]; 
		
			fprintf (stdout, "choice starts:\n");
			cudaMallocCount((void **) &choice_starts_dev,nc*sizeof(int));
			err = cudaMemcpy(choice_starts_dev,choice_starts,nc*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));

			fprintf (stdout, "non_zeros:\n");
			cudaMallocCount((void **) &non_zeros_dev,cols_size*sizeof(double));
			err = cudaMemcpy(non_zeros_dev,non_zeros,cols_size*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "cols_dev:\n");
			cudaMallocCount((void **) &cols_dev,cols_size*sizeof(unsigned int));
			err = cudaMemcpy(cols_dev,cols,cols_size*sizeof(unsigned int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln_dev:\n");
			cudaMallocCount((void **) &soln_dev,n*sizeof(double));
			err = cudaMemcpy(soln_dev,soln,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln2_dev:\n");
			cudaMallocCount((void **) &soln2_dev,n*sizeof(double));

			fprintf (stdout, "y:\n");
			cudaMallocCount((void **) &yes_vec_dev,n*sizeof(double));
			err = cudaMemcpy(yes_vec_dev,yes_vec,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			printf("copied %i KB to GPU \n",vmem/1024);
			printf("copied %i MB to GPU \n",vmem/1024/1024);
		}

		// datacpy for non-det InstReward
		void  datacpy_ndi (int n,
			int *row_starts, int *choice_starts, double *non_zeros, int nc, double *soln,
			unsigned int *cols) {

			cudaError_t err = cudaSuccess;

			printf("copying data \n");

			int cols_size=0;
		 
			fprintf (stdout, "row starts:\n");
			cudaMallocCount((void **) &row_starts_dev,(n+1)*sizeof(int));
			err = cudaMemcpy(row_starts_dev,row_starts,(n+1)*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			cols_size = row_starts[n]; 
		
			fprintf (stdout, "choice starts:\n");
			cudaMallocCount((void **) &choice_starts_dev,nc*sizeof(int));
			err = cudaMemcpy(choice_starts_dev,choice_starts,nc*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));

			fprintf (stdout, "non_zeros:\n");
			cudaMallocCount((void **) &non_zeros_dev,cols_size*sizeof(double));
			err = cudaMemcpy(non_zeros_dev,non_zeros,cols_size*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "cols_dev:\n");
			cudaMallocCount((void **) &cols_dev,cols_size*sizeof(unsigned int));
			err = cudaMemcpy(cols_dev,cols,cols_size*sizeof(unsigned int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln_dev:\n");
			cudaMallocCount((void **) &soln_dev,n*sizeof(double));
			err = cudaMemcpy(soln_dev,soln,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln2_dev:\n");
			cudaMallocCount((void **) &soln2_dev,n*sizeof(double));

			printf("copied %i KB to GPU \n",vmem/1024);
			printf("copied %i MB to GPU \n",vmem/1024/1024);
		}

		// datacpy for non-det reach reward
		void  datacpy_ndr (int n,
			int *row_starts, int *choice_starts, double *non_zeros, int nc, double *soln,
			unsigned int *cols, int *choice_starts_r, double *non_zeros_r, int nc_r, unsigned int *cols_r,
			double *sr_vec, double *inf_vec, int export_adv, int *adv) {

			cudaError_t err = cudaSuccess;

			printf("copying data \n");

			int cols_size=0;
		 
			fprintf (stdout, "row starts:\n");
			cudaMallocCount((void **) &row_starts_dev,(n+1)*sizeof(int));
			err = cudaMemcpy(row_starts_dev,row_starts,(n+1)*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			cols_size = row_starts[n]; 
		
			fprintf (stdout, "choice starts:\n");
			cudaMallocCount((void **) &choice_starts_dev,nc*sizeof(int));
			err = cudaMemcpy(choice_starts_dev,choice_starts,nc*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));

			fprintf (stdout, "non_zeros:\n");
			cudaMallocCount((void **) &non_zeros_dev,cols_size*sizeof(double));
			err = cudaMemcpy(non_zeros_dev,non_zeros,cols_size*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "cols_dev:\n");
			cudaMallocCount((void **) &cols_dev,cols_size*sizeof(unsigned int));
			err = cudaMemcpy(cols_dev,cols,cols_size*sizeof(unsigned int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));

			cols_size=0;
			fprintf (stdout, "choice starts rewards:\n");
			cudaMallocCount((void **) &choice_starts_r_dev,nc_r*sizeof(int));
			err = cudaMemcpy(choice_starts_r_dev,choice_starts_r,nc_r*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			cols_size = choice_starts_r[n];

			fprintf (stdout, "non_zeros rewards:\n");
			cudaMallocCount((void **) &non_zeros_r_dev,cols_size*sizeof(double));
			err = cudaMemcpy(non_zeros_r_dev,non_zeros_r,cols_size*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "cols_dev rewards:\n");
			cudaMallocCount((void **) &cols_r_dev,cols_size*sizeof(unsigned int));
			err = cudaMemcpy(cols_r_dev,cols_r,cols_size*sizeof(unsigned int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln_dev:\n");
			cudaMallocCount((void **) &soln_dev,n*sizeof(double));
			err = cudaMemcpy(soln_dev,soln,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln2_dev:\n");
			cudaMallocCount((void **) &soln2_dev,n*sizeof(double));

			fprintf (stdout, "state rewards:\n");
			cudaMallocCount((void **) &sr_vec_dev,n*sizeof(double));
			err = cudaMemcpy(sr_vec_dev,sr_vec,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));

			fprintf (stdout, "inf states:\n");
			cudaMallocCount((void **) &inf_vec_dev,n*sizeof(double));
			err = cudaMemcpy(inf_vec_dev,inf_vec,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			
			if (export_adv) {
				fprintf (stdout, "adv_dev:\n");
				cudaMallocCount((void **) &adv_dev,n*sizeof(int));
				err = cudaMemcpy(adv_dev,adv,n*sizeof(int),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
		
			printf("copied %i KB to GPU \n",vmem/1024);
			printf("copied %i MB to GPU \n",vmem/1024/1024);
		}

		// datacpy for non-det until
		void  datacpy_ndu (int n,
			int *row_starts, int *choice_starts, double *non_zeros, int nc, double *soln,
			unsigned int *cols,
			double *yes_vec, int export_adv, int *adv) {

			cudaError_t err = cudaSuccess;

			printf("copying data \n");

			int cols_size=0;
		 
			fprintf (stdout, "row starts:\n");
			cudaMallocCount((void **) &row_starts_dev,(n+1)*sizeof(int));
			err = cudaMemcpy(row_starts_dev,row_starts,(n+1)*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			cols_size = row_starts[n]; 
		
			fprintf (stdout, "choice starts:\n");
			cudaMallocCount((void **) &choice_starts_dev,nc*sizeof(int));
			err = cudaMemcpy(choice_starts_dev,choice_starts,nc*sizeof(int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));

			fprintf (stdout, "non_zeros:\n");
			cudaMallocCount((void **) &non_zeros_dev,cols_size*sizeof(double));
			err = cudaMemcpy(non_zeros_dev,non_zeros,cols_size*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "cols_dev:\n");
			cudaMallocCount((void **) &cols_dev,cols_size*sizeof(unsigned int));
			err = cudaMemcpy(cols_dev,cols,cols_size*sizeof(unsigned int),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln_dev:\n");
			cudaMallocCount((void **) &soln_dev,n*sizeof(double));
			err = cudaMemcpy(soln_dev,soln,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
		
			fprintf (stdout, "soln2_dev:\n");
			cudaMallocCount((void **) &soln2_dev,n*sizeof(double));

			fprintf (stdout, "y:\n");
			cudaMallocCount((void **) &yes_vec_dev,n*sizeof(double));
			err = cudaMemcpy(yes_vec_dev,yes_vec,n*sizeof(double),cudaMemcpyHostToDevice);
			if (err) printf("%s \n", cudaGetErrorString(err));
			
			if (export_adv) {
				fprintf (stdout, "adv_dev:\n");
				cudaMallocCount((void **) &adv_dev,n*sizeof(int));
				err = cudaMemcpy(adv_dev,adv,n*sizeof(int),cudaMemcpyHostToDevice);
				if (err) printf("%s \n", cudaGetErrorString(err));
			}
		
			printf("copied %i KB to GPU \n",vmem/1024);
			printf("copied %i MB to GPU \n",vmem/1024/1024);
		}

		int  mv (int n, void * b,bool compact_a, bool compact_b, bool compact_d, double *soln,
						int dist_mask, int dist_shift, jdouble omega,
						int max_iters, int term_crit, double term_crit_param, bool power) {


			cudaError_t err = cudaSuccess;

			dim3 threads(CTA_SIZE, 1, 1);

			unsigned int numRowBlocks =
				max(1, (int)ceil((double) n / ((double)ELTS_PER_THREAD * CTA_SIZE)));
			
			dim3 gridRows(max(1, numRowBlocks), 1, 1);
		
			int iters = 0;
			double *tmpsoln;
			char done = 0;
			char one = 1;
			
			while (!done && iters < max_iters) {

				cudaMemcpy (done_dev, &one, sizeof(char), cudaMemcpyHostToDevice);
				mv_kernel<<<gridRows, threads>>>(n, compact_a, row_starts_dev, non_zeros_dev, soln_dev, soln2_dev, cols_dev,
												dist_dev, dist_mask, dist_shift);		
				cudaError_t error = cudaGetLastError();
				if (error != cudaSuccess) fprintf (stdout, "%s\n", cudaGetErrorString(error));
								
				if (!power) {
					gather<<<gridRows, threads>>>(soln2_dev, n, soln_dev, b, compact_b, compact_d, b_vec_dev, ptrs_dev,
							b_dist_dev, diags_vec_dev, diags_ptrs_dev, diags_dist_dev, omega, term_crit, term_crit_param, done_dev);
				}
				else {
					gather_power<<<gridRows, threads>>>(soln2_dev, n, soln_dev, b, compact_b, b_vec_dev, ptrs_dev, b_dist_dev,
														term_crit, term_crit_param, done_dev);
				}
				//cudaThreadSynchronize();
				
												
				iters++;
			
				cudaMemcpy(&done, done_dev, sizeof(char), cudaMemcpyDeviceToHost);
			
				tmpsoln = soln_dev;
				soln_dev = soln2_dev;
				soln2_dev = tmpsoln;
			}

			err = cudaSuccess;
			err = cudaMemcpy(soln,soln2_dev,n*sizeof(double),cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) printf("%s \n", cudaGetErrorString(err));
		
			return iters;
		}

		// matrix/vector multiplication for prob bounded until checks
		int  mv_pbu (int n, bool compact_tr, bool compact_y, double *soln,
						int dist_mask, int dist_shift, int bound, bool use_yes) {

			cudaError_t err = cudaSuccess;

			dim3 threads(CTA_SIZE, 1, 1);

			unsigned int numRowBlocks =
				max(1, (int)ceil((double) n / ((double)ELTS_PER_THREAD * CTA_SIZE)));
			
			dim3 gridRows(max(1, numRowBlocks), 1, 1);

			int iters = 0;
			double *tmpsoln;
		
			while (iters < bound) {
		
				mv_kernel<<<gridRows, threads>>>(n,compact_tr, row_starts_dev, non_zeros_dev, soln_dev, soln2_dev, cols_dev,
											dist_dev, dist_mask, dist_shift);		
				//cudaThreadSynchronize();
		
				cudaError_t error = cudaGetLastError();
				if (error != cudaSuccess) fprintf (stdout, "%s\n", cudaGetErrorString(error));
			
				if (use_yes) {
					gather_bu<<<gridRows, threads>>>(soln2_dev, n, compact_y, yes_vec_dev, yes_ptrs_dev,
						yes_dist_dev);
					//cudaThreadSynchronize();
				}
		
				iters++;
			
				tmpsoln = soln_dev;
				soln_dev = soln2_dev;
				soln2_dev = tmpsoln;
			}

			err = cudaSuccess;
			err = cudaMemcpy(soln,soln2_dev,n*sizeof(double),cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) printf("%s \n", cudaGetErrorString(err));
		
			return iters;
		}

		// matrix/vector multiplication for prob cumul reward
		int  mv_pcr (int n, bool compact_tr, bool compact_r, double *soln,
						int dist_mask, int dist_shift, int bound) {

			cudaError_t err = cudaSuccess;

			dim3 threads(CTA_SIZE, 1, 1);

			unsigned int numRowBlocks =
				max(1, (int)ceil((double) n / ((double)ELTS_PER_THREAD * CTA_SIZE)));
			
			dim3 gridRows(max(1, numRowBlocks), 1, 1);

			int iters = 0;
			double *tmpsoln;
		
			while (iters < bound) {
		
				mv_kernel<<<gridRows, threads>>>(n,compact_tr, row_starts_dev, non_zeros_dev, soln_dev, soln2_dev, cols_dev,
											dist_dev, dist_mask, dist_shift);		
				//cudaThreadSynchronize();
		
				cudaError_t error = cudaGetLastError();
				if (error != cudaSuccess) fprintf (stdout, "%s\n", cudaGetErrorString(error));
			
				gather_pcr<<<gridRows, threads>>>(soln2_dev, n, compact_r, rew_vec_dev, rew_ptrs_dev,
					rew_dist_dev);
				//cudaThreadSynchronize();
		
				iters++;
			
				tmpsoln = soln_dev;
				soln_dev = soln2_dev;
				soln2_dev = tmpsoln;
			}

			err = cudaSuccess;
			err = cudaMemcpy(soln,soln2_dev,n*sizeof(double),cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) printf("%s \n", cudaGetErrorString(err));
		
			return iters;
		}	

		// Matrix/vector multiplication for ProbTransient
		int  mv_pt (int n, bool compact_tr, double *soln, int dist_mask, int dist_shift, int bound,
						int term_crit, double term_crit_param, bool do_ss_detect) {

			cudaError_t err = cudaSuccess;

			dim3 threads(CTA_SIZE, 1, 1);

			unsigned int numRowBlocks =
				max(1, (int)ceil((double) n / ((double)ELTS_PER_THREAD * CTA_SIZE)));
			
			dim3 gridRows(max(1, numRowBlocks), 1, 1);

			int iters = 0;
			double *tmpsoln;
			char done = 0;
			char one = 1;
		
			while (iters < bound && !done) {
		
				mv_kernel<<<gridRows, threads>>>(n,compact_tr, row_starts_dev, non_zeros_dev, soln_dev, soln2_dev, cols_dev,
											dist_dev, dist_mask, dist_shift);		
				//cudaThreadSynchronize();
		
				cudaError_t error = cudaGetLastError();
				if (error != cudaSuccess) fprintf (stdout, "%s\n", cudaGetErrorString(error));
				
				if (do_ss_detect) {
					cudaMemcpy (done_dev, &one, sizeof(char), cudaMemcpyHostToDevice);
		
					gather_pt<<<gridRows, threads>>>(soln2_dev, n, soln_dev,
							term_crit, term_crit_param, done_dev);
					//cudaThreadSynchronize();
					cudaMemcpy(&done, done_dev, sizeof(char), cudaMemcpyDeviceToHost);
				}
				
				iters++;
			
				tmpsoln = soln_dev;
				soln_dev = soln2_dev;
				soln2_dev = tmpsoln;
			}

			err = cudaSuccess;
			err = cudaMemcpy(soln,soln2_dev,n*sizeof(double),cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) printf("%s \n", cudaGetErrorString(err));
		
			return iters;
		}
			
		// Matrix/vector multiplication for stochastic bounded until checks / stochastic cumul reward
		int  mv_stbu (int n, bool compact_tr, bool compact_d, double *sum, int dist_mask, int dist_shift,
						int left, int bound, double *weights, int term_crit, double term_crit_param,
						double unif, bool do_ss_detect, bool cumulreward) {

			double weight = 0;

			cudaError_t err = cudaSuccess;

			dim3 threads(CTA_SIZE, 1, 1);

			unsigned int numRowBlocks =
				max(1, (int)ceil((double) n / ((double)ELTS_PER_THREAD * CTA_SIZE)));
			
			dim3 gridRows(max(1, numRowBlocks), 1, 1);
		
			// if necessary, do 0th element of summation (doesn't require any matrix powers)
			if (left == 0) {
				sparseMatrixVectorSummation_stbu<<<gridRows, threads>>>(sum_dev, soln_dev, n, weights[0]);
			}
			if (cumulreward) {
				if (left != 0) {
					sparseMatrixVectorDivUnif_stbu<<<gridRows, threads>>>(sum_dev, soln_dev, n, unif);
				}
			}

			int iters = 1;
			double *tmpsoln;
			char done = 0;
			char one = 1;
		
			while (iters <= bound && !done) {
		
				mv_kernel<<<gridRows, threads>>>(n,compact_tr, row_starts_dev, non_zeros_dev,soln_dev, soln2_dev, cols_dev,
											dist_dev,dist_mask,dist_shift);		
				//cudaThreadSynchronize();
		
				cudaError_t error = cudaGetLastError();
				if (error != cudaSuccess) fprintf (stdout, "%s\n", cudaGetErrorString(error));
		
				if (do_ss_detect)
					cudaMemcpy (done_dev, &one, sizeof(char), cudaMemcpyHostToDevice);
		
				gather_stbu<<<gridRows, threads>>>(soln2_dev, n, soln_dev, compact_d,
					diags_vec_dev, diags_ptrs_dev, diags_dist_dev, term_crit, term_crit_param, done_dev, do_ss_detect);
				//cudaThreadSynchronize();
		
				iters++;
			
				if (do_ss_detect)
					cudaMemcpy(&done, done_dev, sizeof(char), cudaMemcpyDeviceToHost);

				// special case when finished early (steady-state detected)
				if (done) {
					// work out sum of remaining poisson probabilities
					if (iters <= left) {
						weight = 1.0;
					} else {
						weight = 0.0;
						for (int i = iters; i <= bound; i++) {
							weight += weights[i-left];
						}
					}
					// add to sum
					sparseMatrixVectorSummation_stbu<<<gridRows, threads>>>(sum_dev, soln2_dev, n, weight);
					break;
				}
			
				tmpsoln = soln_dev;
				soln_dev = soln2_dev;
				soln2_dev = tmpsoln;
			
				// add to sum
				if (iters >= left) {
					sparseMatrixVectorSummation_stbu<<<gridRows, threads>>>(sum_dev, soln_dev, n, weights[iters-left]);
				}
				if (cumulreward) {
					if (iters < left) {
						sparseMatrixVectorDivUnif_stbu<<<gridRows, threads>>>(sum_dev, soln_dev, n, unif);
					}
				}
			}

			err = cudaSuccess;
			err = cudaMemcpy(sum,sum_dev,n*sizeof(double),cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) printf("%s \n", cudaGetErrorString(err));
		
			return iters;
		}
	
		// Setup function for stochastic bounded until checks
		double setup_stbu (int n, long nnz, bool compact_tr, bool compact_d, double *diags_vec, DistVector * diags_dist) {
		
			double max_diag, unif;
		
			dim3 threads(CTA_SIZE, 1, 1);

			unsigned int numEltsBlocks =
				max(1, (int)ceil(nnz / ((double)ELTS_PER_THREAD * CTA_SIZE)));

			dim3 gridElts(max(1, numEltsBlocks), 1, 1);

			unsigned int numRowBlocks =
				max(1, (int)ceil((double) n / ((double)ELTS_PER_THREAD * CTA_SIZE)));
			
			dim3 gridRows(max(1, numRowBlocks), 1, 1);
		
			// Calculate another dimension for diags if necessary
			unsigned int numDiagsBlocks;
			if (compact_d && n != diags_dist->num_dist) {
				numDiagsBlocks =
					max(1, (int)ceil((double) (diags_dist->num_dist) / ((double)ELTS_PER_THREAD * CTA_SIZE)));
			}
			else {
				numDiagsBlocks = numRowBlocks;
			}
		
			dim3 gridDiags(max(1, numDiagsBlocks), 1, 1);
		
			// find max diagonal element
			if (!compact_d) {
				max_diag = diags_vec[0];
				for (int i = 1; i < n; i++) if (diags_vec[i] < max_diag) max_diag = diags_vec[i];
			} else {
				max_diag = diags_dist->dist[0];
				for (int i = 1; i < diags_dist->num_dist; i++) if (diags_dist->dist[i] < max_diag) max_diag = diags_dist->dist[i];
			}
			max_diag = -max_diag;
		
			// constant for uniformization
			unif = 1.02*max_diag;
		
			sparseMatrixVectorModifyDiags_stbu<<<gridDiags, threads>>>(compact_d, diags_vec_dev, diags_dist_dev, n, diags_dist->num_dist, unif);
		
			sparseMatrixUniformization_stbu<<<gridElts, threads>>>(compact_tr, non_zeros_dev, dist_dev, nnz, unif);
		
			sparseMatrixVectorSetSumToZero<<<gridRows, threads>>>(sum_dev, n);
		
			return(unif);
		}
	
	
		// Matrix/vector multiplication for non-det bounded until checks / InstReward
		int  mv_nd (int n, int nc, double *soln, int bound, bool min, bool instreward) {

			cudaError_t err = cudaSuccess;

			dim3 threads(CTA_SIZE, 1, 1);

			unsigned int numRowBlocks =
				max(1, (int)ceil((double) n / ((double)ELTS_PER_THREAD * CTA_SIZE)));
			
			dim3 gridRows(max(1, numRowBlocks), 1, 1);

			int iters = 0;
			double *tmpsoln;
		
			while (iters < bound) {
		
				mv_nd_kernel<<<gridRows, threads>>>(n, false, row_starts_dev, non_zeros_dev,soln_dev, soln2_dev, cols_dev,
											dist_dev,0,0, choice_starts_dev, yes_vec_dev, min, instreward);	
				//cudaThreadSynchronize();
		
				cudaError_t error = cudaGetLastError();
				if (error != cudaSuccess) fprintf (stdout, "%s\n", cudaGetErrorString(error));
		
				iters++;
			
				tmpsoln = soln_dev;
				soln_dev = soln2_dev;
				soln2_dev = tmpsoln;
			}
		
			err = cudaSuccess;
			err = cudaMemcpy(soln,soln2_dev,n*sizeof(double),cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) printf("%s \n", cudaGetErrorString(err));
		
			return iters;
		}

		// Matrix/vector multiplication for non-det reach reward
		int  mv_ndr (int n, int nc, double *soln, int nc_r,
						int bound, int term_crit, double term_crit_param, bool min, int export_adv, int *adv, double huge) {

			cudaError_t err = cudaSuccess;

			dim3 threads(CTA_SIZE, 1, 1);

			unsigned int numRowBlocks =
				max(1, (int)ceil((double) n / ((double)ELTS_PER_THREAD * CTA_SIZE)));
			
			dim3 gridRows(max(1, numRowBlocks), 1, 1);

			int iters = 0;
			double *tmpsoln;
			char done = 0;
			char one = 1;
					
			while (!done && iters < bound) {
			
				iters++;
		
				mv_ndr_kernel<<<gridRows, threads>>>(n, row_starts_dev, non_zeros_dev, soln_dev, soln2_dev, cols_dev,
											dist_dev,0,0, choice_starts_dev, choice_starts_r_dev, non_zeros_r_dev, cols_r_dev,
											sr_vec_dev, inf_vec_dev, export_adv, adv_dev, min, huge);	
				//cudaThreadSynchronize();
		
				cudaError_t error = cudaGetLastError();
				if (error != cudaSuccess) fprintf (stdout, "%s\n", cudaGetErrorString(error));
				
				cudaMemcpy (done_dev, &one, sizeof(char), cudaMemcpyHostToDevice);
		
				gather_pt<<<gridRows, threads>>>(soln2_dev, n, soln_dev,
						term_crit, term_crit_param, done_dev);
				//cudaThreadSynchronize();
				cudaMemcpy(&done, done_dev, sizeof(char), cudaMemcpyDeviceToHost);
			
				tmpsoln = soln_dev;
				soln_dev = soln2_dev;
				soln2_dev = tmpsoln;
			}
		
			err = cudaSuccess;
			err = cudaMemcpy(soln,soln2_dev,n*sizeof(double),cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) printf("%s \n", cudaGetErrorString(err));

			if (export_adv) {
				err = cudaSuccess;
				err = cudaMemcpy(adv,adv_dev,n*sizeof(int),cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) printf("%s \n", cudaGetErrorString(err));
			}
			
			return iters;
		}	

		// Matrix/vector multiplication for non-det until
		int  mv_ndu (int n, int nc, double *soln,
						int bound, int term_crit, double term_crit_param, bool min, int export_adv, int *adv) {

			cudaError_t err = cudaSuccess;

			dim3 threads(CTA_SIZE, 1, 1);

			unsigned int numRowBlocks =
				max(1, (int)ceil((double) n / ((double)ELTS_PER_THREAD * CTA_SIZE)));
			
			dim3 gridRows(max(1, numRowBlocks), 1, 1);

			int iters = 0;
			double *tmpsoln;
			char done = 0;
			char one = 1;
					
			while (!done && iters < bound) {
			
				iters++;
		
				mv_ndu_kernel<<<gridRows, threads>>>(n, row_starts_dev, non_zeros_dev,soln_dev, soln2_dev, cols_dev,
											dist_dev,0,0, choice_starts_dev,
											yes_vec_dev, export_adv, adv_dev, min);	
				//cudaThreadSynchronize();
		
				cudaError_t error = cudaGetLastError();
				if (error != cudaSuccess) fprintf (stdout, "%s\n", cudaGetErrorString(error));
				
				cudaMemcpy (done_dev, &one, sizeof(char), cudaMemcpyHostToDevice);
		
				gather_pt<<<gridRows, threads>>>(soln2_dev, n, soln_dev,
						term_crit, term_crit_param, done_dev);
				//cudaThreadSynchronize();
				cudaMemcpy(&done, done_dev, sizeof(char), cudaMemcpyDeviceToHost);
			
				tmpsoln = soln_dev;
				soln_dev = soln2_dev;
				soln2_dev = tmpsoln;
			}
		
			err = cudaSuccess;
			err = cudaMemcpy(soln,soln2_dev,n*sizeof(double),cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) printf("%s \n", cudaGetErrorString(err));

			if (export_adv) {
				err = cudaSuccess;
				err = cudaMemcpy(adv,adv_dev,n*sizeof(int),cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) printf("%s \n", cudaGetErrorString(err));
			}
					
			return iters;
		}	

	void datafree (bool compact_a, bool compact_b, bool compact_d) {
		printf("freeing memory \n");
	
		if (!compact_b) {zzz
			cudaFree(b_vec_dev) ;
		}
		else {
			cudaFree(ptrs_dev);
			cudaFree(b_dist_dev);
		}
		cudaFree(row_starts_dev);
	
		if (!compact_a) {
			cudaFree(non_zeros_dev);
		}
		else {
			cudaFree(dist_dev);
		}
		cudaFree(cols_dev);
		cudaFree(soln_dev);
		cudaFree(soln2_dev);
		if (!compact_d) {
			cudaFree(diags_vec_dev);
		}
		else {
			cudaFree(diags_ptrs_dev);
			cudaFree(diags_dist_dev);
		}
		cudaFree(done_dev);
	}

	void datafree_power (bool compact_a, bool compact_b) {
		printf("freeing memory \n");

		if (!compact_b) {
			cudaFree(b_vec_dev) ;
		}
		else {
			cudaFree(ptrs_dev);
			cudaFree(b_dist_dev);
		}	
		cudaFree(row_starts_dev);
	
		if (!compact_a) {
			cudaFree(non_zeros_dev);
		}
		else {
			cudaFree(dist_dev);
		}
		cudaFree(cols_dev);
		cudaFree(soln_dev);
		cudaFree(soln2_dev);
		cudaFree(done_dev);
	}	
	
	void datafree_pbu (bool compact_tr, bool compact_y) {
		printf("freeing memory \n");
	
		if (!compact_y) {
			cudaFree(yes_vec_dev);
		}
		else {
			cudaFree(yes_ptrs_dev);
			cudaFree(yes_dist_dev);
		}
		cudaFree(row_starts_dev);
	
		if (!compact_tr) {
			cudaFree(non_zeros_dev);
		}
		else {
			cudaFree(dist_dev);
		}
		cudaFree(cols_dev);
		cudaFree(soln_dev);
		cudaFree(soln2_dev);	
	}
	
	void datafree_pcr (bool compact_tr, bool compact_r) {
		printf("freeing memory \n");
	
		if (!compact_r) {
			cudaFree(rew_vec_dev);
		}
		else {
			cudaFree(rew_ptrs_dev);
			cudaFree(rew_dist_dev);
		}
		cudaFree(row_starts_dev);
	
		if (!compact_tr) {
			cudaFree(non_zeros_dev);
		}
		else {
			cudaFree(dist_dev);
		}
		cudaFree(cols_dev);
		cudaFree(soln_dev);
		cudaFree(soln2_dev);	
	}
	
	void datafree_pir (bool compact_tr) {
		printf("freeing memory \n");
	
		cudaFree(row_starts_dev);
	
		if (!compact_tr) {
			cudaFree(non_zeros_dev);
		}
		else {
			cudaFree(dist_dev);
		}
		cudaFree(cols_dev);
		cudaFree(soln_dev);
		cudaFree(soln2_dev);	
	}
	
	void datafree_stbu (bool compact_tr, bool compact_d) {
		printf("freeing memory \n");
	
		cudaFree(row_starts_dev);
	
		if (!compact_tr) {
			cudaFree(non_zeros_dev);
		}
		else {
			cudaFree(dist_dev);
		}
		cudaFree(cols_dev);
		cudaFree(soln_dev);
		cudaFree(soln2_dev);
		cudaFree(sum_dev);
		if (!compact_d) {
			cudaFree(diags_vec_dev);
		}
		else {
			cudaFree(diags_ptrs_dev);
			cudaFree(diags_dist_dev);
		}
		cudaFree(done_dev);
	}
	
	void datafree_nd () {
		printf("freeing memory \n");
	
		cudaFree(row_starts_dev);
		cudaFree(choice_starts_dev);

		cudaFree(yes_vec_dev);
	
		cudaFree(non_zeros_dev);
		cudaFree(cols_dev);
		cudaFree(soln_dev);
		cudaFree(soln2_dev);
	}

	void datafree_ndr () {
		printf("freeing memory \n");
	
		cudaFree(row_starts_dev);
		cudaFree(choice_starts_dev);
	
		cudaFree(non_zeros_dev);
		cudaFree(cols_dev);
		
		cudaFree(choice_starts_r_dev);
		cudaFree(non_zeros_r_dev);
		cudaFree(cols_r_dev);
		
		cudaFree(sr_vec_dev);
		cudaFree(inf_vec_dev);
		
		cudaFree(soln_dev);
		cudaFree(soln2_dev);
	}

	void datafree_ndu () {
		printf("freeing memory \n");
	
		cudaFree(row_starts_dev);
		cudaFree(choice_starts_dev);
	
		cudaFree(non_zeros_dev);
		cudaFree(cols_dev);
		
		cudaFree(yes_vec_dev);
		cudaFree(inf_vec_dev);
		
		cudaFree(soln_dev);
		cudaFree(soln2_dev);
	}
	
	void datafree_ndi () {
		printf("freeing memory \n");
	
		cudaFree(row_starts_dev);
		cudaFree(choice_starts_dev);
	
		cudaFree(non_zeros_dev);
		cudaFree(cols_dev);
		cudaFree(soln_dev);
		cudaFree(soln2_dev);
	}
}
