/*
 * StormcuSpar.cpp
 *
 *  Created on: Sep 25, 2020
 *      Author: hannan
 */

#include "StormcuSpar.h"


template <typename Ind, typename Val>
StormcuSpar<Ind, Val>::StormcuSpar(int DevID) {
	int DeviceCount = 0;

	cudaGetDeviceCount(&DeviceCount);
	DevMeminit = false;
	HostMeminit = false;


	cudaMemGetInfo(&FreeDevMem, &TotalDevMem);

	std::cout <<"Total Mem = "<<TotalDevMem<<" Free Mem = "<<FreeDevMem<<std::endl;

	IndSize = sizeof(Ind);
	ValSize = sizeof(Val);

	if (sizeof(Ind) == sizeof(int))
		Index_type = I_Sprecision;
	else
		Index_type = I_Dprecision;

	if (sizeof(Val) == sizeof(float))
		Val_type = V_Sprecision;
	else
		Val_type = V_Dprecision;



	if (DeviceCount > 0){

		cudaSetDevice(DevID);

		cudaFree(0);

	    for (int i = 0; i < StreamSize; ++i) {
	    	cudaStreamCreate(&stream[i]);
		}

		status = cusparseCreate(&handle);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
		}


	}
	else {
		cout<<"No CUDA device found"<<endl;

		exit(0);
	}
}

template <typename Ind, typename Val>
StormcuSpar<Ind, Val>::~StormcuSpar() {

	cudaError_t cudaFuncResult;

	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(vecY);
	cusparseDestroySpMat(matA);
	cusparseDestroy(handle);

	delete t1;
	delete t2;
	delete t3;

	if (hX != NULL)
		cudaFreeHost(hX);
	if (hY != NULL)
		cudaFreeHost(hY);
	if (hA_csrOffsets != NULL)
		cudaFreeHost(hA_csrOffsets);
	if (hA_columns != NULL)
		cudaFreeHost(hA_columns);
	if (hA_values != NULL)
		cudaFreeHost(hA_values);

	if(Cols != NULL){
		cudaFuncResult = cudaFreeHost(Cols);
		if (cudaFuncResult != cudaSuccess) {
			std::cout << " Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;

		}
	}

	if(Vals != NULL){
		cudaFuncResult = cudaFreeHost(Vals);
		if (cudaFuncResult != cudaSuccess) {
			std::cout << " Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;

		}
	}


	if (dX != NULL)
		cudaFree(dX);
	if (dY != NULL)
		cudaFree(dY);
	if (dA_csrOffsets != NULL)
		cudaFree(dA_csrOffsets);
	if (dA_columns != NULL)
		cudaFree(dA_columns);
	if (dA_values != NULL)
		cudaFree(dA_values);

	for (int var = 0; var < AddressVectorCol.size(); ++var) {
		cudaFree(AddressVectorCol[var]);
		cudaFree(AddressVectorVal[var]);
	}

}

//template <typename Ind, typename Val>
//Val* StormcuSpar<Ind, Val>::getMemPointer(int){
//	cudaFuncResult = cudaMalloc((void **) &dA_csrOffsets, (Mat_Row+1)*sizeof(Ind));
//	if (cudaFuncResult != cudaSuccess) {
//		std::cout << "Could not allocate memory for dA_csrOffsets, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
//		return;
//	}
//}
template <typename Ind, typename Val>
void StormcuSpar<Ind, Val>::HostMemCopyAlloc(double Size){

	cudaError_t cudaFuncResult;

	if(Cols != NULL){
		cudaFuncResult = cudaFreeHost(Cols);
		if (cudaFuncResult != cudaSuccess) {
			std::cout << " Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;

		}
	}

	if(Vals != NULL){
		cudaFuncResult = cudaFreeHost(Vals);
		if (cudaFuncResult != cudaSuccess) {
			std::cout << " Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;

		}
	}



	cudaFuncResult = cudaMallocHost((void **) &Cols, (Size)*sizeof(Ind));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for Cols, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	cudaFuncResult = cudaMallocHost((void **) &Vals, (Size)*sizeof(Val));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for hX, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}


}

template <typename Ind, typename Val>
void StormcuSpar<Ind, Val>::HostAllocFlush(double Size){

	cudaError_t cudaFuncResult;


	cudaFuncResult = cudaMallocHost((void **) &hA_csrOffsets, (Size+1)*sizeof(Ind));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for csroffsets, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}


}

template <typename Ind, typename Val>
void StormcuSpar<Ind, Val>::AddRow(double Size){
	Ind *TempInd;
	Val *TempVal;


//	hA_csrOffsets[(int)Size] = Lastval;



//	for (int var = 0; var < Size+1; ++var) {
//		std::cout<<hA_csrOffsets[var]<<std::endl;
//	}

	cudaError_t cudaFuncResult;
	std::cout << "Cpy2 " <<std::endl;


	cudaFuncResult = cudaMalloc((void**)&dA_csrOffsets, (Size+1)*sizeof(Ind));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for chunk, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}


	cudaFuncResult = cudaMemcpyAsync(dA_csrOffsets, hA_csrOffsets, (Size+1)*sizeof(Ind), cudaMemcpyHostToDevice, stream[3]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	std::cout << "Cpy3 " <<std::endl;


}


template <typename Ind, typename Val>
void StormcuSpar<Ind, Val>::createFixedMems(){

#ifdef FixedMem

	int equiBlocks = FreeDevMem/(5*ValSize);
	cudaError_t cudaFuncResult;

	MemEqui = equiBlocks;

	std::cout << "Fixed Memory of size "<<equiBlocks<<" being generated"<< std::endl;

	cudaFuncResult = cudaMalloc((void **) &dA_columns, (equiBlocks)*sizeof(Ind));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for dA_columns, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	cudaFuncResult = cudaMalloc((void **) &dA_values, (equiBlocks)*sizeof(Val));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for dA_values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	cudaFuncResult = cudaMalloc((void **) &ArbdX, (equiBlocks)*sizeof(Val));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for dA_values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	cudaFuncResult = cudaMallocHost((void **) &hX, (equiBlocks)*sizeof(Val));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for hX, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}


#endif
}






template <typename Ind, typename Val>
void StormcuSpar<Ind, Val>::AddChunk(double Size){
	Ind *TempInd;
	Val *TempVal;

	cudaError_t cudaFuncResult;

	if(Size <= 0)
		return;


	std::cout << "Adding chunk of size= "<<Size <<std::endl;



#ifdef FixedMem
	cudaFuncResult = cudaMemcpyAsync(dA_columns + (LCsizeC), Cols, Size*sizeof(Ind), cudaMemcpyHostToDevice, stream[1]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}


	LCsizeC += Size;


#else

	cudaFuncResult = cudaMalloc((void**)&TempInd, Size*sizeof(Ind));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for chunk, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	cudaFuncResult = cudaMemcpyAsync(TempInd, Cols, Size*sizeof(Ind), cudaMemcpyHostToDevice, stream[1]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}
	AddressVectorCol.emplace_back(TempInd);

#endif





#ifdef FixedMem

	cudaFuncResult = cudaMemcpyAsync(dA_values + (LCsizeV), Vals, Size*sizeof(Val), cudaMemcpyHostToDevice, stream[2]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}


	cudaFuncResult = cudaMemcpyAsync(ArbdX+ (LCsizeV), hX + (LCsizeV), (Size)*sizeof(Val), cudaMemcpyHostToDevice, stream[8]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return ;
	}

	LCsizeV += Size;

#else
	cudaFuncResult = cudaMalloc((void**)&TempVal, Size*sizeof(Val));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for chunk, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	cudaFuncResult = cudaMemcpyAsync(TempVal, Vals, Size*sizeof(Val), cudaMemcpyHostToDevice, stream[2]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	AddressVectorVal.emplace_back(TempVal);
	SizeofCopy.emplace_back(Size);

#endif

}

template <typename Ind, typename Val>
void StormcuSpar<Ind, Val>::HostMemAlloc(){

	cudaError_t cudaFuncResult;



	cudaFuncResult = cudaMallocHost((void **) &hY, (Mat_Larger)*sizeof(Val));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for hY, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

//	cudaFuncResult = cudaMallocHost((void **) &hA_csrOffsets, (Mat_Row+1)*sizeof(Ind));
//	if (cudaFuncResult != cudaSuccess) {
//		std::cout << "Could not allocate memory for hA_csrOffsets, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
//		return;
//	}

#ifndef FixedMem

	cudaFuncResult = cudaMallocHost((void **) &hX, (Mat_Larger)*sizeof(Val));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for hX, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	cudaFuncResult = cudaMallocHost((void **) &hA_columns, (Mat_NNZ)*sizeof(Ind));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for hA_columns, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	cudaFuncResult = cudaMallocHost((void **) &hA_values, (Mat_NNZ)*sizeof(Val));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for hA_values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

#endif

	HostMeminit = true;
	return;
}

template <typename Ind, typename Val>
void StormcuSpar<Ind,Val>::SparInit(){

	cudaError_t cudaFuncResult;
	sleep(1);

	if (Val_type == V_Dprecision){

		if (Index_type == I_Dprecision)
			status = cusparseCreateCsr(&matA, Mat_Row, Mat_Cols, Mat_NNZ,
					dA_csrOffsets, dA_columns, dA_values,
					CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
					CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

		else
			status = cusparseCreateCsr(&matA, Mat_Row, Mat_Cols, Mat_NNZ,
					dA_csrOffsets, dA_columns, dA_values,
					CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
					CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);


		if (status != CUSPARSE_STATUS_SUCCESS) {
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
			return;
		}


		status = cusparseCreateDnVec(&vecX, Mat_Cols, dX, CUDA_R_64F);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
		}



		status = cusparseCreateDnVec(&vecY, Mat_Row, dY, CUDA_R_64F);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
		}

		status = cusparseSpMV_bufferSize(
										  handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
										  &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
										  CUSPARSE_SPMV_CSR_ALG1, &bufferSize);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
		}


	}
	else{
		if (Index_type == I_Dprecision)
			status = cusparseCreateCsr(&matA, Mat_Row, Mat_Cols, Mat_NNZ,
					dA_csrOffsets, dA_columns, dA_values,
					CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
					CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

		else
			status = cusparseCreateCsr(&matA, Mat_Row, Mat_Cols, Mat_NNZ,
					dA_csrOffsets, dA_columns, dA_values,
					CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
					CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);


		if (status != CUSPARSE_STATUS_SUCCESS) {
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
			return;
		}


		status = cusparseCreateDnVec(&vecX, Mat_Cols, dX, CUDA_R_32F);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
		}

		status = cusparseCreateDnVec(&vecY, Mat_Row, dY, CUDA_R_32F);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
		}

		status = cusparseSpMV_bufferSize(
										  handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
										  &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
										  CUSPARSE_SPMV_CSR_ALG1, &bufferSize);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
		}
	}

	cudaFuncResult = cudaMalloc(&dBuffer, bufferSize);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for dBuffer , Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
	}

}

template <typename Ind, typename Val>
void StormcuSpar<Ind,Val>::DevMemAlloc(){

	cudaError_t cudaFuncResult;


//	cudaFuncResult = cudaMalloc((void **) &dA_csrOffsets, (Mat_Row+1)*sizeof(Ind));
//	if (cudaFuncResult != cudaSuccess) {
//		std::cout << "Could not allocate memory for dA_csrOffsets, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
//		return;
//	}

#ifndef FixedMem
	cudaFuncResult = cudaMalloc((void **) &dA_columns, (Mat_NNZ)*sizeof(Ind));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for dA_columns, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	cudaFuncResult = cudaMalloc((void **) &dA_values, (Mat_NNZ)*sizeof(Val));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for dA_values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

#endif
	cudaFuncResult = cudaMalloc((void **) &dX, (Mat_Larger)*sizeof(Val));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for dX, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}

	cudaFuncResult = cudaMalloc((void **) &dY, (Mat_Larger)*sizeof(Val));
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not allocate memory for dY, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return;
	}


	DevMeminit = true;

	return;

}

template <typename Ind, typename Val>
int StormcuSpar<Ind,Val>::InitMemory(int Rows, int Cols, int NNZ){
	Mat_Row = Rows;
	Mat_Cols = Cols;
	Mat_NNZ = NNZ;
	Mat_Larger = (Rows > Cols)?Rows:Cols;


	cudaError_t cudaFuncResult;

	uint64_t ReqSize = (Mat_Larger*sizeof(Val)*2) + (Mat_NNZ * sizeof(Val)) + (Mat_NNZ * sizeof(Ind)) + ((Mat_Row + 1) * sizeof(Ind)) ;

	if (ReqSize > FreeDevMem){
		cout << "Memory Size required : "<<ReqSize<< " Memory available : "<<FreeDevMem<<endl;
		return -1;
	}

	t3 = new boost::thread(boost::bind(&StormcuSpar<Ind, Val>::DevMemAlloc, this));
	t1 = new boost::thread(boost::bind(&StormcuSpar<Ind, Val>::HostMemAlloc, this));
	t2 = new boost::thread(boost::bind(&StormcuSpar<Ind, Val>::SparInit, this));

	t3->join();
	t2->join();
	t1->join();



	return 0;
}


template <typename Ind, typename Val>
int StormcuSpar<Ind,Val>::cuMult2(int rounds){

	cudaError_t cudaFuncResult;



	if (rounds < 1 || !DevMeminit || !HostMeminit){
		std::cout<<"cuMult not ready"<<std::endl;
		return -1;
	}

//	cudaFuncResult = cudaMemcpyAsync(dA_csrOffsets, hA_csrOffsets, (Mat_Row+1)*sizeof(Ind), cudaMemcpyHostToDevice, stream[6]);
//	if (cudaFuncResult != cudaSuccess) {
//		std::cout << "Could not Memcpy csroffsets, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
//		return -1;
//	}
#ifdef FixedMem
	cudaFuncResult = cudaMemcpyAsync(dX, ArbdX, (Mat_Cols)*sizeof(Val), cudaMemcpyDeviceToDevice, stream[4]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return -1;
	}
#endif

#ifndef FixedMem


	cudaFuncResult = cudaMemcpyAsync(dX, hX, (Mat_Cols)*sizeof(Val), cudaMemcpyHostToDevice, stream[7]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return -1;
	}


	int LastCopy = 0;
	for (int var = 0; var < AddressVectorCol.size(); ++var) {


		cudaFuncResult = cudaMemcpyAsync(dA_columns + (LastCopy), AddressVectorCol[var], (SizeofCopy[var])*sizeof(Ind), cudaMemcpyDeviceToDevice, stream[(var%(StreamSize-1))]);
		if (cudaFuncResult != cudaSuccess) {
			std::cout << "Could not Memcpy dA_columns, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
			std::cout << "at block num " << var << " Total blocks " <<AddressVectorCol.size()<< std::endl;
			return -1;
		}

		LastCopy += SizeofCopy[var];


	}
	LastCopy = 0;
	for (int var = 0; var < AddressVectorVal.size(); ++var) {

		cudaFuncResult = cudaMemcpyAsync(dA_values + (LastCopy), AddressVectorVal[var], (SizeofCopy[var])*sizeof(Val), cudaMemcpyDeviceToDevice, stream[(var%(StreamSize-2))]);
		if (cudaFuncResult != cudaSuccess) {
			std::cout << "Could not Memcpy values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
			std::cout << "at block num " << var << " Total blocks " <<AddressVectorVal.size()<< std::endl;
			return -1;
		}

		LastCopy += SizeofCopy[var];
	}


#endif

	boost::chrono::microseconds ams2;
	boost::chrono::microseconds ams50;
	boost::chrono::microseconds ams100;
	boost::chrono::microseconds ams500;
	boost::chrono::microseconds ams1000;
	

	boost::chrono::high_resolution_clock::time_point astart = boost::chrono::high_resolution_clock::now();
	for (int Loop = 0; Loop < rounds; ++Loop) {
		if (Val_type == V_Dprecision){
			status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
											 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
											 CUSPARSE_SPMV_CSR_ALG1, dBuffer);
		}
		else{
			status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
											 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
											 CUSPARSE_SPMV_CSR_ALG1, dBuffer);
		}

		if (status != CUSPARSE_STATUS_SUCCESS) {
				printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
				return -1;
		}

		if (Loop < rounds-1)
		{
			Val *Temp = dX;
			dX = dY;
			dY = Temp;

			status = cusparseDnVecSetValues(vecX, dX);
			status = cusparseDnVecSetValues(vecY, dY);
			if (status != CUSPARSE_STATUS_SUCCESS) {
					printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
					return -1;
			}
		}

	}
	cudaDeviceSynchronize();
	boost::chrono::microseconds ams = boost::chrono::duration_cast<boost::chrono::microseconds> (boost::chrono::high_resolution_clock::now() - astart);
	std::cout << "Kernel took " << ams.count() << "us " << "\n";

	cudaFuncResult = cudaMemcpy(hY, dY, (Mat_Row)*sizeof(Val), cudaMemcpyDeviceToHost);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values back, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return -1;
	}

	return 0;


}



template <typename Ind, typename Val>
int StormcuSpar<Ind,Val>::cuMult(int rounds){

	cudaError_t cudaFuncResult;


	if (rounds < 1 || !DevMeminit || !HostMeminit){
		std::cout<<"cuMult not ready"<<std::endl;
		return -1;
	}

	cudaFuncResult = cudaMemcpyAsync(dA_csrOffsets, hA_csrOffsets, (Mat_Row+1)*sizeof(Ind), cudaMemcpyHostToDevice, stream[0]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy csroffsets, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return -1;
	}
#ifndef FixedMem
	cudaFuncResult = cudaMemcpyAsync(dA_columns, hA_columns, (Mat_NNZ)*sizeof(Ind), cudaMemcpyHostToDevice, stream[1]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy columns, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return -1;
	}
	cudaFuncResult = cudaMemcpyAsync(dA_values, hA_values, (Mat_NNZ)*sizeof(Val), cudaMemcpyHostToDevice, stream[2]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return -1;
	}
#endif
	cudaFuncResult = cudaMemcpyAsync(dX, hX, (Mat_Cols)*sizeof(Val), cudaMemcpyHostToDevice, stream[3]);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return -1;
	}

	for (int Loop = 0; Loop < rounds; ++Loop) {
		if (Val_type == V_Dprecision){
			status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
											 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
											 CUSPARSE_SPMV_CSR_ALG1, dBuffer);
		}
		else{
			status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
											 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
											 CUSPARSE_SPMV_CSR_ALG1, dBuffer);
		}

		if (status != CUSPARSE_STATUS_SUCCESS) {
				printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
				return -1;
		}

		if (Loop < rounds-1)
		{
			Val *Temp = dX;
			dX = dY;
			dY = Temp;

			status = cusparseDnVecSetValues(vecX, dX);
			status = cusparseDnVecSetValues(vecY, dY);
			if (status != CUSPARSE_STATUS_SUCCESS) {
					printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
					return -1;
			}
		}
	}


	cudaFuncResult = cudaMemcpy(hY, dY, (Mat_Row)*sizeof(Val), cudaMemcpyDeviceToHost);
	if (cudaFuncResult != cudaSuccess) {
		std::cout << "Could not Memcpy values back, Error Code " << cudaGetErrorString(cudaFuncResult) << "." << std::endl;
		return -1;
	}

	return 0;

}


template class StormcuSpar<int, double>;
template class StormcuSpar<uint64_t, double>;
template class StormcuSpar<int, float>;
//template class StormcuSpar<uint64_t, float>;
