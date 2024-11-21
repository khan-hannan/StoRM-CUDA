/*
 * StormcuSpar.h
 *
 *  Created on: Sep 25, 2020
 *      Author: hannan
 */

#ifndef STORMCUSPAR_H_
#define STORMCUSPAR_H_


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <boost/thread.hpp>

typedef enum {
    V_Dprecision = 0,
    V_Sprecision  = 1
} cusValueType;

typedef enum {
    I_Dprecision = 0,
    I_Sprecision  = 1
} cusIndexType;

using std::cout;
using std::endl;

#define NUM_THREADS 2
#define StreamSize 8

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
//        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

template <typename Ind, typename Val>
class StormcuSpar {

//private:
public:
	typedef Ind itype;
	typedef Val vtype;

	boost::thread* t1;
	boost::thread* t2;
	boost::thread* t3;

//	const int StreamSize = 4;

	bool DevMeminit;
	bool HostMeminit;

	size_t TotalDevMem;
	size_t FreeDevMem;

	int ID = 0;

    Val alpha = 1;
    Val beta = 0;

	int Mat_Row = 0;
	int Mat_Cols = 0;
	int Mat_NNZ = 0;
	int Mat_Larger = 0;

	int MemEqui = 0;


    Ind   *dA_csrOffsets, *dA_columns;
    Val *dA_values, *dX, *dY, *ArbdX;

	void*                dBuffer    = NULL;
	size_t               bufferSize = 0;

    pthread_t threads[NUM_THREADS];
    int rc;

    cudaStream_t stream[StreamSize];

    std::vector<Ind*> AddressVectorCol;
    std::vector<Val*> AddressVectorVal;
    std::vector<Ind> SizeofCopy;

    Ind *Cols;
    Val *Vals;

    int IndSize;
    int ValSize;
    double lastcopysize = 0;

    Ind LastSize = 0;

    uint32_t FixedLen = 90000;
    int IterationLen = 0;

    int count = 0;

    int LCsizeV = 0;
    int LCsizeC = 0;

    int TherChk = 0;

    cusparseHandle_t     handle = NULL;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;

	cusparseStatus_t status;

	void HostMemCopyAlloc(double);
	void HostAllocFlush(double);

	void HostMemAlloc();
	void DevMemAlloc();

	void SparInit();

	void createFixedMems();

	void AddRow(double);

	void AddChunk(double );

	void CopyDense(Val);

//	Val* getMemPointer(int);

	StormcuSpar(int);

	cusValueType Val_type;
	cusIndexType Index_type;

	Ind *hA_csrOffsets, *hA_columns;
    Val	*hA_values, *hY, *hX;


	virtual ~StormcuSpar();


	int InitMemory(int, int, int);
	int cuMult(int);
	int cuMult2(int);
	int cuMult3(int);

};

#endif /* STORMCUSPAR_H_ */
