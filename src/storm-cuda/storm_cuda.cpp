#include <cstdio>
#include <cuda_runtime.h>

#include <storm/api/storm.h>
#include <storm/storage/SparseMatrix.h>

#include <storm/utility/macros.h>
#include <storm/utility/initialize.h>

#include <storm/utility/cli.h>
#include <storm-cli-utilities/cli.h>
#include <storm-cli-utilities/model-handling.h>

#include <storm-config.h>
#include <storm-parsers/api/storm-parsers.h>

#include <storm/solver/multiplier/Multiplier.h>

#include "storm-cli-utilities/model-handling.h"
#include "storm-cli-utilities/print.h"
#include "storm-cli-utilities/resources.h"

#include "storm-pars/settings/ParsSettings.h"

#include "StormcuSpar.h"

StormcuSpar<int, double> *GlobalCuda;

int main(const int argc,const char **argv)
{

    storm::utility::setLogLevel(l3pp::LogLevel::OFF);
    int LoopTimes = 2;
	
    cudaDeviceProp prop;
	int devId = 0;
	
    cudaGetDeviceProperties(&prop, devId);
	// printf("Device : %s\n", prop.name);
	cudaSetDevice(devId);
	
    // STORM_LOG_DEBUG("my name is hannan khan");

	StormcuSpar<int, double> CudaValTest1(devId);
    GlobalCuda = &CudaValTest1;

    CudaValTest1.createFixedMems();

	for (int i = 0; i < CudaValTest1.MemEqui; ++i) {
		CudaValTest1.hX[i] = i;
	}

    storm::utility::setUp();
    storm::cli::printHeader("Storm-cuda", argc, argv);
    storm::settings::initializeAll("Storm-cuda", "storm-cuda");

    if (!storm::cli::parseOptions(argc, argv)) {
        return -1;
    }

    auto coreSettings = storm::settings::getModule<storm::settings::modules::CoreSettings>();
    auto engine = coreSettings.getEngine();
    STORM_LOG_WARN_COND(
        engine != storm::utility::Engine::Dd || engine != storm::utility::Engine::Hybrid || coreSettings.getDdLibraryType() == storm::dd::DdType::Sylvan,
        "The selected DD library does not support parametric models. Switching to Sylvan...");

    auto symbolicInput = storm::cli::parseSymbolicInput();
    storm::cli::ModelProcessingInformation mpi;
    std::tie(symbolicInput, mpi) = storm::cli::preprocessSymbolicInput(symbolicInput);

	storm::builder::BuilderOptions options;
    auto model = storm::api::buildSparseModel<double>(symbolicInput.model.get(), options);

    storm::storage::SparseMatrix<double> matrix = model->getTransitionMatrix();

    // std::cout << "Engine: "<< engine <<std::endl; 

    // std::cout << "Matrix of size " << matrix.getRowCount() << " x " << matrix.getColumnCount() << std::endl;

    int Status;

    // Create multiplier
    auto factory = storm::solver::MultiplierFactory<double>();
    auto multiplier = factory.create(mpi.env, matrix);

    int A_num_rows = matrix.getRowCount();
    int A_num_cols = matrix.getColumnCount();
    int A_num_nnz  = matrix.getNonzeroEntryCount();

    std::cout<<"########################################################################"<<std::endl;

    storm::cli::printHeader("Storm-cuda", argc, argv);
    std::cout<<"2X NNZ: "<<A_num_nnz<<std::endl;
    std::cout<<"Rows: "<<A_num_rows+1<<std::endl;

    std::cout<<"########################################################################"<<std::endl;



    // std::cout<< "Sparsity = "<<(static_cast<double>((A_num_rows*A_num_cols) - A_num_nnz) / static_cast<double>(A_num_cols*A_num_rows))<<std::endl;

    // return 0; //TODO: USED THIS TO JUST VIEW THE VALUES OF MATRIX SIZE
    // std::cout<<"Non zero values = "<<A_num_nnz<<std::endl;

    // std::cout<< "Sparsity = "<<(static_cast<double>((A_num_rows*A_num_cols) - A_num_nnz) / static_cast<double>(A_num_cols*A_num_rows))<<std::endl;

    if (CudaValTest1.InitMemory(A_num_rows, A_num_cols, A_num_nnz) == -1){
    	std::cout<<"Fail at Init Memory"<<std::endl;
    	return -1;
    }

    cusparseStatus_t status;

    cudaError_t cudaMallocResult;

    for (int i = 0; i < A_num_cols; ++i) {
		CudaValTest1.hX[i] = i/A_num_cols;
	}

    //////////////////////////////////////////////////////////////////////////////////////////////////////
	boost::chrono::high_resolution_clock::time_point astart = boost::chrono::high_resolution_clock::now();
    std::cout << "GPU 2:" << std::endl;

	if(CudaValTest1.cuMult2(2) == -1){
		std::cout<<"Fail at Multiplier"<<std::endl;
		return -1;
	}

	boost::chrono::microseconds ams = boost::chrono::duration_cast<boost::chrono::microseconds> (boost::chrono::high_resolution_clock::now() - astart);
	// std::cout << "GPU 2: time took " << ams.count() << "us " << "\n";
    std::cout << "/////////////////////////////////////" << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////
	astart = boost::chrono::high_resolution_clock::now();
    std::cout << "GPU 50:" << std::endl;

	if(CudaValTest1.cuMult2(50) == -1){
		std::cout<<"Fail at Multiplier"<<std::endl;
		return -1;
	}

	ams = boost::chrono::duration_cast<boost::chrono::microseconds> (boost::chrono::high_resolution_clock::now() - astart);
	// std::cout << "GPU 50 time took " << ams.count() << "us " << "\n";
    std::cout << "/////////////////////////////////////" << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////
	astart = boost::chrono::high_resolution_clock::now();
    std::cout << "GPU 100:" << std::endl;

	if(CudaValTest1.cuMult2(100) == -1){
		std::cout<<"Fail at Multiplier"<<std::endl;
		return -1;
	}

	ams = boost::chrono::duration_cast<boost::chrono::microseconds> (boost::chrono::high_resolution_clock::now() - astart);
	// std::cout << "GPU 100 time took " << ams.count() << "us " << "\n";
    std::cout << "/////////////////////////////////////" << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////
	astart = boost::chrono::high_resolution_clock::now();
    std::cout << "GPU 500:" << std::endl;

	if(CudaValTest1.cuMult2(500) == -1){
		std::cout<<"Fail at Multiplier"<<std::endl;
		return -1;
	}

	ams = boost::chrono::duration_cast<boost::chrono::microseconds> (boost::chrono::high_resolution_clock::now() - astart);
	// std::cout << "GPU 500 time took " << ams.count() << "us " << "\n";
    std::cout << "/////////////////////////////////////" << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////
	astart = boost::chrono::high_resolution_clock::now();
    std::cout << "GPU 1000:" << std::endl;

	if(CudaValTest1.cuMult2(1000) == -1){
		std::cout<<"Fail at Multiplier"<<std::endl;
		return -1;
	}

	ams = boost::chrono::duration_cast<boost::chrono::microseconds> (boost::chrono::high_resolution_clock::now() - astart);
	// std::cout << "GPU 1000 time took " << ams.count() << "us " << "\n";
    std::cout << "/////////////////////////////////////" << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    std::vector<double> x(matrix.getRowCount());

    for (size_t i = 0; i < matrix.getColumnCount(); ++i) {
        x[i] = i/matrix.getColumnCount();
    }


    std::vector<float> b(matrix.getRowCount());
    std::vector<uint_fast64_t> rowGroupIndices = matrix.getRowGroupIndices();

    double endGroups;
    double endRows;

    {
		boost::chrono::high_resolution_clock::time_point start = boost::chrono::high_resolution_clock::now();
		multiplier->repeatedMultiply(mpi.env, x, nullptr, 2);
		boost::chrono::microseconds ms = boost::chrono::duration_cast<boost::chrono::microseconds> (boost::chrono::high_resolution_clock::now() - start);
		std::cout << "CPU 2: time took " << ms.count() << "us " << "\n";
    }

    bool result = true;
    int Hold =0 ;

    for (int i = 0; i < A_num_rows; ++i) {

    	if ((x[i] - CudaValTest1.hY[i]) > 0.000001 ){

    		result = false;
    		std::cout<<"x was : "<<x[i]<<endl;
    		cout<<" Hy was : "<<CudaValTest1.hY[i]<<endl;

    		Hold = i;
    		break;
    	}
	}

    if (result)
    	std::cout<<"MATCHED " <<std::endl;
    else
    	std::cout<<"Break at : "<<Hold<<std::endl;

    return 0;
}
