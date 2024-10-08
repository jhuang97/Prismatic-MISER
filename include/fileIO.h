#ifndef PRISMATIC_FILEIO_H
#define PRISMATIC_FILEIO_H
#include "H5Cpp.h"
#include "params.h"
#include <thread>

struct complex_float_t
{
	PRISMATIC_FLOAT_PRECISION re;
	PRISMATIC_FLOAT_PRECISION im;
};

struct aberration_t
{
    int m;
    int n;
    PRISMATIC_FLOAT_PRECISION mag;
    PRISMATIC_FLOAT_PRECISION angle;
};


namespace Prismatic{

static std::mutex write4D_lock;

void setupOutputFile(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void setup4DOutput(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void setupVDOutput(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void setup2DOutput(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void setupDPCOutput(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void setupSMatrixOutput(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, const int FP);

void setupHRTEMOutput(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void setupHRTEMOutput_virtual(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void sortHRTEMbeams(Parameters<PRISMATIC_FLOAT_PRECISION> &pars); 

void setupProbeOutput(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void writeRealSlice(H5::DataSet dataset, const PRISMATIC_FLOAT_PRECISION *buffer, const hsize_t *mdims);

void writeDatacube3D(H5::DataSet dataset, const PRISMATIC_FLOAT_PRECISION *buffer, const hsize_t *mdims);

void writeDatacube3D(H5::DataSet dataset, const std::complex<PRISMATIC_FLOAT_PRECISION> *buffer, const hsize_t *mdims);

void writeStringArray(H5::DataSet dataset,H5std_string * string_array, hsize_t elements);

void savePotentialSlices(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void saveHRTEM(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, Array3D<PRISMATIC_FLOAT_PRECISION> &net_output);

void saveSTEM(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void save_qArr(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void saveProbe(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void save_complex_2D(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> arr, 
	const std::string &addr, const std::string &dsetname);

void save_real_2D(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, Array2D<PRISMATIC_FLOAT_PRECISION> arr, 
	const std::string &addr, const std::string &dsetname);

void save_real_3D(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, Array3D<PRISMATIC_FLOAT_PRECISION> arr, 
	const std::string &addr, const std::string &dsetname);

void configureImportFP(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

std::string getDigitString(int digit);

void writeMetadata(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

Array2D<PRISMATIC_FLOAT_PRECISION> readDataSet2D(const std::string &filename, const std::string &dataPath);

Array3D<PRISMATIC_FLOAT_PRECISION> readDataSet3D(const std::string &filename, const std::string &dataPath);

Array4D<PRISMATIC_FLOAT_PRECISION> readDataSet4D(const std::string &filename, const std::string &dataPath);

Array4D<PRISMATIC_FLOAT_PRECISION> readDataSet4D_keepOrder(const std::string &filename, const std::string &dataPath);

// template <size_t N, class T>
// ArrayND<N, T> readDataSet(const std::string &filename, const std::string &dataPath, size_t );

void readAttribute(const std::string &filename, const std::string &groupPath, const std::string &attr, PRISMATIC_FLOAT_PRECISION &val);

void readAttribute(const std::string &filename, const std::string &groupPath, const std::string &attr, PRISMATIC_FLOAT_PRECISION *val);

void readAttribute(const std::string &filename, const std::string &groupPath, const std::string &attr, int &val);

void readAttribute(const std::string &filename, const std::string &groupPath, const std::string &attr, int *val);

void readAttribute(const std::string &filename, const std::string &groupPath, const std::string &attr, std::string &val);

void writeComplexDataSet(H5::Group group,
                        const std::string &dsetname,
                        const std::complex<PRISMATIC_FLOAT_PRECISION> *buffer,
                        const hsize_t *mdims,
                        const size_t &rank,
                        std::vector<size_t> &order);

void writeRealDataSet(H5::Group group,
                        const std::string &dsetname,
                        const PRISMATIC_FLOAT_PRECISION *buffer,
                        const hsize_t *mdims,
                        const size_t &rank,
                        std::vector<size_t> &order);

void writeRealDataSet_inOrder(H5::Group group,
                        const std::string &dsetname,
                        const PRISMATIC_FLOAT_PRECISION *buffer,
                        const hsize_t *mdims,
                        const size_t &rank);

void writeComplexDataSet_inOrder(H5::Group group,
                        const std::string &dsetname,
                        const std::complex<PRISMATIC_FLOAT_PRECISION> *buffer,
                        const hsize_t *mdims,
                        const size_t &rank);

void writeScalarAttribute(H5::H5Object &object, const std::string &name, const int &data);

void writeScalarAttribute(H5::H5Object &object, const std::string &name, const PRISMATIC_FLOAT_PRECISION &data);

void writeScalarAttribute(H5::H5Object &object, const std::string &name, const std::string &data);

int countDataGroups(H5::Group group, const std::string &basename);

int countDimensions(H5::Group group, const std::string &basename);

void configureSupergroup(H5::Group &new_sg,
						H5::Group &sourceExample,
						const std::vector<std::vector<PRISMATIC_FLOAT_PRECISION>> &sgdims,
						const std::vector<std::string> &sgdims_name,
						const std::vector<std::string> &sgdims_units);

void writeVirtualDataSet(H5::Group group,
						const std::string &dsetName,
						std::vector<H5::DataSet> &datasets,
						std::vector<std::vector<size_t>> indices);

void depthSeriesSG(H5::H5File &file);

void CCseriesSG(H5::H5File &file);

std::string reducedDataSetName(std::string &fullPath);

void copyDataSet(H5::Group &targetGroup, H5::DataSet &source);

void restrideElements(H5::DataSpace &fspace, std::vector<size_t> &dims, std::vector<size_t> &order);

void restrideElements_subset(H5::DataSpace &fspace, std::vector<size_t> &dims, std::vector<size_t> &order, std::vector<size_t> &offset);

//return coords so they can be altered in place manually
hsize_t* restrideElements_subset(std::vector<size_t> &dims, std::vector<size_t> &order, std::vector<size_t> &offset);

template <size_t N>
void readComplexDataSet(ArrayND<N, std::vector<std::complex<PRISMATIC_FLOAT_PRECISION>>> &output,
                            const std::string &filename, 
                            const std::string &dataPath,
                            std::vector<size_t> &order)
{
	H5::H5File input = H5::H5File(filename.c_str(), H5F_ACC_RDONLY);
	H5::DataSet dataset = input.openDataSet(dataPath.c_str());
	H5::DataSpace dataspace = dataset.getSpace();

	hsize_t dims_out[N];
	int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
	H5::DataSpace mspace(N,dims_out);

    H5::DSetCreatPropList plist = dataset.getCreatePlist();
    H5D_layout_t layout = plist.getLayout();
    bool virtualCheck = layout == H5D_VIRTUAL;

    std::array<size_t, N> data_dims;
    for(auto i = 0; i < N; i++) data_dims[i] = dims_out[N-1-order[i]];

    if(N > 1 && (!virtualCheck))
    {
        std::vector<size_t> rdims;
        for(auto i = 0;i < N; i++) rdims.push_back(dims_out[i]);
        restrideElements(dataspace, rdims, order);
    }

	output = zeros_ND<N, std::complex<PRISMATIC_FLOAT_PRECISION>>(data_dims);
    dataset.read(&output[0], dataset.getDataType(), mspace, dataspace);

    mspace.close();
    dataspace.close();
    dataset.close();
    input.close();
};

template <size_t N>
void readComplexDataSet_inOrder(ArrayND<N, std::vector<std::complex<PRISMATIC_FLOAT_PRECISION>>> &output,
                            const std::string &filename, 
                            const std::string &dataPath)
{
	H5::H5File input = H5::H5File(filename.c_str(), H5F_ACC_RDONLY);
	H5::DataSet dataset = input.openDataSet(dataPath.c_str());
	H5::DataSpace dataspace = dataset.getSpace();

	hsize_t dims_out[N];
	int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
	H5::DataSpace mspace(N,dims_out);

    std::array<size_t, N> data_dims;
    for(auto i = 0; i < N; i++) data_dims[i] = dims_out[i];

	output = zeros_ND<N, std::complex<PRISMATIC_FLOAT_PRECISION>>(data_dims);
    dataset.read(&output[0], dataset.getDataType(), mspace, dataspace);

    mspace.close();
    dataspace.close();
    dataset.close();
    input.close();
};

template <size_t N>
void readRealDataSet(ArrayND<N, std::vector<PRISMATIC_FLOAT_PRECISION>> &output,
                        const std::string &filename,
                        const std::string &dataPath,
                        std::vector<size_t> &order)
{
	H5::H5File input = H5::H5File(filename.c_str(), H5F_ACC_RDONLY);
	H5::DataSet dataset = input.openDataSet(dataPath.c_str());
	H5::DataSpace dataspace = dataset.getSpace();

	hsize_t dims_out[N];
	int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
	H5::DataSpace mspace(N,dims_out);

    H5::DSetCreatPropList plist = dataset.getCreatePlist();
    H5D_layout_t layout = plist.getLayout();
    bool virtualCheck = layout == H5D_VIRTUAL;

    std::array<size_t, N> data_dims;
    for(auto i = 0; i < N; i++) data_dims[i] = dims_out[N-1-order[i]];

    if(N > 1 && (!virtualCheck))
    {
        std::vector<size_t> rdims;
        for(auto i = 0;i < N; i++) rdims.push_back(dims_out[i]);
        restrideElements(dataspace, rdims, order);
    }

	output = zeros_ND<N, PRISMATIC_FLOAT_PRECISION>(data_dims);
    dataset.read(&output[0], dataset.getDataType(), mspace, dataspace);

    mspace.close();
    dataspace.close();
    dataset.close();
    input.close();
};

template <size_t N>
void readRealDataSet_inOrder(ArrayND<N, std::vector<PRISMATIC_FLOAT_PRECISION>> &output,
                        const std::string &filename,
                        const std::string &dataPath)
{
	H5::H5File input = H5::H5File(filename.c_str(), H5F_ACC_RDONLY);
	H5::DataSet dataset = input.openDataSet(dataPath.c_str());
	H5::DataSpace dataspace = dataset.getSpace();

	hsize_t dims_out[N];
	int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
	H5::DataSpace mspace(N,dims_out);

    std::array<size_t, N> data_dims;
    for(auto i = 0; i < N; i++) data_dims[i] = dims_out[i];

	output = zeros_ND<N, PRISMATIC_FLOAT_PRECISION>(data_dims);
    dataset.read(&output[0], dataset.getDataType(), mspace, dataspace);

    mspace.close();
    dataspace.close();
    dataset.close();
    input.close();
};


// void writeDatacube4D(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, PRISMATIC_FLOAT_PRECISION *buffer, const hsize_t *mdims, const hsize_t *offset, const PRISMATIC_FLOAT_PRECISION numFP, const std::string nameString);
template<class T>
void writeDatacube4D(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, T *buffer, T *readBuffer, const hsize_t *mdims, const hsize_t *offset, const PRISMATIC_FLOAT_PRECISION numFP, const std::string nameString)
{
	//for 4D writes, need to first read the data set and then add; this way, FP are accounted for
	//lock the whole file access/writing procedure in only one location
	std::unique_lock<std::mutex> writeGatekeeper(write4D_lock);

    H5::Group dataGroup = pars.outputFile.openGroup(nameString);
    H5::DataSet dataset = dataGroup.openDataSet("data");

    //set up file and memory spaces
    H5::DataSpace fspace = dataset.getSpace();
    H5::DataSpace mspace(4, mdims); //rank = 4

    //read old frozen phonon set
    fspace.selectHyperslab(H5S_SELECT_SET, mdims, offset);

    dataset.read(&readBuffer[0], dataset.getDataType(), mspace, fspace);
    for (auto i = 0; i < mdims[0] * mdims[1] * mdims[2] * mdims[3]; i++)
        readBuffer[i] += buffer[i]/numFP;
        
    dataset.write(&readBuffer[0], dataset.getDataType(), mspace, fspace);

    fspace.close();
    mspace.close();
    dataset.flush(H5F_SCOPE_LOCAL);
    dataset.close();
    dataGroup.flush(H5F_SCOPE_LOCAL);
    dataGroup.close();
    pars.outputFile.flush(H5F_SCOPE_LOCAL);

	writeGatekeeper.unlock();
};

template<class T>
void writeDatacube4D_MS_MISER(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, T *ave_buffer, T *var_buffer, T *readBuffer, const hsize_t *mdims, const hsize_t *offset, const std::string nameString)
{
	// no lock; this function should not be called during multithreading

    H5::Group dataGroup = pars.outputFile.openGroup(nameString);

    // write the average
    H5::DataSet dataset = dataGroup.openDataSet("data");
    //set up file and memory spaces
    H5::DataSpace fspace = dataset.getSpace();
    H5::DataSpace mspace(4, mdims); //rank = 4
    //read old frozen phonon set
    fspace.selectHyperslab(H5S_SELECT_SET, mdims, offset);

    for (auto i = 0; i < mdims[0] * mdims[1] * mdims[2] * mdims[3]; i++)
        readBuffer[i] = ave_buffer[i];
        
    dataset.write(&readBuffer[0], dataset.getDataType(), mspace, fspace);
    fspace.close();
    mspace.close();
    dataset.flush(H5F_SCOPE_LOCAL);
    dataset.close();

    // write the variance
    H5::DataSet dataset2 = dataGroup.openDataSet("variance");
    //set up file and memory spaces
    H5::DataSpace fspace2 = dataset2.getSpace();
    H5::DataSpace mspace2(4, mdims); //rank = 4
    //read old frozen phonon set
    fspace2.selectHyperslab(H5S_SELECT_SET, mdims, offset);

    for (auto i = 0; i < mdims[0] * mdims[1] * mdims[2] * mdims[3]; i++)
        readBuffer[i] = var_buffer[i];
        
    dataset2.write(&readBuffer[0], dataset2.getDataType(), mspace2, fspace2);
    fspace2.close();
    mspace2.close();
    dataset2.flush(H5F_SCOPE_LOCAL);
    dataset2.close();


    dataGroup.flush(H5F_SCOPE_LOCAL);
    dataGroup.close();
    pars.outputFile.flush(H5F_SCOPE_LOCAL);

};

void createScratchFile(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void removeScratchFile(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void updateScratchData(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

} //namespace Prismatic

#endif //PRISMATIC_FILEIO_H