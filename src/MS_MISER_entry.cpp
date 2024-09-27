// by Jeffrey Huang, making use of code by Alan (AJ) Pryor, Jr.

#include "meta.h"
#include "params.h"
#include "ArrayND.h"
#include "configure.h"
#include "MS_MISER_calcOutput.h"
#include "PRISM01_calcPotential.h"
#include "PRISM02_calcSMatrix.h"
#include <algorithm>
#include <random>
#include "utility.h"
#include "fileIO.h"
#include "MS_MISER_entry.h"
#include "aberration.h"
#include <boost/optional.hpp>

namespace Prismatic
{

void MS_MISER_entry(Metadata<PRISMATIC_FLOAT_PRECISION>& meta)
{
    Parameters<PRISMATIC_FLOAT_PRECISION> pars;
	try
	{ // read atomic coordinates
		pars = Parameters<PRISMATIC_FLOAT_PRECISION>(meta);
	}
	catch (...)
	{
		std::cout << "Terminating" << std::endl;
		exit(1);
	}

    MS_MISER_entry_pars(pars);
}

void MS_MISER_entry_pars(Parameters<PRISMATIC_FLOAT_PRECISION> &pars)
{
	pars.outputFile = H5::H5File(pars.meta.filenameOutput.c_str(), H5F_ACC_TRUNC);

	setupOutputFile(pars);

	H5::Group testGroup(pars.outputFile.createGroup("/test"));
	std::cout << "made group /test" << std::endl;
	pars.outputFile.close();

	pars.meta.aberrations = updateAberrations(pars.meta.aberrations, pars.meta.probeDefocus, pars.meta.C3, pars.meta.C5, pars.lambda);

	setupCoordinates_MS_MISER(pars);

	PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> pw = PRISM01_potential_workspace(pars);

	setupDetector_MS_MISER(pars);

	setupProbes_MS_MISER(pars);

	std::cout << "# atoms: " << pw.x.size() << std::endl;

	std::cout << "# probe positions: " << pars.numProbes;
	if (!pars.meta.arbitraryProbes) {
		std::cout << " = " << pars.xp.size() << " x " << pars.yp.size();
	}
	std::cout << std::endl;

	pars.outputFile = H5::H5File(pars.meta.filenameOutput.c_str(), H5F_ACC_RDWR);
	createStack_MS_MISER(pars);
	pars.outputFile.close();

	for (size_t probe_num = 0; probe_num < 2; ++probe_num) {
	// for (size_t probe_num = 0; probe_num < pars.numProbes; ++probe_num) {
		MS_MISER_run_probe_pos(pars, pw, probe_num);
	}
}

void MS_MISER_run_probe_pos(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, 
        PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw, size_t probe_num)
{
	std::cout << "\nProbe position #" << probe_num << std::endl;

	auto shifted_probe = get_shifted_probe(pars, probe_num);
	size_t ndims = 2 * pw.x.size();

	// If you set the limits to 0 to 1, there is a ~1 in 10^8 chance of randomly sampling 
	// something too close to 0 or 1 and getting an overflow error. For single precision,
	// 1e-7 or 1 - 1e-7 is okay, but 1e-8 and 1 - 1e-8 cause an overflow error.
	Array1D<PRISMATIC_FLOAT_PRECISION> intg_lo(vector<PRISMATIC_FLOAT_PRECISION>(ndims, 1e-7f), {{ndims}});
	Array1D<PRISMATIC_FLOAT_PRECISION> intg_hi(vector<PRISMATIC_FLOAT_PRECISION>(ndims, 1 - 1e-7f), {{ndims}});

	vector<size_t> repr_iqx;
	vector<size_t> repr_iqy;
	ms_miser_add_repr_pts(pars, repr_iqx, repr_iqy);

	MiserPlanner planner(pars, pw, shifted_probe, repr_iqx, repr_iqy);
	MiserTree tree = planner.make_plan(intg_lo, intg_hi, pars.meta.numFP);
	auto ave_var = planner.execute_miser_plan(tree);

	pars.outputFile = H5::H5File(pars.meta.filenameOutput.c_str(), H5F_ACC_RDWR);
	save_CBED_miser(pars, ave_var, probe_num);
	pars.outputFile.close();


	// unsigned int seed0 = ((rand() & 0x7fffu)<<17 | (rand() & 0x7fffu)<<2 ) | (rand() & 0x7fffu)>>13;
	// std::mt19937 gen(seed0);

	// auto intOutput = get_MS_MISER_single_uncropped_CPU(pars, pw, shifted_probe, intg_lo, intg_hi, gen);
	// std::cout << "intOutput size: " << intOutput.get_dimj() << " " << intOutput.get_dimi() << std::endl;
	// auto intOutput_small = cropOutput(intOutput, pars);
	// std::cout << "intOutput_small size: " << intOutput_small.get_dimj() << " " << intOutput_small.get_dimi() << std::endl;


	// for (size_t i = 0; i < repr_iqx.size(); ++i) {
	// 	std::cout << repr_iqx[i] << ", " << repr_iqy[i] << ": " << intOutput.at(repr_iqy[i], repr_iqx[i]) 
	// 		<< " | " << abs(shifted_probe.at(repr_iqy[i], repr_iqx[i])) << std::endl;
	// }

	// // =========
	// Array2D<PRISMATIC_FLOAT_PRECISION> repr_vals;
	// Array2D<PRISMATIC_FLOAT_PRECISION> pts_sampled;
	// get_multislice_miser_rect_repr_CPU(pars, pw, shifted_probe, intg_lo, intg_hi, 5, repr_iqx, repr_iqy, pts_sampled, repr_vals);

	// std::cout << "sampled values:\n=========\n";
	// for (size_t i = 0; i < repr_vals.get_dimi(); ++i) {
	// 	std::cout << repr_iqx[i] << ", " << repr_iqy[i] << ": ";
	// 	for (size_t ipt = 0; ipt < 5; ++ipt) {	
	// 		std::cout << repr_vals.at(ipt, i) << " ";
	// 	}
	// 	std::cout << "\n";
	// }
	// std::cout << std::endl;

	// std::cout << "sampled pts:\n=========" << std::endl;
	// for (size_t i = 0; i < pts_sampled.get_dimi(); ++i) {
	// 	std::cout << "dim " << i << ": ";
	// 	for (size_t ipt = 0; ipt < 5; ++ipt) {	
	// 		std::cout << pts_sampled.at(ipt, i) << " ";
	// 	}
	// 	std::cout << "\n";
	// }
	// std::cout << std::endl;
	// // ===========

	// need to have set up the right version of the workspace pw before this
	// auto test_pot = generateProjectedPotentials_miser(pars, pw, intg_lo, intg_hi, gen);

	// PRISMATIC_FFTW_INIT_THREADS();
	// auto test_pot = generateProjectedPotentials3D_miser(pars, pw, intg_lo, intg_hi, gen);
	// PRISMATIC_FFTW_CLEANUP_THREADS();
	
	// pars.outputFile = H5::H5File(pars.meta.filenameOutput.c_str(), H5F_ACC_RDWR);
	// save_real_2D(pars, intOutput, "/test", string_format("test%d", probe_num));
	// // save_real_3D(pars, test_pot, "/test", string_format("test%d", probe_num));
	// // save_complex_2D(pars, shifted_probe, "/test", string_format("test%d", probe_num));
	// pars.outputFile.close();

}


}