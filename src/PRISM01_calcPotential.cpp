// Copyright Alan (AJ) Pryor, Jr. 2017
// Transcribed from MATLAB code by Colin Ophus
// Prismatic is distributed under the GNU General Public License (GPL)
// If you use Prismatic, we kindly ask that you cite the following papers:

// 1. Ophus, C.: A fast image simulation algorithm for scanning
//    transmission electron microscopy. Advanced Structural and
//    Chemical Imaging 3(1), 13 (2017)

// 2. Pryor, Jr., A., Ophus, C., and Miao, J.: A Streaming Multi-GPU
//    Implementation of Image Simulation Algorithms for Scanning
//	  Transmission Electron Microscopy. arXiv:1706.08563 (2017)

#include "PRISM01_calcPotential.h"
#include "params.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <map>
#include <vector>
#include <random>
#include <thread>
#include "ArrayND.h"
#include "projectedPotential.h"
#include "WorkDispatcher.h"
#include "utility.h"
#include "fileIO.h"
#include "fftw3.h"
#include <complex>

#ifdef PRISMATIC_BUILDING_GUI
#include "prism_progressbar.h"
#endif

namespace Prismatic
{

using namespace std;
mutex potentialWriteLock;
extern mutex fftw_plan_lock;

void fetch_potentials(Array3D<PRISMATIC_FLOAT_PRECISION> &potentials,
					  const vector<size_t> &atomic_species,
					  const Array1D<PRISMATIC_FLOAT_PRECISION> &xr,
					  const Array1D<PRISMATIC_FLOAT_PRECISION> &yr)
{
	Array2D<PRISMATIC_FLOAT_PRECISION> cur_pot;
	for (auto k = 0; k < potentials.get_dimk(); ++k)
	{
		Array2D<PRISMATIC_FLOAT_PRECISION> cur_pot = projPot(atomic_species[k], xr, yr);
		for (auto j = 0; j < potentials.get_dimj(); ++j)
		{
			for (auto i = 0; i < potentials.get_dimi(); ++i)
			{
				potentials.at(k, j, i) = cur_pot.at(j, i);
			}
		}
	}
}

void fetch_potentials3D(Array4D<std::complex<PRISMATIC_FLOAT_PRECISION>> &potentials,
					  const vector<size_t> &atomic_species,
					  const Array1D<PRISMATIC_FLOAT_PRECISION> &xr,
					  const Array1D<PRISMATIC_FLOAT_PRECISION> &yr,
					  const Array1D<PRISMATIC_FLOAT_PRECISION> &zr)
{
	Array3D<PRISMATIC_FLOAT_PRECISION> cur_pot;
	PRISMATIC_FFTW_INIT_THREADS();
	for (auto l = 0; l < potentials.get_diml(); l++)
	{
		Array3D<PRISMATIC_FLOAT_PRECISION> cur_pot = kirklandPotential3D(atomic_species[l], xr, yr, zr);
		Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> fstore = zeros_ND<2, std::complex<PRISMATIC_FLOAT_PRECISION>>({{cur_pot.get_dimj(), cur_pot.get_dimi()}});
		for (auto k = 0; k < cur_pot.get_dimk(); k++)
		{
			//fourier transform potentials in K loop since we only transform in x, y
			for(auto j = 0; j < cur_pot.get_dimj(); j ++)
			{
				for(auto i = 0; i < cur_pot.get_dimi(); i++)
				{
					fstore.at(j,i).real(cur_pot.at(k,j,i));
				}
			}
			unique_lock<mutex> gatekeeper(fftw_plan_lock);
			PRISMATIC_FFTW_PLAN plan_forward = PRISMATIC_FFTW_PLAN_DFT_2D(cur_pot.get_dimj(), cur_pot.get_dimi(),
																	reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&fstore[0]),
																	reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&potentials.at(l,k,0,0)),
																	FFTW_FORWARD,
																	FFTW_ESTIMATE);

			gatekeeper.unlock();
			PRISMATIC_FFTW_EXECUTE(plan_forward);

			gatekeeper.lock();
			PRISMATIC_FFTW_DESTROY_PLAN(plan_forward);
		}
	}
	PRISMATIC_FFTW_CLEANUP_THREADS();
}

vector<size_t> get_unique_atomic_species(Parameters<PRISMATIC_FLOAT_PRECISION> &pars)
{
	// helper function to get the unique atomic species
	vector<size_t> unique_atoms = vector<size_t>(pars.atoms.size(), 0);
	for (auto i = 0; i < pars.atoms.size(); ++i)
		unique_atoms[i] = pars.atoms[i].species;
	sort(unique_atoms.begin(), unique_atoms.end());
	vector<size_t>::iterator it = unique(unique_atoms.begin(), unique_atoms.end());
	unique_atoms.resize(distance(unique_atoms.begin(), it));
	return unique_atoms;
}

void generateProjectedPotentials(Parameters<PRISMATIC_FLOAT_PRECISION> &pars,
								 const Array3D<PRISMATIC_FLOAT_PRECISION> &potentialLookup,
								 const vector<size_t> &unique_species,
								 const Array1D<long> &xvec,
								 const Array1D<long> &yvec)
{
	// splits the atomic coordinates into slices and computes the projected potential for each.

	// create arrays for the coordinates
	Array1D<PRISMATIC_FLOAT_PRECISION> x = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	Array1D<PRISMATIC_FLOAT_PRECISION> y = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	Array1D<PRISMATIC_FLOAT_PRECISION> z = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	Array1D<PRISMATIC_FLOAT_PRECISION> ID = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	Array1D<PRISMATIC_FLOAT_PRECISION> sigma = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	Array1D<PRISMATIC_FLOAT_PRECISION> occ = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});

	for (auto i = 0; i < pars.atoms.size(); ++i)
	{

		x[i] = pars.atoms[i].x * pars.tiledCellDim[2];
		y[i] = pars.atoms[i].y * pars.tiledCellDim[1];
		z[i] = pars.atoms[i].z * pars.tiledCellDim[0];

		ID[i] = pars.atoms[i].species;
		sigma[i] = pars.atoms[i].sigma;
		occ[i] = pars.atoms[i].occ;
	}

	// compute the z-slice index for each atom
	long numPlanes = ceil(pars.tiledCellDim[0]/pars.meta.sliceThickness);
	Array1D<PRISMATIC_FLOAT_PRECISION> zPlane(z);
	std::transform(zPlane.begin(), zPlane.end(), zPlane.begin(), [&pars](PRISMATIC_FLOAT_PRECISION &t_z) {
		return round((-t_z + pars.tiledCellDim[0]) / pars.meta.sliceThickness + 0.5) - 1; // If the +0.5 was to make the first slice z=1 not 0, can drop the +0.5 and -1
	});
	// auto max_z = std::max_element(zPlane.begin(), zPlane.end());
	pars.numPlanes = numPlanes;

	//check if intermediate output was specified, if so, create index of output slices
	if (pars.meta.numSlices == 0)
	{
		pars.numSlices = pars.numPlanes;
	}

#ifdef PRISMATIC_BUILDING_GUI
	pars.progressbar->signalPotentialUpdate(0, pars.numPlanes);
#endif

	// initialize the potential array
	pars.pot = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>({{pars.numPlanes, pars.imageSize[0], pars.imageSize[1]}});

	// create a key-value map to match the atomic Z numbers with their place in the potential lookup table
	map<size_t, size_t> Z_lookup;
	for (auto i = 0; i < unique_species.size(); ++i)
		Z_lookup[unique_species[i]] = i;

	//loop over each plane, perturb the atomic positions, and place the corresponding potential at each location
	// using parallel calculation of each individual slice
	std::vector<std::thread> workers;
	workers.reserve(pars.meta.numThreads);

	WorkDispatcher dispatcher(0, pars.numPlanes);
	for (long t = 0; t < pars.meta.numThreads; ++t)
	{
		cout << "Launching thread #" << t << " to compute projected potential slices\n";
		workers.push_back(thread([&pars, &x, &y, &z, &ID, &Z_lookup, &xvec, &sigma, &occ,
								  &zPlane, &yvec, &potentialLookup, &dispatcher, &t]()
		{
			// create a random number generator to simulate thermal effects
			// std::cout<<"random seed = " << pars.meta.randomSeed << std::endl;
			// srand(pars.meta.randomSeed);
			// std::default_random_engine de(pars.meta.randomSeed);
			// normal_distribution<PRISMATIC_FLOAT_PRECISION> randn(0,1);
			Array1D<long> xp;
			Array1D<long> yp;

			size_t currentSlice, stop;
			currentSlice = stop = 0;
            // create a random number generator to simulate thermal effects
            std::cout << "random seed = " << pars.meta.randomSeed + t * 10000 << std::endl;
            srand(pars.meta.randomSeed + 10000*t);
            std::mt19937 de(pars.meta.randomSeed + 10000*t);
            normal_distribution<PRISMATIC_FLOAT_PRECISION> randn(0, 1);

			while (dispatcher.getWork(currentSlice, stop))
			{ // synchronously get work assignment
				Array2D<PRISMATIC_FLOAT_PRECISION> projectedPotential = zeros_ND<2, PRISMATIC_FLOAT_PRECISION>({{pars.imageSize[0], pars.imageSize[1]}});
				const long dim0 = (long)pars.imageSize[0];
				const long dim1 = (long)pars.imageSize[1];
				while (currentSlice != stop)
				{
					for (auto atom_num = 0; atom_num < x.size(); ++atom_num)
					{
						if (zPlane[atom_num] == currentSlice)
						{
							if (pars.meta.includeOccupancy)
							{
								if (static_cast<PRISMATIC_FLOAT_PRECISION>(rand()) / static_cast<PRISMATIC_FLOAT_PRECISION>(RAND_MAX) > occ[atom_num])
								{
									continue;
								}
							}
							//								if ( !pars.meta.includeOccupancy || static_cast<PRISMATIC_FLOAT_PRECISION>(rand())/static_cast<PRISMATIC_FLOAT_PRECISION> (RAND_MAX) <= occ[atom_num]) {
							const size_t cur_Z = Z_lookup[ID[atom_num]];
							PRISMATIC_FLOAT_PRECISION X, Y;
							if (pars.meta.includeThermalEffects)
							{ // apply random perturbations
                                PRISMATIC_FLOAT_PRECISION perturbX = randn(de) * sigma[atom_num];
                                PRISMATIC_FLOAT_PRECISION perturbY = randn(de) * sigma[atom_num];
								X = round((x[atom_num] + perturbX) / pars.pixelSize[1]);
								Y = round((y[atom_num] + perturbY) / pars.pixelSize[0]);
							}
							else
							{
								X = round((x[atom_num]) / pars.pixelSize[1]); // this line uses no thermal factor
								Y = round((y[atom_num]) / pars.pixelSize[0]); // this line uses no thermal factor
							}
							xp = xvec + (long)X;
							for (auto &i : xp)
								i = (i % dim1 + dim1) % dim1; // make sure to get a positive value

							yp = yvec + (long)Y;
							for (auto &i : yp)
								i = (i % dim0 + dim0) % dim0; // make sure to get a positive value
							for (auto ii = 0; ii < xp.size(); ++ii)
							{
								for (auto jj = 0; jj < yp.size(); ++jj)
								{
									// fill in value with lookup table
									projectedPotential.at(yp[jj], xp[ii]) += potentialLookup.at(cur_Z, jj, ii);
								}
							}
							//								}
						}
					}
					// copy the result to the full array
					copy(projectedPotential.begin(), projectedPotential.end(), &pars.pot.at(currentSlice, 0, 0));
					#ifdef PRISMATIC_BUILDING_GUI
					pars.progressbar->signalPotentialUpdate(currentSlice, pars.numPlanes);
					#endif //PRISMATIC_BUILDING_GUI
					++currentSlice;
				}
			}
		}));
	}
	cout << "Waiting for threads...\n";
	for (auto &t : workers)
		t.join();
#ifdef PRISMATIC_BUILDING_GUI
	pars.progressbar->setProgress(100);
#endif //PRISMATIC_BUILDING_GUI
};


Array3D<PRISMATIC_FLOAT_PRECISION> generateProjectedPotentials_miser(Parameters<PRISMATIC_FLOAT_PRECISION> &pars,
	PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw,
    Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi, 
	std::mt19937 &gen, Array1D<PRISMATIC_FLOAT_PRECISION> &pt)
{
	auto pot = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>({{pars.numPlanes, pars.imageSize[0], pars.imageSize[1]}});

	uniform_real_distribution<PRISMATIC_FLOAT_PRECISION> uniform(0.0, 1.0); // not actually using range of 0-1
	
	for (size_t currentSlice = 0; currentSlice < pars.numPlanes; ++currentSlice) {
		Array1D<long> xp;
		Array1D<long> yp;
		for (auto atom_num = 0; atom_num < pw.x.size(); ++atom_num) {
			if (pw.pw2D.zPlane[atom_num] == currentSlice) {
				const size_t cur_Z = pw.Z_lookup[pw.ID[atom_num]];
				PRISMATIC_FLOAT_PRECISION X, Y;
				
				// apply random perturbations
				size_t idim_x = 2 * atom_num;
				size_t idim_y = 2 * atom_num + 1;
				uniform_real_distribution<PRISMATIC_FLOAT_PRECISION>::param_type paramX(range_lo[idim_x], range_hi[idim_x]);
				uniform_real_distribution<PRISMATIC_FLOAT_PRECISION>::param_type paramY(range_lo[idim_y], range_hi[idim_y]);
				auto pX = uniform(gen, paramX);
				auto pY = uniform(gen, paramY);
				pt.at(idim_x) = pX;
				pt.at(idim_y) = pY;
				PRISMATIC_FLOAT_PRECISION perturbX = probit(pX) * pw.sigma[atom_num];
				PRISMATIC_FLOAT_PRECISION perturbY = probit(pY) * pw.sigma[atom_num];
				X = round((pw.x[atom_num] + perturbX) / pars.pixelSize[1]);
				Y = round((pw.y[atom_num] + perturbY) / pars.pixelSize[0]);

				xp = pw.xvec + (long)X;
				for (auto &i : xp)
					i = (i % pw.dim1 + pw.dim1) % pw.dim1; // make sure to get a positive value

				yp = pw.yvec + (long)Y;
				for (auto &i : yp)
					i = (i % pw.dim0 + pw.dim0) % pw.dim0; // make sure to get a positive value

				for (auto ii = 0; ii < xp.size(); ++ii)
				{
					for (auto jj = 0; jj < yp.size(); ++jj)
					{
						// fill in value with lookup table
						pot.at(currentSlice, yp[jj], xp[ii]) += pw.pw2D.potentialLookup.at(cur_Z, jj, ii);
					}
				}
			}
		}
	}

	if (pars.meta.importExtraPotential) { addExtraPotential_miser(pars, pw, pot); }

	return pot;
}

void interpolatePotential(Array3D<PRISMATIC_FLOAT_PRECISION> &potShift,
							const Array3D<PRISMATIC_FLOAT_PRECISION> &potCrop,
							const PRISMATIC_FLOAT_PRECISION &wx,
							const PRISMATIC_FLOAT_PRECISION &wy,
							const PRISMATIC_FLOAT_PRECISION &wz,
							const size_t &xind,
							const size_t &yind,
							const size_t &zind)
{
	for(auto k = 0; k < potCrop.get_dimk(); k++)
	{
		for(auto j = 0; j < potCrop.get_dimj(); j++)
		{
			for(auto i = 0; i < potCrop.get_dimj(); i++)
			{
				potShift.at(k+zind,j+yind,i+xind) += potCrop.at(k,j,i)*wx*wy*wz;
			}
		}
	}
};

void cropLookup(Array3D<PRISMATIC_FLOAT_PRECISION> &potCrop,
				const Array4D<PRISMATIC_FLOAT_PRECISION> &potLookup,
				const size_t &cur_Z)
{
	//crops faces off of potLookup
	for(auto k = 0; k < potCrop.get_dimk(); k++)
	{
		for(auto j = 0; j < potCrop.get_dimj(); j++)
		{
			for(auto i = 0; i < potCrop.get_dimi(); i++)
			{
				potCrop.at(k,j,i) = potLookup.at(cur_Z, k+1, j+1, i+1);
			}
		}
	}

};			

void generateProjectedPotentials3D(Parameters<PRISMATIC_FLOAT_PRECISION> &pars,
								   const Array4D<std::complex<PRISMATIC_FLOAT_PRECISION>> &potLookup,
								   const vector<size_t> &unique_species,
								   const Array1D<long> &xvec,
								   const Array1D<long> &yvec,
								   const Array1D<PRISMATIC_FLOAT_PRECISION> &zvec)
{		
	long numPlanes = ceil(pars.tiledCellDim[0]/pars.meta.sliceThickness);
	//check if intermediate output was specified, if so, create index of output slices
	pars.numPlanes = numPlanes;
	if (pars.meta.numSlices == 0) pars.numSlices = pars.numPlanes;

	pars.pot = zeros_ND<3,PRISMATIC_FLOAT_PRECISION>({{ (size_t) numPlanes, pars.imageSize[0], pars.imageSize[1]}});

	// create arrays for the coordinates
	Array1D<PRISMATIC_FLOAT_PRECISION> x = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	Array1D<PRISMATIC_FLOAT_PRECISION> y = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	Array1D<PRISMATIC_FLOAT_PRECISION> z = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	Array1D<PRISMATIC_FLOAT_PRECISION> ID = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	Array1D<PRISMATIC_FLOAT_PRECISION> sigma = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	Array1D<PRISMATIC_FLOAT_PRECISION> occ = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});

	// populate arrays from the atoms structure
	for (auto i = 0; i < pars.atoms.size(); ++i)
	{
        x[i] = pars.atoms[i].x * pars.tiledCellDim[2];
		y[i] = pars.atoms[i].y * pars.tiledCellDim[1];
		z[i] = pars.atoms[i].z * pars.tiledCellDim[0];
		ID[i] = pars.atoms[i].species;
		sigma[i] = pars.atoms[i].sigma;
		occ[i] = pars.atoms[i].occ;
	}

	const long dim1 = (long) pars.pot.get_dimi();
	const long dim0 = (long) pars.pot.get_dimj();

	// correct z orientation
	auto max_z = pars.tiledCellDim[0];

	std::transform(z.begin(), z.end(), z.begin(), [&max_z](PRISMATIC_FLOAT_PRECISION &t_z) {
		return (-t_z + max_z);
	});

	Array1D<PRISMATIC_FLOAT_PRECISION> zr = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{zvec.get_dimi()}});
	for (auto j = 0; j < zr.size(); ++j) zr[j] = (PRISMATIC_FLOAT_PRECISION)zvec[j] * pars.dzPot;

	//create fourier coordinate arrays for use in fourier shifting
	const PRISMATIC_FLOAT_PRECISION pi = std::acos(-1);
	Array1D<PRISMATIC_FLOAT_PRECISION> qy = makeFourierCoords(yvec.size(), (PRISMATIC_FLOAT_PRECISION) 1.0);
	Array1D<PRISMATIC_FLOAT_PRECISION> qx = makeFourierCoords(xvec.size(), (PRISMATIC_FLOAT_PRECISION) 1.0);
	std::pair<Array2D<PRISMATIC_FLOAT_PRECISION>, Array2D<PRISMATIC_FLOAT_PRECISION>> qmesh = meshgrid(qy,qx);
	Array2D<PRISMATIC_FLOAT_PRECISION> qya = qmesh.first;
	Array2D<PRISMATIC_FLOAT_PRECISION> qxa = qmesh.second;


	Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> qyShift = zeros_ND<2,std::complex<PRISMATIC_FLOAT_PRECISION>>({{qya.get_dimj(), qya.get_dimi()}});
	Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> qxShift = zeros_ND<2,std::complex<PRISMATIC_FLOAT_PRECISION>>({{qya.get_dimj(), qya.get_dimi()}});
	std::complex<PRISMATIC_FLOAT_PRECISION> I(0.0, 1.0);
	PRISMATIC_FLOAT_PRECISION two = 2.0;
	for(auto jj = 0; jj < qya.get_dimj(); jj++)
	{
		for(auto ii = 0; ii < qya.get_dimi(); ii++)
		{
			qyShift.at(jj,ii) = -two*I*pi*qya.at(jj,ii);
			qxShift.at(jj,ii) = -two*I*pi*qxa.at(jj,ii);
		}
	}

	//band limit arrays for soft aperture with shift in realpace and fourier space; hard-coded for now
	PRISMATIC_FLOAT_PRECISION rband_max = 0.95;
	PRISMATIC_FLOAT_PRECISION rband_min = 0.75;
	Array2D<PRISMATIC_FLOAT_PRECISION> q1(qya);
	std::transform(qxa.begin(), qxa.end(),
		qya.begin(), q1.begin(), [](const PRISMATIC_FLOAT_PRECISION& a, const PRISMATIC_FLOAT_PRECISION& b){
		return sqrt(a*a + b*b);
	});

	Array2D<PRISMATIC_FLOAT_PRECISION> qband(q1);

	for(auto jj = 0; jj < qband.get_dimj(); jj++)
	{
		for(auto ii = 0; ii < qband.get_dimi(); ii++)
		{
			qband.at(jj,ii) = (rband_max-2*q1.at(jj,ii))/(rband_max-rband_min);
			qband.at(jj,ii) = std::max(qband.at(jj,ii), (PRISMATIC_FLOAT_PRECISION) 0.0);
			qband.at(jj,ii) = std::min(qband.at(jj,ii), (PRISMATIC_FLOAT_PRECISION) 1.0);
			qband.at(jj,ii) = pow(std::sin(qband.at(jj,ii)*pi/2.0), 2.0);
		}
	}

	std::pair<Array2D<long>, Array2D<long>> rmesh = meshgrid(yvec,xvec);
	Array2D<PRISMATIC_FLOAT_PRECISION> rband(qband); //construct with qband to avoid type mismatch, knowing sizes are the same
	PRISMATIC_FLOAT_PRECISION xl = (PRISMATIC_FLOAT_PRECISION) xvec[xvec.size()-1];
	PRISMATIC_FLOAT_PRECISION yl = (PRISMATIC_FLOAT_PRECISION) yvec[yvec.size()-1];

	for(auto jj = 0; jj < rband.get_dimj(); jj++)
	{
		for(auto ii = 0; ii < rband.get_dimi(); ii++)
		{
			rband.at(jj,ii) = pow((rmesh.first.at(jj,ii) / (yl+0.5)), 2.0) + pow((rmesh.second.at(jj,ii) / (xl+0.5)), 2.0);
			rband.at(jj,ii) = (rband.at(jj,ii) <= 1) ? 1.0 : 0.0;  	
		}
	}


	// create a key-value map to match the atomic Z numbers with their place in the potential lookup table
	map<size_t, size_t> Z_lookup;
	for (auto i = 0; i < unique_species.size(); ++i)
		Z_lookup[unique_species[i]] = i;
		
	std::vector<std::thread> workers;
	size_t numWorkers = pars.meta.numThreads; //std::min(pars.meta.numThreads, (size_t) 4); //heuristic for now, TODO: improve parallelization scheme to segment atoms over regions to avoid write locks
	workers.reserve(numWorkers);
	WorkDispatcher dispatcher(0, pars.atoms.size());
	const size_t print_frequency = std::max((size_t)1, pars.atoms.size() / 10);

	PRISMATIC_FFTW_INIT_THREADS();
	std::cout << "Base random seed = " << pars.meta.randomSeed << std::endl;
	for (long t = 0; t < numWorkers; t++)
	{
		std::cout << "Launching thread #" << t << " to compute projected potential slices\n";
		workers.push_back(thread([&pars, &x, &y, &z, &ID, &sigma, &occ, &print_frequency,
								 &Z_lookup, &xvec, &yvec, &zvec, &zr, &dim0, &dim1,
								 &numPlanes, &potLookup, &rband, &qband, &qxShift, &qyShift, &dispatcher, &t]()
		{
			size_t currentAtom, stop;
			currentAtom = stop = 0;
            // create a random number generator to simulate thermal effects
            std::cout << "random seed = " << pars.meta.randomSeed + t * 10000 << std::endl;
            srand(pars.meta.randomSeed+10000*t);
            std::mt19937 de(pars.meta.randomSeed+10000*t);
            normal_distribution<PRISMATIC_FLOAT_PRECISION> randn(0, 1);

			while (dispatcher.getWork(currentAtom, stop))
			{
				while(currentAtom != stop)
				{
					if(!(currentAtom % print_frequency))
					{
						std::cout << "Computing atom " << currentAtom << "/" << pars.atoms.size() << std::endl;
					}

					
					const size_t cur_Z = Z_lookup[ID[currentAtom]];
					PRISMATIC_FLOAT_PRECISION X, Y, Z;
					PRISMATIC_FLOAT_PRECISION perturbX, perturbY, perturbZ;
					if (pars.meta.includeThermalEffects)
					{ // apply random perturbations
						perturbX = randn(de) * sigma[currentAtom];
						perturbY = randn(de) * sigma[currentAtom];
						perturbZ = randn(de) * sigma[currentAtom];
						X = round((x[currentAtom] + perturbX) / pars.pixelSize[1]);
						Y = round((y[currentAtom] + perturbY) / pars.pixelSize[0]);
						Z = (z[currentAtom] + perturbZ); //z gets rounded and normalized later
					}
					else
					{
						perturbX = perturbY = perturbZ = 0;
						X = round((x[currentAtom]) / pars.pixelSize[1]); // this line uses no thermal factor
						Y = round((y[currentAtom]) / pars.pixelSize[0]); // this line uses no thermal factor
						Z = (z[currentAtom]); // this line uses no thermal factor, z gets rounded and normalized later
					}

					PRISMATIC_FLOAT_PRECISION dxPx = (x[currentAtom] + perturbX)/ pars.pixelSize[1] - X;
					PRISMATIC_FLOAT_PRECISION dyPy = (y[currentAtom] + perturbY)/ pars.pixelSize[0] - Y;

					Array1D<long> xp = xvec + (long) X;
					Array1D<long> yp = yvec + (long) Y;

					for(auto &i : xp) i = (i % dim1 + dim1) % dim1;
					for(auto &i : yp) i = (i % dim0 + dim0) % dim0;
					Array1D<long> zp = zeros_ND<1, long>({{zvec.get_dimi()}});
					std::vector<long> zVals(zp.size(), 0);
					for(auto i = 0; i < zp.size(); i++)
					{
						PRISMATIC_FLOAT_PRECISION tmp = round((Z+zr[i])/pars.meta.sliceThickness + 0.5)-1;
						tmp = std::max(tmp, (PRISMATIC_FLOAT_PRECISION) 0.0);
						zp[i] = std::min((long) tmp, numPlanes-1);
						zVals[i] = zp[i];
					}

					std::sort(zVals.begin(), zVals.end());
					auto last = std::unique(zVals.begin(), zVals.end());
					zVals.erase(last, zVals.end());

					//iterate through unique z slice values
					for(auto cz_ind = 0; cz_ind < zVals.size(); cz_ind++)
					{
						
						//create tmp array to add potential lookup table to
						Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> tmp_pot = zeros_ND<2, std::complex<PRISMATIC_FLOAT_PRECISION>>({{yp.size(), xp.size()}});

						for(auto kk = 0; kk < zp.size(); kk++)
						{
							if(zp[kk] == zVals[cz_ind])
							{
								for(auto jj = 0; jj < yp.size(); jj++)
								{
									for(auto ii = 0; ii < xp.size(); ii++)
									{
										tmp_pot.at(jj,ii) += potLookup.at(cur_Z, kk,jj,ii);
									}
								}
							}
						}

						//apply fourier shift and qband limit
						for(auto jj = 0; jj < yp.size(); jj++)
						{
							for(auto ii = 0; ii < xp.size(); ii++)
							{
								tmp_pot.at(jj,ii) *= qband.at(jj,ii) * exp(qxShift.at(jj,ii)*dxPx + qyShift.at(jj,ii)*dyPy);
							}
						}

						//inverse FFT and normalize by size of array
						unique_lock<mutex> gatekeeper(fftw_plan_lock);
						PRISMATIC_FFTW_PLAN plan_inverse = PRISMATIC_FFTW_PLAN_DFT_2D(tmp_pot.get_dimj(), tmp_pot.get_dimi(),
																				reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&tmp_pot[0]),
																				reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&tmp_pot[0]),
																				FFTW_BACKWARD,
																				FFTW_ESTIMATE);
						gatekeeper.unlock();
						PRISMATIC_FFTW_EXECUTE(plan_inverse);
						gatekeeper.lock();
						PRISMATIC_FFTW_DESTROY_PLAN(plan_inverse);
						gatekeeper.unlock();
						for(auto &t : tmp_pot) t /= tmp_pot.get_dimi()*tmp_pot.get_dimj();

						//apply realspace band limit
						for(auto i = 0; i < tmp_pot.size(); i++) tmp_pot[i] *= rband[i];

						//then write
						//put into a mutex lock to prevent race condition on potential writing when atoms overlap within potential bound
						std::unique_lock<std::mutex> write_gatekeeper(potentialWriteLock);
						for(auto jj = 0; jj < yp.size(); jj++)
						{
							for(auto ii = 0; ii < xp.size(); ii++)
							{
								pars.pot.at(zVals[cz_ind],yp[jj],xp[ii]) += tmp_pot.at(jj,ii).real();
							}
						}
						write_gatekeeper.unlock();
					}
					++currentAtom;
				}
			}
		}));
	}
	std::cout << "Waiting for threads...\n";
	for (auto &t : workers)
		t.join();

	PRISMATIC_FFTW_CLEANUP_THREADS();
};

Array3D<PRISMATIC_FLOAT_PRECISION> generateProjectedPotentials3D_miser(Parameters<PRISMATIC_FLOAT_PRECISION> &pars,
	PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw,
    Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi, 
	std::mt19937 &gen, Array1D<PRISMATIC_FLOAT_PRECISION> &pt)
{
	auto pot = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>({{pars.numPlanes, pars.imageSize[0], pars.imageSize[1]}});

	uniform_real_distribution<PRISMATIC_FLOAT_PRECISION> uniform(0.0, 1.0); // not actually using range of 0-1

	//create tmp array to add potential lookup table to
	Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> tmp_pot = zeros_ND<2, std::complex<PRISMATIC_FLOAT_PRECISION>>({{pw.yvec.size(), pw.xvec.size()}});
	unique_lock<mutex> gatekeeper(fftw_plan_lock);
	PRISMATIC_FFTW_PLAN plan_inverse = PRISMATIC_FFTW_PLAN_DFT_2D(tmp_pot.get_dimj(), tmp_pot.get_dimi(),
															reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&tmp_pot[0]),
															reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&tmp_pot[0]),
															FFTW_BACKWARD,
															FFTW_ESTIMATE);
	gatekeeper.unlock();

	auto print_frequency = pars.atoms.size() / 10;

	for (size_t currentAtom = 0; currentAtom < pars.atoms.size(); ++currentAtom) {
		if(!(currentAtom % print_frequency))
		{
			std::cout << "Computing atom " << currentAtom << "/" << pars.atoms.size() << std::endl;
		}

		const size_t cur_Z = pw.Z_lookup[pw.ID[currentAtom]];
		PRISMATIC_FLOAT_PRECISION X, Y, Z;
		PRISMATIC_FLOAT_PRECISION perturbX, perturbY; //perturbZ;

		// apply random perturbations
		size_t idim_x = 2 * currentAtom;
		size_t idim_y = 2 * currentAtom + 1;
		uniform_real_distribution<PRISMATIC_FLOAT_PRECISION>::param_type paramX(range_lo[idim_x], range_hi[idim_x]);
		uniform_real_distribution<PRISMATIC_FLOAT_PRECISION>::param_type paramY(range_lo[idim_y], range_hi[idim_y]);
		auto pX = uniform(gen, paramX);
		auto pY = uniform(gen, paramY);
		pt.at(idim_x) = pX;
		pt.at(idim_y) = pY;
		perturbX = probit(pX) * pw.sigma[currentAtom];
		perturbY = probit(pY) * pw.sigma[currentAtom];
		// perturbZ = randn(de) * sigma[currentAtom];
		X = round((pw.x[currentAtom] + perturbX) / pars.pixelSize[1]);
		Y = round((pw.y[currentAtom] + perturbY) / pars.pixelSize[0]);

		Z = pw.pw3D.z[currentAtom]; // + perturbZ; //z gets rounded and normalized later

		PRISMATIC_FLOAT_PRECISION dxPx = (pw.x[currentAtom] + perturbX)/ pars.pixelSize[1] - X;
		PRISMATIC_FLOAT_PRECISION dyPy = (pw.y[currentAtom] + perturbY)/ pars.pixelSize[0] - Y;

		Array1D<long> xp = pw.xvec + (long) X;
		Array1D<long> yp = pw.yvec + (long) Y;

		for(auto &i : xp) i = (i % pw.dim1 + pw.dim1) % pw.dim1;
		for(auto &i : yp) i = (i % pw.dim0 + pw.dim0) % pw.dim0;
		Array1D<long> zp = zeros_ND<1, long>({{pw.pw3D.zvec.get_dimi()}});
		std::vector<long> zVals(zp.size(), 0);
		for(auto i = 0; i < zp.size(); i++)
		{
			PRISMATIC_FLOAT_PRECISION tmp = round((Z + pw.pw3D.zr[i])/pars.meta.sliceThickness + 0.5)-1;
			tmp = std::max(tmp, (PRISMATIC_FLOAT_PRECISION) 0.0);
			zp[i] = std::min((size_t) tmp, pars.numPlanes-1);
			zVals[i] = zp[i];
		}

		std::sort(zVals.begin(), zVals.end());
		auto last = std::unique(zVals.begin(), zVals.end());
		zVals.erase(last, zVals.end());

		//iterate through unique z slice values
		for(auto cz_ind = 0; cz_ind < zVals.size(); cz_ind++)
		{
			
			// zero out the temporary array
			for(auto& i : tmp_pot) i = 0.0;

			for(auto kk = 0; kk < zp.size(); kk++)
			{
				if(zp[kk] == zVals[cz_ind])
				{
					for(auto jj = 0; jj < yp.size(); jj++)
					{
						for(auto ii = 0; ii < xp.size(); ii++)
						{
							tmp_pot.at(jj,ii) += pw.pw3D.potLookup.at(cur_Z, kk,jj,ii);
						}
					}
				}
			}

			//apply fourier shift and qband limit
			for(auto jj = 0; jj < yp.size(); jj++)
			{
				for(auto ii = 0; ii < xp.size(); ii++)
				{
					tmp_pot.at(jj,ii) *= pw.pw3D.qband.at(jj,ii) * exp(pw.pw3D.qxShift.at(jj,ii)*dxPx + pw.pw3D.qyShift.at(jj,ii)*dyPy);
				}
			}

			PRISMATIC_FFTW_EXECUTE(plan_inverse);

			for(auto &t : tmp_pot) t /= tmp_pot.get_dimi()*tmp_pot.get_dimj();

			//apply realspace band limit
			for(auto i = 0; i < tmp_pot.size(); i++) tmp_pot[i] *= pw.pw3D.rband[i];

			//then write
			for(auto jj = 0; jj < yp.size(); jj++)
			{
				for(auto ii = 0; ii < xp.size(); ii++)
				{
					pot.at(zVals[cz_ind],yp[jj],xp[ii]) += tmp_pot.at(jj,ii).real();
				}
			}
		}
	}

	gatekeeper.lock();
	PRISMATIC_FFTW_DESTROY_PLAN(plan_inverse);
	gatekeeper.unlock();

	if (pars.meta.importExtraPotential) { addExtraPotential_miser(pars, pw, pot); }

	return pot;
}

PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> PRISM01_potential_workspace(Parameters<PRISMATIC_FLOAT_PRECISION> &pars)
{
	cout << "Entering PRISM01_potential_workspace" << endl;
	PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> out;

	PRISMATIC_FLOAT_PRECISION yleng = std::ceil(pars.meta.potBound / pars.pixelSize[0]);
	PRISMATIC_FLOAT_PRECISION xleng = std::ceil(pars.meta.potBound / pars.pixelSize[1]);
	out.xvec = ArrayND<1, vector<long>>(vector<long>(2 * (size_t)xleng + 1, 0), {{2 * (size_t)xleng + 1}});
	out.yvec = ArrayND<1, vector<long>>(vector<long>(2 * (size_t)yleng + 1, 0), {{2 * (size_t)yleng + 1}});
	{
		PRISMATIC_FLOAT_PRECISION tmpx = -xleng;
		PRISMATIC_FLOAT_PRECISION tmpy = -yleng;
		for (auto &i : out.xvec)
			i = tmpx++;
		for (auto &j : out.yvec)
			j = tmpy++;
	}
	Array1D<PRISMATIC_FLOAT_PRECISION> xr(vector<PRISMATIC_FLOAT_PRECISION>(2 * (size_t)xleng + 1, 0), {{2 * (size_t)xleng + 1}});
	Array1D<PRISMATIC_FLOAT_PRECISION> yr(vector<PRISMATIC_FLOAT_PRECISION>(2 * (size_t)yleng + 1, 0), {{2 * (size_t)yleng + 1}});
	for (auto i = 0; i < xr.size(); ++i)
		xr[i] = (PRISMATIC_FLOAT_PRECISION)out.xvec[i] * pars.pixelSize[1];
	for (auto j = 0; j < yr.size(); ++j)
		yr[j] = (PRISMATIC_FLOAT_PRECISION)out.yvec[j] * pars.pixelSize[0];

	out.unique_species = get_unique_atomic_species(pars);

	out.dim1 = (long) pars.imageSize[1];
	out.dim0 = (long) pars.imageSize[0];

	long numPlanes = ceil(pars.tiledCellDim[0]/pars.meta.sliceThickness);
	//check if intermediate output was specified, if so, create index of output slices
	pars.numPlanes = numPlanes;
	if (pars.meta.numSlices == 0) pars.numSlices = pars.numPlanes;

	// create arrays for the coordinates
	out.x = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	out.y = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	out.ID = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	out.sigma = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
	out.occ = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});

	// populate arrays from the atoms structure
	for (auto i = 0; i < pars.atoms.size(); ++i)
	{
		out.x[i] = pars.atoms[i].x * pars.tiledCellDim[2];
		out.y[i] = pars.atoms[i].y * pars.tiledCellDim[1];
		out.ID[i] = pars.atoms[i].species;
		out.sigma[i] = pars.atoms[i].sigma;
		out.occ[i] = pars.atoms[i].occ;
	}

	// initialize the potential array
	pars.pot = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>({{pars.numPlanes, pars.imageSize[0], pars.imageSize[1]}});

	// create a key-value map to match the atomic Z numbers with their place in the potential lookup table
	for (auto i = 0; i < out.unique_species.size(); ++i)
		out.Z_lookup[out.unique_species[i]] = i;

	if(pars.meta.potential3D)
	{	
		out.t = pw_type::p3D;

		//set up Z coords
		pars.dzPot = pars.meta.sliceThickness/pars.meta.zSampling;
        PRISMATIC_FLOAT_PRECISION zleng = std::ceil(pars.meta.potBound/pars.dzPot);
		out.pw3D.zvec = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{(size_t)zleng*2}});
		for (auto j = -zleng; j < zleng; j++)
		{
			out.pw3D.zvec[j+zleng] = (PRISMATIC_FLOAT_PRECISION) j + 0.5;
		}

		out.pw3D.zr = Array1D<PRISMATIC_FLOAT_PRECISION>(out.pw3D.zvec);
        for (auto j = 0; j < out.pw3D.zr.size(); ++j) out.pw3D.zr[j] = out.pw3D.zvec[j] * pars.dzPot;

		// initialize the lookup table and precompute unique potentials
		out.pw3D.potLookup = zeros_ND<4, std::complex<PRISMATIC_FLOAT_PRECISION>>({{out.unique_species.size(), 2 * (size_t)zleng, 2 * (size_t)yleng + 1, 2 * (size_t)xleng + 1}});
		fetch_potentials3D(out.pw3D.potLookup, out.unique_species, xr, yr, out.pw3D.zr);
		
		// generateProjectedPotentials3D
		// ==============================

		// z coordinates
		out.pw3D.z = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
		for (auto i = 0; i < pars.atoms.size(); ++i)
		{
			out.pw3D.z[i] = pars.atoms[i].z * pars.tiledCellDim[0];
		}

		// correct z orientation
		auto max_z = pars.tiledCellDim[0];

		std::transform(out.pw3D.z.begin(), out.pw3D.z.end(), out.pw3D.z.begin(), [&max_z](PRISMATIC_FLOAT_PRECISION &t_z) {
			return (-t_z + max_z);
		});

		// Array1D<PRISMATIC_FLOAT_PRECISION> zr = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{zvec.get_dimi()}}); // already defined and calculated?
		// for (auto j = 0; j < zr.size(); ++j) zr[j] = (PRISMATIC_FLOAT_PRECISION)zvec[j] * pars.dzPot;

		//create fourier coordinate arrays for use in fourier shifting
		const PRISMATIC_FLOAT_PRECISION pi = std::acos(-1);
		Array1D<PRISMATIC_FLOAT_PRECISION> qy = makeFourierCoords(out.yvec.size(), (PRISMATIC_FLOAT_PRECISION) 1.0);
		Array1D<PRISMATIC_FLOAT_PRECISION> qx = makeFourierCoords(out.xvec.size(), (PRISMATIC_FLOAT_PRECISION) 1.0);
		std::pair<Array2D<PRISMATIC_FLOAT_PRECISION>, Array2D<PRISMATIC_FLOAT_PRECISION>> qmesh = meshgrid(qy,qx);
		Array2D<PRISMATIC_FLOAT_PRECISION> qya = qmesh.first;
		Array2D<PRISMATIC_FLOAT_PRECISION> qxa = qmesh.second;


		out.pw3D.qyShift = zeros_ND<2,std::complex<PRISMATIC_FLOAT_PRECISION>>({{qya.get_dimj(), qya.get_dimi()}});
		out.pw3D.qxShift = zeros_ND<2,std::complex<PRISMATIC_FLOAT_PRECISION>>({{qya.get_dimj(), qya.get_dimi()}});
		std::complex<PRISMATIC_FLOAT_PRECISION> I(0.0, 1.0);
		PRISMATIC_FLOAT_PRECISION two = 2.0;
		for(auto jj = 0; jj < qya.get_dimj(); jj++)
		{
			for(auto ii = 0; ii < qya.get_dimi(); ii++)
			{
				out.pw3D.qyShift.at(jj,ii) = -two*I*pi*qya.at(jj,ii);
				out.pw3D.qxShift.at(jj,ii) = -two*I*pi*qxa.at(jj,ii);
			}
		}

		//band limit arrays for soft aperture with shift in realpace and fourier space; hard-coded for now
		PRISMATIC_FLOAT_PRECISION rband_max = 0.95;
		PRISMATIC_FLOAT_PRECISION rband_min = 0.75;
		Array2D<PRISMATIC_FLOAT_PRECISION> q1(qya);
		std::transform(qxa.begin(), qxa.end(),
			qya.begin(), q1.begin(), [](const PRISMATIC_FLOAT_PRECISION& a, const PRISMATIC_FLOAT_PRECISION& b){
			return sqrt(a*a + b*b);
		});

		out.pw3D.qband = Array2D<PRISMATIC_FLOAT_PRECISION>(q1);

		for(auto jj = 0; jj < out.pw3D.qband.get_dimj(); jj++)
		{
			for(auto ii = 0; ii < out.pw3D.qband.get_dimi(); ii++)
			{
				out.pw3D.qband.at(jj,ii) = (rband_max-2*q1.at(jj,ii))/(rband_max-rband_min);
				out.pw3D.qband.at(jj,ii) = std::max(out.pw3D.qband.at(jj,ii), (PRISMATIC_FLOAT_PRECISION) 0.0);
				out.pw3D.qband.at(jj,ii) = std::min(out.pw3D.qband.at(jj,ii), (PRISMATIC_FLOAT_PRECISION) 1.0);
				out.pw3D.qband.at(jj,ii) = pow(std::sin(out.pw3D.qband.at(jj,ii)*pi/2.0), 2.0);
			}
		}

		std::pair<Array2D<long>, Array2D<long>> rmesh = meshgrid(out.yvec, out.xvec);
		out.pw3D.rband = Array2D<PRISMATIC_FLOAT_PRECISION>(out.pw3D.qband); //construct with qband to avoid type mismatch, knowing sizes are the same
		PRISMATIC_FLOAT_PRECISION xl = (PRISMATIC_FLOAT_PRECISION) out.xvec[out.xvec.size()-1];
		PRISMATIC_FLOAT_PRECISION yl = (PRISMATIC_FLOAT_PRECISION) out.yvec[out.yvec.size()-1];

		for(auto jj = 0; jj < out.pw3D.rband.get_dimj(); jj++)
		{
			for(auto ii = 0; ii < out.pw3D.rband.get_dimi(); ii++)
			{
				out.pw3D.rband.at(jj,ii) = pow((rmesh.first.at(jj,ii) / (yl+0.5)), 2.0) + pow((rmesh.second.at(jj,ii) / (xl+0.5)), 2.0);
				out.pw3D.rband.at(jj,ii) = (out.pw3D.rband.at(jj,ii) <= 1) ? 1.0 : 0.0;  	
			}
		}

	}else{
		out.t = pw_type::p2D;
		// initialize the lookup table
		out.pw2D.potentialLookup = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>({{out.unique_species.size(), 2 * (size_t)yleng + 1, 2 * (size_t)xleng + 1}});

		// // precompute the unique potentials
		fetch_potentials(out.pw2D.potentialLookup, out.unique_species, xr, yr);

		// generateProjectedPotentials(pars, potentialLookup, unique_species, xvec, yvec);
		// ===========================

		// splits the atomic coordinates into slices and computes the projected potential for each.

		// z coordinates
		out.pw2D.z = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{pars.atoms.size()}});
		for (auto i = 0; i < pars.atoms.size(); ++i)
		{
			out.pw2D.z[i] = pars.atoms[i].z * pars.tiledCellDim[0];
		}

		// compute the z-slice index for each atom
		out.pw2D.zPlane = Array1D<PRISMATIC_FLOAT_PRECISION>(out.pw2D.z);
		std::transform(out.pw2D.zPlane.begin(), out.pw2D.zPlane.end(), out.pw2D.zPlane.begin(), [&pars](PRISMATIC_FLOAT_PRECISION &t_z) {
			return round((-t_z + pars.tiledCellDim[0]) / pars.meta.sliceThickness + 0.5) - 1; // If the +0.5 was to make the first slice z=1 not 0, can drop the +0.5 and -1
		});
		// auto max_z = std::max_element(zPlane.begin(), zPlane.end());

	}

	if (pars.meta.importExtraPotential)
	{
		std::cout << "Importing extra potential." << std::endl;
		
		Array1D<PRISMATIC_FLOAT_PRECISION> z_imported;

		//scope out imported tmp_pot as soon as possible
		{
			Array3D<PRISMATIC_FLOAT_PRECISION> tmp_pot;
			if (pars.meta.importPath.size() > 0)
			{
				readRealDataSet_inOrder(tmp_pot, pars.meta.importFile, pars.meta.importPath + "/data");
				readRealDataSet_inOrder(z_imported, pars.meta.importFile, pars.meta.importPath + "/dim3");
			}
			else //read default path
			{
				std::string groupPath = "extra_potential_slices";
				readRealDataSet_inOrder(tmp_pot, pars.meta.importFile, groupPath + "/data");
				readRealDataSet_inOrder(z_imported, pars.meta.importFile, groupPath + "/dim3");
			}

			// assume that the extra potential slices do not need to be tiled in x/y

			// Convention for ND arrays is                  dimk: z, dimj: y, dimi: x.  However,
			// Convention for HDF5 datasets is opposite:       x,       y,       z
			// imageSize[0] y, imageSize[1] x
			if (tmp_pot.get_dimk() != pars.imageSize[1] || tmp_pot.get_dimj() != pars.imageSize[0]) {
				throw std::runtime_error("Number of pixels in x/y direction is not consistent between the calculated and extra imported potential.\n");
			}
			if (tmp_pot.get_dimi() != z_imported.get_dimi()) {
				throw std::runtime_error("Wrong number of z coordinates provided.\n");
			}

			PRISMATIC_FLOAT_PRECISION pot_factor = 0.0;
			if (pars.meta.extraPotentialType == ExtraPotentialType::Angle) {
				pot_factor = pars.meta.extraPotentialFactor / pars.sigma;
			}
			else if (pars.meta.extraPotentialType == ExtraPotentialType::ProjectedPotential) {
				pot_factor = pars.meta.extraPotentialFactor;
			}
			else {
				throw std::runtime_error("Invalid ExtraPotentialType");
			}

			//initialize array and get data in right order
			out.extra_potential = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>({ {tmp_pot.get_dimi(), tmp_pot.get_dimj(), tmp_pot.get_dimk()} });  // the dimensions (k, j, i) are reversed here
			for (auto i = 0; i < tmp_pot.get_dimi(); i++)
			{
				for (auto j = 0; j < tmp_pot.get_dimj(); j++)
				{
					for (auto k = 0; k < tmp_pot.get_dimk(); k++)
					{
						out.extra_potential.at(i, j, k) = tmp_pot.at(k, j, i) * pot_factor;
					}
				}
			}
		}

		//std::cout << pars.meta.cellDim[0] << " " << pars.tiledCellDim[0] << " " << pars.meta.sliceThickness << std::endl;

		std::transform(z_imported.begin(), z_imported.end(), z_imported.begin(), [&pars](PRISMATIC_FLOAT_PRECISION& t_z) {
			return t_z / pars.meta.cellDim[0]; // convert to fractional coordinates
		});

		// Figure out which slices to add the extra potential to
		// (We assume that the imported slices do need to be tiled in z)
		out.ep_tiled_zindex.reserve(z_imported.get_dimi() * pars.meta.tileZ);
		out.ep_imported_index.reserve(z_imported.get_dimi() * pars.meta.tileZ);

		for (auto tz = 0; tz < pars.meta.tileZ; ++tz) {
			for (auto i = 0; i < z_imported.get_dimi(); ++i) {
				out.ep_imported_index.push_back(i);
				out.ep_tiled_zindex.push_back((z_imported.at(i) + tz) / pars.meta.tileZ);
			}
		}

		/*for (auto i : tiled_Z) std::cout << i << ' ';
		std::cout << std::endl;
		for (auto i : import_indices) std::cout << i << ' ';
		std::cout << std::endl;*/
		
		// trying to be consistent with generateProjectedPotentials
		std::transform(out.ep_tiled_zindex.begin(), out.ep_tiled_zindex.end(), out.ep_tiled_zindex.begin(), 
			[&pars](PRISMATIC_FLOAT_PRECISION& t_z) {
				return round((1.0 - t_z) * pars.tiledCellDim[0] / pars.meta.sliceThickness + 0.5) - 1;
			});

		//for (auto i : tiled_Z) std::cout << i << ' ';
		//std::cout << std::endl;
	}

	return out;
}

void PRISM01_calcPotential(Parameters<PRISMATIC_FLOAT_PRECISION> &pars)
{
	//builds projected, sliced potential
	
	// setup some coordinates
	cout << "Entering PRISM01_calcPotential" << endl;
	PRISMATIC_FLOAT_PRECISION yleng = std::ceil(pars.meta.potBound / pars.pixelSize[0]);
	PRISMATIC_FLOAT_PRECISION xleng = std::ceil(pars.meta.potBound / pars.pixelSize[1]);
	ArrayND<1, vector<long>> xvec(vector<long>(2 * (size_t)xleng + 1, 0), {{2 * (size_t)xleng + 1}});
	ArrayND<1, vector<long>> yvec(vector<long>(2 * (size_t)yleng + 1, 0), {{2 * (size_t)yleng + 1}});
	{
		PRISMATIC_FLOAT_PRECISION tmpx = -xleng;
		PRISMATIC_FLOAT_PRECISION tmpy = -yleng;
		for (auto &i : xvec)
			i = tmpx++;
		for (auto &j : yvec)
			j = tmpy++;
	}
	Array1D<PRISMATIC_FLOAT_PRECISION> xr(vector<PRISMATIC_FLOAT_PRECISION>(2 * (size_t)xleng + 1, 0), {{2 * (size_t)xleng + 1}});
	Array1D<PRISMATIC_FLOAT_PRECISION> yr(vector<PRISMATIC_FLOAT_PRECISION>(2 * (size_t)yleng + 1, 0), {{2 * (size_t)yleng + 1}});
	for (auto i = 0; i < xr.size(); ++i)
		xr[i] = (PRISMATIC_FLOAT_PRECISION)xvec[i] * pars.pixelSize[1];
	for (auto j = 0; j < yr.size(); ++j)
		yr[j] = (PRISMATIC_FLOAT_PRECISION)yvec[j] * pars.pixelSize[0];

	vector<size_t> unique_species = get_unique_atomic_species(pars);

	if(pars.meta.potential3D)
	{	//set up Z coords

		pars.dzPot = pars.meta.sliceThickness/pars.meta.zSampling;
        PRISMATIC_FLOAT_PRECISION zleng = std::ceil(pars.meta.potBound/pars.dzPot);
		Array1D<PRISMATIC_FLOAT_PRECISION> zvec = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{(size_t)zleng*2}});
		for (auto j = -zleng; j < zleng; j++)
		{
			zvec[j+zleng] = (PRISMATIC_FLOAT_PRECISION) j + 0.5;
		}

		Array1D<PRISMATIC_FLOAT_PRECISION> zr(zvec);
        for (auto j = 0; j < zr.size(); ++j) zr[j] = zvec[j] * pars.dzPot;

		// initialize the lookup table and precompute unique potentials
		Array4D<std::complex<PRISMATIC_FLOAT_PRECISION>> potentialLookup = zeros_ND<4, std::complex<PRISMATIC_FLOAT_PRECISION>>({{unique_species.size(), 2 * (size_t)zleng, 2 * (size_t)yleng + 1, 2 * (size_t)xleng + 1}});
		fetch_potentials3D(potentialLookup, unique_species, xr, yr, zr);
		//generate potential
		generateProjectedPotentials3D(pars, potentialLookup, unique_species, xvec, yvec, zvec);

	}else{
		// initialize the lookup table
		Array3D<PRISMATIC_FLOAT_PRECISION> potentialLookup = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>({{unique_species.size(), 2 * (size_t)yleng + 1, 2 * (size_t)xleng + 1}});

		// precompute the unique potentials
		fetch_potentials(potentialLookup, unique_species, xr, yr);

		// populate the slices with the projected potentials
		generateProjectedPotentials(pars, potentialLookup, unique_species, xvec, yvec);
	}

	if (pars.meta.importExtraPotential)
	{
		std::cout << "Importing extra potential." << std::endl;
		addExtraPotential(pars);
	}

	if (pars.meta.savePotentialSlices) 
	{
		std::cout << "Writing potential slices to output file." << std::endl;
		savePotentialSlices(pars);
	}
}

void PRISM01_importPotential(Parameters<PRISMATIC_FLOAT_PRECISION> &pars)
{
	std::cout << "Setting up PRISM01 auxilary variables according to " << pars.meta.importFile << " metadata." << std::endl;
	//scope out imported tmp_pot as soon as possible
	{
		Array3D<PRISMATIC_FLOAT_PRECISION> tmp_pot;
		if(pars.meta.importPath.size() > 0)
		{
			readRealDataSet_inOrder(tmp_pot, pars.meta.importFile, pars.meta.importPath);
		}
		else //read default path
		{
			std::string groupPath = "4DSTEM_simulation/data/realslices/ppotential_fp" + getDigitString(pars.fpFlag) + "/data";
			readRealDataSet_inOrder(tmp_pot, pars.meta.importFile, groupPath);
		}

		//initailize array and get data in right order
		pars.pot = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>({{tmp_pot.get_dimi(), tmp_pot.get_dimj(), tmp_pot.get_dimk()}});  // the dimensions (k, j, i) are reversed here
		for(auto i = 0; i < tmp_pot.get_dimi(); i++)
		{
			for(auto j = 0; j < tmp_pot.get_dimj(); j++)
			{
				for(auto k = 0; k < tmp_pot.get_dimk(); k++)
				{
					pars.pot.at(i,j,k) = tmp_pot.at(k,j,i);
				}
			}
		}
	}

	pars.numPlanes = pars.pot.get_dimk();
	if (pars.meta.numSlices == 0)
	{
		pars.numSlices = pars.numPlanes;
	}

	//resample coordinates if PRISM algorithm and size of PS array in not a multiple of 4*fx or 4*fy
	if(pars.meta.algorithm == Algorithm::PRISM)
	{
		if ( (pars.pot.get_dimi() % 4*pars.meta.interpolationFactorX) || (pars.pot.get_dimj() % pars.meta.interpolationFactorY))
		{
			std::cout << "Resampling imported potential to align grid size with requested interpolation factors fx = " 
					  << pars.meta.interpolationFactorX << " and fy = " << pars.meta.interpolationFactorY << std::endl;
			fourierResampling(pars);
		}
	}

	//TODO: metadata from non-prismatic sources?
    std::string groupPath = "4DSTEM_simulation/metadata/metadata_0/original/simulation_parameters";
	PRISMATIC_FLOAT_PRECISION meta_cellDims[3];
	readAttribute(pars.meta.importFile, groupPath, "c", meta_cellDims);

	PRISMATIC_FLOAT_PRECISION meta_tile[3];
	readAttribute(pars.meta.importFile, groupPath, "t", meta_tile);

	pars.tiledCellDim[0] = meta_cellDims[2]*meta_tile[2];
	pars.tiledCellDim[1] = meta_cellDims[1]*meta_tile[1];
	pars.tiledCellDim[2] = meta_cellDims[0]*meta_tile[0];

	std::vector<PRISMATIC_FLOAT_PRECISION> pixelSize{(PRISMATIC_FLOAT_PRECISION) pars.tiledCellDim[1], (PRISMATIC_FLOAT_PRECISION) pars.tiledCellDim[2]};
	pars.imageSize[0] = pars.pot.get_dimj();
	pars.imageSize[1] = pars.pot.get_dimi();
	pixelSize[0] /= pars.imageSize[0];
	pixelSize[1] /= pars.imageSize[1];
	pars.pixelSize = pixelSize;

	if (pars.meta.savePotentialSlices) 
	{
		std::cout << "Writing potential slices to output file." << std::endl;
		savePotentialSlices(pars);
	}

};

void addExtraPotential(Parameters<PRISMATIC_FLOAT_PRECISION>& pars) {
	Array3D<PRISMATIC_FLOAT_PRECISION> imported_slices;
	Array1D<PRISMATIC_FLOAT_PRECISION> z_imported;

	//scope out imported tmp_pot as soon as possible
	{
		Array3D<PRISMATIC_FLOAT_PRECISION> tmp_pot;
		if (pars.meta.importPath.size() > 0)
		{
			readRealDataSet_inOrder(tmp_pot, pars.meta.importFile, pars.meta.importPath + "/data");
			readRealDataSet_inOrder(z_imported, pars.meta.importFile, pars.meta.importPath + "/dim3");
		}
		else //read default path
		{
			std::string groupPath = "extra_potential_slices";
			readRealDataSet_inOrder(tmp_pot, pars.meta.importFile, groupPath + "/data");
			readRealDataSet_inOrder(z_imported, pars.meta.importFile, groupPath + "/dim3");
		}

		// assume that the extra potential slices do not need to be tiled in x/y

		// Convention for ND arrays is                  dimk: z, dimj: y, dimi: x.  However,
		// Convention for HDF5 datasets is opposite:       x,       y,       z
		// imageSize[0] y, imageSize[1] x
		if (tmp_pot.get_dimk() != pars.imageSize[1] || tmp_pot.get_dimj() != pars.imageSize[0]) {
			throw std::runtime_error("Number of pixels in x/y direction is not consistent between the calculated and extra imported potential.\n");
		}
		if (tmp_pot.get_dimi() != z_imported.get_dimi()) {
			throw std::runtime_error("Wrong number of z coordinates provided.\n");
		}

		PRISMATIC_FLOAT_PRECISION pot_factor = 0.0;
		if (pars.meta.extraPotentialType == ExtraPotentialType::Angle) {
			pot_factor = pars.meta.extraPotentialFactor / pars.sigma;
		}
		else if (pars.meta.extraPotentialType == ExtraPotentialType::ProjectedPotential) {
			pot_factor = pars.meta.extraPotentialFactor;
		}
		else {
			throw std::runtime_error("Invalid ExtraPotentialType");
		}

		//initialize array and get data in right order
		imported_slices = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>({ {tmp_pot.get_dimi(), tmp_pot.get_dimj(), tmp_pot.get_dimk()} });  // the dimensions (k, j, i) are reversed here
		for (auto i = 0; i < tmp_pot.get_dimi(); i++)
		{
			for (auto j = 0; j < tmp_pot.get_dimj(); j++)
			{
				for (auto k = 0; k < tmp_pot.get_dimk(); k++)
				{
					imported_slices.at(i, j, k) = tmp_pot.at(k, j, i) * pot_factor;
				}
			}
		}
	}

	//std::cout << pars.meta.cellDim[0] << " " << pars.tiledCellDim[0] << " " << pars.meta.sliceThickness << std::endl;

	std::transform(z_imported.begin(), z_imported.end(), z_imported.begin(), [&pars](PRISMATIC_FLOAT_PRECISION& t_z) {
		return t_z / pars.meta.cellDim[0]; // convert to fractional coordinates
	});

	// Figure out which slices to add the extra potential to
	// (We assume that the imported slices do need to be tiled in z)
	std::vector<PRISMATIC_FLOAT_PRECISION> tiled_Z;
	tiled_Z.reserve(z_imported.get_dimi() * pars.meta.tileZ);
	std::vector<size_t> import_indices;
	import_indices.reserve(z_imported.get_dimi() * pars.meta.tileZ);

	for (auto tz = 0; tz < pars.meta.tileZ; ++tz) {
		for (auto i = 0; i < z_imported.get_dimi(); ++i) {
			import_indices.push_back(i);
			tiled_Z.push_back((z_imported.at(i) + tz) / pars.meta.tileZ);
		}
	}

	/*for (auto i : tiled_Z) std::cout << i << ' ';
	std::cout << std::endl;
	for (auto i : import_indices) std::cout << i << ' ';
	std::cout << std::endl;*/
	
	// trying to be consistent with generateProjectedPotentials
	std::transform(tiled_Z.begin(), tiled_Z.end(), tiled_Z.begin(), [&pars](PRISMATIC_FLOAT_PRECISION& t_z) {
		return round((1.0 - t_z) * pars.tiledCellDim[0] / pars.meta.sliceThickness + 0.5) - 1;
		});

	//for (auto i : tiled_Z) std::cout << i << ' ';
	//std::cout << std::endl;

	// add the extra imported potential slices
	std::cout << "Adding imported extra potential (from slice index -> to slice index)" << std::endl;
	for (size_t to_slice = 0; to_slice < pars.numPlanes; ++to_slice) {
		for (size_t i = 0; i < tiled_Z.size(); ++i) {
			if (tiled_Z[i] == to_slice) {
				size_t from_slice = import_indices[i];
				std::cout << from_slice << " -> " << to_slice << std::endl;
				for (auto jj = 0; jj < pars.pot.get_dimj(); ++jj) {
					for (auto ii = 0; ii < pars.pot.get_dimi(); ++ii) {
						pars.pot.at(to_slice, jj, ii) += imported_slices.at(from_slice, jj, ii);
					}
				}
			}
		}
	}
}

void addExtraPotential_miser(Parameters<PRISMATIC_FLOAT_PRECISION>& pars,
	PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw, Array3D<PRISMATIC_FLOAT_PRECISION> &pot)
{
	// std::cout << "Adding imported extra potential (from slice index -> to slice index)" << std::endl;
	for (size_t to_slice = 0; to_slice < pars.numPlanes; ++to_slice) {
		for (size_t i = 0; i < pw.ep_tiled_zindex.size(); ++i) {
			if (pw.ep_tiled_zindex[i] == to_slice) {
				size_t from_slice = pw.ep_imported_index[i];
				// std::cout << from_slice << " -> " << to_slice << std::endl;
				for (auto jj = 0; jj < pot.get_dimj(); ++jj) {
					for (auto ii = 0; ii < pot.get_dimi(); ++ii) {
						pot.at(to_slice, jj, ii) += pw.extra_potential.at(from_slice, jj, ii);
					}
				}
			}
		}
	}
}

void fourierResampling(Parameters<PRISMATIC_FLOAT_PRECISION> &pars)
{
	int Ni = 0;
	int Nj = 0;

	//get highest multiple of 4*fx and 4*fy to ensure resampling to a smaller grid only
	while(Ni < pars.pot.get_dimi()) Ni += pars.meta.interpolationFactorX*4;
	while(Nj < pars.pot.get_dimj()) Nj += pars.meta.interpolationFactorY*4;
	Ni -= pars.meta.interpolationFactorX*4;
 	Nj -= pars.meta.interpolationFactorY*4;
 	
	Array3D<PRISMATIC_FLOAT_PRECISION> newPot = zeros_ND<3,PRISMATIC_FLOAT_PRECISION>({{pars.pot.get_dimk(), (size_t) Nj, (size_t) Ni}});

	//create storage variables to hold data from FFTs
	Array2D<complex<PRISMATIC_FLOAT_PRECISION>> fstore = zeros_ND<2,complex<PRISMATIC_FLOAT_PRECISION>>({{pars.pot.get_dimj(), pars.pot.get_dimi()}});
	Array2D<complex<PRISMATIC_FLOAT_PRECISION>> bstore = zeros_ND<2,complex<PRISMATIC_FLOAT_PRECISION>>({{(size_t) Nj, (size_t) Ni}});
	Array2D<complex<PRISMATIC_FLOAT_PRECISION>> fpot = zeros_ND<2,complex<PRISMATIC_FLOAT_PRECISION>>({{pars.pot.get_dimj(),pars.pot.get_dimi()}});
	Array2D<complex<PRISMATIC_FLOAT_PRECISION>> bpot = zeros_ND<2,complex<PRISMATIC_FLOAT_PRECISION>>({{(size_t)Nj,(size_t) Ni}});
	
	//create FFT plans 
	PRISMATIC_FFTW_INIT_THREADS();
	PRISMATIC_FFTW_PLAN_WITH_NTHREADS(pars.meta.numThreads);
	
	unique_lock<mutex> gatekeeper(fftw_plan_lock);
	PRISMATIC_FFTW_PLAN plan_forward = PRISMATIC_FFTW_PLAN_DFT_2D(fstore.get_dimj(), fstore.get_dimi(),
															reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&fpot[0]),
															reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&fstore[0]),
															FFTW_FORWARD,
															FFTW_ESTIMATE);

	PRISMATIC_FFTW_PLAN plan_inverse = PRISMATIC_FFTW_PLAN_DFT_2D(bstore.get_dimj(), bstore.get_dimi(),
															reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&bstore[0]),
															reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&bpot[0]),
															FFTW_BACKWARD,
															FFTW_ESTIMATE);
	gatekeeper.unlock();

	//calculate indices for downsampling in fourier space
	int nyqi = std::floor(Ni/2) + 1;
	int nyqj = std::floor(Nj/2) + 1;

	for(auto k = 0; k < newPot.get_dimk(); k++)
	{
		//copy current slice to forward transform
		for(auto i = 0; i < fpot.size(); i++) fpot[i] = pars.pot[k*pars.pot.get_dimj()*pars.pot.get_dimi()+i];
		
		//forward transform 
		PRISMATIC_FFTW_EXECUTE(plan_forward);

		//copy relevant quadrants to backward store
		//manual looping through quadrants
		for(auto j = 0; j < nyqj; j++)
		{
			for(auto i = 0; i < nyqi; i++)
			{
				bstore.at(j, i) = fstore.at(j, i);
			}
		}

		for(auto j = nyqj-Nj; j < 0; j++)
		{
			for(auto i = 0; i < nyqi; i++)
			{
				bstore.at(Nj + j, i) = fstore.at(fstore.get_dimj() + j, i);
			}
		}

		for(auto j = 0; j < nyqj; j++)
		{
			for(auto i = nyqi-Ni; i < 0; i++)
			{
				bstore.at(j, Ni + i) = fstore.at(j, fstore.get_dimi() + i);
			}
		}

		for(auto j = nyqj-Nj; j < 0; j++)
		{
			for(auto i = nyqi-Ni; i < 0; i++)
			{
				bstore.at(Nj + j, Ni + i) = fstore.at(fstore.get_dimj() + j, fstore.get_dimi() + i);
			}
		}

		//inverse transform
		PRISMATIC_FFTW_EXECUTE(plan_inverse);

		//store slice in potential
		for(auto i = 0; i < bpot.size(); i++) newPot[k*newPot.get_dimj()*newPot.get_dimi()+i] = bpot[i].real();
	}

	//store final resort after normalizing FFT, rescaling from transform, and removing negative values
	PRISMATIC_FLOAT_PRECISION orig_x = pars.pot.get_dimi();
	PRISMATIC_FLOAT_PRECISION orig_y = pars.pot.get_dimj();
	PRISMATIC_FLOAT_PRECISION new_x = Ni;
	PRISMATIC_FLOAT_PRECISION new_y = Nj;
	newPot /= Ni*Nj;
	newPot *= (new_x/orig_x)*(new_y/orig_y);

	pars.pot = newPot;
};

} // namespace Prismatic
