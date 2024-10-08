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

#include "utility.h"
#include <complex>
#include "defines.h"
#include "configure.h"
#include <string>
#include <stdio.h>
#ifdef _WIN32
#include <io.h>
#define access _access_s
#else
#include <unistd.h>
#endif
#include <thread>
#include <map>
#include <boost/math/special_functions/erf.hpp>

namespace bm = boost::math;

namespace Prismatic
{

PRISMATIC_FLOAT_PRECISION probit(PRISMATIC_FLOAT_PRECISION p)
{
    static constexpr PRISMATIC_FLOAT_PRECISION root_2 = 1.414213562373095;

    return root_2 * bm::erf_inv(2*p-1);
}

std::pair<Prismatic::Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>>, Prismatic::Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>>>
upsamplePRISMProbe(Prismatic::Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> probe,
				   const long dimj, const long dimi, long ys, long xs)
{
	Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> realspace_probe;
	Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> buffer_probe;
	Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> kspace_probe;

	buffer_probe = zeros_ND<2, std::complex<PRISMATIC_FLOAT_PRECISION>>({{(size_t)dimj, (size_t)dimi}});
	//		std::cout << "dimj = " << dimj << std::endl;
	long ncy = probe.get_dimj() / 2;
	long ncx = probe.get_dimi() / 2;
	for (auto j = 0; j < probe.get_dimj(); ++j)
	{
		for (auto i = 0; i < probe.get_dimi(); ++i)
		{
			buffer_probe.at((dimj + ((j - ncy + ys) % dimj)) % dimj,
							(dimi + ((i - ncx + xs) % dimi)) % dimi) = probe.at(j, i);
		}
	}
	std::unique_lock<std::mutex> gatekeeper(fftw_plan_lock);
	PRISMATIC_FFTW_PLAN plan = PRISMATIC_FFTW_PLAN_DFT_2D(buffer_probe.get_dimj(), buffer_probe.get_dimi(),
														  reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&buffer_probe[0]),
														  reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&buffer_probe[0]),
														  FFTW_FORWARD, FFTW_ESTIMATE);
	gatekeeper.unlock();
	realspace_probe = buffer_probe;
	PRISMATIC_FFTW_EXECUTE(plan);
	kspace_probe = buffer_probe;
	gatekeeper.lock();
	PRISMATIC_FFTW_DESTROY_PLAN(plan);
	gatekeeper.unlock();
	return std::make_pair(realspace_probe, kspace_probe);
}

PRISMATIC_FLOAT_PRECISION computePearsonCorrelation(Prismatic::Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> left,
													Prismatic::Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> right)
{
	PRISMATIC_FLOAT_PRECISION m1, m2, sigma1, sigma2, R;
	m1 = m2 = sigma1 = sigma2 = R = 0;

	for (auto &i : left)
		m1 += std::abs(i);
	for (auto &i : right)
		m2 += std::abs(i);

	m1 /= (left.size());
	m2 /= (right.size());

	for (auto &i : left)
		sigma1 += pow(std::abs(i) - m1, 2);
	for (auto &i : right)
		sigma2 += pow(std::abs(i) - m2, 2);

	sigma1 /= (left.size());
	sigma2 /= (right.size());

	sigma1 = sqrt(sigma1);
	sigma2 = sqrt(sigma2);
	for (auto i = 0; i < std::min(left.size(), right.size()); ++i)
	{
		R = R + (std::abs(left[i]) - m1) * (std::abs(right[i]) - m2);
	}
	R /= sqrt(left.size() * right.size());
	return R / (sigma1 * sigma2);
}
PRISMATIC_FLOAT_PRECISION computeRfactor(Prismatic::Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> left,
										 Prismatic::Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> right)
{
	PRISMATIC_FLOAT_PRECISION accum, diffs;
	accum = diffs = 0;
	for (auto i = 0; i < std::min(left.size(), right.size()); ++i)
	{
		diffs += std::abs(left[i] - right[i]);
		accum += std::abs(left[i]);
	}
	return diffs / accum;
}

int nyquistProbes(Prismatic::Parameters<PRISMATIC_FLOAT_PRECISION> pars, size_t dim)
{
	int nProbes = ceil(4 * (pars.meta.probeSemiangle / pars.lambda) * pars.tiledCellDim[dim]);
	return nProbes;
}

std::string remove_extension(const std::string &filename)
{
	size_t lastdot = filename.find_last_of(".");
	if (lastdot == std::string::npos)
		return filename;
	return filename.substr(0, lastdot);
}

int testFilenameOutput(const std::string &filename)
{
	bool exists = !testExist(filename);
	bool write_ok = !testWrite(filename);
	//Check if file already exists and if we can write to it
	if (exists && write_ok)
	{
		std::cout << "Warning " << filename << " already exists and will be overwritten" << std::endl;
		return 2;
	}
	else if (exists && !write_ok)
	{
		std::cout << filename << " isn't an accessible write destination" << std::endl;
		return 0;
	}
	else
	{
		//If the file does not exist, check to see if we can open a file of that name
		std::ofstream f(filename, std::ios::binary | std::ios::out);
		if (f)
		{
			//If we can open such a file, close the file and delete it.
			f.close();
			std::remove(filename.c_str());
			return 1;
		}
		else
		{
			std::cout << filename << " isn't an accessible write destination" << std::endl;
			return 0;
		}
	}
}

int testWrite(const std::string &filename)
{
	int answer = access(filename.c_str(), 02); //W_OK = 02
	return answer;
}

int testExist(const std::string &filename)
{
	int answer = access(filename.c_str(), 00); //F_OK == 00
	return answer;
}

void updateSeriesParams(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, size_t iter)
{

    //right now this assumes only CC series
	std::map<std::string, PRISMATIC_FLOAT_PRECISION*> valMap{{"probeDefocus", &pars.meta.probeDefocus}};
    pars.currentTag = pars.meta.seriesTags[iter];
	for(auto i = 0; i < pars.meta.seriesKeys.size(); i++)
	{
		PRISMATIC_FLOAT_PRECISION* val_ptr = valMap[pars.meta.seriesKeys[i]];
		*val_ptr = pars.meta.seriesVals[i][iter];
	}
};

} // namespace Prismatic
