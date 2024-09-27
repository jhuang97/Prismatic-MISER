#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>
#include "MS_MISER_calcOutput.h"
#include "PRISM01_calcPotential.h"
#include "fileIO.h"
#include "params.h"
#include "meta.h"
#include "ArrayND.h"
#include "utility.h"
#include "WorkDispatcher.h"
#include <boost/optional.hpp>

namespace Prismatic{
	using namespace std;
    static const PRISMATIC_FLOAT_PRECISION pi = acos(-1);
	static const std::complex<PRISMATIC_FLOAT_PRECISION> i(0, 1);
	extern mutex fftw_plan_lock; // for synchronizing access to shared FFTW resources. This mutex is apparently first defined in Multislice_calcOutput.cpp?
	mutex summ_lock;

    void setupCoordinates_MS_MISER(Parameters<PRISMATIC_FLOAT_PRECISION>& pars){

		// setup coordinates and build propagators
		std::vector<PRISMATIC_FLOAT_PRECISION> xp_d;
		std::vector<PRISMATIC_FLOAT_PRECISION> yp_d;
		if(pars.meta.arbitraryProbes)
		{
			xp_d = pars.meta.probes_x;
			yp_d = pars.meta.probes_y;
			pars.numXprobes = xp_d.size();
			pars.numYprobes = 1;
			pars.numProbes = xp_d.size();
		}
		else
		{
			Array1D<PRISMATIC_FLOAT_PRECISION> xR = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{2}});
			xR[0] = pars.scanWindowXMin * pars.tiledCellDim[2];
			xR[1] = pars.scanWindowXMax * pars.tiledCellDim[2];
			Array1D<PRISMATIC_FLOAT_PRECISION> yR = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{2}});
			yR[0] = pars.scanWindowYMin * pars.tiledCellDim[1];
			yR[1] = pars.scanWindowYMax * pars.tiledCellDim[1];

			PRISMATIC_FLOAT_PRECISION probeStepX;
			PRISMATIC_FLOAT_PRECISION probeStepY;
			if(pars.meta.nyquistSampling){
				int numX = nyquistProbes(pars,2); //x is dim 2
				int numY = nyquistProbes(pars,1); //y is dim 1
				probeStepX = pars.tiledCellDim[2]/numX;
				probeStepY = pars.tiledCellDim[1]/numY;
			}else{
				probeStepX = pars.meta.probeStepX;
				probeStepY = pars.meta.probeStepY;
			}
			
			xp_d = vecFromRange(xR[0], probeStepX, xR[1]);
			yp_d = vecFromRange(yR[0], probeStepY, yR[1]);
		
			pars.numXprobes = xp_d.size();
			pars.numYprobes = yp_d.size();
			pars.numProbes = xp_d.size()*yp_d.size();
		}

		Array1D<PRISMATIC_FLOAT_PRECISION> xp(xp_d, {{xp_d.size()}});
		Array1D<PRISMATIC_FLOAT_PRECISION> yp(yp_d, {{yp_d.size()}});

		pars.xp = xp;
		pars.yp = yp;

        cout << "pars.imageSize: " << pars.imageSize[0] << ", " << pars.imageSize[1] << endl;
        cout << "pars.pixelSize: " << pars.pixelSize[0] << ", " << pars.pixelSize[1] << endl;

        // pars.pot not defined yet, but imageSize should already be determined?

		// cout << "pars.pot size: " << pars.pot.get_dimj() << ", " << pars.pot.get_dimi() << endl;
		// pars.imageSize[0] = pars.pot.get_dimj();
		// pars.imageSize[1] = pars.pot.get_dimi();

		Array1D<PRISMATIC_FLOAT_PRECISION> qx = makeFourierCoords(pars.imageSize[1], pars.pixelSize[1]);
		Array1D<PRISMATIC_FLOAT_PRECISION> qy = makeFourierCoords(pars.imageSize[0], pars.pixelSize[0]);
		pars.qx = qx;
		pars.qy = qy;

		pair< Array2D<PRISMATIC_FLOAT_PRECISION>, Array2D<PRISMATIC_FLOAT_PRECISION> > mesh = meshgrid(qy,qx);
		pars.qya = mesh.first;
		pars.qxa = mesh.second;
		Array2D<PRISMATIC_FLOAT_PRECISION> q2(pars.qya);
		transform(pars.qxa.begin(), pars.qxa.end(),
		          pars.qya.begin(), q2.begin(), [](const PRISMATIC_FLOAT_PRECISION& a, const PRISMATIC_FLOAT_PRECISION& b){
					return a*a + b*b;
				});
		Array2D<PRISMATIC_FLOAT_PRECISION> q1(q2);
		pars.q2 = q2;
		pars.q1 = q1;
		for (auto& q : pars.q1)q=sqrt(q);

		// get qMax
		long long ncx = (long long) floor((PRISMATIC_FLOAT_PRECISION) pars.imageSize[1] / 2);
		PRISMATIC_FLOAT_PRECISION dpx = 1.0 / ((PRISMATIC_FLOAT_PRECISION)pars.imageSize[1] * pars.meta.realspacePixelSize[1]);
		long long ncy = (long long) floor((PRISMATIC_FLOAT_PRECISION) pars.imageSize[0] / 2);
		PRISMATIC_FLOAT_PRECISION dpy = 1.0 / ((PRISMATIC_FLOAT_PRECISION)pars.imageSize[0] * pars.meta.realspacePixelSize[0]);
		pars.qMax = std::min(dpx*(ncx), dpy*(ncy)) / 2;

		pars.qMask = zeros_ND<2, unsigned int>({{pars.imageSize[0], pars.imageSize[1]}});
		{
			long offset_x = pars.qMask.get_dimi()/4;
			long offset_y = pars.qMask.get_dimj()/4;
			long ndimy = (long)pars.qMask.get_dimj();
			long ndimx = (long)pars.qMask.get_dimi();
			for (long y = 0; y < pars.qMask.get_dimj() / 2; ++y) {
				for (long x = 0; x < pars.qMask.get_dimi() / 2; ++x) {
					pars.qMask.at( ((y-offset_y) % ndimy + ndimy) % ndimy,
					               ((x-offset_x) % ndimx + ndimx) % ndimx) = 1;
				}
			}
		}

		// build propagators
		pars.prop     = zeros_ND<2, std::complex<PRISMATIC_FLOAT_PRECISION> >({{pars.imageSize[0], pars.imageSize[1]}});
		pars.propBack = zeros_ND<2, std::complex<PRISMATIC_FLOAT_PRECISION> >({{pars.imageSize[0], pars.imageSize[1]}});
		for (auto y = 0; y < pars.qMask.get_dimj(); ++y) {
			for (auto x = 0; x < pars.qMask.get_dimi(); ++x) {
				if (pars.qMask.at(y,x)==1)
				{
		//					pars.prop.at(y,x)     = exp(-i * pi * complex<PRISMATIC_FLOAT_PRECISION>(pars.lambda, 0) *
		//					                            complex<PRISMATIC_FLOAT_PRECISION>(pars.meta.sliceThickness, 0) *
		//					                            complex<PRISMATIC_FLOAT_PRECISION>(pars.q2.at(y, x), 0));
		//					pars.propBack.at(y,x) = exp(i * pi * complex<PRISMATIC_FLOAT_PRECISION>(pars.lambda, 0) *
		//					                            complex<PRISMATIC_FLOAT_PRECISION>(pars.tiledCellDim[0], 0) *
		//					                            complex<PRISMATIC_FLOAT_PRECISION>(pars.q2.at(y, x), 0));

					pars.prop.at(y,x)     = exp(-i*pi*complex<PRISMATIC_FLOAT_PRECISION>(pars.lambda, 0) *
												complex<PRISMATIC_FLOAT_PRECISION>(pars.meta.sliceThickness, 0) *
												complex<PRISMATIC_FLOAT_PRECISION>(pars.q2.at(y, x), 0) +
												i * complex<PRISMATIC_FLOAT_PRECISION>(2, 0)*pi *
												complex<PRISMATIC_FLOAT_PRECISION>(pars.meta.sliceThickness, 0) *
												(qx[x] * tan(pars.meta.probeXtilt) + qy[y] * tan(pars.meta.probeYtilt)));

				}
			}
		}

	}

    void setupDetector_MS_MISER(Parameters<PRISMATIC_FLOAT_PRECISION>& pars){
		pars.alphaMax = pars.qMax * pars.lambda;
		vector<PRISMATIC_FLOAT_PRECISION> detectorAngles_d = vecFromRange(pars.meta.detectorAngleStep * 1000 / 2,
																	      pars.meta.detectorAngleStep * 1000,
																	      (pars.alphaMax - pars.meta.detectorAngleStep / 2) * 1000);
		Array1D<PRISMATIC_FLOAT_PRECISION> detectorAngles(detectorAngles_d, {{detectorAngles_d.size()}});
		pars.detectorAngles = detectorAngles;
		pars.Ndet = pars.detectorAngles.size();
		Array2D<PRISMATIC_FLOAT_PRECISION> alpha = pars.q1 * pars.lambda;
		pars.alphaInd = (alpha + pars.meta.detectorAngleStep/2) / pars.meta.detectorAngleStep;
		for (auto& q : pars.alphaInd) q = std::round(q);
		pars.dq = (pars.qxa.at(0, 1) + pars.qya.at(1, 0)) / 2;
	}

    void setupProbes_MS_MISER(Parameters<PRISMATIC_FLOAT_PRECISION>& pars){

		PRISMATIC_FLOAT_PRECISION qProbeMax = pars.meta.probeSemiangle/ pars.lambda; // currently a single semiangle
		pars.psiProbeInit = zeros_ND<2, complex<PRISMATIC_FLOAT_PRECISION> >({{pars.q1.get_dimj(), pars.q1.get_dimi()}});

		// erf probe is deprecated, but keeping the source here in case we ever want to flexibly switch
		// transform(pars.psiProbeInit.begin(), pars.psiProbeInit.end(),
		//           pars.q1.begin(), pars.psiProbeInit.begin(),
		//           [&pars, &qProbeMax](std::complex<PRISMATIC_FLOAT_PRECISION> &a, PRISMATIC_FLOAT_PRECISION &q1_t) {
		// 	          a.real(erf((qProbeMax - q1_t) / (0.5 * pars.dq)) * 0.5 + 0.5);
		// 	          a.imag(0);
		// 	          return a;
		//           });

		PRISMATIC_FLOAT_PRECISION dqx = pars.qxa.at(0,1);
		PRISMATIC_FLOAT_PRECISION dqy = pars.qya.at(1,0);
		for(auto j = 0; j < pars.q1.get_dimj(); j++)
		{
			for(auto i = 0; i < pars.q1.get_dimi(); i++)
			{
				PRISMATIC_FLOAT_PRECISION tmp_val = (qProbeMax*pars.q1.at(j,i) - pars.q2.at(j,i));
				tmp_val /= sqrt(dqx*dqx*pow(pars.qxa.at(j,i),2.0)+dqy*dqy*pow(pars.qya.at(j,i),2.0));					
				tmp_val += 0.5; 
				tmp_val = std::max(tmp_val, (PRISMATIC_FLOAT_PRECISION) 0.0);
				tmp_val = std::min(tmp_val, (PRISMATIC_FLOAT_PRECISION) 1.0);
				pars.psiProbeInit.at(j,i).real(tmp_val);
			}
		}

		pars.psiProbeInit.at(0,0).real(1.0);

		//apply aberrations
		pars.qTheta = pars.q1;
		std::transform(pars.qxa.begin(), pars.qxa.end(),
					   pars.qya.begin(), pars.qTheta.begin(), [](const PRISMATIC_FLOAT_PRECISION&a, const PRISMATIC_FLOAT_PRECISION& b){
						   return atan2(b,a);
					   });
		
		Array2D<std::complex<PRISMATIC_FLOAT_PRECISION>> chi = getChi(pars.q1, pars.qTheta, pars.lambda, pars.meta.aberrations);

		transform(pars.psiProbeInit.begin(), pars.psiProbeInit.end(),
				chi.begin(), pars.psiProbeInit.begin(),
				[](std::complex<PRISMATIC_FLOAT_PRECISION> &a, std::complex<PRISMATIC_FLOAT_PRECISION> &b) {
					a = a * exp(-i * b);
					return a;
				});


		PRISMATIC_FLOAT_PRECISION norm_constant = sqrt(accumulate(pars.psiProbeInit.begin(), pars.psiProbeInit.end(),
		                                                      (PRISMATIC_FLOAT_PRECISION)0.0, [](PRISMATIC_FLOAT_PRECISION accum, std::complex<PRISMATIC_FLOAT_PRECISION> &a) {
					return accum + abs(a) * abs(a);
				})); // make sure to initialize with 0.0 and NOT 0 or it won't be a float and answer will be wrong

		transform(pars.psiProbeInit.begin(), pars.psiProbeInit.end(),
		          pars.psiProbeInit.begin(), [&norm_constant](std::complex<PRISMATIC_FLOAT_PRECISION> &a) {
					return a / norm_constant;
				});

		if(pars.meta.saveProbe && pars.fpFlag == 0)
		{
            setupProbeOutput(pars);
            saveProbe(pars);
		}
	}

	void ms_miser_add_repr_pts(Parameters<PRISMATIC_FLOAT_PRECISION> &pars,
    	vector<size_t> &repr_iqx, vector<size_t> &repr_iqy)
	{
		const double pi = acos(-1);

		PRISMATIC_FLOAT_PRECISION qProbeMax = pars.meta.probeSemiangle/ pars.lambda; // currently a single semiangle
		PRISMATIC_FLOAT_PRECISION dqx = pars.qxa.at(0,1);
		PRISMATIC_FLOAT_PRECISION dqy = pars.qya.at(1,0);

		repr_iqx.clear();
		repr_iqy.clear();

		vector<PRISMATIC_FLOAT_PRECISION> ring_rfrac = {0.0, 0.5, 0.9, 1.2};
		vector<size_t> ring_npts = {1, 8, 12, 8};
		// vector<size_t> ring_npts = {1, 4, 4, 4};

		for (size_t i = 0; i < ring_rfrac.size(); ++i) {
			for (size_t j = 0; j < ring_npts[i]; ++j) {
				PRISMATIC_FLOAT_PRECISION qx = qProbeMax * ring_rfrac[i] * cos(2 * pi / ring_npts[i] * j);
				PRISMATIC_FLOAT_PRECISION qy = qProbeMax * ring_rfrac[i] * sin(2 * pi / ring_npts[i] * j);
				int iqx = round(qx / dqx);
				int iqy = round(qy / dqy);
				if (iqx < 0) iqx += pars.imageSize[1];
				if (iqy < 0) iqy += pars.imageSize[0];
				repr_iqx.push_back(iqx);
				repr_iqy.push_back(iqy);
			}
		}
	}

    Array2D<complex<PRISMATIC_FLOAT_PRECISION>> get_shifted_probe(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, size_t probe_num) {
        const size_t ay = (pars.meta.arbitraryProbes) ? probe_num : probe_num / pars.xp.size();
		const size_t ax = (pars.meta.arbitraryProbes) ? probe_num : probe_num % pars.xp.size();

        Array2D<complex<PRISMATIC_FLOAT_PRECISION> > psi(pars.psiProbeInit);
        auto qxa_ptr = pars.qxa.begin();
        auto qya_ptr = pars.qya.begin();
        for (auto& p:psi)p*=exp(-2 * pi * i * ( (*qxa_ptr++)*pars.xp[ax] +
                                                (*qya_ptr++)*pars.yp[ay]));
        return psi;
    }

	void createStack_MS_MISER(Parameters<PRISMATIC_FLOAT_PRECISION>& pars){
		size_t numLayers = (pars.numPlanes / pars.numSlices) + ((pars.numPlanes) % pars.numSlices != 0);
		if(pars.zStartPlane > 0)  numLayers += ((pars.zStartPlane) % pars.numSlices == 0) - (pars.zStartPlane / pars.numSlices) ;

		size_t firstLayer = (pars.zStartPlane / pars.numSlices) + ((pars.zStartPlane) % pars.numSlices != 0);
		if(pars.zStartPlane == 0) firstLayer = 1;

		cout << "Number of layers: " << numLayers << endl;
		cout << "First output depth is at " << firstLayer * pars.meta.sliceThickness * pars.numSlices << " angstroms with steps of " << pars.numSlices * pars.meta.sliceThickness << " angstroms" << endl;
		//store depths in vector
		std::vector<PRISMATIC_FLOAT_PRECISION> depths(numLayers);
		depths[0] = firstLayer * pars.meta.sliceThickness * pars.numSlices;
		for(auto i = 1; i < numLayers; i++) depths[i] = depths[i-1]+pars.numSlices*pars.meta.sliceThickness;
		pars.depths = depths;
		pars.numLayers = numLayers;
		
		pars.output = zeros_ND<4, PRISMATIC_FLOAT_PRECISION>({{numLayers, pars.numYprobes, pars.numXprobes, pars.Ndet}});

		if(pars.meta.saveDPC_CoM) {
			std::cout << "Warning: saveDPC_CoM not implemented for MISER mode" << std::endl;
		}
		if(pars.meta.save4DOutput)
		{
			if(pars.meta.saveComplexOutputWave) {
				std::cout << "Error: pars.meta.saveComplexOutputWave not implemented for MISER mode" << std::endl;
				std::exit(1);
			}

			setup4DOutput(pars);

			pars.cbed_buffer = zeros_ND<2, PRISMATIC_FLOAT_PRECISION>({{pars.imageSize[0]/2, pars.imageSize[1]/2}});

			std::cout << "pars.cbed_buffer size: " << pars.cbed_buffer.get_dimj() << " " << pars.cbed_buffer.get_dimi() << std::endl;

			if(pars.meta.crop4DOutput) {
				pars.cbed_buffer = cropOutput(pars.cbed_buffer, pars);
			} else {
				std::cout << "Error: crop4DOutput == 0 not implemented for MISER mode" << std::endl;
				std::exit(1);
			}

			std::cout << "pars.cbed_buffer size: " << pars.cbed_buffer.get_dimj() << " " << pars.cbed_buffer.get_dimi() << std::endl;
			
		}
	}

	Array2D<PRISMATIC_FLOAT_PRECISION> get_MS_MISER_single_uncropped_CPU(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, 
		PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw,
		Array2D<complex<PRISMATIC_FLOAT_PRECISION>> &shifted_probe,
		Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi,
		std::mt19937 &gen)
	{
		Array3D<complex<PRISMATIC_FLOAT_PRECISION>> transmission;
		{
			Array1D<PRISMATIC_FLOAT_PRECISION> pt = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{range_lo.get_dimi()}});
			auto pot = pars.meta.potential3D ? 
				Prismatic::generateProjectedPotentials3D_miser(pars, pw, range_lo, range_hi, gen, pt) :
				Prismatic::generateProjectedPotentials_miser(pars, pw, range_lo, range_hi, gen, pt);
			transmission = zeros_ND<3, complex<PRISMATIC_FLOAT_PRECISION> >(
				{{pot.get_dimk(), pot.get_dimj(), pot.get_dimi()}});
			
			auto p = pot.begin();
			for (auto &j:transmission)j = exp(i * pars.sigma * (*p++));
		}
		
		Array2D<complex<PRISMATIC_FLOAT_PRECISION> > psi = shifted_probe;
		PRISMATIC_FFTW_INIT_THREADS();
		PRISMATIC_FFTW_PLAN_WITH_NTHREADS(1);
		unique_lock<mutex> gatekeeper(fftw_plan_lock);
		PRISMATIC_FFTW_PLAN plan_forward = PRISMATIC_FFTW_PLAN_DFT_2D(psi.get_dimj(), psi.get_dimi(),
		                                                      reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
		                                                      reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
		                                                      FFTW_FORWARD, FFTW_ESTIMATE);
		PRISMATIC_FFTW_PLAN plan_inverse = PRISMATIC_FFTW_PLAN_DFT_2D(psi.get_dimj(), psi.get_dimi(),
		                                                      reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
		                                                      reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
		                                                      FFTW_BACKWARD, FFTW_ESTIMATE);
		gatekeeper.unlock();

		auto scaled_prop = pars.prop;
		for (auto& jj : scaled_prop) jj /= shifted_probe.size(); // apply FFT scaling factor here once in advance rather than at every plane

		for (auto a2 = 0; a2 < pars.numPlanes; ++a2){
			PRISMATIC_FFTW_EXECUTE(plan_inverse);
			complex<PRISMATIC_FLOAT_PRECISION>* t_ptr = &transmission[a2 * transmission.get_dimj() * transmission.get_dimi()];
			for (auto& p:psi)p *= (*t_ptr++); // transmit
			PRISMATIC_FFTW_EXECUTE(plan_forward);
			auto p_ptr = scaled_prop.begin();
			for (auto& p:psi)p *= (*p_ptr++); // propagate
		}

		gatekeeper.lock();
		PRISMATIC_FFTW_DESTROY_PLAN(plan_forward);
		PRISMATIC_FFTW_DESTROY_PLAN(plan_inverse);
		PRISMATIC_FFTW_CLEANUP_THREADS();
		gatekeeper.unlock();

		Array2D<PRISMATIC_FLOAT_PRECISION> intOutput = zeros_ND<2, PRISMATIC_FLOAT_PRECISION>({{psi.get_dimj(), psi.get_dimi()}});
		auto psi_ptr = psi.begin();
		for (auto& j:intOutput) j = pow(abs(*psi_ptr++),2);

		return intOutput;
	}

	void get_multislice_miser_rect_repr_CPU(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw,
		Array2D<complex<PRISMATIC_FLOAT_PRECISION>> &shifted_probe, 
		Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, 
		Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi, 
		size_t npts, vector<size_t> &repr_iqx, vector<size_t> &repr_iqy,
		Array2D<PRISMATIC_FLOAT_PRECISION> &pts_sampled, Array2D<PRISMATIC_FLOAT_PRECISION> &repr_vals)
	{
		size_t nout = repr_iqx.size();
		pts_sampled = zeros_ND<2, PRISMATIC_FLOAT_PRECISION>({{npts, range_lo.get_dimi()}});
		repr_vals = zeros_ND<2, PRISMATIC_FLOAT_PRECISION>({{npts, nout}});
		unsigned int seed0 = ((rand() & 0x7fffu)<<17 | (rand() & 0x7fffu)<<2 ) | (rand() & 0x7fffu)>>13;

		vector<thread> workers;
		workers.reserve(pars.meta.numThreads); // prevents multiple reallocations
		PRISMATIC_FFTW_INIT_THREADS();
		PRISMATIC_FFTW_PLAN_WITH_NTHREADS(2);
		WorkDispatcher dispatcher(0, npts);

		for (auto t = 0; t < pars.meta.numThreads; ++t){
			workers.push_back(thread([&pars, &pw, &dispatcher, &shifted_probe, &range_lo, &range_hi, 
				&repr_iqx, &repr_iqy, &repr_vals, &pts_sampled,
				&seed0, t]() {

				size_t Nstart, Nstop;
                Nstart=Nstop=0;
				if (dispatcher.getWork(Nstart, Nstop, 1)){ // synchronously get work assignment
					unsigned int seed1 = seed0 + 10000u * (unsigned int) t;
					std::mt19937 gen(seed1);
					std::uniform_real_distribution<PRISMATIC_FLOAT_PRECISION> distr(1.0, 2.0);

					Array2D<complex<PRISMATIC_FLOAT_PRECISION> > psi = shifted_probe;
					unique_lock<mutex> gatekeeper(fftw_plan_lock);
					PRISMATIC_FFTW_PLAN plan_forward = PRISMATIC_FFTW_PLAN_DFT_2D(psi.get_dimj(), psi.get_dimi(),
																		reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
																		reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
																		FFTW_FORWARD, FFTW_ESTIMATE);
					PRISMATIC_FFTW_PLAN plan_inverse = PRISMATIC_FFTW_PLAN_DFT_2D(psi.get_dimj(), psi.get_dimi(),
																		reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
																		reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
																		FFTW_BACKWARD, FFTW_ESTIMATE);
					gatekeeper.unlock();

					auto scaled_prop = pars.prop;
					for (auto& jj : scaled_prop) jj /= shifted_probe.size(); // apply FFT scaling factor here once in advance rather than at every plane

					Array3D<complex<PRISMATIC_FLOAT_PRECISION>> transmission = zeros_ND<3, complex<PRISMATIC_FLOAT_PRECISION> >(
						{{pars.pot.get_dimk(), pars.pot.get_dimj(), pars.pot.get_dimi()}});

					do {
						while (Nstart < Nstop) {
							// copy the shifted_probe over again
							auto psi_ptr = psi.begin();
							for (auto i:shifted_probe)*psi_ptr++ = i;

							{
								Array1D<PRISMATIC_FLOAT_PRECISION> pt = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{range_lo.get_dimi()}});
								auto pot = pars.meta.potential3D ? 
									Prismatic::generateProjectedPotentials3D_miser(pars, pw, range_lo, range_hi, gen, pt) :
									Prismatic::generateProjectedPotentials_miser(pars, pw, range_lo, range_hi, gen, pt);

								for (size_t i = 0; i < range_lo.get_dimi(); ++i) {
									pts_sampled.at(Nstart, i) = pt.at(i);
								}
								
								auto p = pot.begin();
								for (auto &j:transmission)j = exp(i * pars.sigma * (*p++));
							}

							for (auto a2 = 0; a2 < pars.numPlanes; ++a2){
								PRISMATIC_FFTW_EXECUTE(plan_inverse);
								complex<PRISMATIC_FLOAT_PRECISION>* t_ptr = &transmission[a2 * transmission.get_dimj() * transmission.get_dimi()];
								for (auto& p:psi)p *= (*t_ptr++); // transmit
								PRISMATIC_FFTW_EXECUTE(plan_forward);
								auto p_ptr = scaled_prop.begin();
								for (auto& p:psi)p *= (*p_ptr++); // propagate
							}

							Array2D<PRISMATIC_FLOAT_PRECISION> intOutput = zeros_ND<2, PRISMATIC_FLOAT_PRECISION>({{psi.get_dimj(), psi.get_dimi()}});
							psi_ptr = psi.begin();
							for (auto& j:intOutput) j = pow(abs(*psi_ptr++),2);

							for (size_t i = 0; i < repr_iqx.size(); ++i) {
								repr_vals.at(Nstart, i) = intOutput.at(repr_iqy[i], repr_iqx[i]);
							}
							
							Nstart=Nstop;
						}
					} while(dispatcher.getWork(Nstart, Nstop, 1));

					gatekeeper.lock();
					PRISMATIC_FFTW_DESTROY_PLAN(plan_forward);
					PRISMATIC_FFTW_DESTROY_PLAN(plan_inverse);
					gatekeeper.unlock();
				}
			}));
		}
		for (auto& t:workers)t.join();

		PRISMATIC_FFTW_CLEANUP_THREADS();
	}

	void get_multislice_miser_rect_full_CPU(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw,
    Array2D<complex<PRISMATIC_FLOAT_PRECISION>> &shifted_probe, 
    Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, 
    Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi, 
    size_t npts,
    Array3D<PRISMATIC_FLOAT_PRECISION> &summ, 
    Array3D<PRISMATIC_FLOAT_PRECISION> &summ2)
	{
		summ = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>({{pars.numPlanes, pars.cbed_buffer.get_dimj(), pars.cbed_buffer.get_dimi()}});
		summ2 = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>({{pars.numPlanes, pars.cbed_buffer.get_dimj(), pars.cbed_buffer.get_dimi()}});

		unsigned int seed0 = ((rand() & 0x7fffu)<<17 | (rand() & 0x7fffu)<<2 ) | (rand() & 0x7fffu)>>13;

		vector<thread> workers;
		workers.reserve(pars.meta.numThreads); // prevents multiple reallocations
		PRISMATIC_FFTW_INIT_THREADS();
		PRISMATIC_FFTW_PLAN_WITH_NTHREADS(2);
		WorkDispatcher dispatcher(0, npts);

		for (auto t = 0; t < pars.meta.numThreads; ++t){
			workers.push_back(thread([&pars, &pw, &dispatcher, &shifted_probe, &range_lo, &range_hi, 
				&summ, &summ2,
				&seed0, t]() {

				size_t Nstart, Nstop;
                Nstart=Nstop=0;
				if (dispatcher.getWork(Nstart, Nstop, 1)){ // synchronously get work assignment
					unsigned int seed1 = seed0 + 10000u * (unsigned int) t;
					std::mt19937 gen(seed1);
					std::uniform_real_distribution<PRISMATIC_FLOAT_PRECISION> distr(1.0, 2.0);

					Array2D<complex<PRISMATIC_FLOAT_PRECISION> > psi = shifted_probe;
					unique_lock<mutex> gatekeeper(fftw_plan_lock);
					PRISMATIC_FFTW_PLAN plan_forward = PRISMATIC_FFTW_PLAN_DFT_2D(psi.get_dimj(), psi.get_dimi(),
																		reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
																		reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
																		FFTW_FORWARD, FFTW_ESTIMATE);
					PRISMATIC_FFTW_PLAN plan_inverse = PRISMATIC_FFTW_PLAN_DFT_2D(psi.get_dimj(), psi.get_dimi(),
																		reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
																		reinterpret_cast<PRISMATIC_FFTW_COMPLEX *>(&psi[0]),
																		FFTW_BACKWARD, FFTW_ESTIMATE);
					gatekeeper.unlock();

					auto scaled_prop = pars.prop;
					for (auto& jj : scaled_prop) jj /= shifted_probe.size(); // apply FFT scaling factor here once in advance rather than at every plane

					Array3D<complex<PRISMATIC_FLOAT_PRECISION>> transmission = zeros_ND<3, complex<PRISMATIC_FLOAT_PRECISION> >(
						{{pars.pot.get_dimk(), pars.pot.get_dimj(), pars.pot.get_dimi()}});

					do {
						while (Nstart < Nstop) {
							// copy the shifted_probe over again
							auto psi_ptr = psi.begin();
							for (auto i:shifted_probe)*psi_ptr++ = i;

							{
								Array1D<PRISMATIC_FLOAT_PRECISION> pt = zeros_ND<1, PRISMATIC_FLOAT_PRECISION>({{range_lo.get_dimi()}});
								auto pot = pars.meta.potential3D ? 
									Prismatic::generateProjectedPotentials3D_miser(pars, pw, range_lo, range_hi, gen, pt) :
									Prismatic::generateProjectedPotentials_miser(pars, pw, range_lo, range_hi, gen, pt);
								
								auto p = pot.begin();
								for (auto &j:transmission)j = exp(i * pars.sigma * (*p++));
							}

							size_t currentSlice = 0;
							for (auto a2 = 0; a2 < pars.numPlanes; ++a2){
								PRISMATIC_FFTW_EXECUTE(plan_inverse);
								complex<PRISMATIC_FLOAT_PRECISION>* t_ptr = &transmission[a2 * transmission.get_dimj() * transmission.get_dimi()];
								for (auto& p:psi)p *= (*t_ptr++); // transmit
								PRISMATIC_FFTW_EXECUTE(plan_forward);
								auto p_ptr = scaled_prop.begin();
								for (auto& p:psi)p *= (*p_ptr++); // propagate

								if  ( ( (((a2+1) % pars.numSlices) == 0) && ((a2+1) >= pars.zStartPlane) ) || ((a2+1) == pars.numPlanes) ){
									Array2D<PRISMATIC_FLOAT_PRECISION> intOutput = zeros_ND<2, PRISMATIC_FLOAT_PRECISION>({{psi.get_dimj(), psi.get_dimi()}});
									psi_ptr = psi.begin();
									for (auto& j:intOutput) j = pow(abs(*psi_ptr++),2);
									Array2D<PRISMATIC_FLOAT_PRECISION> intOutput_small = cropOutput(intOutput,pars);

									{
										unique_lock<mutex> summ_gatekeeper(summ_lock);
										for (size_t j = 0; j < intOutput_small.get_dimj(); ++j) {
											for (size_t i = 0; i < intOutput_small.get_dimi(); ++i) {
												auto val = intOutput_small.at(j,i);
												summ.at(currentSlice, j, i) += val;
												summ2.at(currentSlice, j, i) += val*val;
											}
										}
										// size_t offset = currentSlice * pars.cbed_buffer.get_dimj() * pars.cbed_buffer.get_dimi();
										// PRISMATIC_FLOAT_PRECISION* s_ptr = &summ[offset];
										// PRISMATIC_FLOAT_PRECISION* s2_ptr = &summ2[offset];
										// for (auto& val : intOutput_small) {
										// 	*s_ptr++ += val;
										// 	*s2_ptr++ += val*val;
										// }
									}
									currentSlice++;
								}
							}
							
							Nstart=Nstop;
						}
					} while(dispatcher.getWork(Nstart, Nstop, 1));

					gatekeeper.lock();
					PRISMATIC_FFTW_DESTROY_PLAN(plan_forward);
					PRISMATIC_FFTW_DESTROY_PLAN(plan_inverse);
					gatekeeper.unlock();
				}
			}));
		}
		for (auto& t:workers)t.join();

		PRISMATIC_FFTW_CLEANUP_THREADS();
	}

	void save_CBED_miser(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, miser_result_t ave_var, size_t probe_num)
	{
		const size_t ay = (pars.meta.arbitraryProbes) ? 0 : probe_num / pars.numXprobes;
		const size_t ax = (pars.meta.arbitraryProbes) ? probe_num : probe_num % pars.numXprobes;
		hsize_t offset[4] = {ax,ay,0,0}; //order by ax, ay so that aligns with py4DSTEM
		hsize_t mdims[4];
		mdims[0] = mdims[1] = {1};

		mdims[2] = {ave_var.first.get_dimj()};
		mdims[3] = {ave_var.first.get_dimi()};

		for (size_t currentSlice = 0; currentSlice < pars.numLayers; ++currentSlice) {
			std::string nameString = "4DSTEM_simulation/data/datacubes/CBED_array_depth" + getDigitString(currentSlice);
			nameString += pars.currentTag;

			writeDatacube4D_MS_MISER(pars,&ave_var.first[0], &ave_var.second[0], &pars.cbed_buffer[0],mdims,offset,nameString.c_str());
		}
	}

    pair<size_t, size_t> MiserTree::subdivide(size_t key, NodeData left_data, NodeData right_data)
    {
		if (key >= nodes.size()) {
			throw std::runtime_error("MiserTree: Index out of range.");
		}
		size_t i1 = nodes.size();
		nodes.push_back(MiserNode(i1, left_data));
		nodes[key].left = i1;

		size_t i2 = nodes.size();
		nodes.push_back(MiserNode(i2, right_data));
		nodes[key].right = i2;
		
		nodes[key].has_child = true;

        return pair<size_t, size_t>(i1, i2);
    }

    size_t MiserTree::_height(size_t idx)
    {
		auto node = nodes[idx];
		if (node.has_child) {
			return 1 + max(_height(node.left), _height(node.right));
		}
        return 1;
    }

	void MiserTree::update_data(size_t step_idx, long npts, int dim_split, PRISMATIC_FLOAT_PRECISION fracl)
	{
		nodes[step_idx].data.npts = npts;
		nodes[step_idx].data.dim_split = dim_split;
		nodes[step_idx].data.fracl = fracl;
	}

	pair<unsigned long, vector<unsigned long>> MiserTree::plan_info(size_t ndim)
	{
		auto curr_ht = height();
		auto curr = nodes[0];
		
		vector<int> path(curr_ht, -1);
		vector<bool> is_left(curr_ht, false);
		vector<long> left_nodes(curr_ht, -1);
		path[0] = 0;
		is_left[0] = true;
		left_nodes[0] = 0;
		vector<bool> left_is_ready(curr_ht, false);

		long right_node = -1;
		size_t curr_depth = 0;

		unsigned long npts_total = 0;
		vector<unsigned long> num_bisections(ndim, 0);

		bool go_up = false;
		while (true) {
			go_up = false;

			if (!curr.has_child) {
				// calculate and go up a level
				npts_total += curr.data.npts;
				for (size_t id = 0; id < curr_depth; ++id) {
					num_bisections[nodes[path[id]].data.dim_split] += curr.data.npts;
				}

				if (is_left[curr_depth]) {
					left_nodes[curr_depth] = curr.key;
					left_is_ready[curr_depth] = true;
				} else {
					right_node = curr.key;
				}
				go_up = true;
			} else if (std::find(left_nodes.begin(), left_nodes.end(), curr.left) == left_nodes.end() || !left_is_ready[curr_depth+1]) {
				// go into left node
				curr_depth += 1;
				curr = nodes[curr.left];
				path[curr_depth] = curr.key;
				is_left[curr_depth] = true;
				left_nodes[curr_depth] = curr.key;
			} else if (right_node == -1) {
				// go into right node
				curr_depth += 1;
				curr = nodes[curr.right];
				path[curr_depth] = curr.key;
			} else {
				// combine left and right children and go up a level
				if (is_left[curr_depth]) {
					left_is_ready[curr_depth] = true;
					right_node = -1;
				} else {
					right_node = curr.key;
				}

				// the left child is used up and is not ready anymore
				left_nodes[curr_depth+1] = -1;
				left_is_ready[curr_depth+1] = false;

				go_up = true;
			}

			if (go_up) {
				if (curr_depth == 0) {
					return pair<unsigned long, vector<unsigned long>>(npts_total, num_bisections);
				}
				path[curr_depth] = -1;
				is_left[curr_depth] = false;
				curr_depth--;
				curr = nodes[path[curr_depth]];
			}
		}
	}

	MiserTree MiserPlanner::make_plan(Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, 
				Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi, size_t npts)
	{
		MiserTree tree = MiserTree();
		miser_plan_helper(tree, range_lo, range_hi, npts, 0);
		return tree;
	}

	void MiserPlanner::miser_plan_helper(MiserTree &tree, Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi,
                            size_t npts, size_t step_idx)
	{
		if (npts < MNBS) {
			return;
		}

		size_t ndims = range_lo.get_dimi();
		vector<PRISMATIC_FLOAT_PRECISION> rmid;
		rmid.reserve(ndims);
		for (size_t i = 0; i < ndims; ++i) {
			auto dith = coin_flip(bounds_gen) ? -DITHER : DITHER;
			rmid.push_back( (0.5 + dith) * range_lo.at(i) + (0.5 - dith) * range_hi.at(i));
		}

		uniform_int_distribution<long> int_distr(0, ndims-1);

		size_t npre_pts = max((size_t) round(npts * PFAC), MNPT);

		// decide along which dimension to bisect
		auto sumb = BIG;
		long jb = -1;
		PRISMATIC_FLOAT_PRECISION siglb = 1.0;
		PRISMATIC_FLOAT_PRECISION sigrb = 1.0;
		{
			const array<size_t, 2> dims = {ndims, nout};
			size_t size = ndims * nout;
			Array2D<PRISMATIC_FLOAT_PRECISION> fmaxl(vector<PRISMATIC_FLOAT_PRECISION>(size, -BIG), dims);
			Array2D<PRISMATIC_FLOAT_PRECISION> fmaxr(vector<PRISMATIC_FLOAT_PRECISION>(size, -BIG), dims);
			Array2D<PRISMATIC_FLOAT_PRECISION> fminl(vector<PRISMATIC_FLOAT_PRECISION>(size, BIG), dims);
			Array2D<PRISMATIC_FLOAT_PRECISION> fminr(vector<PRISMATIC_FLOAT_PRECISION>(size, BIG), dims);

			Array2D<PRISMATIC_FLOAT_PRECISION> repr_vals;
			Array2D<PRISMATIC_FLOAT_PRECISION> pts_sampled;
			get_multislice_miser_rect_repr_CPU(pars, pw, shifted_probe, range_lo, range_hi, npre_pts, repr_iqx, repr_iqy, pts_sampled, repr_vals);

			for (size_t n = 0; n < npre_pts; ++n) {
				for (size_t d = 0; d < ndims; ++d) {
					for (size_t i = 0; i < nout; ++i) {
						if (pts_sampled.at(n, d) <= rmid.at(d)) {
							// left
							fminl.at(d, i) = min(fminl.at(d, i), repr_vals.at(n, i));
							fmaxl.at(d, i) = max(fmaxl.at(d, i), repr_vals.at(n, i));
						} else {
							// right
							fminr.at(d, i) = min(fminr.at(d, i), repr_vals.at(n, i));
							fmaxr.at(d, i) = max(fmaxr.at(d, i), repr_vals.at(n, i));
						}
					}
				}
			}

			vector<PRISMATIC_FLOAT_PRECISION> diff_l(ndims, 0);
			vector<PRISMATIC_FLOAT_PRECISION> diff_r(ndims, 0);
			for (size_t d = 0; d < ndims; ++d) {
				bool dl_nonzero = false;
				bool dr_nonzero = false;
				PRISMATIC_FLOAT_PRECISION dl_sum = 0;
				PRISMATIC_FLOAT_PRECISION dr_sum = 0;
				for (size_t i = 0; i < nout; ++i) {
					diff_l[d] = fmaxl.at(d,i) - fminl.at(d,i);
					diff_r[d] = fmaxr.at(d,i) - fminr.at(d,i);
					if (diff_l[d] > 0) {
						dl_nonzero = true;
					}
					if (diff_r[d] > 0) {
						dr_nonzero = true;
					}
					dl_sum += diff_l[d];
					dr_sum += diff_r[d];
				}

				if (dl_nonzero && dr_nonzero) {
					PRISMATIC_FLOAT_PRECISION sigl = max(TINY, pow(dl_sum, 0.666666f));
					PRISMATIC_FLOAT_PRECISION sigr = max(TINY, pow(dr_sum, 0.666666f));
					PRISMATIC_FLOAT_PRECISION sumlr = sigl + sigr;
					if (sumlr <= sumb) {
						sumb = sumlr;
						jb = d;
						siglb = sigl;
						sigrb = sigr;
					}
				}
			}
		}
		if (jb < 0) {
			jb = int_distr(bounds_gen);
		}

		PRISMATIC_FLOAT_PRECISION rgl = range_lo[jb];
		PRISMATIC_FLOAT_PRECISION rgm = rmid[jb];
		PRISMATIC_FLOAT_PRECISION rgr = range_hi[jb];
		PRISMATIC_FLOAT_PRECISION fracl = abs( (rgm-rgl) / (rgr-rgl) );
		size_t nptl = MNPT + (size_t) round((npts-npre_pts-2*MNPT)*fracl*siglb/(fracl*siglb + (1-fracl)*sigrb));
		size_t nptr = npts - npre_pts - nptl;

		Array1D<PRISMATIC_FLOAT_PRECISION> range_lo1, range_hi1, range_lo2, range_hi2;
		range_lo1 = range_lo;
		range_lo2 = range_lo;
		range_hi1 = range_hi;
		range_hi2 = range_hi;

		// lo --1-- mid --2-- hi
		range_hi1[jb] = rmid[jb];
		range_lo2[jb] = rmid[jb];

		tree.update_data(step_idx, npts, jb, fracl);
		auto indices = tree.subdivide(step_idx, NodeData(nptl, range_lo1, range_hi1), NodeData(nptr, range_lo2, range_hi2));

		std::cout << "step " << step_idx << " -> " << indices.first << " " << indices.second << ", " << npts << " pts, split dim " << jb << " ";
		if (jb % 2 == 0) {
			std::cout << "[" << pars.atoms[jb/2].x << "] " << pars.atoms[jb/2].y << " ; ";
		} else {
			std::cout << " " << pars.atoms[jb/2].x << " [" << pars.atoms[jb/2].y << "]; ";
		}
		std::cout << rgl << ".." << rgr << std::endl;

		miser_plan_helper(tree, range_lo1, range_hi1, nptl, indices.first);
		miser_plan_helper(tree, range_lo2, range_hi2, nptr, indices.second);

		return;
	}

	miser_result_t MiserPlanner::execute_miser_plan(MiserTree tree)
	{
		size_t curr = 0;
		size_t curr_depth = 0;
		size_t total_pts = 0;

		cout << "miser tree height: " << tree.height() << endl;

		vector<long> path(tree.height(), -1);
		vector<bool> is_left(tree.height(), false);
		vector<long> left_nodes(tree.height(), -1);
		path[0] = 0;
		is_left[0] = true; // think of the root node as a "left" node
		left_nodes[0] = 0;
		vector<boost::optional<miser_result_t>> left_results(tree.height(), boost::none);
		
		long right_node = -1;
		boost::optional<miser_result_t> right_result;

		const array<size_t, 3> dims = {pars.numPlanes, pars.cbed_buffer.get_dimj(), pars.cbed_buffer.get_dimi()};
		
		while (true) {
			bool go_up = false;

			// std::cout << "curr: " << curr << ", curr_depth: " << curr_depth << std::endl;

			if (!tree.nodes[curr].has_child) {
				// calculate and go up a level
				size_t npts = tree.nodes[curr].data.npts;
				std::cout << "calculate node " << curr << " with " << npts << " points" << std::endl;
				Array3D<PRISMATIC_FLOAT_PRECISION> summ, summ2;
				get_multislice_miser_rect_full_CPU(pars, pw, shifted_probe,
					tree.nodes[curr].data.range_lo,
					tree.nodes[curr].data.range_hi,
					npts, summ, summ2);
				
				total_pts += npts;

				Array3D<PRISMATIC_FLOAT_PRECISION> ave = summ / npts;
				auto var = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>(dims);

				// auto var_ptr = var.begin();
				// auto s2_ptr = summ2.begin();
				for (size_t i = 0; i < summ.size(); ++i) {
					// var[i] = (summ2[i] - summ[i]*summ[i]/npts) / npts / npts;
					var[i] = max(TINY, (summ2[i] - summ[i]*summ[i]/npts) / npts / npts); // ????
					// var[i] = max(TINY, (summ2[i] - summ[i]*summ[i]/npts) / npts);
				}

				if (is_left[curr_depth]) {
					// we've calculated a left node
					left_nodes[curr_depth] = curr;
					left_results[curr_depth] = miser_result_t(ave, var);
				} else {
					// we've calculated a right node
					right_node = curr;
					right_result = miser_result_t(ave, var);
				}

				go_up = true;
			} else if (find(left_nodes.begin(), left_nodes.end(), tree.nodes[curr].left) == left_nodes.end() ||
				!left_results[curr_depth+1])
			{
				// go into left node
				curr_depth++;
				curr = tree.nodes[curr].left;
				path[curr_depth] = curr;
				is_left[curr_depth] = 1;
				left_nodes[curr_depth] = curr;
			} else if (right_node == -1) {
				// go into right node
				curr_depth++;
				curr = tree.nodes[curr].right;
				path[curr_depth] = curr;
			} else {
				// combine left and right children and go up a level
				auto fracl = tree.nodes[curr].data.fracl;
				
				std::cout << "combine " << left_nodes[curr_depth+1] << " " << right_node << " -> " << curr << "; " << fracl << std::endl;

				auto avel = left_results[curr_depth+1]->first;
				auto varl = left_results[curr_depth+1]->second;
				auto aver = right_result->first;
				auto varr = right_result->second;

				auto ave = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>(dims);
				auto var = zeros_ND<3, PRISMATIC_FLOAT_PRECISION>(dims);
				for (size_t i = 0; i < ave.size(); ++i) {
					ave[i] = fracl*avel[i] + (1-fracl)*aver[i];
					var[i] = fracl*fracl*varl[i] + (1-fracl)*(1-fracl)*varr[i];
				}

				if (is_left[curr_depth]) {
					left_results[curr_depth] = miser_result_t(ave, var);
					right_node = -1;
					right_result = boost::none;
				} else {
					right_node = curr;
					right_result = miser_result_t(ave, var);
				}

				left_nodes[curr_depth+1] = -1;
				left_results[curr_depth+1] = boost::none;

				go_up = true;
			}

			if (go_up) {
				if (curr_depth == 0) {
					cout << "total points: " << total_pts << endl;
					return left_results[0].get();
				}
				path[curr_depth] = -1;
				is_left[curr_depth] = false;
				curr_depth--;
				curr = path[curr_depth];
			}
		}
	}

}