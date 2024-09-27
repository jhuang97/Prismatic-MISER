#ifndef MS_MISER_H
#define MS_MISER_H
#include <random>
#include "ArrayND.h"
#include "params.h"
#include "meta.h"

namespace Prismatic {
using namespace std;

typedef pair<Array3D<PRISMATIC_FLOAT_PRECISION>, Array3D<PRISMATIC_FLOAT_PRECISION>> miser_result_t;

void setupCoordinates_MS_MISER(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void setupDetector_MS_MISER(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void setupProbes_MS_MISER(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

void ms_miser_add_repr_pts(Parameters<PRISMATIC_FLOAT_PRECISION> &pars,
    vector<size_t> &repr_iqx, vector<size_t> &repr_iqy);

Array2D<complex<PRISMATIC_FLOAT_PRECISION>> get_shifted_probe(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, size_t probe_num);

void createStack_MS_MISER(Parameters<PRISMATIC_FLOAT_PRECISION>& pars);

Array2D<PRISMATIC_FLOAT_PRECISION> get_MS_MISER_single_uncropped_CPU(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, 
		PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw,
		Array2D<complex<PRISMATIC_FLOAT_PRECISION>> &shifted_probe,
		Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi,
		std::mt19937 &gen);

void get_multislice_miser_rect_repr_CPU(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw,
    Array2D<complex<PRISMATIC_FLOAT_PRECISION>> &shifted_probe, 
    Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, 
    Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi, 
    size_t npts, vector<size_t> &repr_iqx, vector<size_t> &repr_iqy,
	Array2D<PRISMATIC_FLOAT_PRECISION> &pts_sampled, Array2D<PRISMATIC_FLOAT_PRECISION> &repr_vals);

void get_multislice_miser_rect_full_CPU(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw,
    Array2D<complex<PRISMATIC_FLOAT_PRECISION>> &shifted_probe, 
    Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, 
    Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi, 
    size_t npts,
    Array3D<PRISMATIC_FLOAT_PRECISION> &summ, 
    Array3D<PRISMATIC_FLOAT_PRECISION> &summ2);

void save_CBED_miser(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, miser_result_t ave_var, size_t probe_num);

struct NodeData {
    long npts;
    int dim_split;
    PRISMATIC_FLOAT_PRECISION fracl;
    Array1D<PRISMATIC_FLOAT_PRECISION> range_lo;
    Array1D<PRISMATIC_FLOAT_PRECISION> range_hi;

    NodeData() {
        npts = -1;
        dim_split = -1;
        fracl = -1;
    }

    NodeData(long _npts, Array1D<PRISMATIC_FLOAT_PRECISION> _range_lo, 
        Array1D<PRISMATIC_FLOAT_PRECISION> _range_hi):
        npts(_npts), range_lo(_range_lo), range_hi(_range_hi)
    {
        dim_split = -1;
        fracl = -1;
    }
};

struct MiserNode {
    size_t key = 0;
    NodeData data;
    bool has_child = false;
    size_t left;
    size_t right;

    MiserNode(){}
    MiserNode(size_t key, NodeData data) : key(key), data(data) {}
};

class MiserTree {

public:
    static constexpr PRISMATIC_FLOAT_PRECISION TINY = 1e-30;

    MiserTree()
    {
        nodes.push_back(MiserNode());
    }

    pair<size_t, size_t> subdivide(size_t key, NodeData left_data, NodeData right_data);
    size_t height() { return _height(0); };
    size_t size() { return nodes.size(); };
    void update_data(size_t step_idx, long npts, int dim_split, PRISMATIC_FLOAT_PRECISION fracl);
    pair<unsigned long, vector<unsigned long>> plan_info(size_t ndim);
    vector<MiserNode> nodes;

private:
    size_t _height(size_t idx);

};

class MiserPlanner {

public:
    static constexpr PRISMATIC_FLOAT_PRECISION PFAC = 0.1;
    static const size_t MNPT = 60;
    static const size_t MNBS = 240;
    static constexpr PRISMATIC_FLOAT_PRECISION TINY = 1e-30;
    static constexpr PRISMATIC_FLOAT_PRECISION BIG = 1e30;
    static constexpr PRISMATIC_FLOAT_PRECISION DITHER = 0.1;

    Array2D<complex<PRISMATIC_FLOAT_PRECISION>> &shifted_probe;
    vector<size_t> &repr_iqx;
    vector<size_t> &repr_iqy;
    size_t nout;
    bernoulli_distribution coin_flip = bernoulli_distribution(0.5);
    mt19937 bounds_gen;

    MiserPlanner(Parameters<PRISMATIC_FLOAT_PRECISION> &_pars, 
		PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &_pw,
        Array2D<complex<PRISMATIC_FLOAT_PRECISION>> &_shifted_probe,
        vector<size_t> &_repr_iqx, vector<size_t> &_repr_iqy) :
        pars(_pars), pw(_pw), shifted_probe(_shifted_probe), repr_iqx(_repr_iqx), repr_iqy(_repr_iqy)
    {
        nout = repr_iqx.size();
        coin_flip = bernoulli_distribution(0.5);
        unsigned int seed0 = ((rand() & 0x7fffu)<<17 | (rand() & 0x7fffu)<<2 ) | (rand() & 0x7fffu)>>13;
        bounds_gen = mt19937(seed0);
    }

    MiserTree make_plan(Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi, size_t npts);

    miser_result_t execute_miser_plan(MiserTree tree);

private:
    Parameters<PRISMATIC_FLOAT_PRECISION> &pars;
	PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw;

    void miser_plan_helper(MiserTree &tree, Array1D<PRISMATIC_FLOAT_PRECISION> &range_lo, Array1D<PRISMATIC_FLOAT_PRECISION> &range_hi,
                            size_t npts, size_t step_idx);
};

}

#endif //MS_MISER_H