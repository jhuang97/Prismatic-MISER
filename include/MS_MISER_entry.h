#ifndef MS_MISER_ENTRY_H
#define MS_MISER_ENTRY_H
#include "configure.h"
#include "params.h"

namespace Prismatic {

    void MS_MISER_entry(Metadata<PRISMATIC_FLOAT_PRECISION>& meta);

    void MS_MISER_entry_pars(Parameters<PRISMATIC_FLOAT_PRECISION> &pars);

    void MS_MISER_run_probe_pos(Parameters<PRISMATIC_FLOAT_PRECISION> &pars, 
        PotentialWorkspace<PRISMATIC_FLOAT_PRECISION> &pw, size_t probe_num);

}

#endif //MS_MISER_ENTRY_H