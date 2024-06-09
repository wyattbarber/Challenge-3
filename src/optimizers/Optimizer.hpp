#ifndef _OPTIMIZER_HPP
#define _OPTIMIZER_HPP

#include "Adam.hpp"

namespace optimization
{
    /** Supported optimizers
     *
     */
    enum class OptimizerClass
    {
        None,
        Adam
    };
}

#endif