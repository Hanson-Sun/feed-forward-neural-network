#include "headers/cost.h"

namespace Cost
{
    double L2Cost::func(Math::nVector x, Math::nVector y)
    {
        return (x - y).magSqr();
    }
    Math::nVector L2Cost::funcDx(Math::nVector x, Math::nVector y)
    {
        return (x - y);
    }
}