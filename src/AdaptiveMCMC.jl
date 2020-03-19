# Very simple implementation of some adaptive MCMC algorithms.
module AdaptiveMCMC

export adaptive_rwm, AdaptiveMetropolis, RobustAdaptiveMetropolis,
AdaptiveScalingMetropolis, AdaptiveScalingWithinAdaptiveMetropolis,
ConstantShape, AdaptState, RWMState,
draw!, accept!, adapt!, adapt_rb!,
StepSize, PolynomialStepSize, RAMStepSize, get

using Random, LinearAlgebra, StaticArrays

# Abstract type of the state of an adaptive random walk Metropolis MCMC.
abstract type AdaptState end

include("linearalgebra_helpers.jl")
include("rwm.jl")       # Random walk Metropolis functionality
include("stepsize.jl")  # Step size sequences
include("ram.jl")       # RobustAdaptiveMetropolis
include("am.jl")        # AdaptiveMetropolis
include("asm.jl")       # AdaptiveScalingMetropolis and AdaptiveScalingWithinAdaptiveMetropolis
include("adaptive_rwm.jl")      # Adaptive random walk Metropolis interface `arwm`

end # module
