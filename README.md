# AdaptiveMCMC.jl

This package provides implementations of some general-purpose random-walk based adaptive MCMC algorithms, including the following:

* Adaptive Metropolis, proposal covariance adaptation, ([Haario, Saksman and Tamminen, 2001](https://projecteuclid.org/euclid.bj/1080222083), and [Andrieu and Moulines, 2006](http://dx.doi.org/10.1214/105051606000000286))
* Adaptive scaling Metropolis, acceptance rate adaptation for scale (e.g. as in [Andrieu and Thoms, 2008](https://doi.org/10.1007/s11222-008-9110-y), and [Atchadé and Fort, 2010](https://projecteuclid.org/euclid.bj/1265984706))
* Robust Adaptive Metropolis, acceptance rate adaptation for shape [(Vihola, 2012)](http://dx.doi.org/10.1007/s11222-011-9269-5)
* Adaptive Parallel Tempering, acceptance rate adaptation for temperature levels [(Miasojedow, Moulines and Vihola, 2013)](http://dx.doi.org/10.1080/10618600.2013.778779)

The aim of the package is to provide a simple and modular general-purpose implementation, which may be easily used to sample from a log-target density, but also used in a variety of custom settings.

See also [AdaptiveParticleMCMC.jl](https://github.com/mvihola/AdaptiveParticleMCMC.jl) which uses this package with [SequentialMonteCarlo.jl](https://github.com/awllee/SequentialMonteCarlo.jl) for adaptive particle MCMC.

## Getting the package

To get the latest registered version:
```julia
using Pkg
Pkg.add("AdaptiveMCMC")
```
To install the latest development version:
```julia
using Pkg
Pkg.add(url="https://github.com/mvihola/AdaptiveMCMC.jl")
```

## Quick start

```julia
# Load the package
using AdaptiveMCMC

# Define a function which returns log-density values:
log_p(x) = -.5*sum(x.^2)

# Run 10k iterations of the Adaptive Metropolis:
out = adaptive_rwm(zeros(2), log_p, 10_000; algorithm=:am)

# Calculate '95% credible intervals':
using Statistics
mapslices(x->"$(mean(x)) ± $(1.96std(x))", out.X, dims=2)
```

The function `adaptive_rwm` ('Adaptive Random walk Metropolis') is a simple implenentation
which does sampling for a given log-target density with the chosen method.

## Adaptive parallel tempering

The `adaptive_rwm` also implements tempering, which is used if an optional argument `L≥2` (number of temperature levels) is supplied. Here is a simple multimodal distribution sampled with APT:

```julia
# Multimodal target of dimension d.
function multimodalTarget(d::Int, sigma2=0.1^2, sigman=sigma2)
    # The means of mixtures
    m = [2.18 5.76; 3.25 3.47; 5.41 2.65; 4.93 1.50; 8.67 9.59;
         1.70 0.50; 2.70 7.88; 1.83 0.09; 4.24 8.48; 4.59 5.60;
         4.98 3.70; 2.26 0.31; 8.41 1.68; 6.91 5.81; 1.14 2.39;
         5.54 6.86; 3.93 8.82; 6.87 5.40; 8.33 9.50; 1.69 8.11]'
    n_m = size(m,2)
    @assert d>=2 "Dimension should be >= 2"
    let m=m, n_m=size(m,2), d=d
        function log_p(x::Vector{Float64})
            l_dens = -0.5*(mapslices(sum, (m.-x[1:2]).^2, dims=1)/sigma2)
            if d>2
                l_dens .-= 0.5*mapslices(sum, x[3:d].^2, dims=1)/sigman
            end
            l_max = maximum(l_dens) # Prevent underflow by log-sum trick
            l_max + log(sum(exp.(l_dens.-l_max)))
        end
    end
end

using AdaptiveMCMC
n = 100_000; L = 2
rwm = adaptive_rwm(zeros(2), multimodalTarget(2), n; thin=10)
apt = adaptive_rwm(zeros(2), multimodalTarget(2), div(n,L); L = L, thin=10)

# Assuming you have 'Plots' installed:
using Plots
plot(scatter(rwm.X[1,:], rwm.X[2,:], title="w/o tempering", legend=:none),
scatter(apt.X[1,:], apt.X[2,:], title="w/ tempering", legend=:none), layout=(1,2))
```

## Using with Distributions and LabelledArrays

MCMC is often useful with hierarchical models. These may be conveniently built using `Distributions` and `LabelledArrays` packages. The following example assumes these packages to be installed.

```julia
using Distributions, LabelledArrays, AdaptiveMCMC
# Define convenience log-transform for continuous univariate distributions
struct LogTransformedDistribution{Dist <: ContinuousUnivariateDistribution}
        d::Dist
end
import Distributions.logpdf
logpdf(d::LogTransformedDistribution, x) = logpdf(d.d, exp(x)) + x
import Base.log
log(d::ContinuousUnivariateDistribution) = LogTransformedDistribution(d)

# This example is modified from Turing Getting Started:
# https://turing.ml/dev/docs/using-turing/get-started
function buildModel(x=0.0, y=1.0)
    let x=x, y=y
        function(v)
            p = 0.0
            p += logpdf(log(InverseGamma(2,3)), v.log_s)
            ss = exp(.5*v.log_s)
            p += logpdf(Normal(0, ss), v.m)
            p += logpdf(Normal(v.m, ss), x)
            p += logpdf(Normal(v.m, ss), y)
            p
        end
    end
end

# Initial state vector (labelled with keys `s` and `m`)
x0 = LVector(log_s=1.0, m=0.0); log_p = buildModel(3.3, 4.14)
# Hint: If you do not have a good guess of the mode of log_p (which is
# a good initial value for MCMC), you may use optimisation:
#using Optim; o = optimize(x -> -log_p(x), x0); x0 = o.minimizer
out = adaptive_rwm(x0, log_p, 1_000_000; thin=20)
using StatsPlots # Assuming installed
corrplot(out.X', labels=[keys(x0)...])
```

## Custom sampler

The package provides also simple building blocks which you can use within a 'custom' MCMC sampler. Here is an example:

```julia
using AdaptiveMCMC

# Sampler in R^d
function mySampler(log_p, n, x0)

    # Initialise random walk sampler state: r.x current state, r.y proposal
    r = RWMState(x0)

    # Initialise Adaptive Metropolis state (with default parameters)
    s = AdaptiveMetropolis(x0)
    # Other adaptations are: AdaptiveScalingMetropolis,
    # AdaptiveScalingWithinAdaptiveMetropolis, and RobustAdaptiveMetropolis

    X = zeros(eltype(x0), length(x0), n) # Allocate output storage
    p_x = log_p(r.x)                     # = log_p(x0); the initial log target
    for k = 1:n

        # Draw new proposal r.x -> r.y:
        draw!(r, s)

        p_y = log_p(r.y)                      # Calculate log target at proposal
        alpha = min(one(p_x), exp(p_y - p_x)) # The Metropolis acceptance probability

        if rand() <= alpha
            p_x = p_y

            # This 'accepts', or interchanges r.x <-> r.y:
            # (NB: do not do r.x = r.y; these are (pointers to) vectors!)
            accept!(r)
        end

        # Do the adaptation update:
        adapt!(s, r, alpha, k)

        X[:,k] = r.x   # Save the current sample
     end
    X
end

# Standard normal target for testing
normal_log_p(x) = -mapreduce(e->e*e, +, x)/2

# Run 1M iterations of the sampler targetting 30d standard Normal:
X = mySampler(normal_log_p, 1_000_000, zeros(30))
```

## More information & citations

The algorithms implemented in the package are discussed in the following reference:

* M. Vihola. Ergonomic and reliable Bayesian inference with adaptive Markov chain Monte Carlo. In <i>Wiley StatsRef: Statistics Reference Online</i>, Davidian, M., Kenett, R.S., Longford, N.T., Molenberghs, G., Piegorsch, W.W., and Ruggeri, F. (eds.), Article No. stat08286, 2020.
[doi.org/10.1002/9781118445112.stat08286](https://doi.org/10.1002/9781118445112.stat08286)

The above is also published as the following book chapter (which can also be cited):

* M. Vihola. Bayesian inference with Adaptive Markov chain Monte Carlo. In <i>Computational Statistics in Data Science</i>, Piegorsch, W.W., Levine, R.A., Zhang, H.H., and Lee, T.C.M. (eds.), Chichester: John Wiley & Sons, ISBN: 978-1-119-56107-1, 2022.

Details of the implementation, such as about arguments and options, there are help fields written to the functions, which you may access, for instance, by typing `? adaptive_mcmc` in the Julia REPL.
