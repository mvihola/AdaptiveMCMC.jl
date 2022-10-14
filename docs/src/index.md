# Introduction

This package provides implementations of some general-purpose random-walk based
adaptive MCMC algorithms, including the following:

* Adaptive Metropolis, proposal covariance adaptation, ([Haario, Saksman and Tamminen, 2001](https://projecteuclid.org/euclid.bj/1080222083), and [Andrieu and Moulines, 2006](http://dx.doi.org/10.1214/105051606000000286))
* Adaptive scaling Metropolis, acceptance rate adaptation for scale (e.g. as in [Andrieu and Thoms, 2008](https://doi.org/10.1007/s11222-008-9110-y), and [Atchadé and Fort, 2010](https://projecteuclid.org/euclid.bj/1265984706))
* Robust Adaptive Metropolis, acceptance rate adaptation for shape [(Vihola, 2012)](http://dx.doi.org/10.1007/s11222-011-9269-5)
* Adaptive Parallel Tempering, acceptance rate adaptation for temperature levels [(Miasojedow, Moulines and Vihola, 2013)](http://dx.doi.org/10.1080/10618600.2013.778779)

The aim of the package is to provide a simple and modular general-purpose implementation, which may be easily used to sample from a log-target density, but also used in a variety of custom settings.

See also [AdaptiveParticleMCMC.jl](https://github.com/mvihola/AdaptiveParticleMCMC.jl) which uses this package with [SequentialMonteCarlo.jl](https://github.com/awllee/SequentialMonteCarlo.jl) for adaptive particle MCMC.

## Installation

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

## Sampling from log-posteriors

The package provides an easy-to-use adaptive random-walk Metropolis sampler, which samples (in principle) from any probability distribution $p$, whose log-density values can be evaluated point-wise.

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

See [Adaptation state](@ref) for explanation of the different `algorithm` options:

* `:am` = `AdaptiveMetropolis`
* `:ram` = `RobustAdaptiveMetropolis`
* `:asm` = `AdaptiveScalingMetropolis`
* `:aswam` = `AdaptiveScalingWithinAdaptiveMetropolis`

There are a number of other optional keyword arguments, too:

```@docs
adaptive_rwm
```

<!--The `adaptive_rwm` uses a [Random-walk sampler state](@ref) and an [Adaptation state](@ref) chosen based on the `algorithm` option.-->

## With adaptive parallel tempering

If the keyword argument `L` is greater than one, then the adaptive parallel tempering algorithm (APT) of [Miasojedow, Moulines & Vihola (2013)](http://dx.doi.org/10.1080/10618600.2013.778779) is used. This can greatly improve mixing with multimodal distributions.

Here is a simple multimodal distribution sampled with normal adaptive random walk Metropolis, and with APT:

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

What the APT is actually based on? Parallel tempering is a MCMC algorithm which samples from a product density proporitional to:
```math
\prod_{i=1}^L p^{\beta(i)}(x^{(i)}),
```
where (the 'inverse temperatures') $1 = \beta(1) > \beta(2) > \cdots > \beta(L) > 0$.

In the end, the 'first level' is of interest (and samples of the first level are usually used for estimation), whereas the tempered levels $i=2,\ldots,L$ are auxiliary, which help the sampler to move between modes of a multi-modal target. The easier moving is because the tempered densities $p^{\beta(i)}$ are 'flatter' than $p$ for any $\beta(i)<1$.

The sampler consists of two types of MCMC moves:

* Independent adaptive random-walk Metropolis moves on individual levels $i$, targetting tempered densities $p^{\beta(i)}$. 
* Switch moves, where swaps of adjacent levels $x^{(i)} \leftrightarrow x^{(i+1)}$ are proposed, and the moves are accepted with (Metropolis-Hastings) probability $\min\big\{1, \frac{p^{\beta(i)-\beta(i+1)}(x^{(i+1)})}{p^{\beta(i)-\beta(i+1)}(x^{(i)})}\big\}$.

In the APT, each random walk sampler for each individual level is adapted totally independently, following exactly the same mechanism as before. Additionally, the APT adapts the inverse temperatures $\beta(2),\ldots,\beta(L)$, in order to reach the average switch probability $0.234$. More precisely, adaptation mechanism tunes the parameters $\rho^{(i)}$, which determine
```math
\frac{1}{\beta^{(i)}} = \frac{1}{\beta^{(i-1)}} + e^{\rho^{(i)}},
```
and the adaptation is similar to [Adaptive scaling Metropolis](@ref): if swap $x^{(i-1)}\leftrightarrow x^{(i)}$ is proposed the $k$:th time, the parameter is updated as follows:
```math
\rho_k^{(i)} = \rho_{k-1}^{(i)} + \gamma_k (\alpha_k^{(\text{swap }i)} - \alpha_*),
```
where $\alpha_k^{(\text{swap }i)}$ is the swap probability.

