# Sampling from log-posteriors

The package provides an easy-to-use adaptive random-walk Metropolis sampler, which samples (in principle) from any probability distribution $p$, whose log-density values can be evaluated point-wise.

```@docs
adaptive_rwm
```

The `adaptive_rwm` uses a [Random-walk sampler state](@ref) and an [Adaptation state](@ref) chosen based on the `algorithm` option.

If the parameter `L` is greater than one, then the adaptive parallel tempering algorithm (APT) of [Miasojedow, Moulines & Vihola (2013)](http://dx.doi.org/10.1080/10618600.2013.778779) is used. 

Parallel tempering is a MCMC algorithm which samples from a product density for the following form:
```math
\prod_{i=1}^L p^{\beta(i)}(x^{(i)}),
```
where $1 = \beta(1) > \beta(2) > \cdots > \beta(L) > 0$. Here, the 'first level' is the level of interest, whereas the tempered levels $i=2,\ldots,L$ are auxiliary, and help the sampler switch between modes of a multi-modal target.

The sampler consists of two types of MCMC moves:

* Independent random-walk moves on individual levels $i$, targetting tempered densities $p^{\beta(i)}$.
* Switch moves, where swaps $x^{(i)} \leftrightarrow x^{(i+1)}$ are proposed, and accepted with probability $\min\big\{1, \frac{p\big\}$.

The APT adapts the independent random-walk samplers on levels $i=1,\ldots,L$, and also tunes the $\beta(2),\ldots,\beta(L)$ during simulation, so that the average switch probabilities tend to the desired level $0.234$.

