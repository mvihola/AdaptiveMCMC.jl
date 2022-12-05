
# Adaptation state

The state of each adaptive algorithms is encapsulated to a subtype of abstract type `AdaptState`.
Each adaptive algorithm implements two methods

* `draw!(::RWMState, ::AdaptState)`
* `adapt!(::AdaptState, ::RWMState, α)`

The `draw!` method forms proposals of the form
```math
Y_k = X_{k-1} + S_{k-1} U_k, \quad U_k \sim q,
```
where $q$ is the 'prototype proposal' as discussed in [Random walk sampler state](@ref), and the form of the 'shape' matrix factor $S_{k-1}$ depends on the chosen adaptive algorithm.

The `adapt!` method updates the adapted state, based on the state of the random walk sampler. Some of the adaptation algorithms use the third argument of `adapt!`: the acceptance probability $\alpha\in[0,1]$.

## Adaptive Metropolis

```@docs
AdaptiveMetropolis
```

This is the seminal Adaptive Metropolis (AM) algorithm of [Haario, Saksman & Tamminen (2001)](https://doi.org/10.2307/3318737), where the proposal increment shape is
```math
S_{k-1} + s_d L_{k-1},
```

where $s_d$ is a fixed scaling defaulting to $s_d = \frac{2.38}{\sqrt{d}}$, where $d$ is the dimension, and $L_{k-1}$ is the Cholesky factor of estimated covariance matrix.

The covariance matrix is estimated using the recursive Robbins-Monro stochastic approximation update of [Andrieu & Moulines (2006)](https://doi.org/10.1214/105051606000000286), that is,
```math
\begin{aligned}
\mu_k &= (1-\gamma_k)\mu_{k-1} + \gamma_k X_k, \\
\Sigma_k &= (1-\gamma_k)\Sigma_{k-1} + \gamma_k (X_k - \mu_{k-1})(X_k - \mu_{k-1})^T,
\end{aligned}
```
where $\gamma_k$ is one of the [Step sizes](@ref).

In fact, the $\Sigma_k$ fators are not calculated directly as given above, but their Cholesky factors are updated sequentially, using rank-1 updates. This is more efficient, as calculating a full Cholesky factor is $O(d^3)$ but rank-1 updates are $O(d^2)$.

The empirical covariances $\Sigma_k$ of the AM converge to the true target covariance (under technical conditions).

## Robust adaptive Metropolis

```@docs
RobustAdaptiveMetropolis
```

This is the Robust Adaptive Metropolis (RAM) of [Vihola, 2010](http://dx.doi.org/10.1007/s11222-011-9269-5). In this algorithm, the shape $S_{k}$ is adapted directly by Cholesky rank-1 updates:
```math
S_k S_k^T = S_{k-1} \bigg( I + \gamma_k (\alpha_k - \alpha_*) \frac{U_k U_k^T}{\| U_k \|^2} \bigg) S_{k-1},
```
where $\alpha_k$ is the acceptance probability of the move $X_{k-1} \to Y_k$, and $U_k$ is the 'prototype proposal' used when forming the proposal $Y_k$, and $\alpha_*\in(0,1)$ is the desired acceptance rate, defaulting to $0.234$.

In the case of finite-variance elliptically symmetric target distribution (and technical conditions), the shape parameter $S_k$ converges to a Cholesky factor of a matrix which is proportional to the covariance. However, in general, the limiting shapes of the RAM and the AM can differ.

The RAM seems to have empirical advantage over AM in certain cases, such as higher dimensions, but RAM is not directly applicable for instance with [Pseudo-marginal](https://en.wikipedia.org/wiki/Pseudo-marginal_Metropolis%E2%80%93Hastings_algorithm) algorithms.

## Adaptive scaling Metropolis

```@docs
AdaptiveScalingMetropolis
```

Adaptive scaling Metropolis (ASM) implements a version of adaptive scaling first suggested in the [Andrieu and Robert, 2001](https://crest.science/RePEc/wpstorage/2001-33.pdf) preprint, but the version implemented by `AdaptiveMCMC.jl` is due to [Andrieu and Thoms, 2008](https://doi.org/10.1007/s11222-008-9110-y) and [Atchadé and Fort, 2010](https://projecteuclid.org/euclid.bj/1265984706).

In the ASM, the adaptation parameter $S_k$ is a scalar, and it has the following dynamic:
```math
\log S_k = \log S_{k-1}  + \gamma_k (\alpha_k - \alpha_*).
```
The update has similarities with RAM, which may be regarded as a multivariate generalisation of this rule.

## Adaptive scaling within adaptive Metropolis

```@docs
AdaptiveScalingWithinAdaptiveMetropolis
```
The adaptive scaling within adaptive Metropolis combines the AM and ASM, as suggested (at least) in 
[Andrieu and Thoms, 2008](https://doi.org/10.1007/s11222-008-9110-y). That is:

* The scaling $\theta_k$ is updated as in the ASM
* The covariance $\Sigma_k$ is updated as in AM, with Cholesky factor $L_k$
* The scaling parameter is both combined: $S_k = \theta_k L_k$

This algorithm is often more robust than AM, but often outperformed by a simpler RAM.

## 'Rao-Blackwellised' update

The covariance states of `AdaptiveMetropolis` and `AdaptiveScalingWithingAdaptiveMetropolis` also support the alternative adaptation suggested in [Andrieu and Thoms, 2008](https://doi.org/10.1007/s11222-008-9110-y). The alternative update call is the following:

```@docs
adapt_rb!
```

This update is not based on new, accepted state, but involves a convex combination of the current and proposed states, weighted by the acceptance probability:
```math
\begin{aligned}
\mu_k &= (1-\gamma_k)\mu_{k-1} + \gamma_k \big[(1-\alpha_k) X_{k-1} + \alpha_k Y_k\big], \\
\Sigma_k &= (1-\gamma_k)\Sigma_{k-1} + \gamma_k \big[(1-\alpha_k)(X_{k-1} - \mu_{k-1})(X_{k-1} - \mu_{k-1})^T + \alpha_k (Y_{k} - \mu_{k-1})(Y_{k} - \mu_{k-1})^T\big].
\end{aligned}
```
This update is 'Rao-Blackwellised' in the sense that it may be regarded as a conditional expectation of the AM adaptation rule, integrating the accept/reject decision. The Rao-Blackwellised rule can stabilise the covariance adaptation slightly.
