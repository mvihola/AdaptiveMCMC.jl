
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

The `adapt!` method updates the adapted state, based on the state of the random walk sampler, and the acceptance probability of the step $\alpha$.

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

In fact, the $\Sigma_k$ are not calculated as above, but their Cholesky factors are updated directly using rank-1 updates. This is more efficient, as calculating a Cholesky factor is $O(d^3)$ but rank-1 updates are $O(d^2)$. The empirical covariances $\Sigma_k$ converge to the true covariance of the target distribution (under technical conditions).

## Robust adaptive Metropolis

```@docs
RobustAdaptiveMetropolis
```

This is the Robust Adaptive Metropolis (RAM) of [Vihola, 2010](http://dx.doi.org/10.1007/s11222-011-9269-5). In this algorithm, the shape $S_{k}$ is adapted directly by Cholesky rank-1 updates:
```math
S_k S_k^T = S_{k-1} \bigg( I + \gamma_k (\alpha_k - \alpha_*) \frac{U_k U_k^T}{\| U_k \|^2} \bigg) S_{k-1},
```
where $\alpha_k$ is the acceptance probability of the move $X_{k-1} \to Y_k$, and $U_k$ is the 'prototype proposal' used when forming the proposal $Y_k$, and $\alpha_*\in(0,1)$ is the desired acceptance rate, defaulting to $0.234$.

In the case of finite-variance elliptically symmetric target distribution (and technical conditions), the shape parameter $S_k$ converge to a Cholesky factor of a matrix which is proportional to the covariance. However, in general, the shapes of RAM and AM can differ.

## Adaptive scaling Metropolis

```@docs
AdaptiveScalingMetropolis
```

## Adaptive scaling within adaptive Metropolis

```@docs
AdaptiveScalingWithinAdaptiveMetropolis
```

## 'Rao-Blackwellised' update

The states of `AdaptiveMetropolis` and `AdaptiveScalingWithingAdaptiveMetropolis` also support the alternative adaptation suggested by implemented by:

```@docs
adapt_rb!
```
