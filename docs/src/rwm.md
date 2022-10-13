
# Random-walk sampler state

The state of a random-walk Metropolis sampler is encapsulated in the data structure `RWMState`, which can be constructed as follows:

```@docs
RWMState
```

By default, the `RWMState` will provide a Gaussian random walk sampler, as the 'prototype' increment proposal density `q!` corresponds to a standard normal $N(0,I)$. The increment proposal density can be any other distribution, as long as it is symmetric: it is as likely to produce a random vector $z$ as it is to produce $-z$.

## Forming a proposal

Assuming `s=RWMState`, we can use two methods. The first is `draw!`, which forms a proposal. This means that starting from $X_{k-1}=$`s.x`, we produce
```math
Y_k \sim X_{k-1} + C U_k, \quad U_k \sim q,
```
where $q$ corresponds to the distribution `s.q!` simulates from. The increment vector $U_k$ is stored to an internal `s.u` (which is used by some of the adaptive algorithms), and the proposal $Y_k$ is stored to `s.y`.

The factor $C$ can be a scalar, which controls the increment scale, or a scalar times Cholesky factor, which controls the shape of the increment distribution.

```@docs
draw!
```

## Forming a proposal with adaptive state

Calling `draw(r::RWMState, s::AdaptState)`, where `s` is one of the [Adaptation state](@ref) will use the adapted scale/shape when forming the proposal.

## Accepting a proposal

After `draw!`, the proposal `s.x` can be accessed (and for instance copied to storage), but should not be modified directly. Instead, the 'acceptance', which means that `s.y` will replace `s.x` should be done as follows:

```@docs
accept!
```
