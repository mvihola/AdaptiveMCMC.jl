# Step sizes

All adaptation algorithms are based on similar sequential stochastic gradient like (or [Robbins-Monro](https://en.wikipedia.org/wiki/Stochastic_approximation#Robbins%E2%80%93Monro_algorithm)) updates, which rely in decreasing step size or 'learning rate' sequence $\gamma_k$.

The `AdaptiveMCMC.jl` uses an abstract `StepSize` type. Each concrete subtype should have a method `get(stepsize, k)`, which returns $\gamma_k$ corresponding to `stepsize`.

The current implementation `AdaptiveMCMC.jl` implements essentially only the step size of the form:
```math
   \gamma_k = c k^{-\eta}
```
where $c>0$ and $1/2 < \eta\le 1$ are the two parameters of the sequence. (The given range for $\eta$ ensures that $\sum_k \eta_k = \infty$ and $\sum_k \eta_k^2 <\infty$, which are desirable properties for the step size sequence...)

```@docs
PolynomialStepSize
```

There is also a variant of the `PolynomialStepSize` for the RAM: 
```math
   \gamma_k = \min\{1, d k^{-\eta}\},
```
where $d$ is the state dimension. The RAM step size can be constructed as follows:
```@docs
RAMStepSize
```

