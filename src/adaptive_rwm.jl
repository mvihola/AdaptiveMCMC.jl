# Container for log-density values
mutable struct PVals{FT <: AbstractFloat}
    x::FT
    pr_x::FT
end

# Initialise random walk adaptation
function init_rwm_adapt(algorithm, x0)
    if algorithm==:ram
        s = RobustAdaptiveMetropolis(x0)
    elseif algorithm==:am
        s = AdaptiveMetropolis(x0)
    elseif algorithm==:asm
        s = AdaptiveScalingMetropolis(x0)
    elseif algorithm==:aswam
        s = AdaptiveScalingWithinAdaptiveMetropolis(x0)
    else
        error("No such algorithm: $algorithm")
    end
    s
end

# Initialise temperature adaptation based on swap rates
function init_sw_adapt(L::Int, alpha_opt::FT=0.234, sc=FT(1.0),
    gamma=FT(0.66)) where {FT <: AbstractFloat}
    step = PolynomialStepSize(gamma, FT(L-1))
    [AdaptiveScalingMetropolis(alpha_opt, sc, step) for _ in 1:L-1]
end

# Update inverse temperatures based on adapted parameters
@inline function rho2beta!(Betas::Vector{FT}, Rhos) :: Nothing where {
    FT <: AbstractFloat}
    T = one(FT)
    Betas[1] = T
    @inbounds for i = 1:length(Rhos)
        T += exp(Rhos[i].sc[])
        Betas[i+1] = one(FT)/T
    end
    nothing
end

# Single random walk update & possible adaptation
@inline function arwm_step!(r::RWMState, s::AdaptState, p::PVals{FT},
    beta::FT, log_p::Function, log_pr::Function, k::Int, do_adapt::Bool) :: Int where {
    FT <: AbstractFloat }
    # Draw proposal, evaluate log-target, acc.rate
    draw!(r, s)
    p_y = log_p(r.y); p_pr_y = log_pr(r.y)
    dp = beta*(p_y - p.x) + p_pr_y - p.pr_x
    #alpha = min(one(FT), exp(dp))
    rxy = exp(dp); alpha = (rxy<one(FT)) ? rxy : one(FT) # This makes r=NaN -> alpha=1
    if rand(r.rng) <= alpha
        # Accept: interchange current/proposed states & log-target values
        accept!(r)
        p.x = p_y; p.pr_x = p_pr_y
        acc = 1
    else
        acc = 0
    end
    if do_adapt
        adapt!(s, r, alpha, k)
    end
    acc
end

# Attempt swap move between levels lev_ <-> lev & possible adaptation
@inline function swap_step!(R, Rhos, P, Betas::Vector{FT}, lev_::Int, lev::Int,
    k::Real, do_adapt::Bool) :: Int where {FT <: AbstractFloat}
    p_ = P[lev_]; beta_ = Betas[lev_]
    p  = P[lev];  beta = Betas[lev]

    # Acceptance probability:
    dp = (beta_ - beta) * (p.x - p_.x)
    #alpha = min(one(FT), exp(dp))
    rxy = exp(dp); alpha = (rxy<one(FT)) ? rxy : one(FT) # This makes r=NaN -> alpha=1

    # Accept swap:
    if rand(R[lev_].rng) <= alpha
        R[lev], R[lev_] = R[lev_], R[lev]
        P[lev], P[lev_] = p_, p
        acc = 1
    else
        acc = 0
    end

    # Do swap adaptation:
    if do_adapt
        adapt!(Rhos[lev_], R[lev_], alpha, k)
        rho2beta!(Betas, Rhos)
    end
    acc
end

# Initialise the key variables in the adaptive random walk Metropolis algoritm
@inline function init_arwm(x0::T, algorithm, rng, q, L, all_levels, nX,
                           log_p, log_pr) where {FT <: AbstractFloat,
                           T <: AbstractVector{FT}}
    d = length(x0)
    # RWM states, adaptation states...
    R = [RWMState(x0, rng, q) for i = 1:L]
    P = Vector{PVals{FT}}(undef, L)
    for lev = 1:L
        r = R[lev]
        P[lev] = PVals(log_p(r.x), log_pr(r.x))
    end
    
    # Initialise adaptation
    if typeof(algorithm) == Symbol
        S = [init_rwm_adapt(algorithm, x0) for _ in 1:L]
    else
        S = algorithm
    end
    l = all_levels ? L : 1
    # Initialise output variables
    X = [Matrix{FT}(undef, d, nX) for _ in 1:l]
    D = [Vector{FT}(undef, nX) for _ in 1:l]
    X, D, R, S, P
end

"""
    out = adaptive_rwm(x0, log_p, n; kwargs)

Generic adaptive random walk Metropolis algorithm from initial state vector
`x0` targetting log probability density `log_p` run for `n` iterations,
including adaptive parallel tempering.

# Arguments
- `x0::Vector{<:AbstractFloat}`: The initial state vector
- `log_p::Function`: Function that returns log probability density values
                     (up to an additive constant) for any state vector.
- `n::Int`: Total number of iterations

# Keyword arguments
- `algorithm::Symbol`: The random walk adaptation algorithm; current choices are
  `:ram` (default), `:am`, `:asm`, `:aswam` and `:rwm`.
  (Alternatively, if algorithm is a vector of AdaptState, then this will be used as an
  initial state for adaptation.)
- `b::Int`: Burn-in length: `b`:th sample is the first saved sample. Default `⌊n/5⌋`
- `thin::Int`: Thinning factor; only every `thin`:th sample is stored; default `1`
- `fulladapt::Bool`: Whether to adapt after burn-in; default `true`
- `Sp`: Saved adaptive state from output to restart MCMC; default `nothing`
- `Rp`: Saved rng state from output to restart MCMC; default `nothing`
- 'indp`: Index of saved adaptive state to restart MCMC; default `0`
- `rng::AbstractRNG`: Random number generator; default `Random.GLOBAL_RNG`
- `q::Function`: Zero-mean symmetric proposal generator (with arguments `x` and `rng`);
   default `q=randn!(x, rng)`
- `L::Int`: Number of parallel tempering levels
- `acc_sw::AbstractFloat`: Desired acceptance rate between level swaps; default `0.234`
- `all_levels::Bool`: Whether to store output of all levels; default `false`
- `log_pr::Function`: Log-prior density function; default `log_pr(x) = 0.0`.
- `swaps::Symbol`: Swap strategy, one of:
   `:single` (default, single randomly picked swap)
   `:randperm` (swap in random order)
   `:sweep` (up- or downward sweep, picked at random)
   `:nonrev` (alternate even/odd sites as in Syed, Bouchard-Côté, Deligiannidis,
   Doucet, 	arXiv:1905.02939)

Note that if `log_pr` is supplied, then `log_p(x)` is regarded as the
log-likelihood (or, equivalently, log-target is `log_p(x) + log_pr(x)`).
Tempering is only applied to `log_p`, not to `log_pr`.

The output `out.X contains the simulated samples (column vectors).
`out.allX[k]` for `k>=2` contain higher temperature auxiliary chains (if requested)

# Examples
```
log_p(x) = -.5*sum(x.^2)
o = adaptive_rwm(zeros(2), log_p, 10_000; algorithm=:am)
using MCMCChains, StatsPlots # Assuming MCMCChains & StatsPlots are installed...
c = Chains(o.X[1]', start=o.params.b, thin=o.params.thin); plot(c)
```
"""
function adaptive_rwm(x0::T, log_p::Function, n::Int;
    algorithm::Union{Symbol,Vector{<:AdaptState}}=:ram,
    thin::Int=1, b::Int=max(1,Int(floor(n/5))), fulladapt::Bool=true, 
    Sp=nothing, Rp=nothing, indp=nothing,
    q::Function=randn!, L::Int=1, log_pr::Function = (x->zero(FT)),
    all_levels::Bool=false, acc_sw::FT = FT(0.234), swaps::Symbol = :single,
    rng::AbstractRNG=Random.GLOBAL_RNG) where {FT <: AbstractFloat,
    T <: AbstractVector{FT}}

    args = (x0, log_p, n)
    params = (algorithm=algorithm, thin=thin, b=b, fulladapt=fulladapt,
              q=q, L=L, log_pr=log_pr, all_levels=all_levels, acc_sw=acc_sw,
              swaps=swaps, rng=copy(rng))

    # Initialise key variables
    nX = length(b:thin:n) # Number of output
    X, D, R, S, P = init_arwm(x0, algorithm, rng, q, L,
                              all_levels, nX, log_p, log_pr)

    # Restore adaptation & sampler states
    if Sp != nothing
        S = Sp
    end
    if Rp != nothing
        R = Rp
    end
    if indp == nothing
        indp = 0
        if Sp != nothing
            warning("When you restart from a previous sampler state, please also supply the index of the sampler state `indp`. Otherwise, repeated restarts might lead to biased algorithm.")
        end
    end

    adaptive_rwm_(X, D, R, S, P, args, params, x0, log_p, n,
        thin, b, fulladapt, indp,
        L, log_pr,
        all_levels, acc_sw, swaps,
        rng)
end

function adaptive_rwm_(X, D, R, S, P, args, params, x0::T, log_p::Function, n::Int,
    thin::Int, b::Int, fulladapt::Bool, indp::Int,
    L::Int, log_pr::Function,
    all_levels::Bool, acc_sw::FT, swaps::Symbol,
    rng::AbstractRNG) where {FT <: AbstractFloat,
    T <: AbstractVector{FT}}

    # Acceptance statistics
    accRWM = zeros(Int, L)  # How many are accepted
    accSW = zeros(Int, L-1); nSW = zeros(Int, L-1)

    # Shorthands for all levels, swap levels, stored levels
    allLevels = 1:L
    oddLevels = 1:2:L-1; evenLevels = 2:2:L-1
    upSweep = 1:1:(L-1); downSweep = L-1:-1:1
    if swaps == :single
        sc_sw = FT(L-1)
    elseif swaps == :randperm
        sc_sw = one(FT)
        sweepLevels = Vector{Int}(undef, L-1)
    elseif swaps == :nonrev
        sc_sw = one(FT)*2
    else # :sweep
        sc_sw = one(FT)
    end
    Rhos = init_sw_adapt(L, acc_sw, sc_sw)

    # Which ones are stored
    storeLevels = all_levels ? (1:L) : (1:1)

    # Initialise inverse temperatures
    Betas = zeros(FT, L)
    rho2beta!(Betas, Rhos)

    for k = 1:n
        # The 'real' index, to ensure valid adaptation with restarts
        k_real = k + indp

        # Random walk moves:
        for lev = allLevels
            accRWM[lev] += arwm_step!(R[lev], S[lev], P[lev], Betas[lev],
                                 log_p, log_pr, k_real, fulladapt || k <= b)
        end

        # Swap move between random adjacent levels
        if L>1
            if swaps == :single
                lev = rand(rng, upSweep);
                nSW[lev] += 1
                accSW[lev] += swap_step!(R, Rhos, P, Betas, lev, lev+1,
                                      k_real/L, fulladapt || k <= b)
            else
                if swaps == :nonrev
                    if k_real % 2 == 1
                        sweepLevels = oddLevels
                    else
                        sweepLevels = evenLevels
                    end
                elseif swaps == :randperm
                    randperm!(rng, sweepLevels)
                else # :sweep
                    if rand(rng) < 0.5
                        sweepLevels = upSweep
                    else
                        sweepLevels = downSweep
                    end
                end
                for lev in sweepLevels
                    nSW[lev] += 1
                    accSW[lev] += swap_step!(R, Rhos, P, Betas, lev, lev+1,
                                          k_real, fulladapt || k <= b)
                end
            end
        end

        # Store output
        for lev in storeLevels
            if k>=b && rem(k-b, thin) == 0
                i = Int((k-b)/thin)+1
                X[lev][:,i] .= R[lev].x
                D[lev][i] = Betas[lev]*P[lev].x + P[lev].pr_x
            end
        end
    end

    (X=X[1], allX=X, D=D, R=R, S=S, Rhos=Rhos, accRWM=accRWM/n, accSW=accSW./nSW,
    args=args, params=params)
end
