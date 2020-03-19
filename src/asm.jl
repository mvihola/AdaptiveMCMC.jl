# Adaptive scaling Metropolis
struct AdaptiveScalingMetropolis{StepT <: StepSize, FT <: AbstractFloat} <: AdaptState
    sc::Base.RefValue{FT} # The log-scaling which is adapted
    α_opt::FT             # The desired acceptance rate
    step::StepT           # The step size sequence
end

"""
    r = AdaptiveScalingMetropolis(x0, α_opt, sc, step)
    r = AdaptiveScalingMetropolis(α_opt, sc, step)

Constructor for AdaptiveScalingMetropolis state.

# Arguments
- `x0`: The initial state vector (not used).
- `α_opt`: Desired mean accept rate; default `0.234`.
- `sc`: Initial scaling; default `1.0`.
- `step`: Step size object; default `PolynomialStepSize(0.66)`.

If `s` is `RWMState`, then proposal samples may be drawn calling
 `draw!(s, r)` and adaptation is performed with `adapt!(r, s, α)`.
"""
function AdaptiveScalingMetropolis(α_opt::FT=0.234, sc::FT=FT(1.0),
      step=PolynomialStepSize(FT(0.66))) where {FT <: AbstractFloat}
    AdaptiveScalingMetropolis(Ref(log(sc)), α_opt, step)
end
function AdaptiveScalingMetropolis(x0::VT,
      α_opt::FT=length(x0)==1 ? FT(0.44) : FT(0.234),
      sc::FT=FT(1.0), step=PolynomialStepSize(FT(0.66))) where {FT <: AbstractFloat, VT <: AbstractVector{FT}}
    AdaptiveScalingMetropolis(α_opt, sc, step)
end

@inline function adapt!(s::AdaptiveScalingMetropolis, sr::RWMState,
      α::AbstractFloat, k::Real) :: Nothing
    dα = α - s.α_opt
    γ = get(s.step, k)
    s.sc[] += γ*dα
    nothing
end
@inline function draw!(sr::RWMState, sa::AdaptiveScalingMetropolis) :: Nothing
    draw!(sr, exp(sa.sc[]))
    nothing
end

# Adaptive scaling within AM
struct AdaptiveScalingWithinAdaptiveMetropolis{d,AMT <: AdaptiveMetropolis,
      ASMT <: AdaptiveScalingMetropolis} <: AdaptState
    AM::AMT
    ASM::ASMT
end

"""
    r = AdaptiveScalingWithinAdaptiveMetropolis(x0, α_opt, sc, stepAM, stepASM)

Constructor for AdaptiveScalingWithinAdaptiveMetropolis state.

# Arguments
- `x0`: The initial state vector.
- `α_opt`: Desired mean accept rate; default `0.234`.
- `sc`: Initial scaling; default `2.38/sqrt(d)` where `d` is the dimension.
- `stepAM`: Step size object for covariance adaptation;
             default `PolynomialStepSize(0.66)`.
- `stepASM`: Step size object for scaling adaptation;
             default `PolynomialStepSize(0.66)`.

If `s` is `RWMState`, then proposal samples may be drawn calling
 `draw!(s, r)` and adaptation is performed with `adapt!(r, s, α)` or
 `adapt_rb!(r, s, α)`.
"""
function AdaptiveScalingWithinAdaptiveMetropolis(x0::T,
      α_opt=length(x0)==1 ? FT(0.44) : FT(0.234),
      sc=FT(2.38/sqrt(length(x0))),
      stepAM::StepSize=PolynomialStepSize(FT(0.66)),
      stepASM::StepSize=PolynomialStepSize(FT(0.66))) where {FT <: AbstractFloat,
      T<:AbstractVector{FT}}
    d = length(x0)
    ASM = AdaptiveScalingMetropolis(α_opt, sc, stepASM)
    AM = AdaptiveMetropolis(x0, one(FT), stepAM)
    AdaptiveScalingWithinAdaptiveMetropolis{d,typeof(AM),typeof(ASM)}(AM, ASM)
end

@inline function adapt!(s::AdaptiveScalingWithinAdaptiveMetropolis,
      sr::RWMState, α::AbstractFloat, k::Real) :: Nothing
    adapt!(s.AM, sr, α, k)
    adapt!(s.ASM, sr, α, k)
    nothing
end
@inline function draw!(sr::RWMState,
      s::AdaptiveScalingWithinAdaptiveMetropolis) :: Nothing
    draw!(sr, s.AM.L, exp(s.ASM.sc[]))
end

@inline function adapt_rb!(s::AdaptiveScalingWithinAdaptiveMetropolis,
      sr::RWMState, α::AbstractFloat, k::Real) :: Nothing
    adapt_rb!(s.AM, sr, α, k)
    adapt!(s.ASM, sr, α, k)
    nothing
end
