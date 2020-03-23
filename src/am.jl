struct AdaptiveMetropolis{d, FT <: AbstractFloat, T <:
    Union{MVector{d,FT}, Vector{FT}}, SST <: StepSize,
    CT <: Cholesky} <: AdaptState
    m::T
    L::CT
    sc::FT
    dx::T
    step::SST
end

"""
    r = AdaptiveMetropolis(x0, [sc, [step]])

Constructor for AdaptiveMetropolis state.

# Arguments
- `x0`: The initial state vector
- `sc`: Scaling parameter; default `2.38/sqrt(d)` where `d` is dimension.
- `step`: Step size object; default `PolynomialStepSize(1.0)`

If `s` is `RWMState`, then proposal samples may be drawn calling
 `draw!(s, r)` and adaptation is performed with `adapt!(r, s)` or
 `adapt_rb!(r, s, α)`.
"""
function AdaptiveMetropolis(x0::T,
                            sc::FT = FT(2.38/sqrt(length(x0))),
                            step = PolynomialStepSize(one(FT))) where {
                            FT <: AbstractFloat, T<:AbstractVector{FT}}
    d = length(x0)
    L = Cholesky(eye(FT,d), :L, 0)
    AdaptiveMetropolis{d,FT,Vector{FT},typeof(step),typeof(L)}(
    Vector{FT}(x0), L, sc, Vector{FT}(x0), step)
end

# Specialisation to StaticArrays
function AdaptiveMetropolis(x0::T,
                            sc::FT = FT(2.38/sqrt(length(x0))),
                            step = PolynomialStepSize(one(FT))) where {d,
                            FT <: AbstractFloat, T<:StaticArray{Tuple{d}, FT}}
    L = Cholesky(MMatrix{d,d}(eye(FT,d)), :L, 0)
    AdaptiveMetropolis{d,FT,MVector{d,FT},typeof(step),typeof(L)}(
      MVector{d,FT}(x0), L, sc, MVector{d,FT}(x0), step)
end

@inline adapt!(sa, sr, α, k) = adapt!(sa, sr, k)
# Update function -- not yet implemented in the best possible way...
@inbounds function adapt!(sa::AdaptiveMetropolis{d},
      sr::RWMState{d}, k::Real) :: Nothing where {d}
    gamma = get(sa.step, k)
    sa.dx .= sr.x .- sa.m
    sa.m .+= gamma.*sa.dx
    rmul!(sa.dx, sqrt(gamma))
    rmul!(sa.L.factors, sqrt(1.0-gamma))
    lowrankupdate!(sa.L, sa.dx)
    nothing
end

@inbounds function draw!(sr::RWMState{d},
      s::AdaptiveMetropolis{d}) :: Nothing where {d}
    #draw!(sr, s.sc[], s.L.L)
    draw!(sr, s.L, s.sc)
    nothing
end

"""
adapt_rb!(s, r, α)

Rao-Blackwellised adaptation step of Adaptive Metropolis as suggested by
Andrieu & Thoms (Statist. Comput. 2008).

# Arguments:
- `s`: AdaptiveMetropolis object
- `r`: RWMState object
- `α`: Acceptance rate

NB: This function should be called *before* calling accept!(r).
"""
function adapt_rb!(sa::AdaptiveMetropolis{d},
      sr::RWMState{d}, α::FT, k::Real) :: Nothing where {d, FT<:AbstractFloat}
    gamma = get(sa.step, k)
    wx = (one(FT)-α)*gamma
    wy = α*gamma
    one_gamma = one(FT) - gamma
    # Update covariance Cholesky factor:
    rmul!(sa.L.factors, sqrt(one_gamma))
    sa.dx .= sr.x .- sa.m
    rmul!(sa.dx, sqrt(wx))
    lowrankupdate!(sa.L, sa.dx)
    sa.dx .= sr.y .- sa.m
    rmul!(sa.dx, sqrt(wy))
    lowrankupdate!(sa.L, sa.dx)
    # Update mean:
    sa.m .= one_gamma.*sa.m .+ wx.*sr.x .+ wy.*sr.y
    nothing
end
