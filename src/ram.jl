struct RobustAdaptiveMetropolis{d, FT<:AbstractFloat, T <:
    Union{MVector{d,FT}, Vector{FT}}, SST<:StepSize,
    CT<:Cholesky{FT}} <: AdaptState
    L::CT
    α_opt::FT
    dx::T
    n_u::T
    step::SST
end

"""
    r = RobustAdaptiveMetropolis(x0, [α_opt, [step]])

Constructor for RobustAdaptiveMetropolis state.

# Arguments
- `x0`: The initial state vector
- `α_opt`: Desired mean accept rate; default `0.234`.
- `step`: Step size object; default `RAMStepSize(0.66,d)` where
  `d` is state dimension.

If `s` is `RWMState`, then proposal samples may be drawn calling
 `draw!(s, r)` and adaptation is performed with `adapt!(r, s, α)`.
"""
function RobustAdaptiveMetropolis(x0::T, α_opt::FT=FT(0.234),
  step::StepSize=RAMStepSize(FT(0.66),length(x0))) where {
      FT <: AbstractFloat, T <: AbstractVector{FT}}
    d = length(x0)
    L = Cholesky(eye(FT, d), :L, 0)
    dx = zeros(FT, d)
    n_u = zeros(FT, d)
    RobustAdaptiveMetropolis{d,FT,typeof(dx),typeof(step),typeof(L)}(L,
      α_opt, dx, n_u, step)
end

function RobustAdaptiveMetropolis(x0::T, α_opt::FT=FT(0.234),
  step::StepSize=RAMStepSize(FT(0.66),d)) where {d,
      FT <: AbstractFloat, T<:StaticArray{Tuple{d}, FT}}
    L = Cholesky(MMatrix{d,d}(eye(FT, d)), :L, 0)
    dx = zeros(MVector{d,FT})
    n_u = zeros(MVector{d,FT})
    RobustAdaptiveMetropolis{d,FT,typeof(dx),typeof(step),typeof(L)}(L,
      α_opt, dx, n_u, step)
end

@inline function adapt!(s::RobustAdaptiveMetropolis{d}, sr::RWMState{d},
      α::FT, k::Real) :: Nothing where {d, FT <: AbstractFloat}
    γ::FT = get(s.step, k)
    norm_u::FT = BLAS.nrm2(sr.u)
    norm_u = (norm_u == zero(FT)) ? one(FT) : norm_u

    s.n_u .= sr.u ./ norm_u
    #copy!(s.n_u, sr.u)
    #lmul!(one(FT)/norm_u, s.n_u)

    #mul!(s.dx, s.L.L, s.n_u)
    #lmul!(sqrt(γ*abs(α-s.α_opt)), s.dx)
    lowerTriInplaceMultiplyAdd!(s.dx, sqrt(γ*abs(α-s.α_opt)), s.L.factors,
              s.n_u, zero(FT), s.dx)
    if α >= s.α_opt
        lowrankupdate!(s.L, s.dx)
    else
        lowrankdowndate!(s.L, s.dx)
    end
    nothing
end

@inline function draw!(sr::RWMState, s::RobustAdaptiveMetropolis) :: Nothing
    draw!(sr, s.L)
    nothing
end
