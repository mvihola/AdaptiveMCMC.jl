# Abstract type of a 'step size sequence'
abstract type StepSize end

# Polynomially decaying step size sequence
struct PolynomialStepSize{FT <: AbstractFloat} <: StepSize
    eta::FT
    c::FT
end
function PolynomialStepSize(eta::FT, c=one(FT)) where {FT <: AbstractFloat}
    PolynomialStepSize(eta, c)
end
@inline function get(s::PolynomialStepSize{FT}, k::Real) where {FT <: AbstractFloat}
    s.c*(k+one(FT))^(-s.eta)
end

# Specialization to the RAM default rule:
struct RAMStepSize{FT<:AbstractFloat} <: StepSize
    p::PolynomialStepSize{FT}
end
function RAMStepSize(eta::FT, d::Int) where {FT <: AbstractFloat}
    RAMStepSize{FT}(PolynomialStepSize(eta, FT(d)))
end
@inline function get(s::RAMStepSize{FT}, k::Real) where {FT}
    min(FT(0.5), get(s.p, k))
end
