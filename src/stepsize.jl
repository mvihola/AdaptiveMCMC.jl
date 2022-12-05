# Abstract type of a 'step size sequence'
abstract type StepSize end

# Polynomially decaying step size sequence
struct PolynomialStepSize{FT <: AbstractFloat} <: StepSize
    eta::FT
    c::FT
end

"""
    gamma = PolynomialStepSize(eta::AbstractFloat, [c::AbstractFloat=1.0]))

Constructor for PolynomialStepSize.

# Arguments
- `eta`: The step size exponent, should be within (1/2,1].
- `c`: Scaling factor; default `1.0`.
"""
function PolynomialStepSize(eta::FT, c=one(FT)) where {FT <: AbstractFloat}
    PolynomialStepSize(eta, c)
end

@inline function get(s::PolynomialStepSize{FT}, k::Real) where {FT <: AbstractFloat}
    s.c*(k+one(FT))^(-s.eta)
end

struct RAMStepSize{FT<:AbstractFloat} <: StepSize
    p::PolynomialStepSize{FT}
end

"""
    gamma = RAMStepSize(eta::AbstractFloat, d::Int)

Constructor for RAM step size.

# Arguments
- `eta`: The step size exponent, should be within (1/2,1].
- `d`: State dimension.
"""
function RAMStepSize(eta::FT, d::Int) where {FT <: AbstractFloat}
    RAMStepSize{FT}(PolynomialStepSize(eta, FT(d)))
end

@inline function get(s::RAMStepSize{FT}, k::Real) where {FT}
    min(FT(0.5), get(s.p, k))
end
