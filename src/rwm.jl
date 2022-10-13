# Inner state of a random walk Metropolis algorithm in R^d
struct RWMState{d,FT<:AbstractFloat, T <: AbstractVector{FT},
    FunT <: Function,  RngT <: Random.AbstractRNG}
    rng::RngT # The random number generator
    q!::FunT  # The proposal function
    x::T      # Current state
    u::T      # Draw from q!
    y::T      # Proposal state
end


"""
    s = RWMState(x0, [rng, [q!]])

Constructor for RWMState (random walk Metropolis state).

# Arguments
- `x0`: The initial state vector
- `rng`: Random number generator; default `Random.GLOBAL_RNG`.
- `q!`: Symmetric proposal distribution; default `randn!`.
  Called by `q!(rng, u)` which puts a draw to vector `u`

If `s` is `RWMState`, then `s.x` is the current state. New proposal
may be drawn to `s.y` by calling `draw!`, and the proposal is
accepted by calling `accept!(s)`.
"""
function RWMState(x0::AbstractVector{FT}, rng::AbstractRNG=Random.GLOBAL_RNG,
    q::Function=randn!) where {FT <: AbstractFloat}
    state = RWMState{length(x0),FT,typeof(similar(x0)),typeof(q),typeof(rng)}(rng, 
    q,  similar(x0), similar(x0), similar(x0))
    copyto!(state.x, x0)
    state
end

# Draw u from proposal
@inline function draw_u!(s::RWMState) :: Nothing
    s.q!(s.rng, s.u)
    nothing
end

"""
   draw!(s::RWMState, sc::AbstractFloat)

Draw proposal from `s.x` to `s.y` using scaling `sc`.
"""
@inline function draw!(s::RWMState, sc::AbstractFloat) :: Nothing
    draw_u!(s)
    s.y .= sc.*s.u .+ s.x
    nothing
end

# Draw proposal using scalar multiplier
@inline function draw!(s::RWMState{1}, sc::AbstractFloat) :: Nothing
    draw_u!(s)
    s.y[] = sc*s.u[] + s.x[]
    nothing
end


"""
   draw!(s::RWMState, L::Cholesky, sc::AbstractFloat)

Draw proposal from `s.x` to `s.y` using scaling `sc*L.L`, with default `sc=1.0`.
"""
@inline function draw!(s::RWMState{d,FT}, L::Cholesky{FT}, sc::FT=one(FT)) where {
    d, FT <: AbstractFloat}
    draw_u!(s)
    if L.uplo == 'L'
        lowerTriInplaceMultiplyAdd!(s.y, sc, L.factors, s.u, one(FT), s.x)
    else
        mul!(s.y, L.L, s.u)
        s.y .= sc .* s.y .+ s.x
    end
    nothing
end


"""
   accept!(s::RWMState)

Accept the proposal, that is, set copy `s.y` to `s.x`.
"""
@inline function accept!(sr::RWMState) :: Nothing
    sr.x .= sr.y
    nothing
end

@inline function accept!(sr::RWMState{1}) :: Nothing
    sr.x[] = sr.y[]
    nothing
end
