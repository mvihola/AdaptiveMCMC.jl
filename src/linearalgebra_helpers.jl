import LinearAlgebra: lowrankupdate!, lowrankdowndate!

# Matlab style identity matrix convenience function
eye(FT, d) = diagm(0 => fill(one(FT),d))
eye(d) = eye(Float64, d)

# Helper to calculate z = sc_y*y  + sc*A*x (in place), where A assumed
# lower triangular. Argument y can be same as z.
# NB: No checks about dimensions!
@inline function lowerTriInplaceMultiplyAdd!(z::zT,
    sc::FT, A::AT, x::xT, sc_y::FT, y::yT) ::Nothing where {
    FT <: AbstractFloat,
    zT <: AbstractVector{FT},
    AT <: AbstractMatrix{FT},
    xT <: AbstractVector{FT},
    yT <: AbstractVector{FT}
    }
    n::Int64 = length(x)
    if (sc_y == zero(FT))
        @inbounds @simd for i = 1:n
            u::FT = zero(FT)
            for j = 1:i
                u += A[i,j]*x[j]
            end
            z[i] = sc*u
        end
    else
        @inbounds @simd for i = 1:n
            u::FT = zero(FT)
            for j = 1:i
                u += A[i,j]*x[j]
            end
            z[i] = sc_y*y[i] + sc*u
        end
    end
    nothing
end

# These are from cholesky.jl: just to make these work with MVector...
function lowrankupdate!(C::Cholesky{FT,<:MMatrix{n,n}},
    v::MVector{n,FT}) ::Nothing where {n, FT<:AbstractFloat}

    A = C.factors
    if C.uplo == 'U'
        conj!(v)
    end

    @inbounds for i = 1:n

        # Compute Givens rotation
        c, s, r = LinearAlgebra.givensAlgorithm(A[i,i], v[i])

        # Store new diagonal element
        A[i,i] = r

        # Update remaining elements in row/column
        if C.uplo == 'U'
            @simd for j = i + 1:n
                Aij = A[i,j]
                vj  = v[j]
                A[i,j]  =   c*Aij + s*vj
                v[j]    = -s'*Aij + c*vj
            end
        else
            @simd for j = i + 1:n
                Aji = A[j,i]
                vj  = v[j]
                A[j,i]  =   c*Aji + s*vj
                v[j]    = -s'*Aji + c*vj
            end
        end
    end
    nothing
end

function lowrankdowndate!(C::Cholesky{FT,<:MMatrix{n,n}},
    v::MVector{n,FT}) ::Nothing where {n, FT<:AbstractFloat}
    A = C.factors
    if C.uplo == 'U'
        conj!(v)
    end

    @inbounds for i = 1:n

        Aii = A[i,i]

        # Compute Givens rotation
        s = conj(v[i]/Aii)
        s2 = abs2(s)
        if s2 > one(FT)
            throw(LinearAlgebra.PosDefException(i))
        end
        c = sqrt(one(FT) - abs2(s))

        # Store new diagonal element
        A[i,i] = c*Aii

        # Update remaining elements in row/column
        if C.uplo == 'U'
            @simd for j = i + 1:n
                vj = v[j]
                Aij = (A[i,j] - s*vj)/c
                A[i,j] = Aij
                v[j] = -s'*Aij + c*vj
            end
        else
            @simd for j = i + 1:n
                vj = v[j]
                Aji = (A[j,i] - s*vj)/c
                A[j,i] = Aji
                v[j] = -s'*Aji + c*vj
            end
        end
    end
    nothing
end
