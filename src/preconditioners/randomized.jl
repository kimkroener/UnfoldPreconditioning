# Randomized Preconditioners using RandomizedPreconditioners.jl
# implemented methods: NystromPreconditioner, NystromPreconditionerInverse, NystromSketch

# --- mul! and ldiv!
function LinearAlgebra.mul!(
    y::AbstractVector{T}, 
    P::RandomizedPreconditioners.NystromPreconditioner{T}, 
    x::AbstractVector{T},
    α::Number,
    β::Number
) where {T <: Real}
    # NystromPreconditioner doesn't have a direct 3-arg mul!, so we use Matrix form
    if β == 0
        mul!(y, Matrix(P), x)
        α != 1 && lmul!(α, y)
    else
        temp = similar(y)
        mul!(temp, Matrix(P), x)
        @. y = α * temp + β * y
    end
    return y
end

# ----
function setup_nystrom_preconditioner(X; μ::Float64=1e-6, kwargs...)
    # Convert sparse matrix to dense for SVD if needed
    P = RandomizedPreconditioners.NystromPreconditioner(Matrix(X); μ=μ)
    return P, nothing
end

nystrom_pm = PreconditionerMethod(
    :nystrom,
    setup_nystrom_preconditioner,
    PreconditionerProperties(
        supports_rectangular_matrices=false,  # requires square SPD matrix
        side=:left,
        ldiv=true,
        supports_sparse=true, # converts to dense
        supports_dense=true,
        supported_backends=Set([:Krylov, :IterativeSolvers])
    ),
    "Randomized Nyström preconditioner P ≈ A + μI for symmetric positive (semi-)definite systems. Uses low-rank approximation via randomized sketching.",
    "https://github.com/tjdiamandis/RandomizedPreconditioners.jl"
)


function setup_nystrom_preconditioner_inverse(X; μ::Float64=1e-6, kwargs...)
    Pinv = RandomizedPreconditioners.NystromPreconditionerInverse(Matrix(X); μ=μ)
    return Pinv, nothing
end

nystrom_inv_pm = PreconditionerMethod(
    :nystrom_inv,
    setup_nystrom_preconditioner_inverse,
    PreconditionerProperties(
        supports_rectangular_matrices=false,
        side=:left,
        ldiv=false, 
        supports_sparse=true, # converts to dense
        supports_dense=true,
        supported_backends=Set([:Krylov])
    ),
    "Inverse of the randomized Nyström preconditioner P⁻¹ ≈ (A + μI)⁻¹. Applies via multiplication (mul!) rather than solve.",
    "https://github.com/tjdiamandis/RandomizedPreconditioners.jl"
)



"""
    setup_nystrom_sketch_preconditioner(A; k=nothing, r=nothing, μ=1e-6, kwargs...)

Construct a Nyström sketch Â ≈ A and wrap it as a preconditioner.

The Nyström sketch provides a low-rank approximation of A that can be used
for preconditioning or other purposes. This allows explicit control over
the rank and sketch size.

Complexity of O(n²r), where the parameter k can truncate the sketch to improve numberical performance. 

# Arguments
- `A`: Symmetric positive semi-definite matrix
- `k`: Truncation rank (default: auto-selected as min(n, max(10, n÷10)))
- `r`: Sketch size, must be > k (default: k + 10)
- `μ`: Regularization for the preconditioner wrapper (default: 1e-6)


"""
function setup_nystrom_sketch_preconditioner(A; k::Union{Nothing,Int}=nothing, r::Union{Nothing,Int}=nothing, μ::Float64=1e-6, kwargs...)
    #https://github.com/tjdiamandis/RandomizedPreconditioners.jl

    

    # set k, r based on matrix size if not provided
    n = size(A, 1)
    if k === nothing
        k = min(n, max(10, n ÷ 10))
    end
    if r === nothing
        r = min(n, k + 10)
    end
    
    # Ensure r > k as required by NystromSketch
    if r <= k
        r = k + max(5, k ÷ 5)
        r = min(n, r)  # Don't exceed matrix dimension
        if r <= k
            @warn "Cannot satisfy r > k constraint with matrix size $n and k=$k. Using k=$(r-1) instead."
            k = r - 1
        end
    end
    
    # Create the sketch
    Â = RandomizedPreconditioners.NystromSketch(Matrix(A), k, r)
    # Wrap as preconditioner
    P = RandomizedPreconditioners.NystromPreconditioner(Â, μ)
    return P, nothing
end

nystrom_sketch_pm = PreconditionerMethod(
    :nystrom_sketch,
    setup_nystrom_sketch_preconditioner,
    PreconditionerProperties(
        supports_rectangular_matrices=false,
        side=:left,
        ldiv=true,
        supports_sparse=true,
        supports_dense=true,
        supported_backends=Set([:all])
    ),
    "Nyström sketch-based preconditioner. First constructs low-rank sketch Â ≈ A, then uses it for preconditioning",
    "https://github.com/tjdiamandis/RandomizedPreconditioners.jl"
)

register_preconditioner_method!(nystrom_pm)
register_preconditioner_method!(nystrom_inv_pm)
register_preconditioner_method!(nystrom_sketch_pm)


# # randomized preconditioner registry
# const randomized_precond_dict = Dict{Symbol,PreconditionerMethod}(
#     :nystrom => nystrom_pm,
#     :nystrom_inv => nystrom_inv_pm,
#     :nystrom_sketch => nystrom_sketch_pm,
# )