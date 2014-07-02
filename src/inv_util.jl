using NumericExtensions
using Devectorize

function subinv(X::Matrix{Float64}, i::Int64)
    N = size(X, 1)
        
    z = trues(N)
    z[i] = false
    v = X[z, i]
    u = X[i, z]
    subAinv = X[z, z] - v * u / X[i, i]
    
    return subAinv
end

function inv_update_row!(X::Matrix{Float64}, i::Int64, v::Vector{Float64})
    N = size(X, 1)
    b = reshape(X[:, i], N, 1)
    c = 1. + dot(v, b)
    X[:, :] -= (b * v' * X) / c
end

function inv_update_row!(X::Matrix{Float64}, i::Int64, v::Matrix{Float64})
    N = size(X, 1)
    dX = zeros(N, N)

    prepare_update_row!(X, dX, i, v)
    add!(X, dX)
end

function prepare_update_row!(X::Matrix{Float64}, dX::Matrix{Float64}, i::Int64, v::Vector{Float64})
    b = X[:, i]
    c = 1. + dot(v, b)

    divide!(v, c)
    N = size(X, 1)
    dX[:, :] = b * (v' * X)
    negate!(dX)
end

#assuming b is a vector
function spdot(a::DenseArray{Float64}, b::SparseMatrixCSC{Float64, Int64})
    r = 0.
    K = size(b.rowval, 1)
    for i=1:K
        r += b.nzval[i] * a[b.rowval[i]]
    end
    return r
end

function mul_sprow_mat(a::SparseMatrixCSC{Float64, Int64}, A::Matrix{Float64})
    N = size(A, 1)
    b = zeros(1, N)

    for i=1:N
        b[i] = spdot(unsafe_view(A, :, i), a)
    end

    return b
end

#function something(X::Matrix{Float64}, b::UnsafeViewVector{Float64}i

function inv_update_row!(X::Matrix{Float64}, i::Int64, v::SparseMatrixCSC{Float64, Int64})
    N = size(X, 1)
    b = unsafe_view(X, :, i)
    c = 1. + spdot(b, v)

    @devec v.nzval[:] ./= c

    u = mul_sprow_mat(v, X)
    for j=1:N
        if i == j continue end
        @devec X[:, j] -= b .* u[j]
    end
    @devec X[:, i] -= b .* u[i]
end

function seq_inv_update_row!(X::Matrix{Float64}, i::Int64, v::SparseMatrixCSC{Float64, Int64})
    N = size(X, 1)
    b = unsafe_view(X, :, i)
    c = 1. + spdot(b, v)

    @devec v.nzval[:] ./= c

    u = mul_sprow_mat(v, X)
    for j=1:N
        if i == j continue end
        @devec X[j:N, j] -= b[j:N] .* u[j]
    end
    @devec X[:, i] -= b .* u[i]
end

#assuming A[i, k] = 0 for each k except k = j
function inv_update_row!(X::Matrix{Float64}, i::Int64, k::Int64)
    N = size(X, 1)
    b = unsafe_view(X, :, i)
#    b = X[:, i]
    c = 1. - b[k]

    u = -X[k, :] ./ c
#    subtract!(X, b * u)
    for j=1:N
        if i == j continue end
        @devec X[j:N, j] -= b[j:N] .* u[j]
    end
    @devec X[:, i] -= b .* u[i]
end
