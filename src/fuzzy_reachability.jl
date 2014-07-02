immutable SequentialRestaurant
    c::Matrix{Float64}
    Linv::Matrix{Float64}
    alpha::Vector{Float64}
    z::Matrix{Float64}

    function SequentialRestaurant(c::Matrix{Float64})
        N = size(c, 1)
        @assert sum(c .< 0.) == 0 "negative probabilities"

        for i=1:N
            if sum(c[i, i+1:N]) > 1e-7 
                error("non-sequential at $i")
            end
            if abs(sum(c[i, :]) - 1.) > 1e-7 
                error("not normalized. i=", i, ", sum=", sum(c[i, :]))
            end
        end

        F = copy(c)
        alpha = diag(c)
        for i=1:N F[i, i] = 0. end

        r = new(F, zeros(N, N), alpha, zeros(N, N))
        refresh!(r)

        r
    end
end

function update!(r::SequentialRestaurant, i::Int64, v::Vector{Float64})
#    @assert abs(sum(v) - 1.) < 1e-7 string("not normalized probabilities provided")

    N = size(r.alpha, 1)
    r.alpha[i] = v[i]
    v[i] = 0.
        
    u = zeros(1, N)
    @devec u[:] = r.c[i, :] .- v

    @devec r.c[i, :] = v
    v[i] = r.alpha[i]

    inv_update_row!(r.Linv, i, u)
    update_z!(r)

#    check(r)    
end

function update!(r::SequentialRestaurant, i::Int64, v::SparseMatrixCSC{Float64, Int64})
    if abs(sum(v.nzval) - 1.) > 1e-7 
        error("not normalized probabilities provided\n$v")
    end

    N = size(r.alpha, 1)
        
    u = copy(v)
    K = length(u.nzval)

    for j=1:K
        k = u.rowval[j]
        if k == i 
            r.alpha[i] = v.nzval[j]
            u.nzval[j] = 0.
            continue 
        end
        u.nzval[j] = r.c[i, k] - v.nzval[j]
        r.c[i, k] = v.nzval[j]
    end

    seq_inv_update_row!(r.Linv, i, u)
#    refresh!(r)
    update_z!(r)
end

function check(r::SequentialRestaurant)
    N = size(r.alpha, 1)

    C = r.c + diagm(r.alpha)

    for i=1:N
        if abs(sum(r.z[i, :]) - 1.) > 1e-7
            error("Cluster assignments are not normalized $i\n$(repr(C[i, :]))")
        end

        for j=1:N
            if r.z[i, j] < 0.
                error("Negative probabilities $i $j\n$(C[i, j])")
            end
        end
    end

    for j=1:N
        for i in j-1:-1:1
            if r.Linv[i, j] - r.Linv[i+1, j] > 1e-7
                error("Non-monotone $j $i\n$(repr(C[:, j]))")
            end
        end
    end
end

function update!(r::SequentialRestaurant, i::Int64, j::Int64)
    N = size(r.alpha, 1)
    
    @devec r.c[i, :] = 0.
    r.alpha[i] = 0.
 
    if i == j
        r.alpha[i] = 1.
    else
        r.c[i, j] = 1.
    end

    if i != j
        inv_update_row!(r.Linv, i, j)
    end
#    update_z!(r)
end


function update_z!(r::SequentialRestaurant)
    N = size(r.c, 1)
    for j=1:N
        r.Linv[1:j-1, j] = 0.
    end
    r.Linv[r.Linv .< 0.] = 0.
    r.z[:, :] = r.Linv
    vbroadcast1!(Multiply(), r.z, r.alpha, 2)

#    for i=1:N
#        @assert (abs(1. - sum(r.z[i, :])) < 1e-7) string("incorrect table assignments $i")
#    end
end

function refresh!(r::SequentialRestaurant)
    N = size(r.c, 1)
    for i=1:N
        si = 0.
        for j=1:N
            if r.c[i, j] < 0 
                error(i, " ", j, ": ", r.c[i, j])
            end
            si += r.c[i, j]
        end
        @assert r.c[i, i] < eps()
        if r.alpha[i] < 0.
            error("alpha[$i] = ", r.alpha[i])
        end
        si += r.alpha[i]
        if abs(si - 1.) > 1e-7
            error("non-normalized $i $si")
        end
    end

#    println("b")    
    r.Linv[:, :] = inv(eye(N) - r.c)
#    println("a")
    abs!(r.Linv)
    update_z!(r)
end
