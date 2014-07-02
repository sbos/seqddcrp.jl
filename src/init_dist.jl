function get_nzs(F::Matrix{Float64})
    N = size(F, 1)
    nzs = Array(Tsp, N)
    for i=1:N
        pattern = sparsevec(vec(F[i, :]))
        pattern.nzval[:] = 0.
        nzs[i] = pattern
    end
    return nzs
end 

singular_init_dist(C::Matrix{Float64}, nzs::Vector{Tsp}) = eye(size(C, 1))

function rand_init_dist(C::Matrix{Float64}, nzs::Vector{Tsp})
    N = size(C, 1)
    C = zeros(N, N)
    for i=1:N
        rsum = 0.
        for j in nzs[i].rowval
            C[i, j] = rand()
            rsum += C[i, j]
        end

        C[i, :] /= rsum 
    end

    return C
end

function spec_init_dist(C::Matrix{Float64}, nzs::Vector{Tsp})
    N = size(C, 1)
    C = zeros(N, N)
    C[1, 1] = 1.
    for i in 2:N
        if rand() >= 0.1
            C[i, i] = 1.
        else
            C[i, i-1] = 1.
        end
    end

    return C
end


function reweighted_init_dist(F::Matrix{Float64}, nzs::Vector{Tsp})
    N = size(F, 1)
    C = zeros(N, N)
    for i=1:N
        rsum = 0.
        for j in nzs[i].rowval
            C[i, j] = F[i, j]
            rsum += C[i, j]
        end

        C[i, :] /= rsum 
    end

    return C
end

function uniform_init_dist(C::Matrix{Float64}, nzs::Vector{Tsp})
    N = size(C, 1)
    C = zeros(N, N)
    for i=1:N
        C[i, nzs[i].rowval] = 1. / length(nzs[i].rowval)
    end
    return C
end

export singular_init_dist, rand_init_dist, uniform_init_dist, reweighted_init_dist, spec_init_dist
