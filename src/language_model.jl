function get_nzs(w::Vector{Int64}, F::Matrix{Float64})
    N = size(F, 1)
    nzs = Array(Tsp, N)
    for i=1:N
        nz = Bool[w[i] == w[j] && F[i, j] > eps() for j=1:N]
        oksa = zeros(N)
        oksa[nz] = 1.
        pattern = sparsevec(oksa)
        pattern.nzval[:] = 0.
        nzs[i] = pattern
    end
    return nzs
end

function var_language_model(w::Vector{Int64}, log_word_freq::Array{Float64}, decay::Function, alpha::Float64, a)
    N = length(w)

    D = time_dist(N)
    F = seqddcrp(decay, D, alpha, a, false)

    return var_language_model(w, log_word_freq, F)
end

function var_language_model(w::Vector{Int64}, log_word_freq::Array{Float64}, F::Matrix{Float64}, init_c::Function=rand_init_dist)
    N = length(w)

    crp_cluster_init(x...) = false

    crp_cluster_update(x...) = false

    function crp_cluster_likelihood(j::Int64, z::Union(UnsafeVectorView{Float64}, Vector{Float64}))
        for i=j+1:N
            if z[i] > realmin() && w[i] != w[j]
                return -Inf
            end
        end

        return log_word_freq[w[j]] * z[j]
    end

    crp_cluster_entropy(x...) = 0.

    function crp_likelihood_gradient(s::Int64, t::Int64, alpha::Vector{Float64}, r::Matrix{Float64}, treshold::Float64)
        if s == t
            for i=s+1:N
                if r[i, s] > realmin() && w[i] != w[s]
                    @assert false
                end
            end
            return log_word_freq[w[s]]
        end

        for j in 1:N
            for i in j+1:N
                if r[i, s] * r[t, j] > realmin() && w[i] != w[j]
                    @assert false
                    return -Inf
                end
            end
        end

        return 0.
    end

    nzs = get_nzs(w, F)

    crpx = seqddCRP(F, init_c, crp_cluster_likelihood, crp_likelihood_gradient, crp_cluster_update, crp_cluster_entropy, nzs)

    return crpx
end

function alt_var_language_model(w::Vector{Int64}, log_word_freq::Array{Float64}, F::Matrix{Float64}, init_c::Function=rand_init_dist)
    N = size(F, 1)
    M = length(w)

    theta = w

    crp_cluster_update(x...) = false

    function crp_cluster_likelihood(theta::Int64, z::Array{Float64})
        for i=j+1:N
            if z[i] > realmin() && w[i] != theta
                return -Inf
            end
        end

        return log_word_freq[theta] * z[j]
    end

    object_likelihood(theta::Int64, i::Int64) = w[i] == theta ? 0. : -Inf

    crp_cluster_entropy(x...) = 0.

    function crp_likelihood_gradient(s::Int64, t::Int64, alpha::Vector{Float64}, r::Matrix{Float64}, treshold::Float64)
        if s == t
            for i=s+1:N
                if r[i, s] > realmin() && w[i] != w[s]
                    @assert false
                end
            end
            return log_word_freq[w[s]]
        end

        for j in 1:N
            for i in j+1:N
                if r[i, s] * r[t, j] > realmin() && w[i] != w[j]
                    @assert false
                    return -Inf
                end
            end
        end

        return 0.
    end

    nzs = get_nzs(w, F)

    crpx = seq_graph_z(F, w, crp_cluster_update, crp_cluster_likelihood, crp_cluster_entropy; nzs=nzs, init_c=init_c)

    return crpx
end

function gibbs_language_model(w::Vector{Int64}, log_word_freq::Array{Float64}, F::Matrix{Float64}, burnin::Int64, niter::Int64; init_c::Function=singular_init_discrete, iter_callback::Function = (x...) -> true)
    nzs = get_nzs(w, F)

    function cluster_likelihood(z::Vector{Int64})
        for i in z[2:end]
            if w[i] != w[z[1]]
                return -Inf
            end
        end

        return log_word_freq[w[z[1]]]
    end

    return collapsed_gibbs(F, cluster_likelihood, burnin, niter; iter_callback=iter_callback, nzs=nzs, init_c=init_c), cluster_likelihood
end

export var_language_model, gibbs_language_model, get_nzs
