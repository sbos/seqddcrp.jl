#uncollapsed gibbs sampler

function conditional!{Tp}(cp::Vector{Float64}, G::Matrix{Float64}, graph::RestrauntGraph, theta::Vector{Tp}, i::Int64, cluster_likelihood::Function, nzs::Vector{Tsp})
    N = size(G, 1)

    cp[:] = -Inf
    for j in nzs[i].rowval
        if i == j
            cp[j] = G[i, i]
            continue
        end

        graph = remove_link!(graph, i)
        K = numcomponents(graph)

        k_cluster = [graph.components[vertex_component(graph, i)]...]
        l_cluster = [graph.components[vertex_component(graph, j)]...]

        graph = add_link!(graph, i, j)
        if numcomponents(graph) == K
            cp[j] = G[i, j]
            continue
        end

        assert(K == numcomponents(graph)+1)

        kl_cluster = vcat(k_cluster, l_cluster)

        kl_clust_l = cluster_likelihood(theta[j], kl_cluster)
        k_clust_l = cluster_likelihood(theta[i], k_cluster)
        l_clust_l = cluster_likelihood(theta[j], l_cluster)
 
        cp[j] = G[i, j] + kl_clust_l - k_clust_l - l_clust_l
    end

    adjust!(cp)
end

function sample_theta!{Tp}(theta::Vector{Tp}, z::Vector{IntSet}, sample_theta::Function)
    N = length(theta)
    for zk in z
        zka = [zk...]
        theta[zka] = sample_theta(zka)
    end
    return theta
end

function gibbs{Tp}(::Type{Tp}, F::Matrix{Float64}, cluster_likelihood::Function, sample_theta::Function, burnin::Int64, niter::Int64; nzs::Union(Vector{Tsp}, Nothing)=nothing, iter_callback::Function=(c, z, theta, ll) -> true, init_c::Function = singular_init_discrete, init_theta::Function = sample_theta)
    N = size(F, 1)
    G = log(F)

    c_samples = zeros(Int64, N, niter)
    theta_samples = Array(Tp, N, niter)

    loglike = zeros(niter)

    if nzs == nothing
        nzs = Array(Tsp, N)
        for i=1:N
            nzs[i] = sparsevec(vec(F[i, :]))
        end
    end

    c = init_c(F, nzs)
    graph = RestrauntGraph(c)
    theta = Array(Tp, N) 
    sample_theta!(theta, graph.components, init_theta)

    cp = zeros(N)

    function iterate()
        for i in randperm(N)
            conditional!(cp, G, graph, theta, i, cluster_likelihood, nzs)
            remove_link!(graph, i)
            j = rand(Categorical(cp))
            add_link!(graph, i, j)

            i_cluster = [graph.components[vertex_component(graph, i)]...]
            if i != j
                theta[i_cluster] = theta[j]
            else
#                println("born $i")
                theta[i_cluster] = sample_theta(i_cluster)
            end
        end
        sample_theta!(theta, graph.components, sample_theta)
    end

    for iter=1:burnin
        iterate()
    end

    for iter=1:niter
        @time iterate()
        c_samples[:, iter] = graph.c
        theta_samples[:, iter] = theta
        loglike[iter] = loglikelihood(graph.c, theta, G, graph.components, cluster_likelihood)

        if !iter_callback(graph.c, graph.components, theta, loglike[iter])
            break
        end
    end

    return c_samples, theta_samples, loglike
end

function loglikelihood{Tp}(graph::RestrauntGraph, theta::Vector{Tp}, G::Matrix{Float64}, cluster_likelihood::Function)
    return loglikelihood(graph.c, theta, G, graph.components, cluster_likelihood)    
end

function loglikelihood{Tp}(c::Vector{Int64}, theta::Vector{Tp}, G::Matrix{Float64}, z::Vector{Ts}, cluster_likelihood::Function)
    N = length(c)
    ll = 0.
    
    for i=1:N
        ll += G[i, c[i]]
    end

    ll += sublikelihood(c, theta, z, cluster_likelihood)
    return ll
end

function sublikelihood{Tp}(c::Vector{Int64}, theta::Vector{Tp}, z::Vector{Ts}, cluster_likelihood::Function)
    ll = 0.

    K = length(z)
    for k=1:K
        zk = [z[k]...]
        ll += cluster_likelihood(theta[zk[1]], zk)
    end

    return ll
end

function alt_predictive_posterior{Tp}(samples::Matrix{Int64}, theta::Matrix{Tp}, F::Matrix{Float64}, likelihood::Function, niter::Int64, nzs::Union(Vector{Tsp}, Nothing)=nothing)
    G = log(F)
    N, S = size(samples)
    M = size(F, 1)

    esamples = zeros(Int64, M-N, niter)
    cp = zeros(M, M-N)
    for iter=1:niter 
        for i=N+1:M
            cp[:, i-N] = 0.
            if nzs != nothing
                cs = 0.
                for j in nzs[i].rowval
                    cp[j, i-N] = F[i, j]
                    cs += F[i, j]
                end
                cp[:, i-N] /= cs
            else
                cp[:, i-N] = F[i, :]
            end
            esamples[i-N, iter] = rand(Categorical(cp[:, i-N]))
        end
    end

    results = zeros(S)
    R = zeros(niter, S)

    function sublikelihood(s::Int64, c::Vector{Int64})
        ll = 0.

        for i in N+1:M
            j = c[i]
            while j != c[j]
                j = c[j]
            end

            ll += likelihood(theta[j, s], i)
        end

        return ll
    end

    for iter=1:niter
        c = zeros(Int64, M)
        c[N+1:M] = esamples[:, iter]

        ks = KahanState()

        for s=1:S
            c[1:N] = samples[:, s]
            add!(ks, exp(sublikelihood(s, c)))
            R[iter, s] = ks.sum
        end

#        println("Predictive $iter/$niter")
    end

    for s=1:S
        ks = KahanState()
        for iter=1:niter
            add!(ks, R[iter, s])
        end
        results[s] = log(ks.sum) - log(s) - log(niter)
    end

    return results
end

export gibbs, alt_predictive_posterior
