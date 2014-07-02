import Distributions.MvNormalStats, Distributions.lpgamma, Distributions.suffstats, Distributions.mean

export object_loglikelihood

function gibbs_gaussian_mixture(x::Matrix{Float64}, F::Matrix{Float64}, prior::NormalWishart, burnin::Int64, niter::Int64; init_c::Function=singular_init_discrete, iter_callback::Function = (x...) -> true)
    N = size(F, 1)
    D = size(x, 1)

    function cluster_likelihood(theta::MvNormal, z::Vector{Int64})
#        post = posterior_cool(prior, suffstats(MvNormal, x[:, z]))
#        return marginal_loglikelihood(prior, post, float(length(z)))

        ll = logpdf(prior, theta.μ, full(inv(theta.Σ)))
        ll += sum(logpdf(theta, x[:, z]))
        
        return ll
    end

    function init_theta(z::Vector{Int64})
        return MvNormal(vec(mean(x, 2)), eye(D))
    end

    function sample_theta(z::Vector{Int64})        
        post = posterior_cool(prior, suffstats(MvNormal, x[:, z]))
        mu, lamb = rand(post)

        return MvNormal(mu, inv(lamb))
    end

    return gibbs(MvNormal, F, cluster_likelihood, sample_theta, burnin, niter; iter_callback = iter_callback, init_c = init_c, init_theta = init_theta)
end

function collapsed_gibbs_gaussian_mixture(x::Matrix{Float64}, F::Matrix{Float64}, prior::NormalWishart, burnin::Int64, niter::Int64; init_c::Function=singular_init_discrete, iter_callback::Function = (x...) -> true)
    N = size(F, 1)

    function cluster_likelihood12(z::Vector{Int64})
        n = float(length(z))
        post = posterior_cool(prior, suffstats(MvNormal, x[:, z]))
        ll = marginal_loglikelihood(prior, post, n)

#        println(repr(z), " ", ll)    
        
        return ll
    end

    return collapsed_gibbs(F, cluster_likelihood12, burnin, niter; iter_callback = iter_callback, init_c = init_c)
end

function var_gaussian_mixture(x::Matrix{Float64}, F::Matrix{Float64}, prior::NormalWishart, init_c::Function = singular_init_dist)
    N = size(F, 1)
    dim = size(x, 1)

    M = size(x, 2)
    w = zeros(M)

    function cluster_update(z::Array{Float64})
        w[:] = 0.
        w[1:length(z)] = z
        return posterior_cool(prior, suffstats_cool(MvNormal, x, w))
    end

    function cluster_entropy(nw::NormalWishart)
        return entropy(nw)
    end

    function cluster_loglikelihood(nw::NormalWishart, z::Array{Float64})
        w[:] = 0.
        w[1:length(z)] = z
        return expected_loglikelihood(prior, nw, x, w)
    end
    
    function object_likelihood(nw::NormalWishart, i::Int64)
        return object_loglikelihood(nw, x[:, i])
    end

    theta = Array(NormalWishart, M)
    for k in 1:N
        theta[k] = NormalWishart(x[:, k], 1e-7, eye(2) / 4, 4.0001) #prior
    end
    crp = seq_graph_z(theta[1:N], F, cluster_update, cluster_loglikelihood, object_likelihood, cluster_entropy; init_c = init_c)
    #global crpx = crp

    function predictive_likelihood(F1::Matrix{Float64}, nsamples::Int64, restrict_new_clusters::Bool=true)
        function predictive(theta::NormalWishart, z::Array{Float64})
            post = posterior_cool(theta, suffstats_cool(MvNormal, x, z))
            return marginal_loglikelihood(theta, post, sum(z))
        end

        return sample_test_likelihood(crp, F1, nsamples, predictive, nothing)
            #restrict_new_clusters ? nothing : prior)
    end

    return crp, predictive_likelihood
end

export var_gaussian_mixture, gibbs_gaussian_mixture, collapsed_gibbs_gaussian_mixture
