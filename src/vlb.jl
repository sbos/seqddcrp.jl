#push!(LOAD_PATH, "../src/")

require("seqddcrp.jl")

using seqddCRP
using Distributions

prior = NormalWishart(zeros(2), 1e-7, eye(2) / 4, 4.0001)
#out = open("5.csv", "a") 

x = readdlm("/Users/sbos/projects/seqddcrp/dataset/balley_7_3.csv", ',')
x = x'

N = size(x, 2)

F = crp_F(N, 1e-1)

crp, predictive_likelihood = var_gaussian_mixture(x, F, prior)
ll = infer(crp, 100)

println(ll)
