module seqddCRP

using Distributions
using NumericExtensions
using Devectorize

include("util.jl")
include("kahan.jl")
include("inv_util.jl")
include("prior.jl")
include("init_dist.jl")
include("Reachability.jl")
include("gibbs.jl")
include("fullgibbs.jl")
include("language_model.jl")
include("student.jl")
include("seqgraphz.jl")
include("gauss_util.jl")
include("gaussian_mixture.jl")

end
