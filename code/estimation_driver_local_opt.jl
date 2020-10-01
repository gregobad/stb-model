### OPTIONS: change things here ###
## number of cores to use in parallel
const numprocs = 6;
using Distributed
addprocs(numprocs)

## Set your local directory
@everywhere local_dir = "/home/gregorymartin/Dropbox/STBNews"

### END OPTIONS ###

## Directory locations for code and data
@everywhere using Printf
@everywhere code_dir = @sprintf("%s/code/model/julia/", local_dir)
@everywhere data_dir = @sprintf("%s/data/model/", local_dir)

### LOAD OBJECTIVE AND DATA ###
## parallel version ##
@everywhere cd(code_dir)
@everywhere include("parallel_jacobian.jl")
@everywhere include("load_model_data.jl")


### SETUP INITIAL PARAMETER VECTOR ###
# to read from last MCMC run:
@everywhere cd(code_dir)
@everywhere include("read_par_from_mcmc.jl")

# # to read direct from csv:
# @everywhere par_init = CSV.read("par_init.csv");

# # dictionary indexed by parameter, with initial values and bounds for each
# @everywhere pb = DataStructures.OrderedDict(zip(
#     par_init.par,
#     DataFrames.eachrow(par_init[:, [:value, :lb, :ub]]),
# ));


### DIRECT OPTIMIZATION

# store as initial parameter vector in mprob object
@everywhere SMM.addSampledParam!(mprob,pb);


@everywhere function f!(m, x)
    # wrap param vector in OrderedDict
    pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))

    # evaluate, extracting moments
    m .= collect(values(stb_obj(SMM.Eval(mprob, pb_val); dt=stbdat, store_moments=true).simMoments))
end

### use finite dif jacobian in standard optim
jaccache = FiniteDiff.JacobianCache(zeros(Float64, length(to_optimize)), zeros(Float64, length(SMM.dataMoment(SMM.Eval(mprob)))), Val{:forward});
@everywhere jac = SharedArrays.SharedArray(zeros(Float64, length(SMM.dataMoment(SMM.Eval(mprob))), length(to_optimize)));


@everywhere function fgh!(F, G, H, x)
    # wrap param vector in OrderedDict
    pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))

    # wrap in Eval object
    ev = SMM.Eval(mprob, pb_val)
    if ((F != nothing) & (G == nothing) & (H == nothing))
        return stb_obj(ev; dt=stbdat).value
    end
    if (G != nothing)
        # data moments
        m0 = SMM.dataMoment(ev)

        # Jacobian from finite differences
        parallel_jacobian!(jac,f!,x,jaccache; relstep = Real(0.001), absstep= Real(0.01))
        # jac = parallel_jacobian(f!,x,jaccache)

        # Weights vector
        w = SMM.dataMomentW(ev)

        # gradient is 2 J' W (m - m0)
        G .= 2 .* jac' * (w .* (jaccache.fx .- m0));
        if (H != nothing)
            # Hessian is 2 J' W J
            H .= 2 .* jac' * (w .* jac)
        end
        if (F != nothing)
            # return objective value
            return 4128 / 2 * sum(((jaccache.fx .- m0) .^ 2) .* w)
        end
    end
    if (H != nothing)
        m0 = SMM.dataMoment(ev)
        parallel_jacobian!(jac,f!,x,jaccache; relstep = Real(0.001), absstep= Real(0.01))
        # jac = parallel_jacobian(f!,x,jaccache)
        w = SMM.dataMomentW(ev)
        H .= 2 .* jac' * (w .* jac)
        if (F != nothing)
            # return objective value
            return 4128 / 2 * sum(((jaccache.fx .- m0) .^ 2) .* w)
        end
    end

    nothing

end

## define subset of pars to optimize over
@everywhere to_optimize = String.(par_init.par[findall(occursin.(r"info|slant", par_init.par))])
# @everywhere to_optimize = String.(par_init.par)


# ## grid search
# x0 = [pb[k].value for k in to_optimize]
# x0_save = copy(x0)
# f = 0.0
#
# @everywhere using Plots
#
#
# grid = @distributed (hcat) for p in to_optimize
#     println(p)
#     x0 .= x0_save
#     i = findfirst(p .== to_optimize)
#     beg_seq = min((x0[i] * 0.8), (x0[i] * 1.2))
#     end_seq = max((x0[i] * 0.8), (x0[i] * 1.2))
#     step = (end_seq - beg_seq) / 50
#     xs = beg_seq:step:end_seq
#     fs = [fgh!(f, nothing, nothing, setindex!(x0, x, i)) for x in xs]
#     png(plot(xs, fs), "/home/gregorymartin/Dropbox/STBNews/data/model/grid_plots/" * p)
#     fs
# end
#
#
# CSV.write(file="/home/gregorymartin/Dropbox/STBNews/data/model/grid_plots/grid_search.csv", DataFrame(grid))

### use LeastSquaresOptim
import LeastSquaresOptim
x0 = [pb[k].value for k in to_optimize]


m0 = SMM.dataMoment(SMM.Eval(mprob))
w = SMM.dataMomentW(SMM.Eval(mprob))

@everywhere function f1!(m :: Vector{Float64}, x :: Vector{Float64})
    # wrap param vector in OrderedDict
    pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))

    # evaluate, extracting moments
    m .= sqrt(w) .* (collect(values(stb_obj(SMM.Eval(mprob, pb_val); dt=stbdat, store_moments=true).simMoments)) - m0)
end

lsopt = LeastSquaresOptim.optimize!(LeastSquaresOptim.LeastSquaresProblem(;x=x0, f! = f1!, output_length=length(m0), autodiff = :central))


### Levenberg-Marquardt
import LinearAlgebra
lm_lambda = 1.0
lm_lambda_up = 2.0
lm_lambda_down = 3.0
cur_fv = 0.0
last_fv = 0.0
g = zeros(Float64, length(x0));
JtJ = zeros(Float64, length(x0), length(x0));
DtD = LinearAlgebra.Diagonal(ones(Float64, length(x0)));
it = 0
it_max = 10

while it < it_max
    it += 1
    println(x0)
    cur_fv = fgh!(0, g, JtJ, x0)
    last_fv = copy(cur_fv)
    lm_step = -(JtJ + lm_lambda .* DtD) \ g

    if (sum(g.^2) < 1e-4)
        break
    end

    while ((lm_lambda >= 1e-6) & (lm_lambda <= 100))
        cur_fv = fgh!(0, nothing, nothing, x0 .+ lm_step);
        if (cur_fv < last_fv)
            lm_lambda /= lm_lambda_down
            x0 .= x0 .+ lm_step
            break
        else
            lm_lambda *= lm_lambda_up
            lm_step = -(JtJ + lm_lambda .* DtD) \ g
        end
    end

    if abs(last_fv - cur_fv) <= 1e-4
        break
    end
end


### pure newton method (bad)

optimized = Optim.optimize(Optim.only_fgh!(fgh!),
    [pb[k].value for k in to_optimize],
    Optim.Newton(;linesearch=LineSearches.BackTracking(iterations=10)),
    Optim.Options(show_trace=true, iterations=100, show_every=1)
    )

CSV.write(file="/home/gregorymartin/Dropbox/STBNews/data/model/optim_params.csv", DataFrame(Optim.minimizer(optimized)))


@everywhere function f0(x)
    # wrap param vector in OrderedDict
    pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))

    # wrap in Eval object
    ev = SMM.Eval(mprob, pb_val)

    stb_obj(ev; dt=stbdat).value
end

BlackBoxOptim.bboptimize(f0; SearchSpace = BlackBoxOptim.ContinuousRectSearchSpace([10, -0.1], [20, -0.0001]),
    PopulationSize = 72, MaxFuncEvals = 200, TraceInterval = 1.0, TraceMode = :verbose)


### TESTING ZONE ###
# # to evaluate at start point
# obj = stb_obj(SMM.Eval(mprob); dt=stbdat)
