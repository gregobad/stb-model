### OPTIONS: change things here ###

## number of cores to use in parallel
# const numprocs = 2;
const numprocs = 1;
using Distributed
addprocs(numprocs)

# Set your local directory
# @everywhere local_dir = "/home/gregorymartin/Dropbox/STBNews"
# @everywhere local_dir = "/Users/mlinegar/Dropbox/STBNews"
@everywhere local_dir = "/home/cfuser/mlinegar"
## Directory locations for code and data
@everywhere using Printf
@everywhere code_dir = @sprintf("%s/code/model/julia/", local_dir)
@everywhere data_dir = @sprintf("%s/data/model/", local_dir)
## number of iterations for each MCMC chain
const niter = 1;
### END OPTIONS ###


## number of chains to run
nchains = numprocs;

## module loads
@everywhere import CSV
@everywhere import Random
@everywhere import Statistics
@everywhere import Distributions
@everywhere import StatsFuns
@everywhere import StatsBase
# in pkg mode:
# remove SMM
# add https://github.com/gregobad/SMM.jl#better_parallel
@everywhere import SMM
@everywhere import DataFrames
@everywhere import DataStructures
@everywhere using FileIO
@everywhere using DataFrames

## data structure containing objective function arguments
# note: don't have linebreaks here, causes it to break on server for some reason
@everywhere struct STBData
    # number of channels, days, topics, etc.
    C::Int64
    T::Int64
    K::Int64
    t_i::Int64
    D::Int64
    N::Int64
    N_national::Int64
    N_stb::Int64
    election_day::Int64
    S::Int64
    # show index arrays
    channel_show_etz::Array{Int64,2}
    channel_show_ctz::Array{Int64,2}
    channel_show_mtz::Array{Int64,2}
    channel_show_ptz::Array{Int64,2}
    # topic weight arrays
    channel_topic_coverage_etz::Array{Float64,3}
    channel_topic_coverage_ctz::Array{Float64,3}
    channel_topic_coverage_mtz::Array{Float64,3}
    channel_topic_coverage_ptz::Array{Float64,3}
    # consumer attributes
    consumer_tz::Vector{Int64}
    consumer_r_prob::Vector{Float64}
    i_etz::Vector{Int64}
    i_ctz::Vector{Int64}
    i_mtz::Vector{Int64}
    i_ptz::Vector{Int64}
    i_national::Vector{Int64}
    i_stb::Vector{Int64}
    # show to channel index
    show_to_channel::Vector{Int64}
    # random draws
    channel_report_errors::Array{Float64,3}
    consumer_choice_draws::Array{Float64,2}
    consumer_free_errors::Array{Float64,3}
    pre_consumer_channel_draws::Array{Float64,2}
    # pre_consumer_news_draws::Array{Float64}
    # pre_consumer_channel_zeros::Array{Float64,2}
    pre_consumer_news_zeros::Array{Float64}
    # symbols to access parts of the parameter vector
    keys_lambda::Array{Symbol,1}
    keys_leisure::Array{Symbol,1}
    keys_mu::Array{Symbol,1}
    keys_channel_loc::Array{Symbol,1}
    keys_show::Array{Symbol,1}
    keys_channel_mu::Array{Symbol,1}
    keys_channel_sigma::Array{Symbol,1}
    # keys_zeros::Array{Symbol,1}
    keys_etz::Array{Symbol,1}
    keys_ctz::Array{Symbol,1}
    keys_mtz::Array{Symbol,1}
    keys_ptz::Array{Symbol,1}
    keys_innovations::Array{Symbol,1}
    # path::Array{Float64,2}
end



### LOAD OBJECTIVE ###
## parallel version ##
@everywhere cd(code_dir)

@everywhere include("helper_functions.jl")
@everywhere include("sim_viewership_polling.jl")
@everywhere include("stb_objective.jl")
@everywhere include("algoSTB.jl")

@everywhere cd(data_dir)

## sequential version ##
# cd(code_dir)
# include("helper_functions.jl")
# include("sim_viewership_polling.jl")
# include("stb_objective.jl")
# include("algoSTB.jl")
# cd(data_dir)

## channels in choice set
const chans = ["cnn"; "fnc"; "msnbc"];
const C = length(chans);

## read topic matrices
channel_topic_and_show_etz = read_topics(chans, "ETZ");
channel_topic_and_show_ctz = read_topics(chans, "CTZ");
channel_topic_and_show_mtz = read_topics(chans, "MTZ");
channel_topic_and_show_ptz = read_topics(chans, "PTZ");

## dimensions
const T = size(channel_topic_and_show_etz[1])[1];
const K = size(channel_topic_and_show_etz[1])[2] - 1;
const t_i = 24;
const D = convert(Int, T / t_i);

## extract show id's (first col of topic matrices)
# result is (num_times x num_channels)
const channel_show_etz = permutedims(
    convert(
        Array{Int,2},
        hcat(map(x -> x[:, 1], channel_topic_and_show_etz)...),
    ),
    [2, 1],
);
const channel_show_ctz = permutedims(
    convert(
        Array{Int,2},
        hcat(map(x -> x[:, 1], channel_topic_and_show_ctz)...),
    ),
    [2, 1],
);
const channel_show_mtz = permutedims(
    convert(
        Array{Int,2},
        hcat(map(x -> x[:, 1], channel_topic_and_show_mtz)...),
    ),
    [2, 1],
);
const channel_show_ptz = permutedims(
    convert(
        Array{Int,2},
        hcat(map(x -> x[:, 1], channel_topic_and_show_ptz)...),
    ),
    [2, 1],
);

## remaining cols are topic weights
# size: (num_topics x num_channels x num_times)
const channel_topic_coverage_etz = reshape(
    vcat(map(x -> x[:, 2:end]', channel_topic_and_show_etz)...),
    K,
    C,
    T,
);
const channel_topic_coverage_ctz = reshape(
    vcat(map(x -> x[:, 2:end]', channel_topic_and_show_ctz)...),
    K,
    C,
    T,
);
const channel_topic_coverage_mtz = reshape(
    vcat(map(x -> x[:, 2:end]', channel_topic_and_show_mtz)...),
    K,
    C,
    T,
);
const channel_topic_coverage_ptz = reshape(
    vcat(map(x -> x[:, 2:end]', channel_topic_and_show_ptz)...),
    K,
    C,
    T,
);

#@ read STB household data (STB sample)
stb_hh = CSV.read("stb_hh_sample.csv");
const N_stb = size(stb_hh, 1);

## read CCES household data (national sample)
national_hh = CSV.read("cces_2012.csv");
const N_national = size(national_hh, 1);

const N = N_national + N_stb;

## individual indices
const consumer_tz = [national_hh[:, :timezone]; stb_hh[:, :timezone]];
const consumer_sample_index = cat(
    zeros(Int, N_national),
    ones(Int, N_stb);
    dims = 1,
);
const i_etz = findall(consumer_tz .== 1);
const i_ctz = findall(consumer_tz .== 2);
const i_mtz = findall(consumer_tz .== 3);
const i_ptz = findall(consumer_tz .== 4);
const i_national = findall(consumer_sample_index .== 0);
const i_stb = findall(consumer_sample_index .== 1);

const consumer_r_prob = [national_hh[:, :r_prop]; stb_hh[:, :r_prop]];

## read (national) viewership data
viewership = CSV.read("nielsen_ratings.csv");
viewership_etz = convert(Matrix, viewership[:, [3, 7, 11]]) ./ 100;  # nielsen viewership numbers are in percentage points
viewership_ctz = convert(Matrix, viewership[:, [4, 8, 12]]) ./ 100;
viewership_mtz = convert(Matrix, viewership[:, [5, 9, 13]]) ./ 100;
viewership_ptz = convert(Matrix, viewership[:, [6, 10, 14]]) ./ 100;

## read polls
polling = CSV.read("polling.csv")[:, :obama_2p];
const election_day = findlast(polling .> 0);

## read individual moments
viewership_indiv_rawmoments = CSV.read("viewership_indiv_rawmoments.csv");

## read in show to channel mapping
const show_to_channel = CSV.read("show_to_channel.csv")[:, :channel_index];
const S = length(show_to_channel) - C;

## simulated draws
rng = Random.MersenneTwister(72151835);

const channel_report_errors = Random.randn(rng, K, C, T);
const consumer_choice_draws = Random.rand(rng, N_stb + N_national, T);
const consumer_free_errors = Random.randn(rng, N_stb + N_national, K, D);
const pre_consumer_news_draws = Random.randexp(rng, N_stb + N_national, 1); # not used
const pre_consumer_channel_draws = Random.randn(rng, N_stb + N_national, C);
const pre_consumer_news_zeros = Random.rand(rng, N_stb + N_national, 1);
const pre_consumer_channel_zeros = Random.rand(rng, N_stb + N_national, C);  # not used

## to read saved MATLAB version for comparison purposes
# using MAT
# mt = matread("/Users/mlinegar/Dropbox/STBNews/Mitchell_work/v9/test_inside_stb_obj_func_split_simple.mat")
# kmt=collect(keys(mt));
# const channel_report_errors = mt["channel_report_errors"];
# const consumer_choice_draws = mt["consumer_choice_draws"];
# const pre_consumer_channel_draws = mt["pre_consumer_channel_draws"];
# const consumer_free_errors = mt["consumer_free_errors"];
# const pre_consumer_news_draws = mt["pre_consumer_news_draws"];

# const N_national = mt["N_national"];
# const consumer_tz = round.(Int,mt["consumer_tz"])[:];
# const N_national = mt["N_national"];
# const N_stb = mt["N_stb"];
# const N = mt["N"];
# const consumer_sample_index = round.(Int,mt["consumer_sample_index"])[:];
# const i_etz = findall(consumer_tz.==1);
# const i_ctz = findall(consumer_tz.==2);
# const i_mtz = findall(consumer_tz.==3);
# const i_ptz = findall(consumer_tz.==4);
# const i_national = findall(consumer_sample_index.==0);
# const i_stb = findall(consumer_sample_index.==1);
# const consumer_r_prob = mt["consumer_r_prob"][:];


## construct moment vector in data
raw_mean_data_ratings = transpose((viewership_etz + viewership_ctz +
                                   viewership_mtz + viewership_ptz) / 4);
const data_moments = cat(
    viewership_indiv_rawmoments[:, :value],
    reshape(raw_mean_data_ratings, C * T),
    polling[1:110];
    dims = 1,
);
w = ones(Float64, length(data_moments));
w[22:30] .= 1/100000;  # weight average-minutes-by-tercile moments by 1/1000

moms = DataFrames.DataFrame(
    name = [
        viewership_indiv_rawmoments[:, :stat]
        repeat(chans; outer = T) .* " block " .*
        string.(repeat(1:T; inner = length(chans)))
        "poll day " .* string.(1:election_day)
    ],
    value = data_moments,
    weight = w,
)


## read initial parameter vector and bounds
par_init_og = CSV.read("parameter_bounds.csv");
par_init_og.ub = Float64.(par_init_og.ub);
par_init_og.lb = Float64.(par_init_og.lb);

## replace with parameter values from last accepted change from previous run of MCMC
# par_init = CSV.read("mcmc_params.csv");
# par_init = CSV.read("fit_channel_pct_viewing_pars.csv");
# par_init = CSV.read("best_zeroed_out_utility_pars_2.csv");
# par_init = CSV.read("manual_calibration_params.csv");

par_init = CSV.read("MCMC_chain1.csv");
# par_init.accepted = par_init.accepted .== "TRUE" # reading in as string, not logical
par_init = par_init[(par_init.accepted.==1),:];
best_draw = argmin(par_init.curr_val);
par_init_val = collect(Float64, par_init[best_draw,8:size(par_init, 2)]);
par_init_par = string.(names(par_init[:,8:size(par_init, 2)]));
 par_init = DataFrames.DataFrame(
    par = par_init_par,
    value = par_init_val
);
par_init = join(par_init, par_init_og, on = :par, kind=:right);
par_init.value = coalesce.(par_init.value, 0.0)

# the following 2 lines zero out the tz-hour and show dummies, respectively
filter!(row -> !occursin(r"beta:h\d", row[:par]), par_init)
# filter!(row -> !occursin("beta:show", row[:par]), par_init)


# set start value for channel_beta and channel_mu for first run
if (par_init[par_init.par.=="beta:channel_mu:cnn",:value][1] == 0.0)
    par_init.value[occursin.("beta:channel_mu", par_init.par)] .= [ -2.36240221908309, -3.5994403225633507,-2.6661747358361767];
    par_init.value[occursin.("beta:channel_sigma", par_init.par)] .=[1.0, 1.0,1.0];
end




## dictionary indexed by parameter, with initial values and bounds for each
pb = DataStructures.OrderedDict(zip(
    par_init.par,
    DataFrames.eachrow(par_init[:, [:value, :lb, :ub]]),
));


## construct data object to pass to objective fun
stbdat = STBData(
    C,
    T,
    K,
    t_i,
    D,
    N,
    N_national,
    N_stb,
    election_day,
    S,
    channel_show_etz,
    channel_show_ctz,
    channel_show_mtz,
    channel_show_ptz,
    channel_topic_coverage_etz,
    channel_topic_coverage_ctz,
    channel_topic_coverage_mtz,
    channel_topic_coverage_ptz,
    consumer_tz,
    consumer_r_prob,
    i_etz,
    i_ctz,
    i_mtz,
    i_ptz,
    i_national,
    i_stb,
    show_to_channel,
    channel_report_errors,
    consumer_choice_draws,
    consumer_free_errors,
    pre_consumer_channel_draws,
    pre_consumer_news_zeros,
    Symbol.([k for k in keys(pb) if occursin("topic_lambda", k)]),
    Symbol.([k for k in keys(pb) if occursin("topic_leisure", k)]),
    Symbol.([k for k in keys(pb) if occursin("topic_mu", k)]),
    Symbol.([k for k in keys(pb) if occursin("channel_location", k)]),
    Symbol.([k for k in keys(pb) if occursin("beta:show", k)]),
    Symbol.([k for k in keys(pb) if occursin("beta:channel_mu", k)]),
    Symbol.([k for k in keys(pb) if occursin("beta:channel_sigma", k)]),
    Symbol.([k for k in keys(pb) if occursin(":etz", k)]),
    Symbol.([k for k in keys(pb) if occursin(":ctz", k)]),
    Symbol.([k for k in keys(pb) if occursin(":mtz", k)]),
    Symbol.([k for k in keys(pb) if occursin(":ptz", k)]),
    Symbol.([k for k in keys(pb) if occursin(r"^[0-9]+_t[0-9]", k)])
    # path,
);



# Initialize an empty MProb() object:
#------------------------------------
mprob = SMM.MProb();

# Add structural parameters to MProb():
# specify starting values and support
#--------------------------------------

# # run on full set of parameters
SMM.addSampledParam!(mprob,pb);

# run only on subset of parameters
# SMM.addSampledParam!(mprob, pb_nonpath);
# SMM.addParam!(mprob, pb_path_value); # not run on these


# Add moments to be matched to MProb():
#--------------------------------------
SMM.addMoment!(mprob, moms);

# Attach an objective function to MProb():
#----------------------------------------
# SMM.addEvalFunc!(mprob, stb_obj, options  => stbdat);
SMM.addEvalFunc!(mprob, stb_obj);
SMM.addEvalFuncOpts!(mprob, Dict(:dt => stbdat));

## estimation options:
opts = Dict(
    "N" => nchains,
    # "maxiter" => 5,
    "maxiter"=>niter,
    "maxtemp" => numprocs,
    "sigma" => 0.005,
    "sigma_update_steps" => 10,
    "sigma_adjust_by" => 0.05,
    "smpl_iters" => 1000,
    "parallel" => true,
    "min_improve" => [0.0 for i = 1:nchains],
    "acc_tuners" => [1.0 for i = 1:nchains],
    "animate" => false,
    "batch_size" => 8,
    "single_par_draw" => true     # set true to enable one-parameter-at-a-time jumps
    # "save_frequency"=>100
    # "filename"=>"MCMC_chain_state.jld2"
    # dist_fun => -
);

## set-up BGP algorithm (modified version that respects lambdas summing to 1 constraint):
MA = MAlgoSTB(mprob, opts);
wp = CachingPool(workers());

### MAIN MCMC LOOP ###
# not run
# SMM.run!(MA);


### POST PROCESSING OF RESULTS ###
# summary(MA)
# chain1 = history(MA.chains[1]);

# CSV.write("MCMC_chain1_greg_test_1000.csv", chain1)
# var = [MA.chains[1].sigma]
# var = DataFrame(var = var)
# CSV.write("MCMC_variance.csv", var, writeheader=false)

# MA.chains[1].evals[].simMoments
# MA.chains[1].evals[].dataMoments
# datamoments = SMM.dataMomentd(MA.chains[1].evals[])
# simmoments = SMM.check_moments(MA.chains[1].evals[])
# CSV.write("julia_MCMC_moments.csv", simmoments)



### TESTING ZONE ###

# stb_obj(SMM.Eval(mprob); dt=stbdat, save_output=true, store_moments=true)

obj = stb_obj(SMM.Eval(mprob); dt=stbdat, save_output=true, store_moments=true)
# MA.chains[1].evals[].simMoments
# MA.chains[1].evals[].dataMoments
# datamoments = SMM.dataMomentd(MA.chains[1].evals[])
# simmoments = SMM.check_moments(MA.chains[1].evals[])
simmoments = SMM.check_moments(obj)
CSV.write("julia_MCMC_moments_temp.csv", simmoments)

STOP
#
# ## test objective function
# @time obj = stb_obj(SMM.Eval(mprob); dt=stbdat);
#
# ## to produce saved output for std output graphs
#
# # get result of MCMC run
# accepted_draws = chain1[(chain1.accepted.==1),:];
# median_val = collect(Float64,
#     mapcols(Statistics.median, chain1[:,8:size(chain1,2)])[1,:]
# );
# median_val[1:K] .= median_val[1:K] ./ sum(median_val[1:K])
# median_par = string.(names(chain1[:,8:size(chain1, 2)]));
# med_pb = DataStructures.OrderedDict(zip(
#     median_par,
#     median_val)
# );
#
# shows = filter((k,v) -> in(k, string.(stbdat.keys_show)), med_pb)
#
# stb_obj(SMM.Eval(mprob, med_pb); dt=stbdat, save_output=false)
# med_pb_tweak = copy(med_pb)
# med_pb_tweak["beta:show:foxreportwithshepardsmith"] = -17
#
#
# stb_obj(SMM.Eval(mprob, med_pb_tweak); dt=stbdat, save_output=false)
#
#
# Juno.@enter()
#
#
#
# ## test objective function
# @time obj = stb_obj(SMM.Eval(mprob); dt=stbdat);
#
# ## to produce saved output for std output graphs
#
# # get result of MCMC run
# accepted_draws = chain1[(chain1.accepted.==1),:];
# median_val = collect(Float64,
#     mapcols(Statistics.median, chain1[:,8:size(chain1,2)])[1,:]
# );
# median_val[1:K] .= median_val[1:K] ./ sum(median_val[1:K])
# median_par = string.(names(chain1[:,8:size(chain1, 2)]));
# med_pb = DataStructures.OrderedDict(zip(
#     median_par,
#     median_val)
# );
#
# shows = filter((k,v) -> in(k, string.(stbdat.keys_show)), med_pb)
#
# stb_obj(SMM.Eval(mprob, med_pb); dt=stbdat, save_output=false)
# med_pb_tweak = copy(med_pb)
# med_pb_tweak["beta:show:foxreportwithshepardsmith"] = -17
#
#
# stb_obj(SMM.Eval(mprob, med_pb_tweak); dt=stbdat, save_output=false)

## experimenting with hour dummies
pb_val = DataStructures.OrderedDict(k => pb[k][1] for k = keys(pb))
ev1 = stb_obj(SMM.Eval(mprob, pb_val); dt=stbdat, save_output=false)
pb_val_tweak = copy(pb_val)

## grid search
function obj_mod(p::DataStructures.OrderedDict{Any, Any}, par_to_tweak::String, tweak_val::Float64; save_output=false)
    p[par_to_tweak] = tweak_val
    ev = stb_obj(SMM.Eval(mprob, p); dt=stbdat, save_output=save_output)
    ev.value
end

# obj_mod(pb_val_tweak, "beta:h7:mtz", -20.0)
# obj_mod(pb_val_tweak, "consumer_state_var_0", 0.001, save_output=true)

function obj_mod2(p::DataStructures.OrderedDict{Any, Any}, pars_to_tweak::Array{String,1}, tweak_val::Float64; save_output=false)
    [p[pars_to_tweak] = tweak_val for par in pars_to_tweak] # set all pars to same value (will come back and improve)
    ev = stb_obj(SMM.Eval(mprob, p); dt=stbdat, save_output=save_output)
    ev.value
end

# some_pars = ["zero:news", "zero:channel:cnn", "zero:channel:fnc", "zero:channel:msnbc"]
# some_jitters = [0.0]
# obj_mod2(pb_val_tweak, some_pars, some_jitters, save_output=true)


function obj_mod3(p::DataStructures.OrderedDict{Any, Any}, pars_to_tweak::Array{String,1}, tweak_val::Array{Float64,1}; save_output=false, store_moments=false)
    p_copy = copy(p)
    [p_copy[pars_to_tweak[i]] = p_copy[pars_to_tweak[i]] * tweak_val[i] for i in 1:length(pars_to_tweak)]
    # [println(p[pars_to_tweak[i]]) for i in 1:length(pars_to_tweak)]
    [println(pars_to_tweak[i], " * ", tweak_val[i], " = ", p_copy[pars_to_tweak[i]]) for i in 1:length(pars_to_tweak)]
    ev = stb_obj(SMM.Eval(mprob, p_copy); dt=stbdat, save_output=save_output, store_moments=store_moments)
    println("Objective Value: ", ev.value)
    ev
end

function par_copy(p::DataStructures.OrderedDict{Any, Any}, pars_to_tweak::Array{String,1}, tweak_val::Array{Float64,1}; save_output=false, store_moments=false)
    p_copy = copy(p)
    [p_copy[pars_to_tweak[i]] = p_copy[pars_to_tweak[i]] * tweak_val[i] for i in 1:length(pars_to_tweak)]
    p_copy
end

function moment_distance(df::DataFrame, regex_moms)
    # regex_moms = @sprintf(r"%s", moms)
    filtered_moments = df[[occursin(regex_moms, String(moment)) for moment in df[:moment]], :]
    println(Statistics.mean(filtered_moments[:distance]))
end

some_pars = ["beta:slant", "zero:news", "beta:info", "beta:channel_mu:cnn", "beta:channel_mu:fnc", "beta:channel_mu:msnbc", "beta:channel_sigma:cnn", "beta:channel_sigma:fnc", "beta:channel_sigma:msnbc", "topic_leisure:foreign_policy", "topic_leisure:economy",   "topic_leisure:crime",  "topic_leisure:horse_race"]
# some_pars = ["beta:slant", "zero:news", "beta:info", "beta:channel:cnn", "beta:channel:fnc", "beta:channel:msnbc", "channel_location:cnn", "channel_location:fnc", "channel_location:msnbc", "topic_leisure:foreign_policy", "topic_leisure:economy",	"topic_leisure:crime",	"topic_leisure:horse_race"]
# some_pars = ["beta:slant", "zero:news", "beta:info", "beta:channel:cnn", "beta:channel:fnc", "beta:channel:msnbc", "beta:show:cnn", "beta:show:fnc", "beta:show:msnbc", "topic_leisure:foreign_policy", "topic_leisure:economy",	"topic_leisure:crime",	"topic_leisure:horse_race", "109_t01", "109_t02", "109_t03", "109_t04", "086_t01", "086_t02", "086_t03", "086_t04", "092_t01", "092_t02", "092_t03", "092_t04", "095_t01", "095_t02", "095_t03", "095_t04", "099_t01", "099_t02", "099_t03", "099_t04"] # "channel_location:cnn", "channel_location:fnc", "channel_location:msnbc",
# some_pars = ["059_t01", "059_t02", "059_t03", "059_t04", "060_t01", "060_t02", "060_t03", "060_t04", "061_t01", "061_t02", "061_t03", "061_t04"]
# some_pars = ["zero:news", "beta:news", "zero:channel:cnn", "beta:channel:cnn", "zero:channel:fnc", "beta:channel:fnc", "zero:channel:msnbc", "beta:channel:msnbc"]
# some_pars = ["channel_report_var", "109_t01", "109_t02", "109_t03", "109_t04", "086_t01", "086_t02", "086_t03", "086_t04", "092_t01", "092_t02", "092_t03", "092_t04", "095_t01", "095_t02", "095_t03", "095_t04", "099_t01", "099_t02", "099_t03", "099_t04"]
# some_jitters = [0.1, 100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
# some_pars = ["beta:info"]
# some_jitters = [150.0, 15.0, 70.0, 13.0, 0.1] # makes zeroes too close to 1
# some_jitters = [1.0, 2.1, 1.0, 5.0, 10.0, 10.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
# some_jitters = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# some_jitters = [0.8, 1.2, 3.0, 0.53, 0.45, 0.4, 0.9, 0.7, 0.7, -0.45, -0.45, -0.45, 15.0, 10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0] # 1.0, 1.0, 1.0,
# some_pars = ["beta:slant", "zero:news", "beta:info", "beta:channel:cnn", "beta:channel:fnc", "beta:channel:msnbc", "channel_location:cnn", "channel_location:fnc", "channel_location:msnbc", "topic_leisure:foreign_policy", "topic_leisure:economy",	"topic_leisure:crime",	"topic_leisure:horse_race"]
# some_jitters = fill(1.0, length(some_pars))
# # some_jitters = [1.0, 1.0, 1000.0, 0.0001, 500.0, 0.85, 0.9, 0.9, 1.0, 1.0, 1.0, -0.05, -0.05, -0.05, 1.0, 100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0] # 1.0, 1.0, 1.0,
# some_jitters = [1.0, 1.5, 1.0, 1.2, 1.35, 1.23, 1.5, 1.5, 1.5, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0]
# some_jitters = fill(0.00001, length(some_pars))
# some_jitters = [15.0]

# some_jitters = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

# some_jitters = [0.0, 0.5, 0.0, -0.35, -0.3, -0.3, 0.4, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0];

some_jitters = fill(1.0, length(some_pars))

obj = obj_mod3(pb_val_tweak, some_pars, some_jitters, save_output=true,  store_moments = true)
simmoments = SMM.check_moments(obj);
simmoments[1:16,:]
simmoments[17:21,:]
simmoments[22:30,:]
simmoments[31:40,:]
simmoments[7880:7900,:]

CSV.write("julia_MCMC_moments_temp_temp.csv", simmoments)
# how well do we match overall viewership?
moment_distance(simmoments, r"cnn block")
moment_distance(simmoments, r"fnc block")
moment_distance(simmoments, r"msnbc block")

# how well do we match viewership on day 60 (too high on cnn)
moment_distance(simmoments, r"cnn block 14(4[0-9]|5[0-9]|6[0-3])")
println(" ") # break up output if running many times to make more clear

good_pars = par_copy(pb_val_tweak, some_pars, some_jitters)

# using DataStructures
# # https://github.com/JuliaData/DataFrames.jl/issues/591
# function createdataframe(input::OrderedCollections.OrderedDict{Any,Any})  
#   parsedinput = Dict()
#   for x in keys(input)
#     parsedinput[Symbol(x)] = [input[x]]
#   end
#   return DataFrame(parsedinput)
# end

# good_pars = DataFrame(;good_pars...)
header = ["par", "value"]
CSV.write("manual_calibration_params.csv", good_pars)

good_pars2 = CSV.read("manual_calibration_params.csv")

names!(good_pars2, Symbol.(header))

CSV.write("manual_calibration_params.csv", good_pars2, header = header)

good_pars3 = CSV.read("manual_calibration_params.csv")

