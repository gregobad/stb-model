### Loads all data necessary to evaluate objective ###
### expects variables code_dir, data_dir

## module loads
import CSV
import Tables
import Random
import Statistics
import Distributions
import StatsFuns
import StatsBase
# in pkg mode:
# add https://github.com/gregobad/Halton.jl
import Halton
# in pkg mode:
# add https://github.com/gregobad/SMM.jl#better_parallel
import SMM
import DataStructures
using FileIO
using DataFrames
using LinearAlgebra

## data structure containing objective function arguments
# note: don't have linebreaks here, causes it to break on server for some reason
struct STBData
    ## number of channels, days, topics, etc.
    B::Int64
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
    ## show index arrays
    channel_show_etz::Array{Int64,2}
    channel_show_ctz::Array{Int64,2}
    channel_show_mtz::Array{Int64,2}
    channel_show_ptz::Array{Int64,2}
    ## topic weight arrays
    channel_topic_coverage_etz::Array{Float64,3}
    channel_topic_coverage_ctz::Array{Float64,3}
    channel_topic_coverage_mtz::Array{Float64,3}
    channel_topic_coverage_ptz::Array{Float64,3}
    ## consumer attributes
    consumer_tz::Vector{Int64}
    consumer_r_prob::Vector{Float64}
    r_prob_stb::Vector{Float64}
    i_etz::Vector{Int64}
    i_ctz::Vector{Int64}
    i_mtz::Vector{Int64}
    i_ptz::Vector{Int64}
    i_national::Vector{Int64}
    i_stb::Vector{Int64}
    sorted_inds::Vector{Int64}
    ## show to channel index
    show_to_channel::Vector{Int64}
    ## random draws for simulating
    channel_report_draws::Array{Float64,3}
    consumer_choice_draws::Array{Float64,2}
    pre_consumer_channel_draws::Array{Float64,2}
    pre_consumer_news_zeros::Array{Float64}
    pre_consumer_topic_draws::Array{Float64}
    pre_path_draws::Array{Float64}
    ## symbols to access parts of the parameter vector
    keys_lambda::Array{Symbol,1}
    keys_rho::Array{Symbol,1}
    keys_leisure::Array{Symbol,1}
    keys_mu::Array{Symbol,1}
    keys_channel_q_D::Array{Symbol,1}
    keys_channel_q_R::Array{Symbol,1}
    keys_channel_q_0::Array{Symbol,1}
    keys_show::Array{Symbol,1}
    keys_channel_mu::Array{Symbol,1}
    keys_channel_sigma::Array{Symbol,1}
    keys_etz::Array{Symbol,1}
    keys_ctz::Array{Symbol,1}
    keys_mtz::Array{Symbol,1}
    keys_ptz::Array{Symbol,1}
    keys_news::Array{Symbol,1}
    keys_r_news::Array{Symbol,1}
end

### LOAD OBJECTIVE ###
cd(code_dir)

include("helper_functions.jl")
include("sim_viewership_polling.jl")
include("stb_objective.jl")
include("algoSTB.jl")

cd(data_dir)

## channels in choice set
const chans = ["cnn"; "fnc"; "msnbc"];
const C = length(chans);
const topics = ["foreign_policy","economy","crime","horse_race"]

## read topic matrices
channel_topic_and_show_etz = read_topics(chans, "ETZ");
channel_topic_and_show_ctz = read_topics(chans, "CTZ");
channel_topic_and_show_mtz = read_topics(chans, "MTZ");
channel_topic_and_show_ptz = read_topics(chans, "PTZ");

## limit days input here if desired
const t_i = 24;
periods_to_use = reshape(((days_to_use.-1).*24)' .+ collect(1:24), length(days_to_use) * 24)


channel_topic_and_show_etz = [channel_topic_and_show_etz[i][periods_to_use, :] for i in 1:C]
channel_topic_and_show_ctz = [channel_topic_and_show_ctz[i][periods_to_use, :] for i in 1:C]
channel_topic_and_show_mtz = [channel_topic_and_show_mtz[i][periods_to_use, :] for i in 1:C]
channel_topic_and_show_ptz = [channel_topic_and_show_ptz[i][periods_to_use, :] for i in 1:C]

## dimensions
const T = size(channel_topic_and_show_etz[1])[1];
const K = size(channel_topic_and_show_etz[1])[2] - 1;
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
stb_hh = CSV.File("stb_hh_sample.csv")  |> DataFrame;
const N_stb = size(stb_hh, 1);

## read CCES household data (national sample)
national_hh = CSV.File("cces_2012.csv")  |> DataFrame;
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
viewership = CSV.File("nielsen_ratings.csv"; type=Float64, drop=[:date,:time_block]) |> Tables.matrix;
viewership = viewership[periods_to_use, :]

## read polls
polling = CSV.read("polling.csv", DataFrame)[days_to_use, :obama_2p];
const election_day = findlast(polling .> 0);

## read individual moments
viewership_indiv_rawmoments = CSV.File("viewership_indiv_rawmoments.csv")  |> DataFrame;

## read in show to channel mapping
const show_to_channel = CSV.read("show_to_channel.csv", DataFrame).channel_index;
const S = length(show_to_channel) - C;

## simulated draws
rng = Random.MersenneTwister(72151835);

const channel_report_draws = Random.rand(rng, C, T, K);
const consumer_choice_draws = Random.rand(rng, N_stb + N_national, T);
const pre_consumer_channel_draws = Random.randn(rng, N_stb + N_national, C);
const pre_consumer_news_zeros = Random.rand(rng, N_stb + N_national, 1);
# const pre_consumer_topic_draws = Random.randexp(rng, N_stb + N_national, K);  # heterogeneous topic tastes
const pre_consumer_topic_draws = ones(Float64, N_stb + N_national, K);  # homogeneous topic tastes

if !@isdefined B
    B = 1
end
pre_news_draws = zeros(Float64, K, D, B)

use_primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173]

for b in 1:B
    for k in 1:K
        @views Halton.HaltonSeq!(pre_news_draws[k,:,b], use_primes[k+(b-1)*K], skip = 401)
        @views Random.shuffle!(rng, pre_news_draws[k,:,b])
    end
end

const data_moments = cat(
    viewership_indiv_rawmoments[:, :value],     # STB moments
    reshape(transpose(viewership), C * T),      # block ratings (nielsen data)
    polling[1:election_day],                             # daily polling (up to election day)
    0;                                          # ridge penalty term for innovations
    dims = 1,
);

## weighting by 1/SE of the moment in data
w_df = CSV.read("moment_ses.csv", DataFrame)
w_df.block = replace.(w_df.moment, r"\w+_block_(\d+)" => s"\1")
w_df.poll_day = replace.(w_df.moment, r"poll_day_(\d+)" => s"\1")
block_moments_delete = [occursin(r"\w+_block_(\d+)", w_df.moment[i]) ? (parse(Int, w_df.block[i]) ∉ periods_to_use) : false for i in 1:nrow(w_df)]
poll_moments_delete = [occursin(r"poll_day_(\d+)", w_df.moment[i]) ? (parse(Int, w_df.poll_day[i]) ∉ days_to_use) : false for i in 1:nrow(w_df)]
delete!(w_df, cat(findall(block_moments_delete), findall(poll_moments_delete); dims=1))

w = 1 ./ w_df.se
push!(w, 1e-4)


## equal weighted version
# w = ones(Float64, length(data_moments));
# w[22:30] .= 1/10;  # weight average-minutes-by-tercile moments by 1/1000
# w[end] = 1e-4      # small weight for innovation penalty


## put data moments and weights into data frame
moms = DataFrames.DataFrame(
    name = [
        viewership_indiv_rawmoments[:, :stat]
        repeat(chans; outer = T) .* " block " .*
        string.(repeat(1:T; inner = length(chans)))
        "poll day " .* string.(1:election_day)
        "path penalty"
    ],
    value = data_moments,
    weight = w,
)


## read parameter definition file
par_init_og = CSV.File("parameter_bounds.csv"; delim=',') |> DataFrame
par_init_og.ub = Float64.(par_init_og.ub);
par_init_og.lb = Float64.(par_init_og.lb);

# zero out the tz-hour dummies
par_init_og = par_init_og[findall(.! occursin.(r"beta:h\d", par_init_og.par)),:]

# # eliminate channel heterogeneity
# par_init_og = par_init_og[findall(.! occursin.(r"beta:channel_", par_init_og.par)),:]

# keep only path parameters corresponding to selected days
day_index = match.(r"(\d+)_t(\d+)$", par_init_og.par)
par_init_og.day_index = [m == nothing ? 0 : parse(Int, m.captures[1]) for m in day_index]
delete!(par_init_og, [i for i=1:nrow(par_init_og) if (par_init_og.day_index[i] > 0) & !(par_init_og.day_index[i] ∈ days_to_use)])

## read sampling weights files (for MCMC)
cd(sampling_dir)
sampling_files = readdir(pwd())
sampling_tree_nodes = [m.captures[1] for m in match.(r"sample_tree_?(\w*)\.csv",sampling_files)]

deleteat!(sampling_files, [i for i=1:length(sampling_tree_nodes) if !occursin(r"^" * tree_base, sampling_tree_nodes[i])])
deleteat!(sampling_tree_nodes, [i for i=1:length(sampling_tree_nodes) if !occursin(r"^" * tree_base, sampling_tree_nodes[i])])

sampling_file_dict = Dict(zip(sampling_tree_nodes, sampling_files))

tree_depth = length.(split.(sampling_tree_nodes, "_"))
base_node = findfirst(tree_depth.==1)
sample_tree = read_node(sampling_file_dict, sampling_files[base_node], 1.0, tree_base)

# eliminate nodes of sample tree corresponding to path parameters for days not in days_to_use
if sample_tree.name == "path"
    path_node = sample_tree
elseif sample_tree.name == ""
    path_node = sample_tree.children[2]
else
    path_node = nothing
end

if path_node != nothing
    path_node.children = path_node.children[days_to_use]
    norm_probs = [n.prob for n in path_node.children]
    norm_probs ./= sum(norm_probs)
    path_node.conditional_dist = Distributions.Categorical(norm_probs)
end

if sample_tree.name == ""
    sample_tree.children[2] = path_node
end



## construct data object to pass to objective fun
stbdat = STBData(
    B,
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
    consumer_r_prob[i_stb],
    i_etz,
    i_ctz,
    i_mtz,
    i_ptz,
    i_national,
    i_stb,
    sortperm(consumer_r_prob[i_stb]),
    show_to_channel,
    channel_report_draws,
    consumer_choice_draws,
    pre_consumer_channel_draws,
    pre_consumer_news_zeros,
    pre_consumer_topic_draws,
    pre_news_draws,
    Symbol.([k for k in par_init_og.par if occursin("topic_lambda", k)]),
    Symbol.([k for k in par_init_og.par if occursin("topic_rho", k)]),
    Symbol.([k for k in par_init_og.par if occursin("topic_leisure", k)]),
    Symbol.([k for k in par_init_og.par if occursin("topic_mu", k)]),
    Symbol.([k for k in par_init_og.par if occursin("channel_q_D", k)]),
    Symbol.([k for k in par_init_og.par if occursin("channel_q_R", k)]),
    Symbol.([k for k in par_init_og.par if occursin("channel_q_0", k)]),
    Symbol.([k for k in par_init_og.par if occursin("beta:show", k)]),
    Symbol.([k for k in par_init_og.par if occursin("beta:channel_mu", k)]),
    Symbol.([k for k in par_init_og.par if occursin("beta:channel_sigma", k)]),
    Symbol.([k for k in par_init_og.par if occursin(":etz", k)]),
    Symbol.([k for k in par_init_og.par if occursin(":ctz", k)]),
    Symbol.([k for k in par_init_og.par if occursin(":mtz", k)]),
    Symbol.([k for k in par_init_og.par if occursin(":ptz", k)]),
    Symbol.([k for k in par_init_og.par if occursin(r"pr_news_[0-9]+_t[0-9]", k)]),
    Symbol.([k for k in par_init_og.par if occursin(r"pr_rep_[0-9]+_t[0-9]", k)])
);


# Initialize an empty MProb() object:
#------------------------------------
mprob = SMM.MProb();

# Add moments to be matched to MProb():
#--------------------------------------
SMM.addMoment!(mprob, moms);

# Attach an objective function to MProb():
#----------------------------------------
SMM.addEvalFunc!(mprob, stb_obj);
SMM.addEvalFuncOpts!(mprob, Dict(:dt => stbdat));
