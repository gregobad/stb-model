### Loads all data necessary to evaluate objective ###
### expects variables code_dir, data_dir

## module loads
import CSV
import Random
import Statistics
import Distributions
import StatsFuns
import StatsBase
# in pkg mode:
# remove SMM
# add https://github.com/gregobad/SMM.jl#better_parallel
import SMM
import DataStructures
using FileIO
using DataFrames

## data structure containing objective function arguments
# note: don't have linebreaks here, causes it to break on server for some reason
struct STBData
    ## number of channels, days, topics, etc.
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
    i_etz::Vector{Int64}
    i_ctz::Vector{Int64}
    i_mtz::Vector{Int64}
    i_ptz::Vector{Int64}
    i_national::Vector{Int64}
    i_stb::Vector{Int64}
    ## show to channel index
    show_to_channel::Vector{Int64}
    ## random draws
    channel_report_errors::Array{Float64,3}
    consumer_choice_draws::Array{Float64,2}
    consumer_free_errors::Array{Float64,3}
    pre_consumer_channel_draws::Array{Float64,2}
    # pre_consumer_news_draws::Array{Float64}
    # pre_consumer_channel_zeros::Array{Float64,2}
    pre_consumer_news_zeros::Array{Float64}
    ## symbols to access parts of the parameter vector
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
    ## indices of nonzero topic innovations
    innov_index::Vector{Int64}
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
viewership = convert(Matrix, CSV.read("nielsen_ratings.csv")[:,3:5]);

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


const data_moments = cat(
    viewership_indiv_rawmoments[:, :value],     # STB moments
    reshape(transpose(viewership), C * T),      # block ratings (nielsen data)
    polling[1:110],                             # daily polling (up to election day)
    0;                                          # ridge penalty term for innovations
    dims = 1,
);
w = ones(Float64, length(data_moments));
w[22:30] .= 1/10;  # weight average-minutes-by-tercile moments by 1/1000
w[end] = 1e-6      # small weight for innovation penalty
# w[40:12423] .= 500;  # weight daily viewership by 500
# w[12424:12533] .= 500;  # weight polling by 500

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
par_init_og = CSV.read("parameter_bounds.csv");
par_init_og.ub = Float64.(par_init_og.ub);
par_init_og.lb = Float64.(par_init_og.lb);

# zero out the tz-hour dummies
par_init_og = par_init_og[findall(.! occursin.(r"beta:h\d", par_init_og.par)),:]

## read non-zero innovation indices
non_zero_indices = CSV.read("topic_path_sparsity.csv")
par_init_og_main = filter(row -> !occursin(r"^\d+_t\d+", row[:par]), par_init_og)
par_init_og = [par_init_og_main; join(par_init_og, non_zero_indices, on =:par, kind=:semi)]


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
    Symbol.([k for k in par_init_og.par if occursin("topic_lambda", k)]),
    Symbol.([k for k in par_init_og.par if occursin("topic_leisure", k)]),
    Symbol.([k for k in par_init_og.par if occursin("topic_mu", k)]),
    Symbol.([k for k in par_init_og.par if occursin("channel_location", k)]),
    Symbol.([k for k in par_init_og.par if occursin("beta:show", k)]),
    Symbol.([k for k in par_init_og.par if occursin("beta:channel_mu", k)]),
    Symbol.([k for k in par_init_og.par if occursin("beta:channel_sigma", k)]),
    Symbol.([k for k in par_init_og.par if occursin(":etz", k)]),
    Symbol.([k for k in par_init_og.par if occursin(":ctz", k)]),
    Symbol.([k for k in par_init_og.par if occursin(":mtz", k)]),
    Symbol.([k for k in par_init_og.par if occursin(":ptz", k)]),
    Symbol.([k for k in par_init_og.par if occursin(r"^[0-9]+_t[0-9]", k)]),
    non_zero_indices.index
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
