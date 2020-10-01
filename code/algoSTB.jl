###################################
# Start defining STBChain
###################################
# for testing increase in size of variables
function debug_list_vals(m::Module=Main)
    res = DataFrames.DataFrame()
    vs = [v for v in sort!(names(m)) if isdefined(m, v)]
    for v in vs
        value = getfield(m, v)
        if !(value===Base || value===Main || value===Core ||
            value===InteractiveUtils || value===debug_list_vals)
            append!(res, DataFrame(v=v,size=Base.summarysize(value),
                                    summary=summary(value)))
        end
    end
    res
end



"""
# `STBChain`

MCMC Chain storage for BGP algorithm. This is the main datatype for the implementation of Baragatti, Grimaud and Pommeret (BGP) in [Likelihood-free parallel tempring](http://arxiv.org/abs/1108.3423)

## Fields

* `evals`: Array of `Eval`s
* `best_id`: index of best `eval.value` so far
* `best_val`: best eval.value so far
* `curr_val` : current value
* `probs_acc`: vector of probabilities with which to accept current value
* `id`: Chain identifier
* `iter`: current iteration
* `accepted`: `Array{Bool}` of `length(evals)`
* `accept_rate`: current acceptance rate
* `acc_tuner`: Acceptance tuner. `acc_tuner > 1` means to be more restrictive: params that yield a *worse* function value are *less likely* to get accepted, the higher `acc_tuner` is.
* `exchanged`: `Array{Int}` of `length(evals)` with index of chain that was exchanged with
* `m`: `MProb`
* `sigma`: `Float64` shock variance
* `sigma_update_steps`:  update sampling vars every `sigma_update_steps` iterations. setting `sigma_update_steps > maxiter` means to never update the variances.
* `sigma_adjust_by`: adjust sampling vars by `sigma_adjust_by` percent up or down
* `smpl_iters`: max number of trials to get a new parameter from MvNormal that lies within support
* `min_improve`: minimally required improvement in chain `j` over chain `i` for an exchange move `j->i` to talk place.
* `batches`: in the proposal function update the parameter vector in batches. [default: update entire param vector]

"""
mutable struct STBChain <: SMM.AbstractChain
    evals     :: Array{SMM.Eval}
    best_id   :: Vector{Int}   # index of best eval.value so far
    best_val  :: Vector{Float64}   # best eval.value so far
    curr_val  :: Vector{Float64}   # current value
    probs_acc :: Vector{Float64}    # vector of probabilities with which to accept
    id        :: Int64
    iter      :: Int64
    accepted  :: Array{Bool}
    accept_rate :: Float64
    acc_tuner :: Float64
    exchanged :: Array{Int}
    m         :: SMM.MProb
    sigma     :: Float64
    sigma_update_steps :: Int64   # update sampling vars every sigma_update_steps iterations
    sigma_adjust_by :: Float64   # adjust sampling vars by sigma_adjust_by percent up or down
    smpl_iters :: Int64   # max number of trials to get a new parameter from MvNormal that lies within support
    min_improve  :: Float64
    batches  :: Vector{UnitRange{Int}}  # vector of indices to update together.

    """
        STBChain(id::Int=1,n::Int=10;
            m::MProb=MProb(),sig::Float64=0.5,upd::Int64=10,upd_by::Float64=0.01,smpl_iters::Int=1000,
            min_improve::Float64=10.0,acc_tuner::Float64=2.0,batch_size=1)

    Constructor of a STBChain. Keyword args:
        * `acc_tuner`: Acceptance tuner. `acc_tuner > 1` means to be more restrictive: params that yield a *worse* function value are *less likely* to get accepted, the higher `acc_tuner` is.
        * `exchanged`: `Array{Int}` of `length(evals)` with index of chain that was exchanged with
        * `m`: `MProb`
        * `sig`: `Float64` shock variance
        * `upd`:  update sampling vars every `upd` iterations
        * `upd_by`: adjust sampling vars by `upd_by` percent up or down
        * `smpl_iters`: max number of trials to get a new parameter from MvNormal that lies within support
        * `min_improve`: minimally required improvement in chain `j` over chain `i` for an exchange move `j->i` to talk place.
        * `batch_size`: size of batches in which to update parameter vector.
    """
    function STBChain(id::Int=1,n::Int=10;m::SMM.MProb=SMM.MProb(),sig::Float64=0.5,upd::Int64=10,upd_by::Float64=0.01,smpl_iters::Int=1000,min_improve::Float64=10.0,acc_tuner::Float64=2.0,batch_size=1)
        np = length(m.params_to_sample)
        this           = new()
        this.evals     = Array{SMM.Eval}(undef,n)
        this.best_val  = ones(n) * Inf
        this.best_id   = -ones(Int,n)
        this.curr_val  = ones(n) * Inf
        this.probs_acc = rand(n)
        this.evals[1]  = SMM.Eval(m)    # set first eval
        this.accepted  = falses(n)
        this.accept_rate = 0.0
        this.acc_tuner = acc_tuner
        this.exchanged = zeros(Int,n)
        this.id        = id
        this.iter      = 0
        this.m         = m
        # how many bundles + rest
        nb, rest = divrem(np,batch_size)
        this.sigma = sig
        this.batches = UnitRange{Int}[]
        i = 1
        for ib in 1:nb
            j = (ib==nb && rest > 0) ? length(sig) :  i + batch_size - 1
            push!(this.batches,i:j)
            i = j + 1
        end
        this.sigma_update_steps = upd
        this.sigma_adjust_by = upd_by
        this.smpl_iters = smpl_iters
        this.min_improve = min_improve
        return this
    end
end

"""
    allAccepted(c::STBChain)

Get all accepted `Eval`s from a chain
"""
allAccepted(c::STBChain) = c.evals[c.accepted]

# return a dict of param values as arrays
function params(c::STBChain;accepted_only=true)
    if accepted_only
        e = allAccepted(c)
    else
        e = c.evals
    end
    d = Dict{Symbol,Vector{Float64}}()
    for k in keys(e[1].params)
        d[k] = Float64[e[i].params[k] for i in 1:length(e)]
    end
    return d
end

"""
    history(c::STBChain)

Returns a `DataFrame` with a history of the chain.
"""
function history(c::STBChain)
    N = length(c.evals)
    cols = Any[]
    # d = DataFrame([Int64,Float64,Bool,Int64],[:iter,:value,:accepted,:prob],N)
    d = DataFrames.DataFrame()
    d[:iter] = collect(1:c.iter)
    d[:exchanged] = c.exchanged
    d[:accepted] = c.accepted
    d[:best_val] = c.best_val
    d[:curr_val] = c.curr_val
    d[:best_id] = c.best_id
    # get fields from evals
    nms = [:value,:prob]
    for n in nms
        d[n] = eltype(getfield(c.evals[1],n))[getfield(c.evals[i],n) for i in 1:N]
    end
    # get fields from evals.params
    for (k,v) in c.evals[1].params
        d[k] = eltype(v)[c.evals[i].params[k] for i in 1:N]
    end

    return d[[:iter,:value,:accepted,:curr_val, :best_val, :prob, :exchanged,collect(keys(c.evals[1].params))...]]
end

"""
    best(c::STBChain) -> (val,idx)

Returns the smallest value and index stored of the chain.
"""
best(c::STBChain) = findmin([c.evals[i].value for i in 1:length(c.evals)])

"""
    mean(c::STBChain)

Returns the mean of all parameter values stored on the chain.
"""
mean(c::STBChain) = Dict(k => mean(v) for (k,v) in params(c))

"""
    median(c::STBChain)

Returns the median of all parameter values stored on the chain.
"""
median(c::STBChain) = Dict(k => median(v) for (k,v) in params(c))

"""
    CI(c::STBChain;level=0.95)

Confidence interval on parameters
"""
CI(c::STBChain;level=0.95) = Dict(k => quantile(v,[(1-level)/2, 1-(1-level)/2]) for (k,v) in params(c))



"""
    summary(c::STBChain)

Returns a summary of the chain. Condensed [`history`](@ref)
"""
function summary(c::STBChain)
    ex_with = c.exchanged[c.exchanged .!= 0]
    if length(ex_with) == 1
        ex_with  = [ex_with]
    end
    d = DataFrames.DataFrame(id =c.id, acc_rate = c.accept_rate,perc_exchanged=100*sum(c.exchanged .!= 0)/length(c.exchanged),
        exchanged_most_with=length(ex_with)>0 ? StatsBase.mode(ex_with) : 0,
        best_val=c.best_val[end])
    return d
end


function lastAccepted(c::STBChain)
    if c.iter==1
        return 1
    else
        return findlast(c.accepted[1:(c.iter)])
    end
end
getIterEval(c::STBChain,i::Int) = c.evals[i]
getLastAccepted(c::STBChain) = c.evals[lastAccepted(c)]
# set_sigma!(c::STBChain,s::Vector{Float64}) = length(s) == length(c.m.params_to_sample) ? c.sigma = PDiagMat(s) : ArgumentError("s has wrong length")
set_sigma!(c::STBChain,s::Float64) = c.sigma = s
function set_eval!(c::STBChain,ev::SMM.Eval)
    c.evals[c.iter] = deepcopy(ev)
    c.accepted[c.iter] =  ev.accepted
    # set best value
    if c.iter == 1
        c.best_val[c.iter] = ev.value
        c.curr_val[c.iter] = ev.value
        c.best_id[c.iter] = c.iter

    else
        if ev.accepted
            c.curr_val[c.iter] = ev.value
        else
            c.curr_val[c.iter] = c.curr_val[c.iter-1]
        end
        if (ev.value < c.best_val[c.iter-1])
            c.best_val[c.iter] = ev.value
            c.best_id[c.iter]  = c.iter
        else
            # otherwise, keep best and current from last iteration
            c.best_val[c.iter] = c.best_val[c.iter-1]
            c.best_id[c.iter]  = c.best_id[c.iter-1]
        end
    end
    return nothing
end
function set_exchanged!(c::STBChain,i::Int)
    c.exchanged[c.iter] = i
    return nothing
end


"set acceptance rate on chain. considers only iterations where no exchange happened."
function set_acceptRate!(c::STBChain)
    noex = c.exchanged[1:c.iter] .== 0
    acc = c.accepted[1:c.iter]
    c.accept_rate = Statistics.mean(BitArray(acc[noex]))
end


"""
    next_eval(c::STBChain)

Computes the next `Eval` for chain `c`:

1. Get last accepted param
2. get a new param via [`proposal`](@ref)
3. [`evaluateObjective`](@ref)
4. Accept or Reject the new value via [`doAcceptReject!`](@ref)
5. Store `Eval` on chain `c`.

"""
function next_eval(c::STBChain)
    # generate new parameter vector from last accepted param

    # increment interation
    c.iter += 1
    @debug "iteration = $(c.iter)"

    # returns an OrderedDict
    if get(algo.opts, "single_par_draw", false)
        pp = proposal_1(c)
    else
        pp = proposal(c)
    end

    # evaluate objective
    ev = SMM.evaluateObjective(c.m,pp)


    # accept reject
    doAcceptReject!(c,ev)

    # save eval on STBChain
    set_eval!(c,ev)

    return c

end

"""
    next_acceptreject(c::STBChain, ev::Eval)

Stores `ev` to chain `c`, given already computed obj fn value:

1. Accept or Reject the new value via [`doAcceptReject!`](@ref)
2. Store `Eval` on chain `c`.

"""
function next_acceptreject(c::STBChain, ev::SMM.Eval)
    @debug "iteration = $(c.iter)"

    # accept reject
    doAcceptReject!(c,ev)

    # save eval on STBChain
    set_eval!(c,ev)

    return c

end


"""
    doAcceptReject!(c::STBChain,eval_new::Eval)

Perform a Metropolis-Hastings accept-reject operation on the latest `SMM.Eval` and update the sampling variance, if so desired (set via `sigma_update_steps` in [`STBChain`](@ref) constructor.)
"""
function doAcceptReject!(c::STBChain,eval_new::SMM.Eval)
            @debug "doAcceptReject!"
    if c.iter == 1
        # accept everything.
        eval_new.prob =1.0
        eval_new.accepted = true
        eval_new.status = 1
        c.accepted[c.iter] = eval_new.accepted
        set_acceptRate!(c)
    else
        eval_old = getLastAccepted(c)

        if eval_new.status < 0
            eval_new.prob = 0.0
            eval_new.accepted = false
        else

            eval_new.value >= 0 || error("AlgoBGP assumes that your objective function returns a non-negative number. Value = " * string(eval_new.value))
            # this forumulation: old - new
            # because we are MINIMIZING the value of the objective function
            eval_new.prob = minimum([1.0,exp( c.acc_tuner * ( eval_old.value - eval_new.value) )]) #* (eval_new.value < )
            @debug "eval_new.value = $(eval_new.value)"
            @debug "eval_old.value = $(eval_old.value)"
            @debug "eval_new.prob = $(round(eval_new.prob,digits = 2))"
            @debug "c.probs_acc[c.iter] = $(round(c.probs_acc[c.iter],digits = 2))"

            if !isfinite(eval_new.prob)
                eval_new.prob = 0.0
                eval_new.accepted = false
                eval_new.status = -1

            elseif !isfinite(eval_old.value)
                # should never have gotten accepted
                @debug "eval_old is not finite"
                eval_new.prob = 1.0
                eval_new.accepted = true
            else
                # status = 1
                eval_new.status = 1
                if eval_new.prob > c.probs_acc[c.iter]
                    eval_new.accepted = true
                else
                    eval_new.accepted = false
                end
            end
            @debug "eval_new.accepted = $(eval_new.accepted)"

        end

        c.accepted[c.iter] = eval_new.accepted
        set_acceptRate!(c)

        # update sampling variances every x periods
        # -----------------------------------------

        # update shock variance. want to achieve a long run accpetance rate of 23.4% (See Casella and Berger)

        if mod(c.iter,c.sigma_update_steps) == 0
            too_high = c.accept_rate > 0.234
            if too_high
                @debug "acceptance rate on STBChain $(c.id) is too high at $(c.accept_rate). increasing variance of each param by $(100* c.sigma_adjust_by)%."
                set_sigma!(c,c.sigma .* (1.0+c.sigma_adjust_by) )
            else
                @debug "acceptance rate on STBChain $(c.id) is too low at $(c.accept_rate). decreasing variance of each param by $(100* c.sigma_adjust_by)%."
                set_sigma!(c,c.sigma .* (1.0-c.sigma_adjust_by) )
            end
        end
    end
end



"""
    proposal(c::STBChain)

Gaussian Transition Kernel centered on current parameter value.

1. Map all ``k`` parameters into ``\\mu \\in [0,1]^k``.
2. update all parameters by sampling from `MvNormal`, ``N(\\mu,\\sigma)``, where ``sigma`` is `c.sigma` until all params are in ``[0,1]^k``
3. Map ``[0,1]^k`` back to original parameter spaces.

"""
function proposal(c::STBChain)

    if c.iter==1
        return c.m.initial_value
    else
        ev_old = getLastAccepted(c)
        mu  = SMM.paramd(ev_old) # dict of params
        lb = [v[:lb] for (k,v) in c.m.params_to_sample]
        ub = [v[:ub] for (k,v) in c.m.params_to_sample]

        # map into [0,1]
        # (x-a)/(b-a) = z \in [0,1]
        mu01 = SMM.mapto_01(mu,lb,ub)

        # Transition Kernel is q(.|theta(t-1)) ~ TruncatedN(theta(t-1), Sigma,lb,ub)

        # if there is only one batch of params
        if length(c.batches) == 1
            pp = SMM.mysample(Distributions.MvNormal(mu01,c.sigma),0.0,1.0,c.smpl_iters)
        else
            # do it in batches of params
            pp = zero(mu01)
            for (sig_ix,i) in enumerate(c.batches)
                try
                    pp[i] = SMM.mysample(Distributions.MvNormal(mu01[i],c.sigma),0.0,1.0,c.smpl_iters)
                catch err
                    @error "caught exception $err. this is param index $sig_ix, mean = $(mu01[i]), sigma $(c.sigma), lb,ub = $((0,1))"
                end
            end
        end
        # map [0,1] -> [a,b]
        # z*(b-a) + a = x \in [a,b]
        newp = DataStructures.OrderedDict(zip(collect(keys(mu)),SMM.mapto_ab(pp,lb,ub)))

        #### enforce lambda parameters sum to 1 ###
        #### this is the only difference from AlgoBGP / BGPChain

        lambdas = [(i,newp[i]) for i in keys(newp) if occursin("topic_lambda", string(i))]
        lambda_norm = sum(map(last, lambdas))

        if lambda_norm < 1e-8
            for (ls, l) in lambdas
                newp[ls] = 1/length(lambdas)
            end
        else
            for (ls, l) in lambdas
                newp[ls] = l / lambda_norm
            end
        end


        @debug "iteration $(c.iter)"
        @debug "old param: $(ev_old.params)"
        @debug "new param: $newp"
        for (k,v) in newp
            @debug "step for $k = $(v-ev_old.params[k])"
        end

        # flat kernel: random choice in each dimension.
        # newp = Dict(zip(collect(keys(mu)),rand(length(lb)) .* (ub .- lb)))

        return newp
    end

end

"""
mysample_1(d::Distributions.MultivariateDistribution,lb::Vector{Float64},ub::Vector{Float64},iters::Int)
sample from distribution `d` until something found in support. This is a crude version of a truncated distribution: It just samples until all draws are within the admissible domain.
"""
function mysample_1(d::Distributions.UnivariateDistribution,lb::Float64,ub::Float64,iters::Int,par_name::Symbol)

    # draw until in support
    for i in 1:iters
        x = rand(SMM.RAND,d)
        if (x>=lb) && (x<=ub)
            return x
        end
    end
    error("no draw in support after $iters trials when sampling $par_name at value $(Distributions.location(d)). increase either opts[smpl_iters] or opts[bound_prob].")
end


"""
    proposal_1(c::STBChain)

Gaussian Transition Kernel centered on current parameter value.
Update only one parameter (selected uniformly at random)

1. Map all ``k`` parameters into ``\\mu \\in [0,1]^k``.
2. Select one parameter to sample
3. Update this parameter by sampling from `Normal`, ``N(\\mu,\\sigma)``, where ``sigma`` is `c.sigma` until all params are in ``[0,1]^k``
4. Map ``[0,1]^k`` back to original parameter spaces.

"""
function proposal_1(c::STBChain)

    if c.iter==1
        return c.m.initial_value
    else
        # ignore = r"\d+_t\d+|consumer_free_var"       # regex for params to not sample
        # keep = r"(cnn|fnc|msnbc|zero|\d+_t\d+)"                     # regex for params to sample
        ev_old = getLastAccepted(c)
        mu  = SMM.paramd(ev_old) # dict of params
        lb = [v[:lb] for (k,v) in c.m.params_to_sample]
        ub = [v[:ub] for (k,v) in c.m.params_to_sample]

        # map into [0,1]
        # (x-a)/(b-a) = z \in [0,1]
        mu01 = SMM.mapto_01(mu,lb,ub)

        # Transition Kernel is q(.|theta(t-1)) ~ TruncatedN(theta(t-1), Sigma,lb,ub)
        # pick parameter to update
        # i_upd = rand(SMM.RAND, 1:length(mu01))
        all_keys = collect(keys(mu))
        # keys_to_propose = filter(x -> !occursin(ignore, string(x)), all_keys)
        # keys_to_propose = filter(x -> occursin(keep, string(x)), all_keys)
        keys_to_propose = all_keys

        k_upd = rand(SMM.RAND, keys_to_propose)
        i_upd = findfirst(all_keys .== k_upd)

        # update only this one
        mu01[i_upd] = mysample_1(Distributions.Normal(mu01[i_upd],c.sigma), 0.0, 1.0, c.smpl_iters, k_upd)
        muab = SMM.mapto_ab(mu01,lb,ub)

        # # special behavior if we chose either channel mu or sigma
        # if (occursin(r"channel_sigma", string(k_upd)))
        #     thisch = replace(string(k_upd), r"beta:channel_sigma:([a-z]+)" => s"\g<1>")
        #     i_ch_mu = findfirst(occursin.(Regex("beta:channel_mu:"*thisch), string.(all_keys)))
        #     muab[i_ch_mu] = -(muab[i_upd]^2)/2
        # end
        # if (occursin(r"channel_mu", string(k_upd)))
        #     thisch = replace(string(k_upd), r"beta:channel_mu:([a-z]+)" => s"\g<1>")
        #     i_ch_sig = findfirst(occursin.(Regex("beta:channel_sigma:"*thisch), string.(all_keys)))
        #     muab[i_ch_sig] = sqrt(-2*muab[i_upd])
        # end


        # map [0,1] -> [a,b]
        # z*(b-a) + a = x \in [a,b]
        newp = DataStructures.OrderedDict(zip(collect(keys(mu)),muab))

        #### enforce lambda parameters sum to 1 ###
        #### this is the only difference from AlgoBGP / BGPChain

        lambdas = [(i,newp[i]) for i in keys(newp) if occursin("topic_lambda", string(i))]
        lambda_norm = sum(map(last, lambdas))

        if lambda_norm < 1e-8
            for (ls, l) in lambdas
                newp[ls] = 1/length(lambdas)
            end
        else
            for (ls, l) in lambdas
                newp[ls] = l / lambda_norm
            end
        end

        @debug "iteration $(c.iter)"
        @debug "old param: $(ev_old.params)"
        @debug "new param: $newp"
        for (k,v) in newp
            @debug "step for $k = $(v-ev_old.params[k])"
        end

        # flat kernel: random choice in each dimension.
        # newp = Dict(zip(collect(keys(mu)),rand(length(lb)) .* (ub .- lb)))

        return newp
    end

end


###################################
# end STBChain
###################################


"""
# MAlgoSTB: BGP MCMC Algorithm

This implements the [BGP MCMC Algorithm Likelihood-Free Parallel Tempering](http://fr.arxiv.org/abs/1108.3423) by Baragatti, Grimaud and Pommeret (BGP):

> Approximate Bayesian Computational (ABC) methods (or likelihood-free methods) have appeared in the past fifteen years as useful methods to perform Bayesian analyses when the likelihood is analytically or computationally intractable. Several ABC methods have been proposed: Monte Carlo Markov STBChains (MCMC) methods have been developped by Marjoramet al. (2003) and by Bortotet al. (2007) for instance, and sequential methods have been proposed among others by Sissonet al. (2007), Beaumont et al. (2009) and Del Moral et al. (2009). Until now, while ABC-MCMC methods remain the reference, sequential ABC methods have appeared to outperforms them (see for example McKinley et al. (2009) or Sisson et al. (2007)). In this paper a new algorithm combining population-based MCMC methods with ABC requirements is proposed, using an analogy with the Parallel Tempering algorithm (Geyer, 1991). Performances are compared with existing ABC algorithms on simulations and on a real example.


## Fields

* `m`: [`MProb`](@ref)
* `opts`: a `Dict` of options
* `i`: current iteration
* `chains`: An array of [`STBChain`](@ref)
* `anim`: `Plots.Animation`
* `dist_fun`: function to measure distance between one evaluation and the next.

"""
mutable struct MAlgoSTB <: SMM.MAlgo
    m               :: SMM.MProb # an MProb
    opts            :: Dict	# list of options
    i               :: Int 	# iteration
    chains         :: Array{STBChain} 	# collection of STBChains: if N==1, length(STBChains) = 1
    anim           :: SMM.Plots.Animation
    dist_fun   :: Function

    function MAlgoSTB(m::SMM.MProb,opts=Dict("N"=>3,"maxiter"=>100,"maxtemp"=> 2,"sigma"=>0.05,"sigma_update_steps"=>10,"sigma_adjust_by"=>0.01,"smpl_iters"=>1000,"parallel"=>false,"min_improve"=>[0.0 for i in 1:3],"acc_tuners"=>[2.0 for i in 1:3]))

        if opts["N"] > 1
    		temps     = range(1.0,stop=opts["maxtemp"],length=opts["N"])
            # initial std dev for each parameter to achieve at least bound_prob on the bounds
            # println("opts=$opts")
            # println("pars = $( m.params_to_sample)")

            # choose inital sd for each parameter p
            # such that Pr( x \in [init-b,init+b]) = 0.975
            # where b = (p[:ub]-p[:lb])*opts["coverage"] i.e. the fraction of the search interval you want to search around the initial value
            STBChains = STBChain[STBChain(i,opts["maxiter"],
                m = m,
                sig = get(opts,"sigma",0.05) .* temps[i],
                upd = get(opts,"sigma_update_steps",10),
                upd_by = get(opts,"sigma_adjust_by",0.01),
                smpl_iters = get(opts,"smpl_iters",1000),
                min_improve = get(opts,"min_improve",[0.5 for j in 1:opts["N"]])[i],
                acc_tuner = get(opts,"acc_tuners",[2.0 for j in 1:opts["N"]])[i],
                batch_size = get(opts,"batch_size",length(m.params_to_sample))) for i in 1:opts["N"]]
        else
            # println(init_sd)
            STBChains = STBChain[STBChain(1,opts["maxiter"],
                m = m,
                sig = get(opts,"sigma",0.05),
                upd = get(opts,"sigma_update_steps",10),
                upd_by = get(opts,"sigma_adjust_by",0.01),
                smpl_iters = get(opts,"smpl_iters",1000),
                min_improve = get(opts,"min_improve",[0.5 for j in 1:opts["N"]])[i],
                acc_tuner = get(opts,"acc_tuners",[2.0 for j in 1:opts["N"]])[i],
                batch_size = get(opts,"batch_size",length(m.params_to_sample))) for i in 1:opts["N"]]
        end
	    return new(m,opts,0,STBChains, SMM.Animation(),get(opts,"dist_fun",-))
    end
end

function summary(m::MAlgoSTB)
    s = map(summary,m.chains)
    df = s[1]
    if length(s) > 1
        for i in 2:length(s)
            df = vcat(df,s[i])
        end
    end
    return df
end


# return current param spaces on algo
cur_param(m::MAlgoSTB) = iter_param(m,m.i)


# return param spaces on algo at iter
function iter_param(m::MAlgoSTB,iter::Int)
    r = Dict()
    for ic in 1:length(m.chains)
        if m.i == 0
            r[ic] = Dict(:mu => m.m.initial_value,:sigma => m.chains[ic].sigma)
        else
            ev_old = getIterEval(m.chains[ic],iter)
            r[ic] = Dict(:mu => paramd(ev_old),:sigma => m.chains[ic].sigma)
        end
    end
    r

end



"""
    computeNextIteration!( algo::MAlgoSTB )

computes new candidate vectors for each [`STBChain`](@ref)
accepts/rejects that vector on each STBChain, according to some rule. The evaluation objective functions is performed in parallel, is so desired.

1. On each chain `c`:
    * computes new parameter vectors
    * applies a criterion to accept/reject any new params
    * stores the result in STBChains
2. Calls [`exchangeMoves!`](@ref) to swap chains
"""
function SMM.computeNextIteration!( algo::MAlgoSTB )
    # here is the meat of your algorithm:
    # how to go from p(t) to p(t+1) ?

    # incrementSTBChainIter!(algo.chains)


    # TODO
    # this is probably inefficeint
    # ideally, would only pmap evaluateObjective, to avoid large data transfers to each worker (now we're transferring each chain back and forth to each worker.)
    # GJM update: do proposal serially to avoid transferring the entire BGPChain object

    if get(algo.opts, "parallel", false)
        for i in 1:length(algo.chains)
            algo.chains[i].iter +=1   #increment iteration on master
        end

        if get(algo.opts, "single_par_draw", false)
            pps = map(proposal_1, algo.chains)  # proposals on master, single parameter step
        else
            pps = map(proposal, algo.chains)  # proposals on master, multivariate step
        end

        evs = pmap(x -> SMM.evaluateObjective(algo.m, x), wp, pps) # pmap only the objective function evaluation step

        cs = map(next_acceptreject, algo.chains, evs) # doAcceptRecject, set_eval

    else
        # for i in algo.chains
        #     @debug(logger," ")
        #     @debug(logger," ")
        #     @debug(logger,"debugging chain id $(i.id)")
        #     next_eval!(i)
        # end
        cs = map( x->next_eval(x), algo.chains ) # this does proposal, evaluateObjective, doAcceptRecject
    end
    # reorder and insert into algo
    for i in 1:algo.opts["N"]
        algo.chains[i] = cs[map(x->getfield(x,:id) == i,cs)][1]
        # println("i=$i ::", debug_list_vals())
        @assert algo.chains[i].id == i
    end
    if get(algo.opts, "animate", false)
        p = plot(algo,1);
        frame(algo.anim)
    end
    # p = plot(algo,1)
    # display(p)
    # sleep(.1)

    # check algo index is the same on all STBChains
    for ic in 1:algo["N"]
        @assert algo.i == algo.chains[ic].iter
    end

    # Part 2) EXCHANGE MOVES only on master
    # ----------------------
    # starting mixing in period 3
    if algo.i>=2 && algo["N"] > 1
        exchangeMoves!(algo)
    end
end

"""
    exchangeMoves!(algo::MAlgoSTB)

Exchange chain `i` and `j` with if `dist_fun(evi.value,evj.value)` is greate than a threshold value `c.min_improve`. Commonly, this means that we only exchange if `j` is better by *at least* `c.min_improve`.
"""
function exchangeMoves!(algo::MAlgoSTB)

    # i is new index
    # j is old index

    # algo["N"] exchange moves are proposed
    props = [(i,j) for i in 1:algo["N"], j in 1:algo["N"] if (i<j)]
    # N pairs of chains are chosen uniformly in all possibel pairs with replacement
    samples = algo["N"] < 3 ? algo["N"]-1 : algo["N"]
    pairs = StatsBase.sample(props,samples,replace=false)

    # @debug(logger,"")
    # @debug(logger,"exchangeMoves: proposing pairs")
    # @debug(logger,"$pairs")

    for p in pairs
        i,j = p
        evi = getLastAccepted(algo.chains[i])
        evj = getLastAccepted(algo.chains[j])
        # my version
        # if rand() < algo["mixprob"]
            # if (evj.value < evi.value)  # if j's value is better than i's
            #     @debug(logger,"$j better than $i")
            #     # @debug(logger,"$(abs(j.value)) < $(algo.chains[p[1]].min_improve)")
            #     # swap_ev!(algo,p)
            #     set_ev_i2j!(algo,i,j)
            # else
            #     @debug(logger,"$i better than $j")
            #     set_ev_i2j!(algo,j,i)
            # end
        # end

        # BGP version
        # exchange i with j if rho(S(z_j),S(data)) < epsilon_i
        # @debug(logger,"Exchanging $i with $j? Distance is $(algo.dist_fun(evj.value, evi.value))")
        # @debug(logger,"Exchange: $(algo.dist_fun(evj.value, evi.value)  < algo.chains[i].min_improve)")
        # println("Exchanging $i with $j? Distance is $(algo.dist_fun(evj.value, evi.value))")
        # println("Exchange: $(algo.dist_fun(evj.value, evi.value)  > algo["min_improve"][i])")
        # this formulation assumes that evi.value > 0 always, for all i.
        # swap for sure if there is an improvement, i.e. algo.dist_fun(evj.value, evi.value) > 0
        # swap even if there is a deterioration, but only up to threshold min_improve[i]
        if algo.dist_fun(evi.value, evj.value) > algo.chains[i].min_improve
            swap_ev_ij!(algo,i,j)
        end
    end

	# for ch in 1:algo["N"]
 #        e1 = getLastAccepted(algo.chains[ch])
	# 	# 1) find all other STBChains with value +/- x% of STBChain ch
	# 	close = Int64[]  # vector of indices of "close" STBChains
	# 	for ch2 in 1:algo["N"]
	# 		if ch != ch2
	# 			e2 = getLastAccepted(algo.chains[ch2])
	# 			tmp = abs(e2.value - e1.value) / abs(e1.value)
	# 			# tmp = abs(evals(algo.chains[ch2],algo.chains[ch2].i)[1] - oldval) / abs(oldval)	# percent deviation
	# 			if tmp < dtol
 #                    @debug(logger,"perc dist $ch and $ch2 is $tmp. will label that `close`.")
	# 				push!(close,ch2)
	# 			end
	# 		end
	# 	end
	# 	# 2) with y% probability exchange with a randomly chosen STBChain from close
	# 	if length(close) > 0
	# 		ex_with = rand(close)
	# 		@debug(logger,"making an exchange move for STBChain $ch with STBChain $ex_with set: $close")
	# 		swap_ev!(algo,Pair(ch,ex_with))
	# 	end
	# end

end

function set_ev_i2j!(algo::MAlgoSTB,i::Int,j::Int)
    @debug "setting ev of $i to ev of $j"
    ci = algo.chains[i]
    cj = algo.chains[j]

    ei = getLastAccepted(ci)
    ej = getLastAccepted(cj)

    # set ei -> ej
    set_eval!(ci,ej)

    # make a note
    set_exchanged!(ci,j)
end

"replace the current `Eval` of chain ``i`` with the one of chain ``j``"
function swap_ev_ij!(algo::MAlgoSTB,i::Int,j::Int)
    @debug "swapping ev of $i with ev of $j"
    ci = algo.chains[i]
    cj = algo.chains[j]

    ei = getLastAccepted(ci)
    ej = getLastAccepted(cj)

    # set ei -> ej
    set_eval!(ci,ej)
    set_eval!(cj,ei)

    # make a note
    set_exchanged!(ci,j)
    set_exchanged!(cj,i)
end



"""
  extendSTBChain!(chain::STBChain, algo::MAlgoSTB, extraIter::Int64)

Starting from an existing [`MAlgoSTB`](@ref), allow for additional iterations
by extending a specific chain. This function is used to restart a previous estimation run via [`restart!`](@ref)
"""
function extendSTBChain!(chain::STBChain, algo::MAlgoSTB, extraIter::Int64)

  initialIter = algo.i
  finalIter = algo.i + extraIter

  # stores the original chain:
  #---------------------------
  copyOriginalChain = deepcopy(chain)


  # I have to change the following fields:
  #---------------------------------------
  # 1. Change the size:
  #--------------------
  chain.evals    = Array{SMM.Eval}(undef,finalIter)
  chain.best_val  = ones(finalIter) * Inf
  chain.best_id   = -ones(Int, finalIter)
  chain.curr_val  = ones(finalIter) * Inf
  chain.probs_acc = rand(finalIter)
  chain.accepted  = falses(finalIter)
  chain.exchanged = zeros(Int,finalIter)


  # 2. Push the previous values in:
  #--------------------------------
  for iterNumber = 1:initialIter
    chain.evals[iterNumber] = copyOriginalChain.evals[iterNumber]
    chain.best_val[iterNumber] = copyOriginalChain.best_val[iterNumber]
    chain.best_id[iterNumber] =  copyOriginalChain.best_id[iterNumber]
    chain.curr_val[iterNumber] = copyOriginalChain.curr_val[iterNumber]
    chain.probs_acc[iterNumber] =  copyOriginalChain.probs_acc[iterNumber]
    chain.accepted[iterNumber] = copyOriginalChain.accepted[iterNumber]
    chain.exchanged[iterNumber] = copyOriginalChain.exchanged[iterNumber]
  end



end

"""
  extendSTBChain_deletehistory!(chain::STBChain, algo::MAlgoSTB, extraIter::Int64)

Starting from an existing [`MAlgoSTB`](@ref), allow for additional iterations
by extending a specific chain. This deletes the previous history of moments and parameters,
so save first!
This function is used to restart a previous estimation run via [`restart!`](@ref)
"""
function extendSTBChain_deletehistory!(chain::STBChain, algo::MAlgoSTB, extraIter::Int64)

  initialIter = algo.i
  # finalIter = algo.i + extraIter
  finalIter = extraIter + 1
  # stores the original chain:
  #---------------------------
  copyOriginalChain = deepcopy(chain)


  # I have to change the following fields:
  #---------------------------------------
  # 1. Change the size:
  #--------------------
  chain.evals    = Array{SMM.Eval}(undef,finalIter)
  chain.best_val  = ones(finalIter) * Inf
  chain.best_id   = -ones(Int, finalIter)
  chain.curr_val  = ones(finalIter) * Inf
  chain.probs_acc = rand(finalIter)
  chain.accepted  = falses(finalIter)
  chain.exchanged = zeros(Int,finalIter)


  # 2. Push the previous values in:
  #--------------------------------
  # for iterNumber = 1:initialIter
  # store only the last chain, so that we can extend
  chain.evals[1] = copyOriginalChain.evals[initialIter]
  chain.best_val[1] = copyOriginalChain.best_val[initialIter]
  chain.best_id[1] =  copyOriginalChain.best_id[initialIter]
  chain.curr_val[1] = copyOriginalChain.curr_val[initialIter]
  chain.probs_acc[1] =  copyOriginalChain.probs_acc[initialIter]
  chain.accepted[1] = copyOriginalChain.accepted[initialIter]
  chain.exchanged[1] = copyOriginalChain.exchanged[initialIter]
  # end



end

"""
  restart!(algo::MAlgoSTB, extraIter::Int64)

Starting from an existing AlgoBGP, restart the optimization from where it
stopped. Add `extraIter` additional steps to the optimization process.
"""
function restart!(algo::MAlgoSTB, extraIter::Int64)

  @info "Restarting estimation loop with $(extraIter) iterations."
  @info "Current best value on chain 1 before restarting $(SMM.summary(algo)[:best_val][1])"
  t0 = time()

  # Minus 1, to follow the SMM convention
  initialIter = algo.i
  finalIter = initialIter + extraIter

  # Extend algo.chain[].evals
  #---------------------------
  # Loop over chains
  for chainNumber = 1:algo.opts["N"]
    # extendSTBChain!(algo.chains[chainNumber], algo, extraIter)
    extendSTBChain_deletehistory!(algo.chains[chainNumber], algo, extraIter)
  end

  # To follow the conventions in SMM:
  #----------------------------------------
  for ic in 1:algo["N"]
      algo.chains[ic].iter =   algo.i - 1
   end

  #change maxiter in the dictionary storing options
  #------------------------------------------------
  @debug "Setting algo.opts[\"maxiter\"] = $(finalIter)"
  algo.opts["maxiter"] = finalIter

  # do iterations, starting at initialIter
  # and not at i=1, as in run!
    @info "restarting estimation"
    SMM.@showprogress for i in initialIter:finalIter

    algo.i = i


    # try
      SMM.computeNextIteration!( algo )

      # If the user stops the execution (control + C), save algo in a file
      # with a special name.
      # I leave commented for the moment
      #-------------------------------------------------------------------
      # catch x
      #   @warn(logger, "Error = ", x)
      #
      #   if isa(x, InterruptException)
      #     @warn(logger, "User interupted the execution")
      #     @warn(logger, "Saving the algorithm to disk.")
      #     save(algo, "InterruptedAlgorithm")
      #   end
      # end

      # Save the process every $(save_frequency) iterations:
      #----------------------------------------------------
      # save at certain frequency
    	if haskey(algo.opts,"save_frequency") == true
        # if the user provided a filename in the options dictionary
    		if haskey(algo.opts,"filename") == true
				if mod(i,algo.opts["save_frequency"]) == 0
					save(algo,algo.opts["filename"])
					@info(logger,"saved data at iteration $i")
				end
            end
    	end
        t1 = round((time()-t0)/60,digits = 1)
    	algo.opts["time"] = t1
    	if haskey(algo.opts,"filename")
    		save(algo,algo.opts["filename"])
    	else
    		@warn "could not find `filename` in algo.opts - not saving"
    	end

    	@info "Done with estimation after $t1 minutes"

    	if get(algo.opts,"animate",false)
    		gif(algo.anim,joinpath(dirname(@__FILE__),"../../proposals.gif"),fps=2)
    	end
    end

end



function show(io::IO,MA::MAlgoSTB)
	print(io,"\n")
	print(io,"BGP Algorithm with $(MA["N"]) STBChains\n")
	print(io,"============================\n")
	print(io,"\n")
	print(io,"Algorithm\n")
	print(io,"---------\n")
	print(io,"Current iteration: $(MA.i)\n")
	print(io,"Number of params to estimate: $(length(MA.m.params_to_sample))\n")
	print(io,"Number of moments to match: $(length(MA.m.moments))\n")
	print(io,"\n")
end
