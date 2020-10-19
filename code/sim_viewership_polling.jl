# do one day of viewership / polling sim loop

function sim_viewership_polling!(d,
	# these arguments will be modified (preallocated in objective for speed)
	# these are model outputs
	consumer_watched_history,	# simulated viewing history
    μ_τ,						# consumer-specific probs that state of topic τ favors R
	μ_star,   					# consumer-specific estimate of aggregate state
	predicted_viewership,		# predicted ratings by block
	daily_polling,				# daily poll
	u_0, u_vote, u_suspense, u_slant, u_channel, u_exp,  # channel x block x individual utilities
	channel_reports,									# realized signal reported each topic x channel x block
	choice_prob_per_channel,
	consumer_choice_threshold,
	pr_report_R,
	update_if_report_R, update_if_report_D, update_needed,
	# inputs not modified
	# parameters:
	q_R, q_D,		# prob chan reports R | signal R, D | signal D, each is 1 x C
	ρ_τ, 			# transition probability for topic τ, 1 x K
	λ_τ,			# weight of topic τ in aggregate state, K x 1
	β_0, β_hour, β_slant, β_vote, β_suspense,   # utility weights (possibly heterogeneous) of each component
	topic_news_today,  # for each topic either -1 (D favoring), 0 (no news), or 1 (R favoring)
	look_forward,      # K x 2 matrix containing first row of exponentiated transition matrix for each τ
	# data:
	consumer_viewed_topic,
	consumer_viewed_show,
	pre_election,
	consumer_ideology,
	consumer_tz,
	i_stb, i_national,
	# random draws:
	choice_draws,
	report_draws,
	# dimensions:
	row_summer, N,K,C,t_i)

	# update beliefs given 1 period of possible transition
	μ_τ .= μ_τ .* ρ_τ .+ (1 .- μ_τ) .* (1 .- ρ_τ)

	# updated belief about the aggregate state, from topic-specific beliefs
	mul!(μ_star, μ_τ, λ_τ)

	# indexes of topics with D / R / neutral news
	d_topics = findall(topic_news_today.==-1)
	null_topics = findall(topic_news_today.==0)
	r_topics = findall(topic_news_today.==1)
	nonnull_topics = union(d_topics, r_topics)

	# any topic with no news has null reports, no updating, no prob of D/R report, no suspense
	channel_reports[:,:,null_topics] .= 0
	update_if_report_R[:,null_topics,:] .= 0
	update_if_report_D[:,null_topics,:] .= 0
	pr_report_R[:,null_topics] .= consumer_ideology    # no slant from topic τ if no news today

	# sim channel reports for topics with news
	if (length(r_topics) > 0)
		@views channel_reports[:,;,r_topics] = 2 .* (report_draws[:,:,r_topics] .<= q_R) .- 1
	end
	if (length(d_topics) > 0)
		@views channel_reports[:,:,d_topics] = -2 .* (report_draws[:,:,d_topics] .<= q_D) .+ 1
	end

	# store forward update matrix components
	look_forward_1 = look_forward[:,1]'
	look_forward_2 = look_forward[:,2]'

	# polling (do before viewing today)
	if pre_election
	  daily_polling[d] = Statistics.mean(μ_star[i_national] .< consumer_ideology[i_national])
	else
	  daily_polling[d] = 0;
	end

	for t=1:t_i
		# the hour index for this block (1:5)
		h_index = cld(t,4)

		for c = 1:C
			for i = 1:N
				# amount of updating conditional on each event
				if (length(nonnull_topics) > 0)
					for τ in nonnull_topics
						# unconditional prob of reporting R on each topic
						pr_report_R[:,τ] = μ_τ[:,τ] .* q_R[c] .+ (1 .- μ_τ[:,τ]) .* (1 .- q_D[c])
						update_if_report_R[:,τ,c] = (μ_τ[:,τ] .* q_R[c]) ./ (μ_τ[:,τ] .* q_R[c] .+ (1 .- μ_τ[:,τ]) .* (1 .- q_D[c])) .- μ_τ[:,τ]
						update_if_report_D[:,τ,c] = ((μ_τ[:,τ] .* (1 .- q_R[c])) ./ ((1 .- μ_τ[:,τ]) .* q_D[c] .+ μ_τ[:,τ] .* (1 .- q_R[c]))) .- μ_τ[:,τ]
					end
					# suspense = variance of the posterior on each topic, weighted by topic weight in this block
					# u_suspense[:,c] = (consumer_viewed_topic[:,:,c,t] .* (pr_report_R .* update_if_report_R[:,:,c] .^ 2 .+ (1 .- pr_report_R) .* update_if_report_D[:,:,c] .^ 2) .* β_suspense) * row_summer
					mul!(u_suspense[:,c], consumer_viewed_topic[:,:,c,t] .* (pr_report_R .* update_if_report_R[:,:,c] .^ 2 .+ (1 .- pr_report_R) .* update_if_report_D[:,:,c] .^ 2) .* β_suspense, row_summer)

					# u slant = abs diff x_i and topic-weighted prob of seeing R signal
					# if no news on some topic today, no distaste on that dimension
					# u_slant[:,c] = abs.(consumer_ideology .- ((consumer_viewed_topic[:,:,c,t] .* pr_report_R) * row_summer))
					mul!(u_slant[:,c], consumer_viewed_topic[:,:,c,t] .* pr_report_R, row_summer)
					u_slant[:,c] .-= consumer_ideology
					abs!(u_slant[:,c])
				else
					u_suspense[:,c] .= 0
					u_slant[:,c] .= 0
				end

				# voting utility
				if pre_election
					# how much does election-day posterior need to move to cross over threshold?
					# update_needed .= ((μ_τ .* look_forward_1 .+ look_forward_2) * λ_τ) .- consumer_ideology
					mul!(update_needed, μ_τ .* look_forward_1 .+ look_forward_2, λ_τ)
					update_needed .-= consumer_ideology

					# if μ < x, R signals are what we care about, o.w. D signals
					# u_vote is expected fraction of distance to threshold
					# u_vote[:,c] = min.(((((update_needed .>= 0) .* update_if_report_R[:,:,c] .+ (update_needed .< 0) .* update_if_report_D[:,:,c]) .*
										# consumer_viewed_topic[:,:,c,t] .*
										# look_forward_1 .*
								   		# ((2 .* (update_needed .>= 0) .- 1) .* pr_report_R .+ (update_needed .< 0)) * λ_τ) ./ update_needed, 1)
					mul!(u_vote[:,c], ((update_needed .>= 0) .* update_if_report_R[:,:,c] .+ (update_needed .< 0) .* update_if_report_D[:,:,c]) .*
										consumer_viewed_topic[:,:,c,t] .*
										look_forward_1 .*
								   		((2 .* (update_needed .>= 0) .- 1) .* pr_report_R .+ (update_needed .< 0)), λ_τ)
					u_vote[:,c] ./= update_needed
					u_vote[:,c] = min.(u_vote[:,c], 1)

				else
					# after election, no voting utility
					u_vote[:, c] .= 0
				end # vote utility calculation

				# hour / show / channel effects
				for i = 1:N
					u_0[i,c] = β_0[i,consumer_viewed_show[i,c,t]] + β_hour[consumer_tz[i], h_index]
				end
		end # c loop

		# add components
		u_channel .= u_0 .+ u_suspense .+ (β_vote .* u_vote) .+ (β_slant .* u_slant)

		# compute choice probabilities
		u_exp .= exp.(min.(600,u_channel))
		choice_prob_per_channel[:,1] = 1 ./ (1 .+ sum(u_exp; dims=2))
	    choice_prob_per_channel[:,2:(C+1)] = u_exp .* choice_prob_per_channel[:,1]
	    consumer_choice_threshold .= cumsum(choice_prob_per_channel, dims=2)

		# simulate viewing decision
		consumer_watched_history[:,:,t] = (choice_draws[:,t] .>= consumer_choice_threshold[:,1:C]) .* (choice_draws[:,t] .< consumer_choice_threshold[:,2:(C+1)])

		# update beliefs over topic states
		# move a fraction of the amount weighted by topic time
		for i in 1:N
			for c in 1:C
				if consumer_watched_history[i,c,t]
					for τ in nonnull_topics
						if channel_reports[c,t,τ] == -1
							μ_τ[i,τ] += consumer_viewed_topic[i,τ,c,t] * update_if_report_D[i,τ,c]
						elseif channel_reports[c,t,τ] == 1
							μ_τ[i,τ] += consumer_viewed_topic[i,τ,c,t] * update_if_report_R[i,τ,c]
						end # if τ in nonnull
					end # τ loop
					break
				end # if consumer_watched_history
			end # c loop
		end # i loop

		# apply updates to aggregate state
		mul!(μ_star, μ_τ, λ_τ)

		# store average viewing prob among national sample
		predicted_viewership[:,t]=mapslices(Statistics.mean, choice_prob_per_channel[i_national,2:end];dims=1);
	end # t loop
end
