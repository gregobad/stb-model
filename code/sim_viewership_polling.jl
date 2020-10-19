# do one day of viewership / polling sim loop

function sim_viewership_polling!(d,
	# these arguments will be modified (preallocated in objective for speed)
	# these are model outputs
	consumer_watched_history,	# simulated viewing history
    μ_τ,						# consumer-specific probs that state of topic τ favors R
	μ_star,   					# consumer-specific estimate of aggregate state
	predicted_viewership,		# predicted ratings by block
	daily_polling,				# daily poll
	u_0, u_vote, u_suspense, u_slant, u_exp_sum,  # channel x block x individual utilities
	channel_reports,									# realized signal reported each topic x channel x block
	choice_prob_per_channel,
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
	consumer_ideology,consumer_r_prob,
	consumer_tz,
	last_national,
	# random draws:
	choice_draws,
	report_draws,
	# dimensions:
	N,K,C,t_i)

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

	# sim channel reports for topics with news
	if (length(r_topics) > 0)
		@views channel_reports[:,:,r_topics] = 2 .* (report_draws[:,:,r_topics] .<= q_R) .- 1
	end
	if (length(d_topics) > 0)
		@views channel_reports[:,:,d_topics] = -2 .* (report_draws[:,:,d_topics] .<= q_D) .+ 1
	end

	# store forward update matrix components
	look_forward_1 = look_forward[:,1]'
	look_forward_2 = look_forward[:,2]'

	# polling (do before viewing today)
	if pre_election
	  daily_polling[d] = Statistics.mean(μ_star[1:last_national] .> consumer_ideology[1:last_national])
	else
	  daily_polling[d] = 0;
	end

	@views for t=1:t_i
		# the hour index for this block (1:5)
		h_index = cld(t,4)

		# in pre election period, compute expected distance to threshold on election day
		if pre_election
			mul!(update_needed, μ_τ .* look_forward_1 .+ look_forward_2, λ_τ)
			update_needed .= consumer_ideology - update_needed
		end

		u_exp_sum .= 1
		choice_prob_per_channel[:,1] .= 1

		for c = 1:C
			u_suspense[:,c] .= 0
			u_slant[:,c] .= 0
			u_vote[:,c] .= 0
			for i = 1:N
				for τ in nonnull_topics
					# unconditional prob of reporting R on each topic
					pr_report_R[i,τ] = μ_τ[i,τ] * q_R[c] + (1 - μ_τ[i,τ]) * (1 - q_D[c])

					# amount of updating conditional on each event
					update_if_report_R[i,τ,c] = (μ_τ[i,τ] * q_R[c]) / (μ_τ[i,τ] * q_R[c] + (1 - μ_τ[i,τ]) * (1 - q_D[c])) - μ_τ[i,τ]
					update_if_report_D[i,τ,c] = ((μ_τ[i,τ] * (1 - q_R[c])) / ((1 - μ_τ[i,τ]) * q_D[c] + μ_τ[i,τ] * (1 - q_R[c]))) - μ_τ[i,τ]

					# topic contribution to suspense util is variance of posterior * topic weight this period
					u_suspense[i,c] += consumer_viewed_topic[i,τ,c,t] * (pr_report_R[i,τ] * update_if_report_R[i,τ,c]^2 + (1 - pr_report_R[i,τ]) * update_if_report_D[i,τ,c]^2) * β_suspense[i,τ]

					# topic contribution to slant util is pr(report R) * topic weight this period
					u_slant[i,c] += consumer_viewed_topic[i,τ,c,t] * abs(pr_report_R[i,τ] - consumer_r_prob[i])

					# topic contribution to voting util is amount beliefs move in direction of threshold
					if pre_election
						u_vote[i,c] += (update_needed[i] >= 0 ? update_if_report_R[i,τ,c] : update_if_report_D[i,τ,c]) *
						 			   (update_needed[i] >= 0 ? pr_report_R[i,τ] : 1 - pr_report_R[i,τ]) *
										look_forward_1[τ] *
										λ_τ[τ] *
										consumer_viewed_topic[i,τ,c,t]
					end
				end
				# scale vote util by dist. to threshold (cap at 1)
				u_vote[i,c] = min(u_vote[i,c] / update_needed[i], 1)

				# hour / show / channel effects
				u_0[i,c] = β_0[i,consumer_viewed_show[i,c,t]] + β_hour[consumer_tz[i], h_index]

			end # i loop

			# add components
			@. choice_prob_per_channel[:,c+1] = exp(min(u_0[:,c] + u_suspense[:,c] + (β_vote * u_vote[:,c]) + (β_slant * u_slant[:,c]), 100))
			u_exp_sum .+= choice_prob_per_channel[:,c+1]
		end # c loop

		# compute choice probabilities
		choice_prob_per_channel ./= u_exp_sum

		# store average viewing prob among national sample
		colmeans!(predicted_viewership[:,t], choice_prob_per_channel[1:last_national,2:end])

		# simulate viewing decisions
		# and update beliefs over topic states
		# move a fraction of the amount weighted by topic time
		for i in 1:N
			if (choice_draws[i,t] > choice_prob_per_channel[i,1])
				# draw > prob (not watch anything)
				sum_choice_prob = choice_prob_per_channel[i,1]
				for c in 1:C
					sum_choice_prob += choice_prob_per_channel[i,c+1]
					if choice_draws[i,t] <= sum_choice_prob
						# record viewing event
						consumer_watched_history[i,c,t] = true

						# update beliefs over topics with news today
						for τ in nonnull_topics
							if channel_reports[c,t,τ] == -1
								μ_τ[i,τ] += consumer_viewed_topic[i,τ,c,t] * update_if_report_D[i,τ,c]
							elseif channel_reports[c,t,τ] == 1
								μ_τ[i,τ] += consumer_viewed_topic[i,τ,c,t] * update_if_report_R[i,τ,c]
							end # if τ in nonnull
						end # τ loop

						break
					end # if choice_draws
				end # c loop
			end # if choice_draws
		end # i loop

		# apply updates to aggregate state estimate
		mul!(μ_star, μ_τ, λ_τ)
	end # t loop
end
