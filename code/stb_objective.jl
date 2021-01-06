function stb_obj(ev::SMM.Eval;
				 dt::STBData,
				 save_output=false,
				 store_moments=false)

	SMM.start(ev)

	## extract parameters
	# topic-related
	topic_lambda  = SMM.param(ev, dt.keys_lambda)
	topic_rho 	  = reshape(SMM.param(ev, dt.keys_rho),1,dt.K)
	topic_leisure = reshape(SMM.param(ev, dt.keys_leisure),1,dt.K)
	topic_mu0     = reshape(SMM.param(ev, dt.keys_mu),1,dt.K)


	inverse_transition_matrices = [inv([ρ (1 - ρ); (1 - ρ) ρ]) for ρ in topic_rho]

	# channel-related: probs of reporting each state
	channel_q_D = SMM.param(ev, dt.keys_channel_q_D)
	channel_q_R = SMM.param(ev, dt.keys_channel_q_R)
	channel_q_0 = SMM.param(ev, dt.keys_channel_q_0)

	# individual-specific
	consumer_beta_vote 	 	= SMM.param(ev, Symbol("beta:vote"))
	consumer_beta_slant     = SMM.param(ev, Symbol("beta:slant"))

	# channel taste heterogeneity
	if (length(dt.keys_channel_mu) == dt.C)
		consumer_beta_channel_mu = reshape(SMM.param(ev, dt.keys_channel_mu),1,dt.C)     	# channel lognormals (mean)
	else
		consumer_beta_channel_mu = ones(Float64, 1, dt.C) .* -Inf  # if missing, set to -Inf => no heterogeneity in channel taste
	end
	if (length(dt.keys_channel_sigma) == dt.C)
		consumer_beta_channel_sigma = reshape(SMM.param(ev, dt.keys_channel_sigma),1,dt.C);     	# channel lognormals (sd)
	else
		consumer_beta_channel_sigma = zeros(Float64, 1, dt.C) # if missing, set to zero => no heterogeneity in channel taste
	end

	# chance of no utility from news
	consumer_zero_news = SMM.param(ev, Symbol("zero:news"))

	# show dummies
	if (length(dt.keys_show)==dt.S)
		consumer_beta_show = [zeros(1,dt.C) reshape(SMM.param(ev, dt.keys_show),1,dt.S)];     				# show dummies; normalize special coverage on every channel to have 0 show effect
	elseif (length(dt.keys_show)==dt.C)
		consumer_beta_show = reshape(SMM.param(ev, dt.keys_show)[dt.show_to_channel], 1, dt.S+dt.C);
	else
		consumer_beta_show = zeros(1,dt.C+dt.S);
	end

	# scale draws for base channel utility.
	# consumer draw is product of Pareto-distributed draw and Bernoulli draw
	# consumer_beta_0 = ((dt.pre_consumer_channel_zeros .>= consumer_zero_channel) .* (exp.(dt.pre_consumer_channel_draws .* consumer_beta_channel) .- 1)) .+
	# 				  ((dt.pre_consumer_news_zeros .>= consumer_zero_news) .* (exp.(dt.pre_consumer_news_draws .* consumer_beta_news) .- 1));               # exponential draw for channel effect

	# incorporate (possible) heterogeneity in channel tastes
	consumer_beta_0 = (dt.pre_consumer_news_zeros .>= consumer_zero_news) .* exp.(dt.pre_consumer_channel_draws .* consumer_beta_channel_sigma .+ consumer_beta_channel_mu)
	consumer_beta_0 = consumer_beta_0[:,dt.show_to_channel] .+ consumer_beta_show  # plus common show effect

	if (length(dt.keys_etz) > 0)
		consumer_beta_hour_etz = SMM.param(ev, dt.keys_etz); # hour dummies (eastern)
		consumer_beta_hour_ctz = SMM.param(ev, dt.keys_ctz); # hour dummies (central)
		consumer_beta_hour_mtz = SMM.param(ev, dt.keys_mtz); # hour dummies (mountain)
		consumer_beta_hour_ptz = SMM.param(ev, dt.keys_ptz); # hour dummies (pacific)

		consumer_beta_hour = [consumer_beta_hour_etz';
		                      consumer_beta_hour_ctz';
		                      consumer_beta_hour_mtz';
		                      consumer_beta_hour_ptz'];
    else
		consumer_beta_hour = zeros(4,6);
	end

	# incorporate (possible) heterogeneity in topic tastes
	consumer_topic_leisure = (dt.pre_consumer_news_zeros .>= consumer_zero_news) .* (dt.pre_consumer_topic_draws .* topic_leisure);


	# daily news events by topic
	news_thresh = reshape(SMM.param(ev, dt.keys_news), K, D)
 	r_news_thresh = news_thresh .* reshape(SMM.param(ev, dt.keys_r_news), K, D)
	news = zeros(Int8, dt.K, dt.D, dt.B)
	for b in 1:dt.B
		@views news[:,:,b] = (dt.pre_path_draws[:,:,b] .<= news_thresh) .* (2 .* (dt.pre_path_draws[:,:,b] .<= r_news_thresh) .- 1)
	end


	#### given model parameters, compute ####
 	# 1. channel ratings by time period
 	# 2. daily polling outcomes
	# repeat for each path draw

	# arrays to track consumer current estimates of P(state favors R)
	consumer_topic_est    = zeros(Float64,dt.N,dt.K);
	consumer_agg_est      = zeros(Float64,dt.N,dt.D);

	# consumer ideology (threshold) is 1 - r_prob
	consumer_ideology = 1 .- dt.consumer_r_prob

	# arrays to track viewing in each period
	consumer_view_history_stb = zeros(Bool, dt.N_stb, dt.C, dt.T);
	consumer_view_history_national = zeros(Bool, dt.N_national, dt.C, dt.T);

	# history of ratings by channel (at block level)
	predicted_channel_ratings = zeros(Float64, dt.C, dt.T);

	# daily poll averages
	daily_polling = zeros(Float64,dt.D);

	# arrays for one day worth of viewing
	consumer_viewed_topic = zeros(Float64,dt.N,dt.K,dt.C,dt.t_i);
	consumer_viewed_show  = zeros(Int8,dt.N,dt.C,dt.t_i);
	consumer_watched_byblock = zeros(Bool,dt.N,dt.C,dt.t_i);

	# arrays to store today's topic weights and show schedule
	temp_topics = zeros(Float64,dt.K,dt.C,dt.t_i)
	temp_shows = zeros(Int64,dt.C,dt.t_i)

	## preallocated arrays for inner loop
	# utility components
	base_util           = zeros(Float64,dt.N,dt.C);
	vote_util           = zeros(Float64,dt.N,dt.C);
	suspense_util       = zeros(Float64,dt.N,dt.C);
	slant_util          = zeros(Float64,dt.N,dt.C);
	sum_exp_util 		= ones(Float64,dt.N);

	# reporting probabilities and simulated channel reports
	update_if_report_R  = zeros(Float64, dt.N, dt.K, dt.C)
	update_if_report_D  = zeros(Float64, dt.N, dt.K, dt.C)
	update_needed       = zeros(Float64, dt.N)
	channel_reports     = zeros(Int8,dt.C,dt.t_i,dt.K)

	# choice probabilities and simulated choices
	choice_prob_per_channel   = zeros(Float64,dt.N,dt.C+1)
	choice_prob_per_channel[:,1] .= 1

	# arrays to store the moments
	view_thresh = zeros(Float64, 4, dt.C)
	view_thresh_cross = zeros(Float64, 3, dt.C)
	r_prob_quantiles = zeros(Float64, 3, dt.C)
	view_by_tercile = zeros(Float64, dt.C, 3)

	# moment vector and locations to store components
	sim_moments = zeros(Float64, length(SMM.dataMoment(ev)))
	cur_ind = 1
	inds_view_thresh = cur_ind:(cur_ind + length(view_thresh) - 1)
	cur_ind += length(view_thresh)
	inds_r_prob_quant = cur_ind:(cur_ind + length(r_prob_quantiles) - 1)
	cur_ind += length(r_prob_quantiles)
	inds_view_by_tercile = cur_ind:(cur_ind + length(view_by_tercile) - 1)
	cur_ind += length(view_by_tercile)
	inds_view_thresh_cross = cur_ind:(cur_ind + length(view_thresh_cross) - 1)
	cur_ind += length(view_thresh_cross)
	inds_predicted_ratings = cur_ind:(cur_ind + length(predicted_channel_ratings) - 1)
	cur_ind += length(predicted_channel_ratings)
	inds_daily_polling = cur_ind:(cur_ind + dt.election_day - 1)
	cur_ind += dt.election_day
	inds_news_penalty = cur_ind


	for b = 1:dt.B
		# reset consumer topic estimates and aggregate state estimates
		consumer_topic_est .= topic_mu0
		consumer_agg_est[:,1] .= consumer_topic_est * topic_lambda

		# reset forward transition matrices
		# first row of exponentiated transition matrix for each topic
		# col 1 of look_forward is m[1,1] - m[1,2], second is m[1,2]
		forward_transition_matrices = [[ρ (1 - ρ); (1 - ρ) ρ]^dt.election_day for ρ in topic_rho]
		look_forward = vcat([reshape(m[1,:],1,2) for m in forward_transition_matrices]...)
		look_forward[:,1] = look_forward[:,1] - look_forward[:,2]

		for d = 1:dt.D
		  t = (d-1) * dt.t_i + 1;
		  last_t = t + dt.t_i - 1;

		  # increment the forward projection matrices down
		  if d > 1
			  for m in 1:length(forward_transition_matrices)
			  	look_forward[m,:] = (forward_transition_matrices[m] * inverse_transition_matrices[m])[1,:]
			  end
			  look_forward[:,1] = look_forward[:,1] - look_forward[:,2]
		  end
		  # create array of topics (possibly) viewed by each consumer,
		  # respecting time zone
		  @views temp_topics .= dt.channel_topic_coverage_etz[:,:,t:last_t];
		  for i = dt.i_etz
			  consumer_viewed_topic[i,:,:,:] = temp_topics;
		  end
		  @views temp_topics .= dt.channel_topic_coverage_ctz[:,:,t:last_t];
		  for i = dt.i_ctz
			  consumer_viewed_topic[i,:,:,:] = temp_topics;
		  end
		  @views temp_topics .= dt.channel_topic_coverage_mtz[:,:,t:last_t];
		  for i = dt.i_mtz
			  consumer_viewed_topic[i,:,:,:] = temp_topics;
		  end
		  @views temp_topics .= dt.channel_topic_coverage_ptz[:,:,t:last_t];
		  for i = dt.i_ptz
			  consumer_viewed_topic[i,:,:,:] = temp_topics;
		  end

		  # create array of shows (possibly) viewed by each consumer,
		  # respecting time zone
		  @views temp_shows .= dt.channel_show_etz[:,t:last_t];
		  for i = dt.i_etz
			  consumer_viewed_show[i,:,:] = temp_shows;
		  end
		  @views temp_shows .= dt.channel_show_ctz[:,t:last_t];
		  for i = dt.i_ctz
			  consumer_viewed_show[i,:,:] = temp_shows;
		  end
		  @views temp_shows .= dt.channel_show_mtz[:,t:last_t];
		  for i = dt.i_mtz
			  consumer_viewed_show[i,:,:] = temp_shows;
		  end
		  @views temp_shows .= dt.channel_show_ptz[:,t:last_t];
		  for i = dt.i_ptz
			  consumer_viewed_show[i,:,:] = temp_shows;
		  end

		  consumer_watched_byblock .= false

		  @views sim_viewership_polling!(d,
			  consumer_watched_byblock,
			  consumer_topic_est,
			  consumer_agg_est[:,d],
			  predicted_channel_ratings[:,t:last_t],
			  daily_polling,
			  base_util,vote_util,suspense_util,slant_util,sum_exp_util,
			  channel_reports,
			  choice_prob_per_channel,
			  update_if_report_R, update_if_report_D, update_needed,
			  channel_q_R, channel_q_D, channel_q_0,
			  topic_rho,
			  topic_lambda,
			  consumer_beta_0, consumer_beta_hour, consumer_beta_slant, consumer_beta_vote, consumer_topic_leisure,
			  news[:,d,b],
			  look_forward,
			  consumer_viewed_topic,
			  consumer_viewed_show,
			  d <= dt.election_day,
	          consumer_ideology, dt.consumer_r_prob,
			  dt.consumer_tz,
			  last(dt.i_national),
			  dt.consumer_choice_draws[:,t:last_t],
			  dt.channel_report_draws[:,t:last_t,:],
			  dt.N,dt.K,dt.C,dt.t_i)

		  @views consumer_view_history_stb[:,:,t:last_t] = consumer_watched_byblock[dt.i_stb,:,:]

		  if save_output
		  	@views consumer_view_history_national[:,:,t:last_t] = consumer_watched_byblock[dt.i_national,:,:]
		  end

		end

		## write to disk if enabled
		if save_output
			save(@sprintf("model_output_path_%i.jld2", b),
				"consumer_view_history_stb", consumer_view_history_stb,
				"consumer_view_history_national", consumer_view_history_national,
				"predicted_channel_ratings", predicted_channel_ratings,
				"daily_polling", daily_polling,
				"news", news[:,:,b],
				"channel_topic_coverage_ctz", dt.channel_topic_coverage_ctz,
				"channel_topic_coverage_etz", dt.channel_topic_coverage_etz,
				"channel_topic_coverage_mtz", dt.channel_topic_coverage_mtz,
				"channel_topic_coverage_ptz", dt.channel_topic_coverage_ptz)
		end


		## Individual viewership moments
		compute_indiv_moments(dt, consumer_view_history_stb, view_thresh, view_thresh_cross, r_prob_quantiles, view_by_tercile)


		sim_moments[inds_view_thresh] .+= vec(view_thresh)
		sim_moments[inds_r_prob_quant] .+= vec(r_prob_quantiles)
		sim_moments[inds_view_by_tercile] .+= vec(view_by_tercile)
		sim_moments[inds_view_thresh_cross] .+= vec(view_thresh_cross)
		sim_moments[inds_predicted_ratings] .+= vec(predicted_channel_ratings)
		sim_moments[inds_daily_polling] .+= daily_polling[1:dt.election_day]
		sim_moments[inds_news_penalty] += sum(news[:,:,b] .^ 2)

	end # b loop

	sim_moments ./= B

	ssq = sum((sim_moments .- SMM.dataMoment(ev)).^2 .* SMM.dataMomentW(ev,collect(keys(ev.dataMomentsW))))

	# Set value of the objective function:
	#------------------------------------
	SMM.setValue!(ev, dt.t_i * dt.D / 2 * ssq)
	# SMM.setValue!(ev, ssq)

	# also return the moments if requested
	#-----------------------
	if store_moments
		for (i,m) in enumerate(keys(SMM.dataMomentd(ev)))
			SMM.setMoments!(ev, m, sim_moments[i]);
		end
	end
	# flag for success:
	#-------------------
	ev.status = 1

	# finish and return
	SMM.finish(ev)

	return ev

end


function compute_indiv_moments(dt, consumer_view_history_stb, view_thresh, view_thresh_cross, r_prob_quantiles, view_by_tercile)
	#  concentration + cross-viewing
	individual_avg_viewing_min = dropdims(sum(consumer_view_history_stb,dims=3),dims=3) .* 15 ./ dt.D
	thresh_one = [0.18, 0.36, 3.6, 36]
	thresh_two = [0.18, 0.36, 3.6]

	@views for t in 1:length(thresh_one)
		cross_ind = 0
		for c in 1:dt.C
			view_thresh[t,c] = sum(individual_avg_viewing_min[:,c] .>= thresh_one[t]) / dt.N_stb
			if (c < dt.C) & (t <= length(thresh_two))
				for c2 in (c+1):dt.C
					cross_ind += 1
					view_thresh_cross[t, cross_ind] = sum((individual_avg_viewing_min[:,c] .>= thresh_two[t]) .& (individual_avg_viewing_min[:,c2] .>= thresh_two[t])) / dt.N_stb
				end
			end
		end
	end

	# ideological segregation
	quantiles = [0.25, 0.5, 0.75]
	@views for c in 1:dt.C
		total_mass = sum(individual_avg_viewing_min[:,c])
		csum = 0
		cur_q = 1
		cur_thresh = quantiles[cur_q] * total_mass
		for i in dt.sorted_inds
			csum += individual_avg_viewing_min[i,c]
			if csum >= cur_thresh
				r_prob_quantiles[cur_q,c] = dt.r_prob_stb[i]
				cur_q += 1

				if cur_q > length(quantiles)
					break
				end
				cur_thresh = quantiles[cur_q] * total_mass
			end
		end
	end


	# average viewing by channel / r_prob tercile
	terciles = [0.3333, 0.6666]
	tercile_inds= cat(0, [ceil(Int64, length(dt.sorted_inds) * t) for t in terciles], dt.N_stb; dims=1)
	@views for t in 1:(length(tercile_inds)-1)
		for c in 1:dt.C
			view_by_tercile[c,t] = Statistics.mean(individual_avg_viewing_min[dt.sorted_inds[(tercile_inds[t]+1):tercile_inds[t+1]],c])
		end
	end

	# [reshape([pct_0005; pct_001; pct_01; pct_1], 4*dt.C);
	# 		 r_prob_cnn_25;r_prob_cnn_50;r_prob_cnn_75;
	# 		 r_prob_fnc_25;r_prob_fnc_50;r_prob_fnc_75;
	# 		 r_prob_msnbc_25;r_prob_msnbc_50;r_prob_msnbc_75;
	# 		 reshape(left_third_viewing, dt.C); reshape(center_third_viewing, dt.C); reshape(right_third_viewing, dt.C);
	# 		 cnn_fnc_joint_pcts;cnn_msn_joint_pcts;msn_fnc_joint_pcts];


end
