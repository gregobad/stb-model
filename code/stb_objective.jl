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

	forward_transition_matrices = [[ρ (1 - ρ); (1 - ρ) ρ]^dt.election_day for ρ in topic_rho]
	inverse_transition_matrices = [inv([ρ (1 - ρ); (1 - ρ) ρ]) for ρ in topic_rho]

	# first row of exponentiated transition matrix for each topic
	# col 1 of look_forward is m[1,1] - m[1,2], second is m[1,2]
	look_forward = vcat([reshape(m[1,:],1,2) for m in forward_transition_matrices]...)
	look_forward[:,1] = look_forward[:,1] - look_forward[:,2]

	# channel-related: probs of reporting each state
	channel_q_D = SMM.param(ev, dt.keys_channel_q_D)
	channel_q_R = SMM.param(ev, dt.keys_channel_q_R)

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
	consumer_zero_news = SMM.param(ev, Symbol("zero:news"));

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
	consumer_beta_0 = (dt.pre_consumer_news_zeros .>= consumer_zero_news) .* exp.(dt.pre_consumer_channel_draws .* consumer_beta_channel_sigma .+ consumer_beta_channel_mu);
	consumer_beta_0 = consumer_beta_0[:,dt.show_to_channel] .+ consumer_beta_show[ones(Int, dt.N),:];  # plus common show effect

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
	news = zeros(Int8, dt.K, dt.D)
	news[dt.nonsparse_index] .= SMM.param(ev, dt.keys_news); # nonzero elements of topic path


	#### given model parameters, compute ####
 	# 1. channel ratings by time period
 	# 2. daily polling outcomes

	# arrays to track consumer current estimates of P(state favors R)
	consumer_topic_est    = zeros(Float64,dt.N,dt.K);
	consumer_agg_est      = zeros(Float64,dt.N,dt.D);

	# initialize to topic_mu0 (add some dispersion here?)
	consumer_topic_est .= topic_mu0
	consumer_agg_est[:,1] .= consumer_topic_est * topic_lambda

	# arrays to track viewing in each period
	consumer_view_history_stb = zeros(Bool, dt.N_stb, dt.C, dt.T);
	consumer_view_history_national = zeros(Bool, dt.N_national, dt.C, dt.T);

	# history of ratings by channel (at block level)
	predicted_channel_ratings = zeros(Float64, dt.C, dt.T);

	# daily poll averages and daily averaged ratings
	daily_polling = zeros(Float64,dt.D);
	daily_viewership = zeros(Float64,dt.D,dt.C);

	# arrays for one day worth of viewing
	predicted_viewership = zeros(Float64, dt.C, dt.t_i);
	consumer_viewed_topic = zeros(Float64,dt.N,dt.K,dt.C,dt.t_i);
	consumer_viewed_show  = zeros(Int8,dt.N,dt.C,dt.t_i);
	consumer_watched_byblock      = zeros(Bool,dt.N,dt.C,dt.t_i);

	# arrays to store today's topic weights and show schedule
	temp_topics = zeros(Float64,dt.K,dt.C,dt.t_i)
	temp_shows = zeros(Int64,dt.C,dt.t_i)
	channel_topic_weights = zeros(Float64, dt.N, dt.K)

	## preallocated arrays for inner loop
	# utility components
	base_util           = zeros(Float64,dt.N,dt.C);
	vote_util           = zeros(Float64,dt.N,dt.C);
	suspense_util        = zeros(Float64,dt.N,dt.C);
	slant_util          = zeros(Float64,dt.N,dt.C);
	channel_util 		= zeros(Float64,dt.N,dt.C);
	exp_util 			= zeros(Float64,dt.N,dt.C);

	# reporting probabilities and simulated channel reports
	pr_report_R 		= zeros(Float64, dt.N, dt.K)
	update_if_report_R  = zeros(Float64, dt.N, dt.K, dt.C)
	update_if_report_D  = zeros(Float64, dt.N, dt.K, dt.C)
	update_needed       = zeros(Float64, dt.N)
	channel_reports     = zeros(Int8,dt.C,dt.t_i,dt.K)

	# choice probabilities and simulated choices
	choice_prob_per_channel   = zeros(Float64,dt.N,dt.C+1);
	consumer_choice_threshold = zeros(Float64,dt.N,dt.C+1);
	consumer_report_last      = zeros(Int8,dt.N,dt.K);

	row_summer = ones(Float64, dt.K)

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
	  temp_topics .= dt.channel_topic_coverage_etz[:,:,t:last_t];
	  for i = dt.i_etz
		  consumer_viewed_topic[i,:,:,:] = temp_topics;
	  end
	  temp_topics .= dt.channel_topic_coverage_ctz[:,:,t:last_t];
	  for i = dt.i_ctz
		  consumer_viewed_topic[i,:,:,:] = temp_topics;
	  end
	  temp_topics .= dt.channel_topic_coverage_mtz[:,:,t:last_t];
	  for i = dt.i_mtz
		  consumer_viewed_topic[i,:,:,:] = temp_topics;
	  end
	  temp_topics .= dt.channel_topic_coverage_ptz[:,:,t:last_t];
	  for i = dt.i_ptz
		  consumer_viewed_topic[i,:,:,:] = temp_topics;
	  end

	  # create array of shows (possibly) viewed by each consumer,
	  # respecting time zone
	  temp_shows .= dt.channel_show_etz[:,t:last_t];
	  for i = dt.i_etz
		  consumer_viewed_show[i,:,:] = temp_shows;
	  end
	  temp_shows .= dt.channel_show_ctz[:,t:last_t];
	  for i = dt.i_ctz
		  consumer_viewed_show[i,:,:] = temp_shows;
	  end
	  temp_shows .= dt.channel_show_mtz[:,t:last_t];
	  for i = dt.i_mtz
		  consumer_viewed_show[i,:,:] = temp_shows;
	  end
	  temp_shows .= dt.channel_show_ptz[:,t:last_t];
	  for i = dt.i_ptz
		  consumer_viewed_show[i,:,:] = temp_shows;
	  end

	  sim_viewership_polling!(d,
		  consumer_watched_byblock,
		  consumer_topic_est,
		  consumer_agg_est[:,d],
		  predicted_viewership,
		  daily_polling,
		  base_util,vote_util,suspense_util,slant_util,channel_util,exp_util,
		  channel_reports,
		  choice_prob_per_channel,
		  consumer_choice_threshold,
		  pr_report_R,
		  update_if_report_R, update_if_report_D, update_needed,
		  channel_q_R, channel_q_D,
		  topic_rho,
		  topic_lambda,
		  consumer_beta_0, consumer_beta_hour, consumer_beta_slant, consumer_beta_vote, consumer_topic_leisure,
		  news[:,d],
		  look_forward,
		  consumer_viewed_topic,
		  consumer_viewed_show,
		  d <= dt.election_day,
          dt.consumer_r_prob,
		  dt.consumer_tz,
		  dt.i_stb, dt.i_national,
		  dt.consumer_choice_draws[:,t:last_t],
		  dt.channel_report_draws[:,t:last_t,:],
		  row_summer, dt.N,dt.K,dt.C,dt.t_i);

	  consumer_view_history_stb[:,:,t:last_t] = consumer_watched_byblock[dt.i_stb,:,:];
	  if(save_output)
	  	consumer_view_history_national[:,:,t:last_t] = consumer_watched_byblock[dt.i_national,:,:];
	  end

	  daily_viewership[d,:] =
	      mapslices(Statistics.mean,predicted_viewership; dims=2);

	  predicted_channel_ratings[:,t:last_t] = predicted_viewership;

	end


	## Individual viewership moments
	model_viewership_indiv_rawmoments = compute_indiv_moments(dt, consumer_view_history_stb, save_output=save_output);

	## write to disk if enabled
	if save_output
		save("post_inside_obj_func.jld2", "consumer_view_history_stb", consumer_view_history_stb, "model_viewership_indiv_rawmoments", model_viewership_indiv_rawmoments, "daily_viewership", daily_viewership, "predicted_channel_ratings", predicted_channel_ratings, "daily_polling", daily_polling, "news", news, "consumer_view_history_national", consumer_view_history_national)
	end


	sim_moments = cat(model_viewership_indiv_rawmoments,
					  reshape(predicted_channel_ratings,dt.C*dt.T),
					  daily_polling[1:dt.election_day],
					  sum(news .^ 2); dims=1);

	ssq = sum((sim_moments .- SMM.dataMoment(ev)).^2 .* SMM.dataMomentW(ev,collect(keys(ev.dataMomentsW))));
	if isnan(ssq)
		ssq = 1e8
	end

	# Set value of the objective function:
	#------------------------------------
	SMM.setValue!(ev, 4128 / 2 * ssq)
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

	# save_output ? save("post_inside_obj_func.jld2", "consumer_view_history_stb", consumer_view_history_stb, "model_viewership_indiv_rawmoments", model_viewership_indiv_rawmoments, "raw_mean_predicted_ratings", raw_mean_predicted_ratings, "sim_moments", sim_moments) : nothing

end


function compute_indiv_moments(dt, consumer_view_history_stb; save_output=false)
	#  concentration
	individual_avg_viewing_min= dropdims(sum(consumer_view_history_stb,dims=3),dims=3) .* 15 ./ dt.D;

	pct_0005=mapslices(Statistics.mean, individual_avg_viewing_min.>=0.18, dims=1);
	pct_001=mapslices(Statistics.mean, individual_avg_viewing_min.>=0.36, dims=1);
	pct_01=mapslices(Statistics.mean, individual_avg_viewing_min.>=3.6, dims=1);
	pct_1=mapslices(Statistics.mean, individual_avg_viewing_min.>=36, dims=1);
	# ideological segregation
	r_prob_stb=dt.consumer_r_prob[dt.i_stb];

	r_prob_stb_sort = sort(r_prob_stb);
	idx = sortperm(r_prob_stb);

	cnn_view_min_mass_vec = cumsum(individual_avg_viewing_min[idx,1],dims=1);
	r_prob_cnn_25 = r_prob_stb_sort[findfirst(cnn_view_min_mass_vec .>= .25*cnn_view_min_mass_vec[end]) ];
	r_prob_cnn_50 = r_prob_stb_sort[findfirst(cnn_view_min_mass_vec .>= .5*cnn_view_min_mass_vec[end]) ];
	r_prob_cnn_75 = r_prob_stb_sort[findfirst(cnn_view_min_mass_vec .>= .75*cnn_view_min_mass_vec[end]) ];

	fnc_view_min_mass_vec = cumsum(individual_avg_viewing_min[idx,2],dims=1);
	r_prob_fnc_25 = r_prob_stb_sort[findfirst(fnc_view_min_mass_vec .>= .25*fnc_view_min_mass_vec[end]) ];
	r_prob_fnc_50 = r_prob_stb_sort[findfirst(fnc_view_min_mass_vec .>= .5*fnc_view_min_mass_vec[end]) ];
	r_prob_fnc_75 = r_prob_stb_sort[findfirst(fnc_view_min_mass_vec .>= .75*fnc_view_min_mass_vec[end]) ];

	msnbc_view_min_mass_vec = cumsum(individual_avg_viewing_min[idx,3],dims=1);
	r_prob_msnbc_25 = r_prob_stb_sort[findfirst(msnbc_view_min_mass_vec .>= .25*msnbc_view_min_mass_vec[end]) ];
	r_prob_msnbc_50 = r_prob_stb_sort[findfirst(msnbc_view_min_mass_vec .>= .5*msnbc_view_min_mass_vec[end]) ];
	r_prob_msnbc_75 = r_prob_stb_sort[findfirst(msnbc_view_min_mass_vec .>= .75*msnbc_view_min_mass_vec[end]) ];

	# cross-channel correlation
	# channel_viewership_corr=Statistics.cor(individual_avg_viewing_min);   # Pearson correlation
	# channel_viewership_corr = Statistics.cor(mapslices(StatsBase.tiedrank, individual_avg_viewing_min, dims=1));  # Spearman correlation
	cnn_fnc_joint_pcts=[Statistics.mean((individual_avg_viewing_min[:,1] .>= x) .& (individual_avg_viewing_min[:,2] .>= x)) for x in [0.18, 0.36, 3.6]];
	cnn_msn_joint_pcts=[Statistics.mean((individual_avg_viewing_min[:,1] .>= x) .& (individual_avg_viewing_min[:,3] .>= x)) for x in [0.18, 0.36, 3.6]];
	fnc_msn_joint_pcts=[Statistics.mean((individual_avg_viewing_min[:,2] .>= x) .& (individual_avg_viewing_min[:,3] .>= x)) for x in [0.18, 0.36, 3.6]];

	# average viewing by channel / r_prob tercile
	terciles = Statistics.quantile(r_prob_stb,[0.333,0.666]);
	left_third_viewing = mapslices(Statistics.mean, individual_avg_viewing_min[r_prob_stb .<= terciles[1],:], dims=1);
	center_third_viewing = mapslices(Statistics.mean, individual_avg_viewing_min[(r_prob_stb .<= terciles[2]) .& (r_prob_stb .> terciles[1]),:], dims=1);
	right_third_viewing = mapslices(Statistics.mean, individual_avg_viewing_min[r_prob_stb .> terciles[2],:], dims=1);

	# write to disk if enabled
	save_output ? save("post_inside_compute_moments.jld2", "consumer_view_history_stb", consumer_view_history_stb, "individual_avg_viewing_min", individual_avg_viewing_min, "pct_0005", pct_0005, "pct_001", pct_001, "pct_01", pct_01, "pct_1", pct_1, "cnn_view_min_mass_vec", cnn_view_min_mass_vec, "fnc_view_min_mass_vec", fnc_view_min_mass_vec, "msnbc_view_min_mass_vec", msnbc_view_min_mass_vec) : nothing

	[reshape([pct_0005; pct_001; pct_01; pct_1], 4*dt.C);
			 r_prob_cnn_25;r_prob_cnn_50;r_prob_cnn_75;
			 r_prob_fnc_25;r_prob_fnc_50;r_prob_fnc_75;
			 r_prob_msnbc_25;r_prob_msnbc_50;r_prob_msnbc_75;
			 reshape(left_third_viewing, dt.C); reshape(center_third_viewing, dt.C); reshape(right_third_viewing, dt.C);
			 cnn_fnc_joint_pcts;cnn_msn_joint_pcts;cnn_fnc_joint_pcts];


end
