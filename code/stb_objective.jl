function stb_obj(ev::SMM.Eval;
				 dt::STBData,
				 save_output=false,
				 store_moments=false)

	SMM.start(ev)

	## extract parameters
	topic_lambda = SMM.param(ev, dt.keys_lambda);
	topic_lambda_squared=topic_lambda.^2;
	topic_leisure=reshape(SMM.param(ev, dt.keys_leisure),1,dt.K);
	topic_mu=SMM.param(ev, dt.keys_mu);

	consumer_state_var_0 = SMM.param(ev, :consumer_state_var_0);
	channel_locations = SMM.param(ev, dt.keys_channel_loc);
	consumer_beta_info_util= SMM.param(ev, Symbol("beta:info"));
	consumer_beta_slant= SMM.param(ev, Symbol("beta:slant"));

	# channel taste heterogeneity
	if (length(dt.keys_channel_mu) == dt.C)
		consumer_beta_channel_mu = reshape(SMM.param(ev, dt.keys_channel_mu),1,dt.C);     	# channel lognormals (mean)
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

	consumer_topic_leisure = (dt.pre_consumer_news_zeros .>= consumer_zero_news) .* (dt.pre_consumer_topic_draws .* topic_leisure);

	channel_report_var=SMM.param(ev, :channel_report_var);      # report error variance
	ch_rept_errs=dt.channel_report_errors .* sqrt(channel_report_var);

	consumer_free_var=SMM.param(ev, :consumer_free_var);
	ch_free_errs=dt.consumer_free_errors .* sqrt(consumer_free_var);

	innovations = zeros(Float64, dt.K, dt.D);						   # innovations may be sparse
	innovations[dt.innov_index] .= SMM.param(ev, dt.keys_innovations); # nonzero elements of topic path

	topic_var = ones(Float64,dt.K,1);
	topic_var_expanded = ones(Float64,dt.K,dt.N);

	# construct consumer ideology to match initial R propensity given model parameters
	agg_var_E = dt.time_inter*dt.election_day * sum(topic_lambda_squared);
	agg_0 = sum(topic_mu .* topic_lambda);
	consumer_ideology = StatsFuns.norminvcdf.(agg_0 .* ones(Float64,dt.N), sqrt(agg_var_E) .* ones(Float64,dt.N), dt.consumer_r_prob);


	#### given model parameters, compute ####
 	# 1. channel ratings by time period
 	# 2. daily polling outcomes

	consumer_topic_est    = zeros(Float64,dt.K,dt.N);
	consumer_topic_var    = zeros(Float64,dt.K,dt.N);
	consumer_agg_est      = zeros(Float64,dt.N,dt.D);
	consumer_agg_var      = zeros(Float64,dt.N,dt.D);
	topic_path = zeros(Float64,dt.K,dt.D);
	topic_path[:,1]=topic_mu;

	consumer_topic_est=rep_col(topic_mu, dt.N);
	consumer_topic_var=consumer_state_var_0 .* ones(Float64,dt.K,dt.N);
	consumer_agg_est[:,1] = consumer_topic_est' * topic_lambda;
	consumer_agg_var[:,1] = consumer_topic_var' * topic_lambda_squared;

	consumer_view_history_stb = zeros(Float64, dt.T, dt.N_stb, dt.C);
	# consumer_view_history_national = Array{Array{Float64,3}}(undef,dt.D);
	consumer_view_history_national = zeros(Float64, dt.T, dt.N_national, dt.C);
	predicted_channel_ratings = zeros(Float64, dt.C, dt.T);

	track_topic_est = zeros(Float64,dt.K,dt.N,dt.D);
	track_polling = zeros(Float64,dt.D,1);
	track_viewership = zeros(Float64,dt.C,dt.D);

	# arrays for one day worth of viewing
	predicted_viewership = zeros(Float64, dt.C, dt.t_i);
	consumer_viewed_topic = zeros(Float64,dt.K,dt.N,dt.C,dt.t_i);
	consumer_viewed_show = zeros(Int,dt.N,dt.C,dt.t_i);
	consumer_watched_history = zeros(Bool,dt.t_i,dt.N,dt.C);

	temp_topics = zeros(Float64,dt.K,dt.C,dt.t_i);
	temp_shows = zeros(Int64,dt.C,dt.t_i);

	# preallocated arrays for inner loop
	base_util           = zeros(Float64,dt.N,dt.C);
	info_util           = zeros(Float64,dt.N,dt.C);
	leisure_util        = zeros(Float64,dt.N,dt.C);
	slant_util          = zeros(Float64,dt.N,dt.C);
	channel_util 		= zeros(Float64,dt.N,dt.C);
	exp_util 			= zeros(Float64,dt.N,dt.C);
	channel_ideology    = zeros(Float64,dt.N,dt.C);
	channel_reports     = zeros(Float64,dt.K,dt.C);
	choice_prob_per_channel = zeros(Float64,dt.N,dt.C+1);
	consumer_choice_threshold = zeros(Float64,dt.N,dt.C+1);
	consumer_watched_last = zeros(Bool,dt.N,dt.C);
	consumer_watched_chan = zeros(Int64, dt.N);
	consumer_free_signals = zeros(Float64, dt.N, dt.K);
	consumer_report_last=zeros(Float64,dt.K,dt.N);
	kalman_gain=zeros(Float64,dt.K,dt.N,dt.C+1);
	topic_signal_var = zeros(Float64,dt.K,dt.N);
	use_kalman = zeros(Float64,dt.K,dt.N);
	kalman_gain_watched = zeros(Float64,dt.K,dt.N);
	topic_updated_var = zeros(Float64,dt.K,dt.N);
	topic_updated_var_E = zeros(Float64,dt.K,dt.N);
	topic_prediction_var_E = zeros(Float64, dt.K,dt.N);
	topic_updated_est_guess = zeros(Float64,dt.K,dt.N);
	topic_surprise = zeros(Float64,dt.N,dt.K);
	agg_state_upd_guess=zeros(Float64,dt.N);
	agg_updated_var_E=zeros(Float64,dt.N);
	agg_prediction_var_E=zeros(Float64,dt.N);


	# matrices for indexing
	one_to_K_N = rep_col(collect(1:dt.K),dt.N);
	one_to_N_K = rep_row(collect(1:dt.N)',dt.K);
	one_to_C_N = rep_col(collect(1:dt.N),dt.C);

	for d = 1:dt.D
	  t = (d-1) * dt.t_i + 1;
	  last_t = t + dt.t_i - 1;
	  time_to_election=dt.time_inter*(dt.election_day-d);

	  # create array of topics (possibly) viewed by each consumer,
	  # respecting time zone
	  temp_topics .= dt.channel_topic_coverage_etz[:,:,t:last_t];
	  for i = dt.i_etz
		  consumer_viewed_topic[:,i,:,:] = temp_topics;
	  end
	  temp_topics .= dt.channel_topic_coverage_ctz[:,:,t:last_t];
	  for i = dt.i_ctz
		  consumer_viewed_topic[:,i,:,:] = temp_topics;
	  end
	  temp_topics .= dt.channel_topic_coverage_mtz[:,:,t:last_t];
	  for i = dt.i_mtz
		  consumer_viewed_topic[:,i,:,:] = temp_topics;
	  end
	  temp_topics .= dt.channel_topic_coverage_ptz[:,:,t:last_t];
	  for i = dt.i_ptz
		  consumer_viewed_topic[:,i,:,:] = temp_topics;
	  end
	  # consumer_viewed_topic[:,dt.consumer_tz .== 1,:,:] =permutedims(repeat(dt.channel_topic_coverage_etz[:,:,t:last_t], outer=[1,1,1,dt.n_etz]), [1,4,2,3]);
	  # consumer_viewed_topic[:,dt.consumer_tz .== 2,:,:] =permutedims(repeat(dt.channel_topic_coverage_ctz[:,:,t:last_t], outer=[1,1,1,dt.n_ctz]), [1,4,2,3]);
	  # consumer_viewed_topic[:,dt.consumer_tz .== 3,:,:] =permutedims(repeat(dt.channel_topic_coverage_mtz[:,:,t:last_t], outer=[1,1,1,dt.n_mtz]), [1,4,2,3]);
	  # consumer_viewed_topic[:,dt.consumer_tz .== 4,:,:] =permutedims(repeat(dt.channel_topic_coverage_ptz[:,:,t:last_t], outer=[1,1,1,dt.n_ptz]), [1,4,2,3]);

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

	  topic_path[:,d]=topic_path[:,max(d-1,1)] .+ innovations[:,d];

	  sim_viewership_polling!(d,
		  consumer_watched_history,
		  consumer_topic_est,consumer_topic_var,consumer_agg_est,consumer_agg_var,
		  predicted_viewership,
		  track_polling,
		  base_util,info_util,leisure_util,slant_util,channel_util,exp_util,
		  channel_reports,channel_ideology,
		  choice_prob_per_channel,consumer_choice_threshold,consumer_watched_last,consumer_watched_chan,
		  consumer_free_signals,consumer_report_last,
		  kalman_gain,topic_signal_var,use_kalman,kalman_gain_watched,
		  topic_updated_var,topic_updated_var_E,topic_prediction_var_E,
		  topic_updated_est_guess,topic_surprise,
		  agg_state_upd_guess,agg_updated_var_E,agg_prediction_var_E,
		  one_to_K_N,one_to_N_K,one_to_C_N,
		  consumer_viewed_topic,consumer_viewed_show,
		  topic_path[:,d],
		  channel_report_var,time_to_election,
		  topic_var,topic_var_expanded,
          consumer_ideology,channel_locations,
		  dt.consumer_tz,
		  dt.i_stb, dt.i_national,
		  consumer_topic_leisure,
          consumer_beta_0,consumer_beta_hour,consumer_beta_info_util,consumer_beta_slant,
		  dt.consumer_choice_draws[:,t:last_t],ch_free_errs[:,:,d],ch_rept_errs[:,:,t:last_t],
          topic_lambda,topic_lambda_squared,
		  dt.N,dt.K,dt.C,dt.t_i);

	  consumer_view_history_stb[t:last_t,:,:] = consumer_watched_history[:,dt.i_stb,:];
	  if(save_output)
	  	consumer_view_history_national[t:last_t,:,:] = consumer_watched_history[:,dt.i_national,:];
	  end

	  if d < dt.D
		  consumer_agg_est[:,d+1]=consumer_agg_est[:,d];
		  consumer_agg_var[:,d+1]=consumer_topic_var' * topic_lambda_squared;
		  # consumer_agg_var[:,d+1]=consumer_agg_var[:,d];
	  end

	  track_topic_est[:,:,d]=consumer_topic_est;
	  track_viewership[:,d]=
	      mapslices(Statistics.mean,predicted_viewership; dims=2);

	  predicted_channel_ratings[:,t:last_t] = predicted_viewership;

	  consumer_topic_var .+= topic_var_expanded;

	end


	## Individual viewership moments
	model_viewership_indiv_rawmoments = compute_indiv_moments(dt, consumer_view_history_stb, save_output=save_output);

	## write to disk if enabled
	save_output ? save("post_inside_obj_func.jld2", "consumer_view_history_stb", consumer_view_history_stb, "model_viewership_indiv_rawmoments", model_viewership_indiv_rawmoments, "track_viewership", track_viewership, "predicted_channel_ratings", predicted_channel_ratings, "track_polling", track_polling, "topic_path", topic_path, "innovations", innovations, "consumer_view_history_national", consumer_view_history_national) : nothing


	sim_moments = cat(model_viewership_indiv_rawmoments,
					  reshape(predicted_channel_ratings,dt.C*dt.T),
					  track_polling[1:dt.election_day],
					  sum(SMM.param(ev, dt.keys_innovations) .^ 2); dims=1);

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
	individual_avg_viewing_min= dropdims(sum(consumer_view_history_stb,dims=1),dims=1) .* 15 ./ dt.D;

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
