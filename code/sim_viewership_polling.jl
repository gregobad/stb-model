# do one day of viewership / polling sim loop

function sim_viewership_polling!(d,
	# these arguments will be modified (preallocated in objective for speed)
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
	# inputs not modified
	consumer_viewed_topic,consumer_viewed_show,
	topic_path_day,
	channel_report_var,time_to_election,
	topic_var,topic_var_expanded,
	consumer_ideology,channel_locations,
	consumer_tz,
	i_stb, i_national,
	consumer_topic_leisure,consumer_beta_0,consumer_beta_hour,consumer_beta_vote,consumer_beta_slant,
	choice_draws,free_errors,report_errors,
	topic_lambda,topic_lambda_squared,
	N,K,C,t_i)

	consumer_free_signals .= reshape(topic_path_day,1,K) .+ free_errors;

	@views for t=1:t_i
	    channel_reports .= topic_path_day .+ report_errors[:,:,t];
		# new addition by ML - prensent in matlab but not here
    	topic_prediction_var_E .= consumer_topic_var .+ (time_to_election .* topic_var_expanded);
    	agg_prediction_var_E .= topic_prediction_var_E' * (topic_lambda_squared);
		# end additions
	    @views for c=1:C
		  hold_view_topic = consumer_viewed_topic[:,:,c,t];
	      channel_ideology[:,c] = sum(topic_lambda .* hold_view_topic .* channel_reports[:,c],dims=1)';

	      # first: expected voting utility gain from viewing
	      # only do this in the preelection period
	      topic_signal_var      .= (1 ./ hold_view_topic .^ 2) .* channel_report_var;  # variance of report today relative to state today

	      use_kalman .= consumer_topic_var ./ (consumer_topic_var .+ topic_signal_var);
	      kalman_gain[:,:,c+1] = use_kalman;
	      topic_updated_var   .= (1 .- use_kalman) .* consumer_topic_var;
	      topic_updated_est_guess .= consumer_topic_est .+ (consumer_free_signals' .- consumer_topic_est) .* use_kalman;
	      agg_state_upd_guess .= topic_updated_est_guess' * topic_lambda;

	      if (time_to_election >= 0)
	        topic_updated_var_E   .= topic_updated_var .+ (time_to_election .* topic_var_expanded);
	        agg_updated_var_E     .= topic_updated_var_E' * (topic_lambda_squared);
	        info_util[:,c]        = abs.(StatsFuns.normcdf.((consumer_ideology .+ agg_state_upd_guess) ./ agg_updated_var_E) .- StatsFuns.normcdf.((consumer_ideology.+consumer_agg_est[:,d]) ./ agg_prediction_var_E));
		  else
			info_util[:,c] .= 0;
		  end

	      # second: leisure value
	      topic_surprise .= abs.(consumer_free_signals .- consumer_topic_est');     # N x K
	      leisure_util[:,c] = sum(consumer_topic_leisure.*topic_surprise.*(hold_view_topic)',dims=2);

	      # third: slant
	      # slant_util[:,c]       = ((channel_ideology[:,c] .+ channel_locations[c]) .- consumer_ideology) .^ 2;
	      slant_util[:,c]       = abs.((channel_ideology[:,c] .+ channel_locations[c]) .- consumer_ideology);
	    end

	  # fourth: hour / show / channel effects
	  base_util .= consumer_beta_0[CartesianIndex.(one_to_C_N, consumer_viewed_show[:,:,t])] .+
	              consumer_beta_hour[CartesianIndex.(consumer_tz[:,ones(Int,C)], cld(t,4).*ones(Int,N,C))];

	  # add components
	  channel_util .= base_util .+ leisure_util .+ (consumer_beta_vote .* info_util) .+ (consumer_beta_slant .* slant_util);

	  ## compute choice probabilities, simulate viewing decision
	  exp_util .= exp.(min.(600,channel_util));

	  choice_prob_per_channel  .= norm_rows([ones(Float64,N,1) exp_util]);
	  consumer_choice_threshold .= cumsum(choice_prob_per_channel, dims=2);

	  consumer_watched_last .= (choice_draws[:,t] .>= consumer_choice_threshold[:,1:C]) .* (choice_draws[:,t] .< consumer_choice_threshold[:,2:(C+1)]);
	  consumer_watched_history[t,:,:] = consumer_watched_last;
	  consumer_report_last .= channel_reports * consumer_watched_last';

	  # combine report signals, if any, with prior beliefs
	  consumer_watched_chan .= consumer_watched_last * collect(1:C);
	  # kalman_gain_watched = kalman_gain[CartesianIndex.(one_to_K_N, one_to_N_K, rep_row(1 .+ reshape(consumer_watched_chan,1,N),K))];
	  index_into!(kalman_gain_watched, kalman_gain, one_to_K_N, one_to_N_K, rep_row(1 .+ reshape(consumer_watched_chan,1,N),K));

	  consumer_topic_est .= consumer_topic_est .+ (consumer_report_last .- consumer_topic_est) .* kalman_gain_watched;
	  consumer_topic_var .= (1 .- kalman_gain_watched) .* consumer_topic_var;   # no topic innovations between time blocks
	  consumer_agg_est[:,d] = consumer_topic_est' * topic_lambda;
	  consumer_agg_var[:,d] = consumer_topic_var' * topic_lambda_squared;

	  predicted_viewership[:,t]=mapslices(Statistics.mean, choice_prob_per_channel[i_national,2:end];dims=1);

end

if (time_to_election >= 0)
  vote_prob = (consumer_ideology .+ consumer_agg_est[:,d]) .> 0;
  track_polling[d] = Statistics.mean(vote_prob[i_national]);
else
  track_polling[d] = 0;
end

end

## do one day of polling (non-mutating version for inner maximization)
function sim_viewership_polling(d,
	# these arguments will be modified (preallocated in objective for speed)
	sim_r_today,
	predicted_viewership,
	base_util,info_util,leisure_util,slant_util,channel_util,exp_util,
	channel_reports,channel_ideology,
	choice_prob_per_channel,consumer_choice_threshold,consumer_watched_last,consumer_watched_chan,
	consumer_free_signals,consumer_report_last,
	kalman_gain,topic_signal_var,use_kalman,kalman_gain_watched,
	topic_updated_var,topic_updated_var_E,topic_prediction_var_E,
	topic_updated_est_guess,topic_surprise,
	agg_state_upd_guess,agg_updated_var_E,agg_prediction_var_E,
	one_to_K_N,one_to_N_K,one_to_C_N,
	# inputs not modified
	consumer_topic_est,consumer_topic_var,consumer_agg_est,consumer_agg_var,
	consumer_viewed_topic,consumer_viewed_show,
	topic_path_day,
	channel_report_var,time_to_election,
	topic_var,topic_var_expanded,
	consumer_ideology,channel_locations,
	consumer_tz,
	i_stb,i_national,
	consumer_topic_leisure,consumer_beta_0,consumer_beta_hour,consumer_beta_vote,consumer_beta_slant,
	choice_draws,free_errors,report_errors,
	topic_lambda,topic_lambda_squared,
	N,K,C,t_i)

	cte_copy = copy(consumer_topic_est);
	ctv_copy = copy(consumer_topic_var);
	cae_copy = copy(consumer_agg_est[:,d]);
	cav_copy = copy(consumer_agg_var[:,d]);

	consumer_free_signals .= reshape(topic_path_day,1,K) .+ free_errors;

	@views for t=1:t_i
	    channel_reports.=topic_path_day .+ report_errors[:,:,t];
		# new addition by ML - prensent in matlab but not here
    	topic_prediction_var_E .= ctv_copy .+ (time_to_election .* topic_var_expanded);
    	agg_prediction_var_E .= topic_prediction_var_E' * (topic_lambda_squared);
		# end additions
	    @views for c=1:C
		  hold_view_topic = consumer_viewed_topic[:,:,c,t];
	      channel_ideology[:,c] = sum(topic_lambda .* hold_view_topic .* channel_reports[:,c],dims=1)';

	      # first: expected voting utility gain from viewing
	      # only do this in the preelection period
	      topic_signal_var      .= (1 ./ hold_view_topic .^ 2) .* channel_report_var;  # variance of report today relative to state today

	      use_kalman .= ctv_copy ./ (ctv_copy .+ topic_signal_var);
	      kalman_gain[:,:,c+1] = use_kalman;
	      topic_updated_var   .= (1 .- use_kalman) .* ctv_copy;
	      topic_updated_est_guess .= cte_copy .+ (consumer_free_signals' .- cte_copy) .* use_kalman;
	      agg_state_upd_guess .= topic_updated_est_guess' * topic_lambda;

	      if (time_to_election >= 0)
	        topic_updated_var_E   .= topic_updated_var .+ (time_to_election .* topic_var_expanded);
	        agg_updated_var_E     .= topic_updated_var_E' * (topic_lambda_squared);
	        info_util[:,c]        = abs.(StatsFuns.normcdf.((consumer_ideology .+ agg_state_upd_guess) ./ agg_updated_var_E) .- StatsFuns.normcdf.((consumer_ideology.+cae_copy) ./ agg_prediction_var_E));
		  else
			info_util[:,c] .= 0;
		  end

	      # second: leisure value
	      topic_surprise .= abs.(consumer_free_signals .- cte_copy');     # N x K
	      leisure_util[:,c] = sum(consumer_topic_leisure.*topic_surprise.*(hold_view_topic)',dims=2);

	      # third: slant
	      slant_util[:,c]       = ((channel_ideology[:,c] .+ channel_locations[c]) .- consumer_ideology) .^ 2;
	    end

	  # fourth: hour / show / channel effects
	  base_util .= consumer_beta_0[CartesianIndex.(one_to_C_N, consumer_viewed_show[:,:,t])] .+
	              consumer_beta_hour[CartesianIndex.(consumer_tz[:,ones(Int,C)], cld(t,4).*ones(Int,N,C))];

	  # add components
	  channel_util .= base_util .+ leisure_util .+ (consumer_beta_vote .* info_util) .+ (consumer_beta_slant .* slant_util);

	  ## compute choice probabilities, simulate viewing decision
	  exp_util .= exp.(min.(600,channel_util));

	  choice_prob_per_channel  .= norm_rows([ones(Float64,N,1) exp_util]);
	  consumer_choice_threshold .= cumsum(choice_prob_per_channel, dims=2);

	  consumer_watched_last .= (choice_draws[:,t] .>= consumer_choice_threshold[:,1:C]) .* (choice_draws[:,t] .< consumer_choice_threshold[:,2:(C+1)]);
	  consumer_report_last .= channel_reports * consumer_watched_last';

	  # combine report signals, if any, with prior beliefs
	  consumer_watched_chan .= consumer_watched_last * collect(1:C);
	  # kalman_gain_watched = kalman_gain[CartesianIndex.(one_to_K_N, one_to_N_K, rep_row(1 .+ reshape(consumer_watched_chan,1,N),K))];
	  index_into!(kalman_gain_watched, kalman_gain, one_to_K_N, one_to_N_K, rep_row(1 .+ reshape(consumer_watched_chan,1,N),K));

	  cte_copy .= cte_copy .+ (consumer_report_last .- cte_copy) .* kalman_gain_watched;
	  ctv_copy .= (1 .- kalman_gain_watched) .* ctv_copy;   # no topic innovations between time blocks
	  cae_copy .= cte_copy' * topic_lambda;
	  cav_copy .= ctv_copy' * topic_lambda_squared;

	  predicted_viewership[:,t]=mapslices(Statistics.mean, choice_prob_per_channel[i_national,2:end];dims=1);

	end

	if (time_to_election >= 0)
	  vote_prob = (consumer_ideology .+ cae_copy) .> 0;
	  vote = Statistics.mean(vote_prob[i_national]);
	else
	  vote = 0;
	end

	sim_r_today[1:(C*t_i)] = reshape(predicted_viewership, C * t_i);
	sim_r_today[(C*t_i)+1] = vote;
end
