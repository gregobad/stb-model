### Export data and regression coefs for Julia use ###
rm(list=ls())
library(tidyverse)
library(data.table)
library(broom)
library(glmnet)
library(Matrix)
library(lubridate)
library(magrittr)

### set date, timeblock, topic ranges to include ###

# dates: weekdays 6/4/2012 - 1/31/2013
setwd("~/Dropbox/STBNews/data/FWM/block_view/")
date_range <- ymd(sub("fwm_block_view_(\\d{4}-\\d{2}-\\d{2})\\.rds", "\\1", list.files(pattern="fwm_block_view_")))
date_range <- date_range[which(date_range==ymd("20120604")):length(date_range)]
tminus <- time_length(ymd("20121106") - date_range, unit="day")
tminus_max <- max(tminus)

pre_elec <- tminus > 0
linear_trend <- ifelse(pre_elec, tminus_max - tminus,0)
data.table(date=as.character(date_range), linear_trend=linear_trend) %>% 
	fwrite(file="~/Dropbox/STBNews/stb-model-discrete/data/linear_trend.csv")


# time blocks: 5PM - 11PM
block_range <- 4:27

# households to sample
hh_samp_size <- 6000

# shows to include
allshows <- c("cnn-specialcoverage",
			"fnc-specialcoverage",
			"msnbc-specialcoverage",
			"andersoncooper",
			"erinburnettoutfront",
			"foxreportwithshepardsmith",
			"hannity",
			"hardballwithchrismatthews",
			"johnkingusa",
			"ontherecordwithgretavansusteren",
			"oreillyfactor",
			"piersmorgan",
			"politicsnation",
			"specialreportwithbretbaier",
			"theedshow",
			"thefive",
			"thelastwordwithlawrenceodonnell",
			"therachelmaddowshow",
			"thesituationroomwithwolfblitzer")

# channels to include
allchans <- c("cnn","fnc","msnbc")

### Channel topic weights ###
setwd("~/Dropbox/STBNews/data/topicmodel")
load("tr_segments_tzadjusted.RData")
topic_names <- colnames(tr_segments %>% ungroup %>% select(-(1:4), -starts_with("time_block")))
non_filler_topics <- grep("filler", topic_names, invert=T, value=T)

tr_seg_15 <- tr_segments %>%
	group_by(channel, show, date, time_block_est_15, time_block_cst_15, time_block_mst_15, time_block_pst_15) %>%
	summarize_at(vars(one_of(topic_names)), mean) %>%
	group_by(channel, date, time_block_est_15, time_block_cst_15, time_block_mst_15, time_block_pst_15) %>%
	slice(1) %>%  # occasionally due to schedule slipping, get multiple shows in one channel x date x time block.
	ungroup %>%
	arrange(channel, date, time_block_est_15)

east <- tr_seg_15 %>%
	ungroup %>%
	filter(date %in% ymd(date_range), time_block_est_15 %in% block_range) %>%
	mutate(timezone = "ETZ") %>%
	rename(time_block_15 = time_block_est_15) %>%
	select(-time_block_cst_15,-time_block_mst_15,-time_block_pst_15)

central <- tr_seg_15 %>%
	ungroup %>%
	filter(date %in% ymd(date_range), time_block_cst_15 %in% block_range) %>%
	mutate(timezone = "CTZ") %>%
	rename(time_block_15 = time_block_cst_15) %>%
	select(-time_block_est_15,-time_block_mst_15,-time_block_pst_15)

mountain <- tr_seg_15 %>%
	ungroup %>%
	filter(date %in% ymd(date_range), time_block_mst_15 %in% block_range) %>%
	mutate(timezone = "MTZ") %>%
	rename(time_block_15 = time_block_mst_15) %>%
	select(-time_block_est_15,-time_block_cst_15,-time_block_pst_15)

west <- tr_seg_15 %>%
	ungroup %>%
	filter(date %in% ymd(date_range), time_block_pst_15 %in% block_range) %>%
	mutate(timezone = "PTZ") %>%
	rename(time_block_15 = time_block_pst_15) %>%
	select(-time_block_est_15,-time_block_cst_15,-time_block_mst_15)

tr_seg_15 <- bind_rows(east, central, mountain, west) %>%
	mutate(timezone = factor(timezone, levels=c("ETZ", "CTZ", "MTZ", "PTZ")),
		   show_index = as.integer(factor(show, levels=allshows)),
		   channel_index = as.integer(factor(channel, levels = allchans)))

# output show to channel index
show2chan <- tr_seg_15 %>%
	distinct(show, show_index, channel, channel_index) %>%
	filter(!is.na(show_index), !is.na(channel_index)) %>%
	arrange(show_index)

show2chan %>% write_csv("~/Dropbox/STBNews/stb-model-discrete/data/show_to_channel.csv")


output_weights <- function(df) {
	chan <- df$channel[1]
	tz <- df$timezone[1]

	fills <- list(
		channel_index = which(allchans==chan),
		show_index = which(allchans==chan)
		)

	# expand to include all segments (including those with missing transcripts)
	df <- df %>%
		ungroup %>%
		complete(date = ymd(date_range), time_block_15 = block_range, fill=fills)

	# fills for missing data
	# 1) debate days 10/3 10/16 10/22: missing live coverage (9-10:30pm) for all three channels.
	#    set to weight 1 on debate topic.
	debate_days <- mdy("10/3/2012", "10/16/2012", "10/22/2012")
	df <- df %>%
		mutate(horse_race = replace(horse_race, (date %in% debate_days) & is.na(horse_race), 1)) %>%
		mutate_at(.vars=vars(one_of(setdiff(topic_names,"horse_race"))),
				  .funs= ~ replace(., (date %in% debate_days) & is.na(.), 0))

	# 2) election day 11/6: missing some coverage, esp. on CNN.
	#    set to weight 1 on horse race topic (by far largest topic in nonmissing segments)
	election_day <- mdy("11/6/2012")
	df <- df %>%
		mutate(horse_race = replace(horse_race, (date == election_day) & is.na(horse_race), 1)) %>%
		mutate_at(.vars=vars(one_of(setdiff(topic_names,"horse_race"))),
				  .funs= ~ replace(., (date == election_day) & is.na(.), 0))

	# 3) remaining missings: replace with average topic weight
	#    for other segments on the same channel, same day
	df <- df %>%
		group_by(date) %>%
		mutate_at(vars(one_of(topic_names)),
				  ~ replace(., is.na(.), mean(.[!is.na(.)])))

	# 4) finally, a few day/channel combos have no transcripts at all
	#    (almost all MSNBC, and almost all holidays)
	#    use channel average for the week
	df <- df %>%
		mutate(wk = isoweek(date)) %>%
		group_by(wk) %>%
		mutate_at(vars(one_of(topic_names)),
				  ~ replace(., is.na(.), mean(.[!is.na(.)])))

 	df %>%
 		ungroup %>%
		mutate(date = as.numeric(date)) %>%
		select(date, time_block_15, show_index, one_of(non_filler_topics)) %>%
		write_csv(path = paste("~/Dropbox/STBNews/stb-model-discrete/data/topic_weights_", chan, "_", tz, ".csv", sep=""))

	df %>% ungroup %>% select(date, time_block_15, show, one_of(non_filler_topics))
}

tr_seg_15_fill <- tr_seg_15 %>%
	filter(channel %in% allchans) %>%
	group_by(channel, timezone) %>%
	do(output_weights(.))


### sample households from STB data ###
stb_hh <- readRDS("~/Dropbox/STBNews/data/FWM/ref_data/master_dev_list.RData")


hhs <- stb_hh %>% filter(!is.na(dma_code)) %>% pull(household_id) %>% unique()
set.seed(2385020)
hh_sample <- sample(hhs, size = hh_samp_size)

stb_hh <- stb_hh %>%
	left_join(dma_timezone) %>%
	mutate(timezone = as.integer(factor(timezone, levels = c("ETZ", "CTZ", "MTZ", "PTZ")))) %>%
	select(household_id, zip, dma_code, timezone, r_prop) %>%
	filter(household_id %in% hh_sample) %>%
	group_by(household_id) %>%
	slice(1) %>%
	ungroup()

stb_hh %>% write_csv(path="~/Dropbox/STBNews/stb-model-discrete/data/stb_hh_sample.csv")


### initial parameter vector, and bounds file setting lower / upper bounds for SMM
## set pr(news) = topic weight each day x topic
names(allchans) <- allchans
topics_model <- imap(allchans, ~ fread(file = paste("~/Dropbox/STBNews/stb-model-discrete/data/topic_weights_", .x, "_ETZ.csv", sep=""))[,channel:=.y]) %>% 
	rbindlist %>%
	setnames(old="time_block_15", new="time_block") %>%
	.[,date := as.Date(date, origin = '1970-01-01')]
ratings <- fread("~/Dropbox/STBNews/stb-model-discrete/data/nielsen_ratings.csv") %>%
	melt(id.vars=c("date", "time_block"), variable.name = "channel", value.name="rating") %>%
	.[,channel:=tolower(channel)] %>%
	.[,date:=ymd(date)]

w_top <- topics_model[ratings,on=.(date, time_block,channel)]

daily_topic_weights <- w_top[,map(.SD, weighted.mean, w = rating), by= .(date), .SDcols=non_filler_topics]

daily_news_pr <- melt(daily_topic_weights, id.vars="date", variable.name="topic", value.name="value")
daily_news_pr[,day_index := match(date, date_range)]
daily_news_pr[,par := paste("pr_news_", str_pad(day_index, width=3, pad="0"), "_t", str_pad(match(topic, non_filler_topics), width=2, pad="0"), sep="")]
daily_news_pr <- daily_news_pr[order(par), .(par, value)]


## set pr(news favorable to Rs) = proportional to poll changes pre-election, =0.5 post-election
polls <- fread("~/Dropbox/STBNews/stb-model-discrete/data/polling.csv") %>%
	.[,date:=ymd(date)]
polls[1:110,poll_chg:=lead(obama_2p) - obama_2p]
polls[, r_favor := (1 - poll_chg / max(abs(poll_chg), na.rm=T)) / 2]
polls[is.na(r_favor), r_favor:= 0.5]
polls[,day_index := match(date, date_range)]

r_favor_pr = polls[,.(value = r_favor, par = paste("pr_rep_", str_pad(day_index, width=3, pad="0"), "_t", c("01", "02", "03", "04"), sep = "")),by = .(date)]
r_favor_pr <- r_favor_pr[order(par), .(par, value)]

## main parameters
topic_lambda <- rep(1 / length(non_filler_topics), length(non_filler_topics))
topic_rho <- rep(0.95, length(non_filler_topics))
topic_mu <- rep(0.57, length(non_filler_topics))
topic_leisure <- rep(10, length(non_filler_topics))
channel_q_D <- c(0.9, 0.85, 0.95)
channel_q_R <- c(0.9, 0.95, 0.85)
betas <- c(10,-10)
beta_show <- c(-5.867, -4.048, -6.682)
channel_mu <- c(-10.416, -4.712, -5.163)
channel_sigma <- c(5.084, 3.551, 3.158)
zero_news <- 0.5



main_pars <- data.table(
	par = c(paste("topic_lambda", non_filler_topics, sep=":"),
	 paste("topic_rho", non_filler_topics, sep=":"),
	 paste("topic_leisure", non_filler_topics, sep=":"),
	 paste("topic_mu", non_filler_topics, sep=":"),
	 paste("channel_q_D", allchans, sep=":"),
	 paste("channel_q_R", allchans, sep=":"),
	 paste("beta", c("vote", "slant"), sep=":"),
	 paste("beta:show", allchans, sep=":"),
	 paste("beta:channel_mu", allchans, sep=":"),
	 paste("beta:channel_sigma", allchans, sep=":"),
	 "zero:news"),
	value = c(topic_lambda, topic_rho, topic_leisure, topic_mu, channel_q_D, channel_q_R, betas, beta_show, channel_mu, channel_sigma, zero_news)
)

initial_parameter <- rbind(main_pars, daily_news_pr, r_favor_pr)
fwrite(initial_parameter, "~/Dropbox/STBNews/stb-model-discrete/data/par_init.csv")

bounds <- initial_parameter[,.(par)] %>%
	.[,lb := c(rep(0, length(topic_lambda)),
			 rep(0.501, length(topic_rho)),
			 rep(0, length(topic_leisure)),
			 rep(0, length(topic_mu)),
			 rep(0, length(channel_q_D)),
			 rep(0, length(channel_q_R)),
			 c(0,-100),
			 rep(-20,length(beta_show)),
			 rep(-2,length(channel_mu)),
			 rep(0, length(channel_sigma)), 
			 0,
			 rep(0, 2 * nrow(daily_news_pr)))] %>%
	.[,ub:= c(rep(1, length(topic_lambda)),
			 rep(1, length(topic_rho)),
			 rep(500, length(topic_leisure)),
			 rep(1, length(topic_mu)),
			 rep(1, length(channel_q_D)),
			 rep(1, length(channel_q_R)),
			 c(100,0),
			 rep(-4.5,length(beta_show)),
			 rep(0,length(channel_mu)),
			 rep(2, length(channel_sigma)), 
			 1,
			 rep(1, 2 * nrow(daily_news_pr)))]

fwrite(bounds, "~/Dropbox/STBNews/stb-model-discrete/data/parameter_bounds.csv")



## setup sampling probabilities for groups of pars
setwd("~/Dropbox/STBNews/stb-model-discrete/sampling")

# use ratings / topic weights to adjust sampling probs
overall_topic_weights <- w_top[,map(.SD, weighted.mean, w=rating), .SDcols = non_filler_topics] %>% melt(measure.vars=non_filler_topics)
daily_topic_weights <- w_top[,map(.SD, weighted.mean, w=rating), .SDcols = non_filler_topics, by = date] %>%
	melt(id.vars="date", variable.name="topic", value.name="weight")
overall_ratings <- ratings[,.(rating = mean(rating)), by =.(channel)]
daily_ratings <- ratings[,.(rating=mean(rating)), by =.(date)]

# tier 0: split 50/50 between main and path pars
sample_tier_0 <- data.table(group = c("main", "path"), prob = c(0.5, 0.5))
fwrite(sample_tier_0, "sample_tree.csv")

# tier 1 (main): split evenly between lambdas, topic pars, consumer pars, channel pars
sample_tier_main <- data.table(group = c("main_lambdas", "main_topics", "main_consumer", "main_channel"),
							   prob = c(0.25, 0.25, 0.25, 0.25))
fwrite(sample_tier_main, "sample_tree_main.csv")

# tier 2 (main/lambda): sample all lambdas at once
sample_tier_main_lambdas <- data.table(keys = paste("topic_lambda:", non_filler_topics, sep=""))
fwrite(sample_tier_main_lambdas, "sample_tree_main_lambdas.csv")

# tier 2 (main/topics): sample groups of mu/rho/leisure for each topic in proportion to topic weight
sample_tier_main_topics <- data.table(group = paste("main_topics_", gsub("_","",non_filler_topics), sep=""), prob = overall_topic_weights[,value] / sum(overall_topic_weights[,value]))
fwrite(sample_tier_main_topics, "sample_tree_main_topics.csv")

# tier 3 (main/topics/topic)
names(non_filler_topics) <- paste("main_topics_", gsub("_","",non_filler_topics), sep="")
sample_tier_main_topics_sub <- map(non_filler_topics, 
	~ data.table(keys = paste(c("topic_rho", "topic_leisure", "topic_mu"), .x, sep=":")))
iwalk(sample_tier_main_topics_sub, ~ fwrite(.x, file=paste("sample_tree_", .y, ".csv", sep="")))


# tier 2 (main/consumer): sample each one at a time, equal probability
sample_tier_main_consumer <- data.table(group = c("main_consumer_vote", "main_consumer_slant", "main_consumer_zero"), prob = c(1/3, 1/3, 1/3))
fwrite(sample_tier_main_consumer, "sample_tree_main_consumer.csv")

# tier 3 (main/consumer/parameter)
sample_tier_main_consumer_vote <- data.table(keys = c("beta:vote"))
fwrite(sample_tier_main_consumer_vote, "sample_tree_main_consumer_vote.csv")
sample_tier_main_consumer_slant <- data.table(keys = c("beta:slant"))
fwrite(sample_tier_main_consumer_slant, "sample_tree_main_consumer_slant.csv")
sample_tier_main_consumer_zero <- data.table(keys = c("zero:news"))
fwrite(sample_tier_main_consumer_zero, "sample_tree_main_consumer_zero.csv")

# tier 2 (main/channels): sample reporting probs and constant for one channel together, weighting more heavily rated chans more
sample_tier_main_channel <- data.table(group = paste("main_channel_", allchans, sep=""), prob=overall_ratings[,rating] / sum(overall_ratings[,rating]))
fwrite(sample_tier_main_channel, "sample_tree_main_channel.csv")
names(allchans) <- paste("main_channel_", allchans, sep="")

# tier 3 (main/channels/channel)
sample_tier_main_channel_sub <- map(allchans, 
	~ data.table(keys=paste(c("channel_q_R", "channel_q_D","channel_q_0", "beta:show", "beta:channel_mu", "beta:channel_sigma"), ., sep=":")))
iwalk(sample_tier_main_channel_sub, ~ fwrite(.x, paste("sample_tree_",.y, ".csv", sep="")))

# tier 1 (path): split across days according to
# cumulative ratings from day d to end (weight earlier days more)
# and same-day ratings (rate higher-rated days more)
# give each component equal weight

sample_tier_path <- copy(daily_ratings)[,group:=paste("path_day_",1:.N,sep="")]
sample_tier_path[,prob:=(rev(cumsum(rating)) + length(date_range)*rating) / (sum(cumsum(rating)) + length(date_range)*sum(rating))]
fwrite(sample_tier_path, file = "sample_tree_path.csv")


# tier 2 (path/day): sample topic pars this day according to topic weights that day
names(date_range) <- paste("path_day_", 1:length(date_range), sep="")
sample_tier_path_sub <- imap(date_range, 
	~ daily_topic_weights[date==.x, ][,`:=` (group=paste(.y, gsub("_","",topic), sep="_"), prob=weight / sum(weight))][,.(group,prob)])
iwalk(sample_tier_path_sub, ~ fwrite(.x, paste("sample_tree_", .y, ".csv", sep="")))

# tier 3 (path/day/topic): sample both together
make_day_topic <- function(df, basename) {
	day_ind <- str_pad(sub(".*_(\\d+)$", "\\1", basename), pad="0", width=3)
	topic_inds <- str_pad(1:length(non_filler_topics), pad="0", width=2)
	topic_days <- paste(day_ind, "_t", topic_inds, sep="")
	sub_list <- map(topic_days, ~ data.table(keys = paste(c("pr_news", "pr_rep"), ., sep="_")))
	names(sub_list) <- sub(".*_(.w+)$","\\1",df$group)
	sub_list
}
sample_tier_path_sub_sub <- imap(sample_tier_path_sub, make_day_topic) %>% flatten
iwalk(sample_tier_path_sub_sub, ~ fwrite(.x, paste("sample_tree_", .y, ".csv", sep="")))
