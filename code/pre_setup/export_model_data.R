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
	fwrite(file="~/Dropbox/STBNews/data/model/linear_trend.csv")


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

show2chan %>% write_csv("~/Dropbox/STBNews/data/model/show_to_channel.csv")


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
		write_csv(path = paste("~/Dropbox/STBNews/data/model/topic_weights_", chan, "_", tz, ".csv", sep=""))

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

stb_hh %>% write_csv(path="~/Dropbox/STBNews/data/model/stb_hh_sample.csv")

### path initialization ###
extract_weights <- function(dt, df) {
	df %>% filter(date==dt) %>% ungroup() %>% select(one_of(non_filler_topics)) %>% as.matrix()
}


topics <- national %>%
	gather(key=ch_tz, value =nielsen_rating, starts_with("cnn"), starts_with("fnc"), starts_with("msnbc")) %>%
	separate(ch_tz, c("channel", "timezone")) %>%
	arrange(date, timezone, channel, time_block_15) %>%
	mutate(date = ymd(date)) %>%
	inner_join(tr_seg_15_fill) %>%
	mutate(hour = as.factor(time_block_15 %/% 4),
		   show = replace(show, is.na(show) & channel == "cnn", "cnn-specialcoverage"),
		   show = replace(show, is.na(show) & channel == "fnc", "fnc-specialcoverage"),
		   show = replace(show, is.na(show) & channel == "msnbc", "msnbc-specialcoverage"))


# first compute deviation from mean coverage by topic, by day
topic_anomaly <- topics %>%
	mutate_at(vars(one_of(non_filler_topics)), ~ . - mean(.)) %>%
	group_by(date) %>%
	summarize_at(vars(one_of(non_filler_topics)), mean) %>%
	gather(key = "topic", value = "covg_dev", one_of(non_filler_topics))


# use relative msn / fnc ratings to guess sign of shocks
topic_ratio <- topics %>%
	group_by(date) %>%
	summarize_at(vars(one_of(non_filler_topics)), ~ 2 * (sum(.[channel=="fnc"]) > sum(.[channel=="msnbc"])) - 1) %>%
	gather(key = "topic", value = "direction", one_of(non_filler_topics))


path_guess <- topic_anomaly %>%
	inner_join(topic_ratio) %>%
	mutate(innovation = pmax(0, covg_dev) * direction) %>%
	select(date, topic, innovation) %>%
	spread(key=topic, value=innovation) %>%
	select(date, one_of(non_filler_topics))

write_csv(path_guess, path="~/Dropbox/STBNews/data/model/path_guess.csv")

### sensible defaults for parameters ###

## simple regression of average viewing on topic

tilde <- function (t) paste("`", t, "`",sep="")
f_main <- paste("nielsen_rating",  paste(c("show", "timezone*as.factor(hour)", tilde(non_filler_topics)), collapse=" + "), sep= " ~ ") %>% as.formula
m_topic_main <- lm(f_main, data=topics %>% mutate(hour = time_block_15 %/% 4))
summary(m_topic_main)

leisure_relative_scale <- tidy(m_topic_main) %>%
	filter(term %in% topic_names | term %in% tilde(topic_names)) %>%
	pull(estimate)

leisure_relative_scale <- leisure_relative_scale / max(leisure_relative_scale)


## regression of poll changes on path estimate
polling %<>%
	mutate(date = ymd(date)) %>%
	inner_join(path_guess)

polling %<>% mutate(poll_chg = lead(obama_2p) - obama_2p)

f_poll <- paste("poll_chg",  paste(tilde(non_filler_topics), collapse=" + "), sep= " ~ ") %>% as.formula
m_poll <- lm(f_poll, data=polling)
summary(m_poll)

lambda_guess <- tidy(m_poll) %>%
	filter(term %in% topic_names | term %in% tilde(topic_names)) %>%
	pull(estimate) %>%
	{abs(.) / sum(abs(.))}


# [1] 0.23012984 0.06353405 0.37477442 0.33156170

view_coef <- read_csv("~/Dropbox/STBNews/data/model/viewership_nielsen_coef.csv")

topic_leis <- exp(view_coef %>% filter(!grepl("^timezone|^hour|^show|linear\\.trend$", term)) %>% pull(estimate)) / 10
topic_lambda <- view_coef %>% filter(grepl(":linear\\.trend", term)) %>% pull(estimate)
topic_lambda <- topic_lambda - min(topic_lambda)
topic_lambda <- topic_lambda / sum(topic_lambda)

load("~/Dropbox/STBNews/data/topicmodel/topic_slant.RData")
topic_mu <- slants %>%
	filter(!grepl("filler", topic)) %>% pull(slant)

avg_ratings <- national %>%
	mutate(hour = (time_block_15 - 4) %/% 4) %>%
	group_by(hour) %>%
	summarize_if(is.double, mean)

consumer_beta <- c(25, -0.25)

consumer_beta_0 <- c(rep(0,length(allshows)),rep(1,length(allchans)))

consumer_beta_hour <- avg_ratings %>%
	mutate(ETZ_avg = (cnn_ETZ + fnc_ETZ + msnbc_ETZ)/100,
		   CTZ_avg = (cnn_CTZ + fnc_CTZ + msnbc_CTZ)/100,
		   PTZ_avg = (cnn_PTZ + fnc_PTZ + msnbc_PTZ)/100,
		   ETZ_avg = log(ETZ_avg/(1-ETZ_avg)),
		   CTZ_avg = log(CTZ_avg/(1-CTZ_avg)),
		   PTZ_avg = log(PTZ_avg/(1-PTZ_avg))) %>%
	select(ends_with("_avg")) %>%
	as.matrix %>%
	c



consumer_state_var_0 <- rep(0.05, length(non_filler_topics))
channel_report_var <- 0.05
consumer_free_var <- 1



initial_parameter <- tibble(
	par = c(paste("topic_lambda", non_filler_topics, sep=":"),
	 paste("topic_leisure", non_filler_topics, sep=":"),
	 paste("topic_mu", non_filler_topics, sep=":"),
	 paste("topic_var", non_filler_topics, sep=":"),
	 paste("consumer_state_var_0", non_filler_topics, sep=":"),
	 paste("beta", c("info", "slant"), sep=":"),
	 paste("beta:show", allshows, sep=":"),
	 paste("beta:channel", allchans, sep=":"),
	 paste("beta", paste("h", 5:10, sep=""), "etz", sep=":"),
	 paste("beta", paste("h", 5:10, sep=""), "ctz", sep=":"),
	 paste("beta", paste("h", 5:10, sep=""), "ptz", sep=":"),
	 "channel_report_var", "consumer_free_var"),
	value = c(topic_lambda, topic_leis, topic_mu, topic_var, consumer_state_var_0, consumer_beta, consumer_beta_0, consumer_beta_hour, channel_report_var, consumer_free_var)
)
write_csv( initial_parameter, "~/Dropbox/STBNews/data/model/initial_parameters_v6.csv")




