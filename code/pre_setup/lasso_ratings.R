library(data.table)
library(glmnet)
library(tidyverse)
library(hdm)
library(lubridate)
library(fixest)

local_dir <- "~/Dropbox/STBNews/"


setwd(sprintf("%s/data/FWM/block_view/", local_dir))
alldates <- ymd(sub("fwm_block_view_(\\d{4}-\\d{2}-\\d{2})\\.rds", "\\1", list.files(pattern="fwm_block_view_")))
alldates <- alldates[which(alldates==ymd("20120604")):length(alldates)]

tzs <- c("ETZ", "CTZ", "MTZ", "PTZ")
chans <- c("cnn", "fnc", "msnbc")

topic_weights <- map2(rep(chans, times=length(tzs)), rep(tzs, each = length(chans)), 
		~ fread(paste(local_dir, "data/model/topic_weights_", .x, "_", .y, ".csv", sep="")) %>% .[,`:=` (timezone = .y, channel = .x, date = ymd('1970-01-01') + days(date))]) %>%
	rbindlist %>%
	setnames(old="time_block_15", new="time_block")

load(sprintf("%s/Alex's work/ratings/output_new/nielsen_rating_df_LPM_SM.RData", local_dir))
dma_timezone <- fread(sprintf("%s/data/geo/dma_timezone.csv",local_dir))

nielsen_rating_df <- nielsen_rating_df %>%
	as.data.table %>%
	.[dma_timezone, on = "dma"] %>%
	.[,timezone := factor(timezone, levels=c("ETZ", "CTZ", "MTZ", "PTZ"))] %>%
	.[,date:=ymd(date)]

national <- nielsen_rating_df %>%
  .[,Intab := as.numeric(gsub(",","",as.character(Intab)))] %>%
  .[,.(nielsen_rating = weighted.mean(x=nielsen_rating, w=Intab, na.rm=T)),by=.(Viewing.Source, date, time_block, timezone)] %>%
  .[,channel:=factor(recode(Viewing.Source, CNN="cnn", `FOX NEWS CH`="fnc", MSNBC="msnbc"))] %>%
  .[date %in% alldates] 	


ratings <- national[topic_weights, on = .(channel, timezone, date, time_block)]


# construct x's
ratings <- ratings[order(date, channel, timezone, time_block)]
ratings_demean <- residuals(feols(nielsen_rating ~ 1 | channel^timezone^time_block, data = ratings))

topic_day <- do.call(bdiag, split(ratings, by = "date") %>% 
		map(~ as.matrix(.[,.(foreign_policy, economy, crime, horse_race)])))


ratings_lasso <- rlasso(topic_day,ratings_demean, post=F, penalty = list(c=1.1), intercept=F)

nonzero <- which(ratings_lasso$index)


topic_names <- c("foreign_policy", "economy", "crime", "horse_race")
nonzero_topics <- data.table(
	date = alldates[1 + (nonzero-1) %/% 4],
	topic = topic_names[1 + (nonzero-1) %% 4],
	par = paste(str_pad(1 + (nonzero-1) %/% 4,width=3, pad="0"), str_pad(1 + (nonzero-1) %% 4, width=2, pad="0"), sep="_t"),
	index = nonzero
	)

fwrite(nonzero_topics, file = "~/Dropbox/STBNews/data/model/topic_path_sparsity.csv")