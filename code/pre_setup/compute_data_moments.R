#### compute unconditional moments in stb data ####


local_dir <- "~/Dropbox/STBNews"
# local_dir <- "/usr/local/ifs/gsb/gjmartin/STBnews/STBNews"
plot_dir = "data/model/standard_output"


library(tidyverse)
library(lubridate)
library(data.table)
library(ggthemes)
library(grid)
library(gridExtra)
library(gglorenz)
theme_set(theme_few())

# set up for plots
nstb = 6000
ntopics = 4
max_rating = 20
max_mins = 30
min_lim=180
max_day_rating = 5
min_hh_hrs_viewed = 36.1/60
alpha=.005

weighted.quantile <- function(x, w, tau=0.5) {
	i <- order(x)
	x <- x[i]
	w <- w[i]
	x[min(which(cumsum(w) >= tau * sum(w)))]
}


select <- dplyr::select
rename <- dplyr::rename

## setup
setwd(sprintf("%s/data/FWM/block_view/", local_dir))
alldates <- ymd(sub("fwm_block_view_(\\d{4}-\\d{2}-\\d{2})\\.rds", "\\1", list.files(pattern="fwm_block_view_")))
alldates <- alldates[which(alldates==ymd("20120604")):length(alldates)]
tminus_max <- as.numeric(max(ymd("20121106") - alldates))

# chans <- c("abc", "cbs", "cnn", "fnc", "msnbc", "nbc", "pbs")
chans <- c("cnn", "fnc", "msnbc")
chans_up = c("CNN", "FNC", "MSNBC")

allshows <- fread(sprintf("%s/data/model/show_to_channel.csv", local_dir)) %>% 
	.[,show]


### Nielsen viewership by date / time-block ###
block_range <- 4:27
load(sprintf("%s/Alex's work/ratings/output_new/nielsen_rating_df_LPM_SM.RData", local_dir))
dma_timezone <- fread(sprintf("%s/data/geo/dma_timezone.csv",local_dir))

nielsen_rating_df <- nielsen_rating_df %>%
	as.data.table %>%
	.[dma_timezone, on = "dma"] %>%
	.[,timezone := factor(timezone, levels=c("ETZ", "CTZ", "MTZ", "PTZ"))] %>%
	.[,date:=ymd(date)]

national <- nielsen_rating_df %>%
  .[,Intab := as.numeric(gsub(",","",as.character(Intab)))] %>%
  .[,.(nielsen_rating = weighted.mean(x=nielsen_rating, w=Intab, na.rm=T)),by=.(Viewing.Source, date, time_block)] %>%
  .[,channel:=factor(recode(Viewing.Source, CNN="CNN", `FOX NEWS CH`="FNC", MSNBC="MSNBC"))] %>%
  .[date %in% alldates & time_block %in% block_range] 

### AVERAGE MINUTES BY DAY GRAPH ###
avg_min_day <- national[,.(avg_min = sum(nielsen_rating / 100 * 15)),by=.(date, channel)]
avg_mins_plot <- ggplot(aes(x=date, y=avg_min, group=channel), data = avg_min_day) +
  geom_line(aes(colour=channel)) +
  theme_bw() +
  ylab("Daily Average Minutes (Nielsen)") +
  xlab("Date") +
  scale_colour_manual(values=c(CNN="mediumpurple2", FNC="darkred", MSNBC="cornflowerblue"))+
  scale_x_date(breaks=ymd(c("20120604", "20120801", "20121001", "20121106", "20130101")),
               labels=c("Jun 4", "Aug 1", "Oct 1", "Nov 6", "Jan 1")) +
  geom_vline(xintercept = ymd("20121106"), linetype="dashed") +
  ylim(0,max_mins) +
  theme(legend.position = "none")

ggsave(plot=avg_mins_plot, filename=sprintf("%s/%s/nielsen_daily_avg_mins.png",local_dir, plot_dir), height=4, width=10)

### MAX RATINGS BY DAY GRAPH ###
max_ratings_day <- national %>% 
	.[,.(max_rat = max(nielsen_rating)), by = .(channel, date)]

max_ratings_plot <- ggplot(aes(x=date, y=max_rat, group=channel), data = max_ratings_day) +
  geom_line(aes(colour=channel)) +
  theme_bw() +
  ylab("Daily Max Block Rating") +
  xlab("Date") +
  scale_colour_manual(values=c(CNN="mediumpurple2", FNC="darkred", MSNBC="cornflowerblue"))+
  scale_x_date(breaks=ymd(c("20120604", "20120801", "20121001", "20121106", "20130101")),
               labels=c("Jun 4", "Aug 1", "Oct 1", "Nov 6", "Jan 1")) +
  geom_vline(xintercept = ymd("20121106"), linetype="dashed") +
  ylim(0,max_day_rating) +
  theme(legend.position = "none")

ggsave(plot=max_ratings_plot, filename=sprintf("%s/%s/nielsen_daily_rating_max.png",local_dir, plot_dir), height=4, width=10)


# output block x channel ratings for model
national_wide <- copy(national) %>%
  .[,nielsen_rating := nielsen_rating / 100] %>%
  dcast(date + time_block ~ channel, value.var="nielsen_rating")
national_wide %>% fwrite(file = sprintf("%s/data/model/nielsen_ratings.csv", local_dir)


### polling ###
polling <- fread(sprintf("%s/Alex's work/polling/polling_df.csv", local_dir)) %>%
	.[,`:=` (date = ymd(date), obama_2p = percent_Obama / (percent_Obama + percent_Romney))] %>%
	.[state == "US" & date %in% alldates] %>%
	.[,.(date, obama_2p)] %>%
	rbind(data.table(date = ymd("20121106"), obama_2p = 51.06 / (51.06 + 47.20))) %>%  # append actual election result

polling <- polling[data.table(date=alldates), on = .(date)] %>%
	.[,obama_2p:=replace_na(obama_2p, 0)]  # pad with zeros for post-election days

fwrite(polling, file = sprintf("%s/data/model/polling.csv", local_dir))



## load STB individual viewing data
load(sprintf("%s/data/geo/dma_timezone.RData", local_dir)) # for time zone adjustments
# load("/ifs/gsb/gjmartin/STBnews/STBNews/data/geo/dma_timezone.RData") # for time zone adjustments

block_files <- paste(local_dir, "/data/FWM/block_view/fwm_block_view_", as.character(alldates), ".rds", sep="")
active_files <- paste(local_dir, "/data/FWM/ref_data/fwm_active_devs_", as.character(alldates), ".rds", sep="")

get_indiv_daily_viewing <- function(dayfile, activefile) {
	cat(dayfile, "\n")
	block_day <- readRDS(dayfile)

	blocks <- grep("bl_", colnames(block_day), value=T)
	block_day[,total:=blocks %>% map(get, block_day) %>% reduce(`+`)]

	block_day <- block_day[,.(date, household_id, dma_code, zipcode, channel, total)] %>%
		dcast(date + household_id + dma_code + zipcode ~ channel, value.var="total", fill=0)

	all_active <- readRDS(activefile) %>% 
		unique(by="household_id") %>%
		.[,.(household_id, r_prop, tercile, date=ymd(str_extract(activefile, "\\d{4}-\\d{2}-\\d{2}")))]

	block_day[all_active, on=.(household_id, date)] %>%
		.[,(chans_up) := map(chans_up, ~replace_na(get(.), 0))] %>%
		.[,c("date", "household_id","r_prop","tercile", chans_up),with=F]

}

all_hh_daily <-  map2(block_files, active_files, get_indiv_daily_viewing) %>% 
	rbindlist

all_hh <- all_hh_daily[,map(.SD, sum), by=.(household_id, r_prop, tercile), .SDcols=chans_up] %>% 
	setnames(old=chans_up, new = paste("mins", chans_up, sep="_"))

# normalize amounts by number of days in sample
days_in <- all_hh_daily[,.(n = .N), by = .(household_id)]
all_hh <- all_hh[days_in, on = .(household_id)]

all_hh[, mins_CNN_per := mins_CNN / n]
all_hh[, mins_FNC_per := mins_FNC / n]
all_hh[, mins_MSNBC_per := mins_MSNBC / n]

# thresholds for frac_viewing_x moments
Thresh1 <- 0.0005 * 360
Thresh2 <- 0.001 * 360
Thresh3 <- 0.01 * 360
Thresh4 <- 0.1 * 360

stats <- all_hh[,.(CNN_pct_0005 = mean(mins_CNN_per >= Thresh1),
		  CNN_pct_001 = mean(mins_CNN_per >= Thresh2),
		  CNN_pct_01 = mean(mins_CNN_per >= Thresh3),
		  CNN_pct_1 = mean(mins_CNN_per >= Thresh4),
		  FNC_pct_0005 = mean(mins_FNC_per >= Thresh1),
		  FNC_pct_001 = mean(mins_FNC_per >= Thresh2),
		  FNC_pct_01 = mean(mins_FNC_per >= Thresh3),
		  FNC_pct_1 = mean(mins_FNC_per >= Thresh4),
		  MSN_pct_0005 = mean(mins_MSNBC_per >= Thresh1),
		  MSN_pct_001 = mean(mins_MSNBC_per >= Thresh2),
		  MSN_pct_01 = mean(mins_MSNBC_per >= Thresh3),
		  MSN_pct_1 = mean(mins_MSNBC_per >= Thresh4),
		  # CNN_mean_rprop = weighted.mean(r_prop, mins_CNN),
		  # CNN_sd_rprop   = sqrt(weighted.mean((r_prop - weighted.mean(r_prop, mins_CNN))^2, mins_CNN)),
		  # FNC_mean_rprop = weighted.mean(r_prop, mins_FNC),
		  # FNC_sd_rprop   = sqrt(weighted.mean((r_prop - weighted.mean(r_prop, mins_FNC))^2, mins_FNC)),
		  # MSN_mean_rprop = weighted.mean(r_prop, mins_MSN),
		  # MSN_sd_rprop   = sqrt(weighted.mean((r_prop - weighted.mean(r_prop, mins_MSN))^2, mins_MSN)),
		  CNN_25_rprop = weighted.quantile(r_prop, mins_CNN_per, 0.25),
		  CNN_50_rprop = weighted.quantile(r_prop, mins_CNN_per, 0.5),
		  CNN_75_rprop = weighted.quantile(r_prop, mins_CNN_per, 0.75),
		  FNC_25_rprop = weighted.quantile(r_prop, mins_FNC_per, 0.25),
		  FNC_50_rprop = weighted.quantile(r_prop, mins_FNC_per, 0.5),
		  FNC_75_rprop = weighted.quantile(r_prop, mins_FNC_per, 0.75),
		  MSN_25_rprop = weighted.quantile(r_prop, mins_MSNBC_per, 0.25),
		  MSN_50_rprop = weighted.quantile(r_prop, mins_MSNBC_per, 0.5),
		  MSN_75_rprop = weighted.quantile(r_prop, mins_MSNBC_per, 0.75),
		  left_CNN = mean(mins_CNN_per[tercile==1]),
		  left_FNC = mean(mins_FNC_per[tercile==1]),
		  left_MSN = mean(mins_MSNBC_per[tercile==1]),
		  center_CNN = mean(mins_CNN_per[tercile==2]),
		  center_FNC = mean(mins_FNC_per[tercile==2]),
		  center_MSN = mean(mins_MSNBC_per[tercile==2]),
		  right_CNN = mean(mins_CNN_per[tercile==3]),
		  right_FNC = mean(mins_FNC_per[tercile==3]),
		  right_MSN = mean(mins_MSNBC_per[tercile==3]),
		  CNN_FNC_pct_0005 = mean(mins_CNN_per >= Thresh1 & mins_FNC_per >= Thresh1),
		  CNN_FNC_pct_001 = mean(mins_CNN_per >= Thresh2 & mins_FNC_per >= Thresh2),
		  CNN_FNC_pct_01 = mean(mins_CNN_per >= Thresh3 & mins_FNC_per >= Thresh3),
		  CNN_MSN_pct_0005 = mean(mins_CNN_per >= Thresh1 & mins_MSNBC_per >= Thresh1),
		  CNN_MSN_pct_001 = mean(mins_CNN_per >= Thresh2 & mins_MSNBC_per >= Thresh2),
		  CNN_MSN_pct_01 = mean(mins_CNN_per >= Thresh3 & mins_MSNBC_per >= Thresh3),
		  FNC_MSN_pct_0005 = mean(mins_FNC_per >= Thresh1 & mins_MSNBC_per >= Thresh1),
		  FNC_MSN_pct_001 = mean(mins_FNC_per >= Thresh2 & mins_MSNBC_per >= Thresh2),
		  FNC_MSN_pct_01 = mean(mins_FNC_per >= Thresh3 & mins_MSNBC_per >= Thresh3)
		  )] %>%
	gather(key=stat, value=value)

write_csv(stats, path = sprintf("%s/data/model/viewership_indiv_rawmoments.csv", local_dir))


### HISTOGRAM OF MINUTES PER DAY ###
histdata <- all_hh %>% 
  melt(measure.vars=paste("mins", chans_up, "per", sep="_"), id.vars=c("household_id"), value.name="mins_per", variable.name="channel") %>%
  .[,channel := sub("mins_([A-Z]+)_per", "\\1", channel)]

mins_per_hist <- ggplot(aes(x=mins_per, after_stat(density)), data=histdata) + 
  geom_freqpoly(binwidth=15/172) + 
  theme(legend.position="none") +
  xlab("Minutes Watched Per Day") +
  ylab("Density") +
  facet_wrap(~ channel) + 
  xlim(0,Thresh3)

 ggsave(mins_per_hist, file = sprintf("%s/%s/stb_mins_per_hh_hist.png", local_dir, plot_dir), height=4, width=10)

### CROSS VIEWING PLOT ###

c1 <- ggplot(all_hh, aes(x = mins_FNC_per, y = mins_CNN_per)) + geom_point(alpha=alpha)  + xlab("Minutes FNC watched per day") + ylab("Minutes CNN watched per day") + geom_point(alpha=alpha) + xlim(0,min_lim) + ylim(0,min_lim)
c2 <- ggplot(all_hh, aes(x = mins_FNC_per, y = mins_MSNBC_per)) + geom_point(alpha=alpha) + xlab("Minutes FNC watched per day") + ylab("Minutes MSNBC watched per day") + geom_point(alpha=alpha) + xlim(0,min_lim) + ylim(0,min_lim)
c3 <- ggplot(all_hh, aes(x = mins_CNN_per, y = mins_MSNBC_per)) + geom_point(alpha=alpha) + xlab("Minutes CNN watched per day") + ylab("Minutes MSNBC watched per day") + geom_point(alpha=alpha) + xlim(0,min_lim) + ylim(0,min_lim)
c1 <- ggplot_gtable(ggplot_build(c1))
c2 <- ggplot_gtable(ggplot_build(c2))
c3 <- ggplot_gtable(ggplot_build(c3))
maxWidth <- unit.pmax(c1$widths[2:3], c2$widths[2:3], c3$widths[2:3])
c1$widths[2:3] <- maxWidth
c2$widths[2:3] <- maxWidth
c3$widths[2:3] <- maxWidth
data_cross_viewership <- grid.arrange(c1, c2, c3, nrow = 1)

ggsave(plot=data_cross_viewership, filename=sprintf("%s/%s/stb_cross_viewership.png",local_dir, plot_dir), height=4, width=10)

#### LORENZ PLOT ####

all_hh_long <- melt(all_hh, id.vars="household_id", measure.vars = paste("mins_",chans_up,"_per", sep="")) %>%
	.[,channel := sub("mins_([A-Z]+)_per" , "\\1", variable)] %>%
	.[value >= Thresh1]

frac_watching <- all_hh_long[,.(frac = .N / nrow(all_hh)), by =.(channel)] %>%
	.[,view_pct:=paste(sprintf("%% HH Watching >= %s min/day:", round(Thresh1,2)), round(frac*100, 0))]


points_to_mark <- all_hh_long[,
	.(p=c(0.5,0.9), L=c(cumsum(sort(value))[floor(0.5*.N)]/sum(value),cumsum(sort(value))[floor(0.9*.N)]/sum(value))),
	  by=.(channel)]

hlines_to_mark <- copy(points_to_mark) %>% .[,`:=`(p0=0, L0=L)]
vlines_to_mark <- copy(points_to_mark) %>% .[,`:=`(p0=p, L0=0)]
lines <- rbind(hlines_to_mark, vlines_to_mark)

prop_watched_50pct <- copy(points_to_mark) %>%
	.[p==0.5] %>%
	.[, view_prop := paste0("Bottom Half of Viewers Watch ", round(L*100, 0), "% Total")]


lorenz_plot <- ggplot(aes(x=value), data=all_hh_long) + 
	stat_lorenz(aes(colour=channel), geom="line") +
  	theme_bw() +
  	theme(legend.position="none") +
  	scale_colour_manual(values=c(CNN="mediumpurple2", FNC="darkred", MSNBC="cornflowerblue")) +
  	geom_segment(data=lines, aes(x = p, y = L, xend=p0, yend=L0), linetype="dashed", alpha=0.8) +
  	geom_text(aes(label = view_pct), data=frac_watching, x = 0, y = 1, hjust = "left", size=4) +
  	geom_text(aes(label = view_prop), data=prop_watched_50pct, x = 0, y = .95, hjust = "left", size=4) +
  	facet_wrap(~channel) +
  	xlab("Percent of Households") + ylab("Percent of Time Watched")


ggsave(plot=lorenz_plot, filename=sprintf("%s/%s/stb_lorenz_plot.pdf",local_dir, plot_dir), height=4, width=10)


### time series graphs ###


# output daily household viewership to compare with model
# fwrite(daily_indiv_ratings, sprintf("%s/data/model/stb_daily_indiv_ratings.csv", local_dir))

all_hh_daily[,party := case_when(tercile==1 ~ "Democrat",
                             tercile==2 ~ "Independent",
                             tercile==3 ~ "Republican")]

### AVG MINUTES ###
daily_minutes <- all_hh_daily[,map(.SD, mean),by=.(date),.SDcols=chans_up] %>%
	melt(measure.vars=chans_up, variable.name="channel", value.name="mins")

day_stb_viewership_plot <- ggplot(aes(x=date, y=mins, group=channel), data = daily_minutes) +
  geom_line(aes(colour=channel)) +
  theme_bw() +
  ylab("Daily Avg Minutes Viewed") +
  xlab("Date") +
  scale_colour_manual(values=c(CNN="mediumpurple2", FNC="darkred", MSNBC="cornflowerblue"))+
  scale_x_date(breaks=ymd(c("20120604", "20120801", "20121001", "20121106", "20130101")),
               labels=c("Jun 4", "Aug 1", "Oct 1", "Nov 6", "Jan 1")) +
  geom_vline(xintercept = ymd("20121106"), linetype="dashed") +
  ylim(0,max_mins) +
  theme(legend.position = "none")


ggsave(day_stb_viewership_plot, filename=sprintf("%s/%s/stb_daily_avg_mins.png",local_dir, plot_dir), height = 4, width=10)

### AVG MINUTES BY PARTY ###
daily_minutes_byparty <- all_hh_daily[,map(.SD, mean),by=.(date, party),.SDcols=chans_up] %>%
	melt(measure.vars=chans_up, variable.name="channel", value.name="mins")

day_stb_viewership_plot_party <- ggplot(aes(x=date, y=mins, group=channel), data = daily_minutes_byparty) +
  geom_line(aes(colour=channel)) +
  theme_bw() +
  ylab("Daily Avg Minutes Viewed") +
  xlab("Date") +
  scale_colour_manual(values=c(CNN="mediumpurple2", FNC="darkred", MSNBC="cornflowerblue"))+
  scale_x_date(breaks=ymd(c("20120604", "20120801", "20121001", "20121106", "20130101")),
               labels=c("Jun 4", "Aug 1", "Oct 1", "Nov 6", "Jan 1")) +
  geom_vline(xintercept = ymd("20121106"), linetype="dashed") +
  ylim(0,max_mins) +
  facet_wrap(~party) +
  theme(legend.position = "none")

ggsave(day_stb_viewership_plot_party, filename=sprintf("%s/%s/stb_daily_avg_mins_party.png",local_dir, plot_dir), height = 4, width=10)

#### FRACTION VIEWING ####
daily_rating <- all_hh_daily[,map(.SD, ~mean(.>15)*100),by=.(date),.SDcols=chans_up] %>%
	melt(measure.vars=chans_up, variable.name="channel", value.name="rat")

day_stb_rating_plot <- ggplot(data = daily_rating, aes(x=date, y=rat, group=channel)) +
  geom_line(aes(colour=channel)) +
  theme_bw() +
  ylab("% STB Households Watching >= 15 Mins") +
  xlab("Date") +
  scale_colour_manual(values=c(CNN="mediumpurple2", FNC="darkred", MSNBC="cornflowerblue"))+
  scale_x_date(breaks=ymd(c("20120604", "20120801", "20121001", "20121106", "20130101")),
               labels=c("Jun 4", "Aug 1", "Oct 1", "Nov 6", "Jan 1")) +
  geom_vline(xintercept = ymd("20121106"), linetype="dashed") +
  ylim(0,max_rating) +
  theme(legend.position = "none")

ggsave(day_stb_rating_plot, filename=sprintf("%s/%s/stb_daily_pct_watched.png", local_dir, plot_dir), height = 4, width=10)

### SHOW SORTING PLOT ###
rm(all_hh_daily)
gc()
get_show_index <- function(chan, tz) {
	fread(sprintf("%s/data/model/topic_weights_%s_%s.csv",local_dir, chan,tz)) %>%
		.[,`:=` (timezone=case_when(tz=="ETZ" ~  "America/New_York", tz=="CTZ" ~  "America/Chicago", tz=="PTZ" ~ "America/Los_Angeles"),
				 time_block_15 = time_block_15 - min(time_block_15) + 1,
				 show = allshows[show_index],
				 date = rep(alldates, each=length(unique(time_block_15))),
				 channel = toupper(chan)				 
				 )] %>%
		.[,.(timezone, date, time_block_15, channel, show)] %>%
		.[,.(block=(time_block_15-1)*3 + 1:3), by=.(timezone, date, channel, time_block_15, show)] # convert 15 min to 5 min
}

show_block_index <- map2(rep(chans, each=3), rep(c("ETZ", "CTZ", "PTZ"), times=3), 
						 get_show_index) %>% 
	rbindlist


sort_day <- function(dayfile, activefile) {
	cat(dayfile, "\n")
	all_active <- readRDS(activefile) %>% 
		unique(by="household_id") %>%
		.[,.(household_id, r_prop)]

	block_day <- readRDS(dayfile)

	blocks <- grep("bl_", colnames(block_day), value=T)
	block_day %>% melt(id.vars=c("date", "household_id", "timezone", "channel"),
					   measure.vars=blocks,
					   variable.name="block",
					   value.name="mins") %>%
		.[mins > 0] %>%
		.[,block:=as.integer(str_extract(block, "\\d+"))] %>%
		.[show_block_index, on=.(timezone, date, channel, block), nomatch=0] %>%
		.[,.(mins=sum(mins)),by=.(timezone, date, channel, show, household_id)] %>%
		.[all_active, on = .(household_id), nomatch=0]

}

all_sorting <- map2(block_files, active_files, sort_day) %>% rbindlist

weighted_quantiles <- all_sorting[,
	.(p25 = weighted.quantile(r_prop, mins, tau=0.25),
	  p50 = weighted.quantile(r_prop, mins, tau=0.5),
	  p75 = weighted.quantile(r_prop, mins, tau=0.75),
	  n = length(unique(household_id))),
	by = .(channel, show)]

weighted_quantiles <- weighted_quantiles[order(p50)]
weighted_quantiles[,show:=factor(show, levels=show)]

show_plot <- ggplot(aes(x = p50, y = show, colour = channel), data=weighted_quantiles) + 
	geom_point(aes(size=n)) +
	geom_errorbarh(aes(y=show, x= p50, xmin = p25, xmax = p75), height=0.2) +
	theme_bw() + 
	scale_colour_manual(values=c(CNN="mediumpurple2", FNC="darkred", MSNBC="cornflowerblue", pbs = "gray40", nbc="gray40", abc ="gray40", cbs="gray40" )) +
	xlab("Estimated R vote propensity") + ylab("Show") +
	xlim(0,1)

ggsave(show_plot, filename=sprintf("%s/%s/stb_show_sorting.png", local_dir, plot_dir), height = 4, width=10)