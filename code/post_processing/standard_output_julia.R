#### INTRO ####
# run with: Rscript code/model/julia/standard_output_julia_v3.R post_inside_obj_func_06212020 (or leave blank to run on current version)
# requires running things in Julia first. Need to run:
# stb_obj(SMM.Eval(mprob); dt=stbdat, save_output=true, store_moments=true)
# in Julia to generate post_inside_obj_func and post_inside_compute_moments
# in the objective function

# intermediate output: data.tables with total viewing by hh*channel, and daily viewing by hh*channel
# install.packages(c("hdf5r", "tidyverse", "ggplot2", "gridExtra", "lattice", "broom", "lubridate", "magrittr", "ggthemes", "data.table", "gglorenz"))
# install.packages(c("vctrs", "purrr", "hms", "haven"))
library(hdf5r)
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(grid)
library(lattice)
library(broom)
library(Matrix)
library(lubridate)
library(magrittr)
library(ggthemes)
library(data.table)
library(gglorenz)

# options
obj_func_name = ifelse(is.na(commandArgs(trailingOnly = TRUE)[1]), 'model_output_path', commandArgs(trailingOnly = TRUE)[1])
# obj_func_name = "post_inside_obj_func" # if you want to label run outputs

# model options - change these as the input data changes
nstb = 6000
nnatl = 6000
ntopics = 4

blocklen=15
nblocks_day=360/blocklen
nblocks=ndays*nblocks_day
npaths = 10
# days_to_use = 1:172
# days_to_use = c(7,17,27,36,50,60,70,80,90,100,110,120,130,140,150,160,170)
days_to_use = c(2,7,12,17,22,27,32,36,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170)
ndays = length(days_to_use)
# set plot axes
max_rating = 20
max_day_rating = 5
max_mins = 30
min_lim=180 # now min_lim
alpha=.05
min_daily_mins_watched = .18


# thresholds for frac_viewing_x moments
Thresh1 <- 0.0005 * 360
Thresh2 <- 0.001 * 360
Thresh3 <- 0.01 * 360
Thresh4 <- 0.1 * 360

all_channels = c("cnn", "fnc", "msnbc")
chans_up <- toupper(all_channels)
nchans <- length(all_channels)

# local_dir = "~/Dropbox/STBNews/stb-model-discrete"
# local_dir = "/home/cfuser/mlinegar"
local_dir = "/home/users/mlinegar/stb-model"
# local_dir = "/usr/local/ifs/gsb/gjmartin/STBnews/STBNews"
output_dir = "output/standard_output_graphs"
# output_dir = "output/standard_output_graphs"

# get dates list for plots
# setwd(sprintf("~/Dropbox/STBNews/data/FWM/block_view/", local_dir))
# alldates <- ymd(sub("fwm_block_view_(\\d{4}-\\d{2}-\\d{2})\\.rds", "\\1", list.files(pattern="fwm_block_view_")))
alldates <- fread(sprintf("%s/data/alldates_full.csv", local_dir))[[1]] %>% ymd()
alldates <- alldates[which(alldates==ymd("20120604")):length(alldates)]
alldates <- alldates[days_to_use]

# get list of show names
show_table <- fread(sprintf("%s/data/show_to_channel.csv", local_dir))
allshows <- show_table[,show]

theme_set(theme_few())


weighted.quantile <- function(x, w, tau=0.5) {
  i <- order(x)
  x <- x[i]
  w <- w[i]
  x[min(which(cumsum(w) >= tau * sum(w)))]
}


select <- dplyr::select
rename <- dplyr::rename


#### LOAD OUTSIDE DATA ####
pars <- fread(sprintf("%s/data/parameter_bounds.csv", local_dir))

stb_hh_sample <- fread(sprintf("%s/data/stb_hh_sample.csv", local_dir))[, id := 1:.N][, r_prob := r_prop]
stb_hh_sample[
  , r_prob_cat := ntile(r_prob, 3)
  ][
    , party := case_when(r_prob_cat==1~"Democrat",
                         r_prob_cat==2~"Independent",
                         r_prob_cat==3~"Republican")
    ]

# load shows

get_show_index <- function(chan, tz, periods_to_use=1:4128) {
  fread(sprintf("%s/data/topic_weights_%s_%s.csv",local_dir, chan,tz)) %>%
    .[periods_to_use] %>%
    .[,`:=` (timezone=case_when(tz=="ETZ" ~  1, tz=="CTZ" ~  2, tz=="PTZ" ~ 4),
         block = time_block_15 - min(time_block_15) + 1,
         show = allshows[show_index],
         date = rep(ymd(alldates), each=length(unique(time_block_15))),
         channel = toupper(chan)
         )] %>%
    .[,.(timezone, date, block, channel, show)]
}

show_block_index <- map2(rep(all_channels, each=3), rep(c("ETZ", "CTZ", "PTZ"), times=3),
             get_show_index, periods_to_use = rep((days_to_use - 1) * 24, each=24) + 1:24) %>%
  rbindlist


#### LOAD MODEL DATA ####
# load output from Julia
sim_viewership_natl <- data.table(period = 1:nblocks, 
                                    id = rep(1:nnatl, each = nblocks), 
                                    CNN = 0, 
                                    FNC = 0, 
                                    MSNBC = 0)

daily_polling <- numeric(ndays)

for (i in 1:npaths) {
  julia_obj_func <- h5file(sprintf("%s/output/%s_%i.jld2", local_dir, obj_func_name,i), "r")

  # topic_path <- julia_obj_func[["news"]][1:ntopics,1:ndays] %>% t() %>% as.data.table() %>% setDT()
  # colnames(topic_path) <- pars[1:ntopics, par] %>% str_remove_all("topic_lambda:")
  # topic_path[, day := 1:.N]
  # topic_path_long <- data.table::melt(topic_path, id.vars = c("day"), variable.name = "topic")


  daily_polling <- daily_polling + (julia_obj_func[["daily_polling"]][1:ndays] / npaths)

  # construct sim viewership channel by channel: national sample
  cnn_view_natl_long <- julia_obj_func[["consumer_view_history_national"]][1:nnatl, 1, 1:nblocks] %>% t %>% as.integer 
  fnc_view_natl_long <- julia_obj_func[["consumer_view_history_national"]][1:nnatl, 2, 1:nblocks] %>% t %>% as.integer 
  msnbc_view_natl_long <- julia_obj_func[["consumer_view_history_national"]][1:nnatl, 3, 1:nblocks] %>% t %>% as.integer 

  sim_viewership_natl[,`:=` (CNN = CNN + cnn_view_natl_long / npaths,
                             FNC = FNC + fnc_view_natl_long / npaths,
                             MSNBC = MSNBC + msnbc_view_natl_long / npaths)]
  
  rm(cnn_view_natl_long, fnc_view_natl_long, msnbc_view_natl_long)
}

get_day <- function(x){((x-1) %/% nblocks_day) + 1}
get_date <- function(x){alldates[((x-1) %/% nblocks_day) + 1]}
get_hour <- function(x){(((x - (get_day(x)-1)*nblocks_day)-1) %/% (60/blocklen)) + 5}
get_block <- function(x){((x-1) %% nblocks_day)}

# testing get_hour functions - make sure these look correct
# get_hour(1)
# get_hour(4)
# get_hour(20)
# get_hour(21)
# get_hour(23)
# get_hour(24)
# get_hour(25)
# get_hour(4128)

# convert block inds to date / block
sim_viewership_natl[,`:=`(
  date = get_date(period),
  hour = get_hour(period),
  block = get_block(period)
)]


#### AVERAGE MINUTES ####
# from national sample
avg_mins_day <- sim_viewership_natl[, map(.SD, ~ sum(.) * 15 / nnatl), by = .(date)] %>%
  melt(id.vars=c("date"), measure.vars=chans_up, variable.name="channel", value.name="avg_min")

avg_min_plot <- ggplot(aes(x=date, y=avg_min, group=channel), data = avg_mins_day) +
  geom_line(aes(colour=channel)) +
  theme_bw() +
  ylab("Daily Average Minutes") +
  xlab("Date") +
  scale_colour_manual(values=c(CNN="mediumpurple2", FNC="darkred", MSNBC="cornflowerblue"))+
  scale_x_date(breaks=ymd(c("20120604", "20120801", "20121001", "20121106", "20130101")),
               labels=c("Jun 4", "Aug 1", "Oct 1", "Nov 6", "Jan 1")) +
  geom_vline(xintercept = ymd("20121106"), linetype="dashed") +
  ylim(0,max_mins) +
  theme(legend.position = "none")

ggsave(plot=avg_min_plot, filename=sprintf("%s/%s/sim_daily_avg_mins_natl.png",local_dir, output_dir), height=4, width=10)

#### MAX BLOCK RATING ####
# from national sample
max_ratings_day <- sim_viewership_natl[,map(.SD, mean), by=.(period, date, hour, block), .SDcols=chans_up] %>%
  .[, map(.SD, max), by = .(date)] %>%
  data.table::melt(id.vars=c("date"), measure.vars=chans_up, variable.name="channel", value.name="max_rat") %>%
  .[,max_rat:=max_rat * 100]

rm(sim_viewership_natl)
gc()

# ## alt. method using predicted_channel_ratings
# colnames(predicted_channel_ratings) <- paste(rep(chans_up, times=4), rep(1:4,each=nchans), sep="_")
# max_ratings_day <- predicted_channel_ratings %>%
#   .[,period := 1:nblocks] %>%
#   .[,`:=`(
#     date = get_date(period),
#     hour = get_hour(period),
#     block = get_block(period)
#     )] %>%
#   melt(id.vars=c("period", "date","hour","block"), value.name="rat") %>%
#   .[,`:=` (channel = sub("([A-Z]+)_\\d","\\1",variable),
#            timezone = as.integer(sub("[A-Z]+_(\\d)","\\1",variable)))] %>%
#   .[,.(rat = mean(rat)),by=.(date, hour, block, channel)] %>%
#   .[,.(max_rat = max(rat)*100), by = .(date, channel)]


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

ggsave(plot=max_ratings_plot, filename=sprintf("%s/%s/sim_daily_rating_max.png",local_dir, output_dir), height=4, width=10)






# construct sim viewership channel by channel: stb sample
sim_viewership <- data.table(period = 1:nblocks, 
                             id = rep(1:nstb, each = nblocks), 
                             CNN = 0, 
                             FNC = 0, 
                             MSNBC = 0)
for (i in 1:npaths) {

  julia_obj_func <- h5file(sprintf("%s/output/%s_%i.jld2", local_dir, obj_func_name, i), "r")


  cnn_view_stb_long <- julia_obj_func[["consumer_view_history_stb"]][1:nstb, 1, 1:nblocks] %>% t %>% as.integer 
  fnc_view_stb_long <- julia_obj_func[["consumer_view_history_stb"]][1:nstb, 2, 1:nblocks] %>% t %>% as.integer 
  msnbc_view_stb_long <- julia_obj_func[["consumer_view_history_stb"]][1:nstb, 3, 1:nblocks] %>% t %>% as.integer 

  sim_viewership[,`:=` (CNN = CNN + cnn_view_stb_long / npaths,
                        FNC = FNC + fnc_view_stb_long / npaths,
                        MSNBC = MSNBC + msnbc_view_stb_long / npaths)] 

  rm(cnn_view_stb_long, fnc_view_stb_long, msnbc_view_stb_long)
}

# convert block inds to date / block
sim_viewership[,`:=`(
  date = get_date(period),
  hour = get_hour(period),
  block = get_block(period)
)]

sim_viewership <- stb_hh_sample[sim_viewership, on=.(id)]

keep_vars <- c("id", "party", "r_prob", "timezone", "date", "hour", "block", "period")
drop_vars <- names(sim_viewership)[!names(sim_viewership) %in% c(keep_vars, chans_up)]
sim_viewership[,(drop_vars) := NULL]

rm(julia_obj_func)
gc()

# aggregate to day, converting blocks to minutes
all_hh_daily <- sim_viewership[,
  map(.SD, compose(sum, ~ . * blocklen)),
  by=.(timezone, id, r_prob, party, date),
  .SDcols=chans_up]

# aggregate to hh totals
all_hh <- all_hh_daily[,
  map(.SD, sum),
  by=.(timezone, id, r_prob, party),
  .SDcols=chans_up]  %>%
  setnames(old=chans_up, new = paste("mins", chans_up, sep="_"))

# rescale to per-day
all_hh[, mins_CNN_per := mins_CNN / ndays]
all_hh[, mins_FNC_per := mins_FNC / ndays]
all_hh[, mins_MSNBC_per := mins_MSNBC / ndays]


### HISTOGRAM OF MINUTES PER DAY ###
histdata <- all_hh %>%
  data.table::melt(measure.vars=paste("mins", chans_up, "per", sep="_"), id.vars=c("id"), value.name="mins_per", variable.name="channel") %>%
  .[,channel := sub("mins_([A-Z]+)_per", "\\1", channel)]

mins_per_hist <- ggplot(aes(x=mins_per, after_stat(density)), data=histdata) +
  geom_freqpoly(binwidth=15/172) +
  theme(legend.position="none") +
  xlab("Minutes Watched Per Day") +
  ylab("Density") +
  facet_wrap(~ channel) +
  xlim(0,Thresh3) +
  ylim(0,10)

ggsave(mins_per_hist, file = sprintf("%s/%s/sim_mins_per_hh_hist.png", local_dir, output_dir), height=4, width=10)

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
sim_cross_viewership <- grid.arrange(c1, c2, c3, nrow = 1)

ggsave(plot=sim_cross_viewership, filename=sprintf("%s/%s/sim_cross_viewership.png",local_dir, output_dir), height=4, width=10)

#### LORENZ PLOT ####


all_hh_long <- data.table::melt(all_hh, id.vars="id", measure.vars = paste("mins_",chans_up,"_per", sep="")) %>%
  .[,channel := sub("mins_([A-Z]+)_per" , "\\1", variable)] %>%
  .[value >= Thresh1]

frac_watching <- all_hh_long[,.(frac = .N / nstb), by =.(channel)] %>%
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

ggsave(plot=lorenz_plot, filename=sprintf("%s/%s/sim_lorenz_plot.png",local_dir, output_dir), height=4, width=10)


### time series graphs ###
### AVG MINUTES (STB) ###
daily_minutes <- all_hh_daily[,map(.SD, mean),by=.(date),.SDcols=chans_up] %>%
  data.table::melt(measure.vars=chans_up, variable.name="channel", value.name="mins")

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

ggsave(plot=day_stb_viewership_plot, filename=sprintf("%s/%s/sim_daily_avg_mins.png",local_dir, output_dir), height=4, width=10)

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
ggsave(plot=day_stb_viewership_plot_party, filename=sprintf("%s/%s/sim_daily_avg_mins_party.png",local_dir, output_dir), height=4, width=10)


#### FRACTION VIEWING ####
daily_rating <- all_hh_daily[,map(.SD, ~mean(.>15)*100),by=.(date),.SDcols=chans_up] %>%
  data.table::melt(measure.vars=chans_up, variable.name="channel", value.name="rat")

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

ggsave(plot=day_stb_rating_plot, filename=sprintf("%s/%s/sim_daily_pct_watched.png",local_dir, output_dir), height=4, width=10)

rm(all_hh_daily, all_hh)
gc()

### SHOW SORTING PLOT ###
# construct weighted quantiles of viewership for show sorting plot
# note: only need viewers for this, so filter before reshape
sim_viewership_long <- sim_viewership %>%
  .[CNN + FNC + MSNBC > 0] %>%
  data.table::melt(measure.vars = chans_up, variable.name = "channel", value.name="watched") %>%
  .[show_block_index, on = .(timezone, date, channel, block), nomatch=0]

weighted_quantiles <- sim_viewership_long %>%
  .[,.(mins = sum(watched) * blocklen),by=.(id, r_prob, channel, show)] %>%
  .[,.(p25 = weighted.quantile(r_prob, mins, tau=0.25),
       p50 = weighted.quantile(r_prob, mins, tau=0.5),
       p75 = weighted.quantile(r_prob, mins, tau=0.75),
       n = length(unique(id))),
    by = .(channel, show)]

weighted_quantiles <- weighted_quantiles[order(p50)]
weighted_quantiles[,show:=factor(show, levels=show)]


show_plot <- ggplot(aes(x = p50, y = show, colour = channel), data=weighted_quantiles) +
  geom_point(aes( x= p50,size=n)) +
  geom_errorbarh(aes(y=show, xmin = p25, xmax = p75), height=0.2) +
  theme_bw() +
  scale_colour_manual(values=c(CNN="mediumpurple2", FNC="darkred", MSNBC="cornflowerblue", pbs = "gray40", nbc="gray40", abc ="gray40", cbs="gray40" )) +
  xlab("Estimated R vote propensity") + ylab("Show") +
  xlim(0,1)

ggsave(plot=show_plot, filename=sprintf("%s/%s/sim_show_sorting.png",local_dir, output_dir), height=4, width=10)










