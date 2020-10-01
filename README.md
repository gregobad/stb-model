# README

# Workflow

First run the Setup script locally, on the Yens, and CloudForest. Then to run the pipeline run `./sync_stb.sh`. In all cases, assumes you are starting working from the project base directory (locally, `~/Dropbox/STBNews/` , on the Yens at `/ifs/gsb/gjmartin/STBnews/STBNews`, or in your home directory on CloudForest). 

Running this file will upload files that need to stay synced (all code, parameter files) and download output model data and standard model plots. On the Yens, it runs the R side of the standard output. Locally, it compiles the results with pdflatex. As currently setup, once the estimation is run on CloudForest, just run `./sync_stb.sh` on either the Yens or CF (doesn’t matter which one) and then again at `~/Dropbox/STBNews/` to sync the results. 


# Setup

Locally and on CloudForest, edit the file `~/.ssh/config` to include the following, possibly substituting for a different Yen:

```
ControlMaster auto
ControlPath ~/.ssh/%r@%h:%p
ControlPersist yes

Host yen3
        HostName yen3.stanford.edu
        User mlinegar
```

On the Yens, edit the file `~/.ssh/config` to include the following, substituting your instance name:

```
Host stb72 
    HostName ec2-35-162-27-152.us-west-2.compute.amazonaws.com 
    User mlinegar
```

You can now run through the pipeline. 



# Pipeline

Starting in the project base directory, at the command line, run:

`./sync_stb.sh`

This downloads output files, and uploads relevant code and paramter files, between the Yens and our local directory. It also compiles our standard output into a PDF. 

Now login to the Yens with

`ssh yen3`

Set your working directory:

`cd /ifs/gsb/gjmartin/STBnews/STBNews/`

Again, upload and download relevant files, this time between the Yens and our CF instance. 

`./sync_stb.sh`

Note that after doing the upload and download, it will run the R part of the standard output pipeline, creating the simulated plots. 

Now login to CloudForest with:

`ssh stb72`

Now run `estimation_driver.jl` with the following line:

`/home/cfuser/mlinegar/julia/./julia /home/cfuser/mlinegar/code/model/julia/estimation_driver.jl`

At the end of the optimization, this will run the objective one more time with parameters from the final accepted iteration of the chain. This creates the file the R side of the standard output needs to run. 

# After Estimation

To update the standard output report, run `./sync_stb.sh` on the Yens and then locally. This will produce the plots in R and compile the plots into the report. 

# End Notes: 

Alternatively, to create the output files required by R by itself, run:

`/home/cfuser/mlinegar/julia/./julia /home/cfuser/mlinegar/code/model/julia/estimation_driver_ml_temp.jl`

This file is also intended for doing manual tuning. Edit the lines defining `some_pars` and `some_jitters` to run the objective function on a new set of parameters. Note: jitters are multiplicative. This will produce the file: `fit_channel_pct_viewing_pars_2.csv`, which is one of the input CSV files for `estimation_driver.jl`. 

Note: if you ran manual tuning on the server (rather than locally), you'll have to sync your files in a way not specified by `sync_stb.sh`. First send changes from CloudForest to the Yens with:

`rsync -avz --files-from=upload_files_to_sync.txt /home/cfuser/mlinegar/ yen3:/ifs/gsb/gjmartin/STBnews/STBNews/`

And then pull these changes locally with:

`rsync -avz --files-from=upload_files_to_sync.txt yen3:/ifs/gsb/gjmartin/STBnews/STBNews/ ~/Dropbox/STBNews/`

Now that your changes are synced up, you can resume syncing your steps with `sync_stb.sh`. If you ran manual tuning locally, you can run `sync_stb.sh` as usual. 
# stb-model
