## Set your local directory
local_dir = "/home/gregorymartin/Dropbox/STBNews"

### END OPTIONS ###

## Directory locations for code and data
using Printf
code_dir = @sprintf("%s/stb-model/code/", local_dir)
data_dir = @sprintf("%s/stb-model/data", local_dir)
output_dir = @sprintf("%s/stb-model/output", local_dir)

### LOAD OBJECTIVE AND DATA ###
## parallel version ##
cd(code_dir)
include("load_model_data.jl")


### READ PARAMETER VECTOR ###
# to read from last MCMC run:
# cd(code_dir)
# include("read_par_from_mcmc.jl")

# to read direct from csv:
par_init = CSV.read("par_init.csv");

# dictionary indexed by parameter, with initial values and bounds for each
pb = DataStructures.OrderedDict(zip(
    par_init.par,
    DataFrames.eachrow(par_init[:, [:value, :lb, :ub]]),
));


# add pars to mprob object
SMM.addSampledParam!(mprob,pb);

# wrap in Eval object
ev = SMM.Eval(mprob, pb_val)

cd(output_dir)

# to produce output for standard plots
stb_obj(ev; dt=stbdat, save_output = true)


# to produce moments
out = stb_obj(ev; dt=stbdat, store_moments = true)
