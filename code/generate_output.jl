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

# update last value from bboptim
to_optimize = ["beta:info","topic_mu:foreign_policy","topic_mu:economy","topic_mu:horse_race","topic_leisure:horse_race","topic_leisure:foreign_policy","topic_mu:crime","topic_leisure:crime","beta:slant","topic_leisure:economy"]
x = [7.23233021191866,-8.811581490250695,-6.80466958590165,4.0889161148819,0.548931380300425,0.503140443178971,5.58226339264457,0.01991359,-0.230365197304856,0.036149553854956]

x = [7.23233021191866,-8.77608721699011,-6.80466958590165,4.0889161148819,0.548931380300425,0.503140443178971,5.58226339264457,0.01991359,-0.230365197304856,0.036149553854956]
pb_val = DataStructures.OrderedDict(k => k ∈ to_optimize ? x[findfirst(k .== to_optimize)] : pb[k].value for k in keys(pb))

pb_val = DataStructures.OrderedDict(k => pb[k].value for k in keys(pb))
# wrap in Eval object
ev = SMM.Eval(mprob, pb_val)

# evaluate
stb_obj(ev; dt=stbdat).value




# to produce output for standard plots
stb_obj(SMM.Eval(mprob); dt=stbdat, save_output = true)
