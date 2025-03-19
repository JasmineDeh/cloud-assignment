import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "pip"])

# Import required modules 
import infofile # local file containing cross-sections, sums of weights, dataset IDs
import numpy as np # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
import matplotlib_inline # to edit the inline plot format
matplotlib_inline.backend_inline.set_matplotlib_formats('pdf', 'svg') # to make plots in pdf (vector) format
from matplotlib.ticker import AutoMinorLocator # for minor ticks
import uproot # for reading .root files
import awkward as ak # to represent nested data in columnar format
import vector # for 4-momentum calculations
import time
import os
import pandas as pd
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

#####################################################################################

# Unit definitions as stored in the data files
MeV = 0.001
GeV = 1.0

# For identification and naming
samples = {

    'data': {
        'list' : ['data_A','data_B','data_C','data_D'], # data is from 2016, first four periods of data taking (ABCD)
    },

    r'Background $Z,t\bar{t}$' : { # Z + ttbar
        'list' : ['Zee','Zmumu','ttbar_lep'],
        'color' : "#6b59d3" # purple
    },

    r'Background $ZZ^*$' : { # ZZ
        'list' : ['llll'],
        'color' : "#ff0000" # red
    },

    r'Signal ($m_H$ = 125 GeV)' : { # H -> ZZ -> llll
        'list' : ['ggH125_ZZ4lep','VBFH125_ZZ4lep','WH125_ZZ4lep','ZH125_ZZ4lep'],
        'color' : "#00cdff" # light blue
    },

}

# ATLAS Open Data directory
path = "https://atlas-opendata.web.cern.ch/Legacy13TeV/4lep/" 

# Define what variables are important to our analysis
variables = ['lep_pt','lep_eta','lep_phi','lep_E','lep_charge','lep_type']


# MC define weights important to analysis
weight_variables = ["mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_LepTRIGGER"]

# MC integrated luminosity
lumi = 0.5 # fb-1 # data_A only
# lumi = 1.9 # fb-1 # data_B only
# lumi = 2.9 # fb-1 # data_C only
# lumi = 4.7 # fb-1 # data_D only
# lumi = 10 # fb-1 # data_A,data_B,data_C,data_D

####################################################################################
# Starting analysis

# Cut lepton type (electron type is 11,  muon type is 13)
def cut_lep_type(lep_type):
    sum_lep_type = lep_type[:, 0] + lep_type[:, 1] + lep_type[:, 2] + lep_type[:, 3]
    lep_type_cut_bool = (sum_lep_type != 44) & (sum_lep_type != 48) & (sum_lep_type != 52)
    return lep_type_cut_bool # True means we should remove this entry (lepton type does not match)

# Cut lepton charge
def cut_lep_charge(lep_charge):
    # first lepton in each event is [:, 0], 2nd lepton is [:, 1] etc
    sum_lep_charge = lep_charge[:, 0] + lep_charge[:, 1] + lep_charge[:, 2] + lep_charge[:, 3] != 0
    return sum_lep_charge # True means we should remove this entry (sum of lepton charges is not equal to 0)

# Calculate invariant mass of the 4-lepton state
# [:, i] selects the i-th lepton in each event
def calc_mass(lep_pt, lep_eta, lep_phi, lep_E):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "E": lep_E})
    invariant_mass = (p4[:, 0] + p4[:, 1] + p4[:, 2] + p4[:, 3]).M * MeV # .M calculates the invariant mass
    return invariant_mass
        
#####################################################################################
# Calculate weights

def calc_weight(weight_variables, sample, events):
    info = infofile.infos[sample]
    xsec_weight = (lumi*1000*info["xsec"])/(info["sumw"]*info["red_eff"]) #*1000 to go from fb-1 to pb-1
    total_weight = xsec_weight 
    for variable in weight_variables:
        total_weight = total_weight * events[variable]
    return total_weight

#####################################################################################
# Function to read, process, and store data

def process(lumi=10, fraction=1.0):
    """
    Function to process data.

    Args:
        lumi (int): Luminosity (default is 10 for all data)
        fraction (float): Controls the fraction of all events analysed (default is 1.0 reduce for quicker runtime, applied in loop over tree)
    Returns:
        (ak.array): processed data.
    """
    # Define empty dictionary to hold awkward arrays
    all_data = {} 

     # Define output directory
    output_dir = "/hzz-analysis/output"
    # Create directory if it doesnt exist
    os.makedirs(output_dir, exist_ok=True)

    # Define a mapping of sample names to file-friendly names
    file_names = {
    "data": "data.parquet",
    "Background $Z,t\bar{t}$": "Background_Zt\bar{t}.parquet",
    "Background $ZZ^*$": "background_ZZ.parquet",
    "Signal ($m_H$ = 125 GeV)": "signal_125GeV.parquet"
    }

    # Loop over samples
    for s in samples:
        # Print which sample is being processed
        print('Processing '+s+' samples') 

        # Define empty list to hold data
        frames = [] 

        # Loop over each file
        for val in samples[s]['list']: 
            if s == 'data': 
                prefix = "Data/" # Data prefix
            else: # MC prefix
                prefix = "MC/mc_"+str(infofile.infos[val]["DSID"])+"."
            fileString = path+prefix+val+".4lep.root" # file name to open


            # start the clock
            start = time.time() 
            print("\t"+val+":") 

            # Open file
            t = uproot.open(fileString, timeout=600)  
            if "mini" not in t:
                raise ValueError(f"Tree 'mini' not found in {fileString}")
            
            tree = t["mini"]  # Keep tree reference
        
            sample_data = []

            # Loop over data in the tree
            for data in tree.iterate(variables + weight_variables, 
                                     library="ak",
                                     entry_stop=tree.num_entries*fraction, # process up to numevents*fraction
                                     step_size = 1000000): 
                # Number of events in this batch
                nIn = len(data) 
                                 
                # Record transverse momenta (see bonus activity for explanation)
                data['leading_lep_pt'] = data['lep_pt'][:,0]
                data['sub_leading_lep_pt'] = data['lep_pt'][:,1]
                data['third_leading_lep_pt'] = data['lep_pt'][:,2]
                data['last_lep_pt'] = data['lep_pt'][:,3]

                # Cuts
                lep_type = data['lep_type']
                data = data[~cut_lep_type(lep_type)]
                lep_charge = data['lep_charge']
                data = data[~cut_lep_charge(lep_charge)]
            
                # Invariant Mass
                data['mass'] = calc_mass(data['lep_pt'], data['lep_eta'], data['lep_phi'], data['lep_E'])

                # Store Monte Carlo weights in the data
                if 'data' not in val: # Only calculates weights if the data is MC
                    data['totalWeight'] = calc_weight(weight_variables, val, data)
                    nOut = sum(data['totalWeight']) # sum of weights passing cuts in this batch 
                else:
                    nOut = len(data)
                elapsed = time.time() - start # time taken to process
                print("\t\t nIn: "+str(nIn)+",\t nOut: \t"+str(nOut)+"\t in "+str(round(elapsed,1))+"s") # events before and after

                # Append data to the whole sample data list
                sample_data.append(data)

            # Once all files are processed for this sample, save the sample_data
            if sample_data:  # Ensure that sample_data is not empty
                # Concatenate the awkward arrays for this sample
                combined_sample_data = ak.concatenate(sample_data)

                # Convert the awkward array to an Arrow table
                arrow_table = ak.to_arrow_table(combined_sample_data)

                # Use mapped filename to avoide spaces/special characters
                filename = file_names.get(s, f"{s.replace(' ', '_').replace('$', '').replace(',', '').replace('(', '').replace(')', '')}.parquet")


                # Define the output file path for this sample
                data_path = os.path.join(output_dir, filename)

                # Save to a Parquet file
                pq.write_table(arrow_table, data_path)

                print(f"{s} processed and saved to {data_path}")

    print(f"All samples processed and saved in {output_dir}")

######################################################################################
# Function to create and save plot of processed data

def plot(lumi=10, fraction=1.0):
    """
    Function to create plot.

    Args:
        lumi (int): Luminosity (default is 10 for all data)
        fraction (float): Controls the fraction of all events analysed (default is 1.0 reduce for quicker runtime, applied in loop over tree)
    
    Returns:
        (matplotlib.pyplot): plot
    """
    # Read in data
    file_paths = {
        "data": "/hzz-analysis/output/data.parquet",
        "Background $Z,t\\bar{t}$": "/hzz-analysis/output/Background_Zt\\bar{t}.parquet",
        "Background $ZZ^*$": "/hzz-analysis/output/background_ZZ.parquet",
        "Signal ($m_H$ = 125 GeV)": "/hzz-analysis/output/signal_125GeV.parquet"
    }

    all_data = {}

    for sample_name, file_path in file_paths.items():
        print(f"Reading {file_path}...")
        
        # Read Parquet file into a PyArrow Table
        arrow_table = pq.read_table(file_path)

        # Covert PyArrow Table to Awkward Array
        awkward_array = ak.from_arrow(arrow_table)

        # Store in dictionary
        all_data[sample_name] = awkward_array

    print("All files loaded into Awkward Arrays.")
    
               
    # x-axis range of the plot 
    xmin = 80 * GeV
    xmax = 250 * GeV

    # Histogram bin setup
    step_size = 5 * GeV
    bin_edges = np.arange(start=xmin, # The interval includes this value
                          stop=xmax+step_size, # The interval doesn't include this value
                          step=step_size ) # Spacing between values
    bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                            stop=xmax+step_size/2, # The interval doesn't include this value
                            step=step_size ) # Spacing between values
    
    # Histogram the data
    data_x,_ = np.histogram(ak.to_numpy(all_data['data']['mass']),
                            bins=bin_edges)
    # Statistical error on the data
    data_x_errors = np.sqrt(data_x)

    signal_x = ak.to_numpy(all_data[r'Signal ($m_H$ = 125 GeV)']['mass']) # histogram the signal
    signal_weights = ak.to_numpy(all_data[r'Signal ($m_H$ = 125 GeV)'].totalWeight)      # get the weights of the signal events
    signal_color = samples[r'Signal ($m_H$ = 125 GeV)']['color'] # get the colour for the signal bar

    mc_x = [] # define list to hold the Monte Carlo histogram entries
    mc_weights = [] # define list to hold the Monte Carlo weights
    mc_colors = [] # define list to hold the colors of the Monte Carlo bars
    mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

    for s in samples: # loop over samples
        if s not in ['data', r'Signal ($m_H$ = 125 GeV)']: # if not data nor signal
            mc_x.append( ak.to_numpy(all_data[s]['mass']) ) # append to the list of Monte Carlo histogram entries
            mc_weights.append( ak.to_numpy(all_data[s].totalWeight) ) # append to the list of Monte Carlo weights
            mc_colors.append( samples[s]['color'] ) # append to the list of Monte Carlo bar colors
            mc_labels.append( s ) # append to the list of Monte Carlo legend labels

    # Main plot 
    main_axes = plt.gca() # get current axes

    # plot the data points
    main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                       fmt='ko', # 'k' means black and 'o' is for circles
                       label='Data') 

    # plot the Monte Carlo bars
    mc_heights = main_axes.hist(mc_x, bins=bin_edges, 
                                weights=mc_weights, stacked=True, 
                                color=mc_colors, label=mc_labels )

    mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value

    # calculate MC statistical uncertainty: sqrt(sum w^2)
    mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges,
                                    weights=np.hstack(mc_weights)**2)[0])

    # plot the signal bar
    signal_heights = main_axes.hist(signal_x, bins=bin_edges, bottom=mc_x_tot,
                                    weights=signal_weights, color=signal_color,
                                    label=r'Signal ($m_H$ = 125 GeV)')

    # plot the statistical uncertainty
    main_axes.bar(bin_centres, # x
                  2*mc_x_err, # heights
                  alpha=0.5, # half transparency
                  bottom=mc_x_tot-mc_x_err, color='none', 
                  hatch="////", width=step_size, label='Stat. Unc.' )

    # set the x-limit of the main axes
    main_axes.set_xlim( left=xmin, right=xmax ) 

    # separation of x axis minor ticks
    main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 

    # set the axis tick parameters for the main axes
    main_axes.tick_params(which='both', # ticks on both x and y axes
                          direction='in', # Put ticks inside and outside the axes
                          top=True, # draw ticks on the top axis
                          right=True ) # draw ticks on right axis

    # x-axis label
    main_axes.set_xlabel(r'4-lepton invariant mass $\mathrm{m_{4l}}$ [GeV]',
                         fontsize=13, x=1, horizontalalignment='right' )

    # write y-axis label for main axes
    main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                         y=1, horizontalalignment='right') 

    # set y-axis limits for main axes
    main_axes.set_ylim( bottom=0, top=np.amax(data_x)*1.6 )

    # add minor ticks on y-axis for main axes
    main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 

    # Add text 'ATLAS Open Data' on plot
    plt.text(0.05, # x
             0.93, # y
             'ATLAS Open Data', # text
             transform=main_axes.transAxes, # coordinate system used is that of main_axes
             fontsize=13 ) 

    # Add text 'for education' on plot
    plt.text(0.05, # x
             0.88, # y
             'for education', # text
             transform=main_axes.transAxes, # coordinate system used is that of main_axes
             style='italic',
             fontsize=8 ) 

    # Add energy and luminosity
    lumi_used = str(lumi*fraction) # luminosity to write on the plot
    plt.text(0.05, # x
             0.82, # y
             '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
             transform=main_axes.transAxes ) # coordinate system used is that of main_axes

    # Add a label for the analysis carried out
    plt.text(0.05, # x
             0.76, # y
             r'$H \rightarrow ZZ^* \rightarrow 4\ell$', # text 
             transform=main_axes.transAxes ) # coordinate system used is that of main_axes

    # draw the legend
    main_axes.legend( frameon=False ) # no box around the legend

    # Save plot
    output_dir = "/hzz-analysis/output"
    os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist 
    plot_path = os.path.join(output_dir, "plot.png")
    plt.savefig(plot_path)
    plt.close()

    # Signal stacked height
    signal_tot = signal_heights[0] + mc_x_tot

    # Peak of signal
    print(signal_tot[8])

    # Neighbouring bins
    print(signal_tot[7:10])

    # Signal and background events
    N_sig = signal_tot[7:10].sum()
    N_bg = mc_x_tot[7:10].sum()

    # Signal significance calculation
    signal_significance = N_sig/np.sqrt(N_bg + 0.3 * N_bg**2) 
    print(f"\nResults:\n{N_sig = :.3f}\n{N_bg = :.3f}\n{signal_significance = :.3f}")
    print(f"Plot saved to {plot_path}")
    
#####################################################################################
# Main analysis

def main(lumi=10, fraction=1.0, function=None):
    """
    Main function to either process data or plot.

    Args:
        lumi (int): Luminosity (default is 10 for all data)
        fraction (float): Controls the fraction of all events analysed (default is 1.0 reduce for quicker runtime, applied in loop over tree)
        function (string): Function to be called.

    Returns:
        (matplotlib.pyplot) or (ak.array)
    """
    if function == 'process':
        process(lumi, fraction)
    elif function == 'plot':
        plot(lumi, fraction)
    else:
        raise ValueError("Invalid function. Must be 'process' or 'plot'.")


####################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the HZZ analysis process or plot.")
    parser.add_argument('function', choices=['process', 'plot', ], help="Function to run")
    args = parser.parse_args()

    main(function=args.function)


