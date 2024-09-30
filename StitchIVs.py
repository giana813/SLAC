# Imports
import numpy as np
import matplotlib.pyplot as plt
import glob
import cdms

# Constants
mA = 1e-3
OHM = 1
uOHM = 1e-6
mOHM = 1e-3
A2uA = 1e6
uA2A = 1e-6
V2nV = 1e9
W2pW = 1e12
ADC_BITS = 16
R_FB = 5000*OHM
R_CABLE = 0*OHM 
R_COLD_SHUNT = 15 * mOHM
ADC_GAIN = 2
ADC_RANGE = 8
R_PARA = 15*mOHM
R_TOTAL = R_CABLE + R_FB
M_FB = 2.4
ADC2A = 1/2**ADC_BITS *ADC_RANGE/(R_FB+R_CABLE) /M_FB/ADC_GAIN
FLUX_JUMP_DETECTION_THRESHOLD = 20e-6
SUPERCONDUCTING_RANGE = 6
EPSILON = 1e-15

# MIDAS channel, names, and colors
chs = ['PBS1', 'PAS1', 'PCS1', 'PFS1', 'PDS1', 'PES1', 'PBS2', 'PFS2', 'PES2', 'PAS2', 'PDS2', 'PCS2']
cs = ["#4A90E2", "#50E3C2", "#B8E986", "#F5A623", "#D0021B",  "#7B92A5", "#BD10E0", "#F8E71C", "#D1D8E0","#9B9B9B","#F4A7B9", "#E94F77"]
NAMES = ['NW A', 'NW B', 'TAMU A', 'TAMU B', 'SiC squares A', 'SiC squares B', 'SiC NFH A', 'SiC NFH B', 'SiC NFC1 A', 'SiC NFC1 B', 'SiC NFC2 A', 'SiC NFC2 B']
det = 1 

# Function to retrieve data from DB
def getibis(datadir, rns, det, verboseFlag):
    ibis = np.zeros((len(rns), 12, 2))
    for i in range(len(rns)):
        rn = rns[i]
        rn_str = f'{rn:05}' # Format the run number to be five digits long with leading zeros
        fns = glob.glob(f'{datadir}RUN{rn_str}*.gz')
        myreader = cdms.rawio.IO.RawDataReader(filepath=fns, verbose=verboseFlag)
        events = myreader.read_events(output_format=1, skip_empty=True, phonon_adctoamps=True)
        series = events.index[0][0]
        evns = events.loc[series].index
        odb = myreader.get_full_odb()
        
        for tes in range(12):
            ch = chs[tes]
            meds = []
            for j in range(40):
                try:
                    evn = evns[j]
                    trace = events.loc[series].loc[evn].loc[(f'Z{det}', ch)]
                    meds.append(np.median(trace))
                except:
                    continue
            qetbias = odb[f'/Detectors/Det0{det}/Settings/Phonon/QETBias (uA)'][tes]
            ibis[i, tes, 0] = qetbias
            ibis[i, tes, 1] = np.median(meds)
    
    return ibis

# Function to plot raw vb vs isig
def none(vb, isig):
    return vb, isig

# Retrieve OLAF run number
def extract_runNumber(datadir): # Extract the run number from the datadir path
    parts = datadir.rstrip('/').split('/')
    return parts[-1]

# Finds data points two sigma outside of the center of the distribution
def find_two_sigma_outliers(data):
    data = np.array(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    lower_bound = mean - 2 * std_dev
    upper_bound = mean + 2 * std_dev
    outliers = [(i, x) for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
    return outliers

# Stitched flux jumps by finding outliers in the differences (ignoring the transition jump)
def stitch_by_diffs(vb,isig): 
    diffs = np.diff(isig) 
    outliers = find_two_sigma_outliers(diffs)
    transition_index, transition_vb = find_SC_transition(vb, isig)
    flux_jumps = outliers.copy()
    if transition_index != 0 and transition_index != None and len(flux_jumps) > 0 and transition_index < len(flux_jumps):
        flux_jumps.pop(transition_index)
    isig_stitched = isig.copy()
    for flux_jump in flux_jumps:
        isig_stitched[flux_jump[0]+1:] = isig_stitched[flux_jump[0]+1:] - flux_jump[1] 
    return vb, isig_stitched

# Finds outliers in the data determined on a threshold
def detect_outliers_std(data, threshold=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    outliers = [x for x in data if abs(x - mean) > threshold * std_dev]
    return outliers

# Finds the transition of the superconducting branch to the normal branch in IV curve
def find_SC_transition(vb, isig, A2uA=1.0):
    if len(isig) < 3:
        return -1, None  # Not enough data for derivative analysis

    dvb = np.diff(vb)
    disig = np.diff(isig)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        first_derivative = np.divide(disig, dvb, out=np.zeros_like(disig), where=np.abs(dvb) > 0)
    
    dfirst_derivative = np.diff(first_derivative)
    dvb2 = np.diff(vb[:-1])  # Adjust length to match first_derivative
    with np.errstate(divide='ignore', invalid='ignore'):
        second_derivative = np.divide(dfirst_derivative, dvb2, out=np.zeros_like(dfirst_derivative), where=np.abs(dvb2) > 0)

    transition_indices = []
    for i in range(1, len(second_derivative)):
        if second_derivative[i] < 0 and second_derivative[i - 1] >= 0:
            transition_indices.append(i)

    if not transition_indices:
        return -1, None  # No significant transition found
    
    # Determine the most significant transition based on the magnitude of the second derivative
    magnitudes = np.abs(second_derivative[transition_indices])
    most_significant_index = transition_indices[np.argmax(magnitudes)]
    
    if most_significant_index + 1 >= len(vb):
        return -1, None
    
    transition_vb = vb[most_significant_index + 1]
    
    if transition_vb == 0:
        return -1, None

    transition_vb = abs(transition_vb)
        
    return most_significant_index + 1, transition_vb

STITCHING_METHODS = {"JumpBuster":JumpBuster, "none":none}

# Finds the slope of the superconducting branch to determine the parasitic resistance
def find_slope_sc_branch(vb, isig, SC_trans_index):
    # Superconducting branch data extraction
    sc_vb = vb[:SC_trans_index+1]
    sc_vb = [v * 1e2 for v in sc_vb]  # Convert vb to proper units (V)
    
    # A2uA = 1e6  # Conversion factor for current (A to µA)
    sc_isig = [i * A2uA for i in isig[:SC_trans_index+1]]
    
    # Check if we have enough data points
    if len(sc_vb) < 2 or len(sc_isig) < 2:
        # print("Not enough data points for fitting.")
        return 0  # or return a default slope

    # Check if all values are the same
    if np.all(np.array(sc_vb) == sc_vb[0]) or np.all(np.array(sc_isig) == sc_isig[0]):
        # print("Insufficient variation in data points for fitting.")
        return 0  # or return a default slope

    # Calculate the slope of the superconducting branch
    try:
        slope, _ = np.polyfit(sc_vb, sc_isig, 1)
    except np.linalg.LinAlgError as e:
        # print(f"Linear fit error: {e}")
        return 0  # or return a default slope

    return slope  # returns slope of superconducting branch adjusted for unit conversions

#  Function for plotting current, resistance, or power through the tes versus the bias voltage
def plot_sweep(ibis, datadir, rns, exclude, include, stitch_type="", plot_type="", axs=None):
    # Title/ Table Info
    runNumber = extract_runNumber(datadir)
    start = rns[0]
    end = rns[-1]
    TES = []
    SC_VB = []

    # Determine plot types
    plot_types = plot_type.split('+')
    num_plots = len(plot_types)
    
    # Create subplots
    if num_plots == 1:
        fig, ax = plt.subplots(figsize=(8, 6)) 
        axs = ax 
    else:
        fig, axs = plt.subplots(1, num_plots, figsize=(5*num_plots, 4), sharey=False)
        axs = np.array(axs)  

    if num_plots == 1:
        axs = [axs] 

    for ax in axs:
        ax.axvline(x=0, color='black', linestyle='--')
        ax.axhline(y=0, color='black', linestyle='--')
    
    for tes in range(len(NAMES)):
        if NAMES[tes] in include:
            # Extract and convert vb and isig
            I_bias = ibis[:, tes, 0]
            vb = I_bias * R_COLD_SHUNT * 1e2 # Convert I_b to v_b by value of shunt resistor
            isig = ibis[:, tes, 1] * A2uA
        
            SC_trans_index, transition_vb = find_SC_transition(vb, isig)

            if NAMES[tes] in exclude or np.all(vb == 0) or np.all(isig == 0):
                TES.append(NAMES[tes])
                SC_VB.append("N/A")
                continue  # Skip further processing for excluded TES
            
            TES.append(NAMES[tes])

            if np.all(vb == 0) or np.all(isig == 0):  # Check if data is logical, if not, skip
                SC_VB.append("N/A")
                continue
            
            trans = "N/A"
            if transition_vb is not None:
                trans = float(transition_vb)
                SC_VB.append(f"{trans:.2f}" if trans != 0.00 else "N/A")
            else:
                SC_VB.append(trans)
            
            # Apply the stitching method if necessary
            if stitch_type in STITCHING_METHODS and stitch_type != "none":
                vb, isig = STITCHING_METHODS[stitch_type](vb, isig)
            elif stitch_type != "none":
                print("Not a valid stitching method.")

            if np.all(isig == 0):  # Check if data is logical, if not, skip
                SC_VB.pop()
                SC_VB.append("N/A")
                continue
            
            # Plotting based on plot_type
            for i, ptype in enumerate(plot_types):
                ax = axs[i]
                
                if ptype == "iv": 
                    ax.plot(vb, isig, '.', color=cs[tes], label=NAMES[tes])  # Plot I-V curve
                    ax.set_ylabel(r'Measured TES branch current ($\mu$A)')
                    ax.set_xlabel(r'TES bias (V)') 
                    # ax.set_xlim(vb.min() * 1e2, vb.max() * 1e2)

                ####### MATH #######
                I_tes = np.where(isig != 0, isig, 1e-16)  # Avoid div by zero error by setting tiny value
                min_length = min(len(vb), len(I_tes))  # Find minimum length to avoid mismatch
                
                if len(vb) != min_length:
                    vb = vb[:min_length]
                if len(I_tes) != min_length:
                    I_tes = I_tes[:min_length]

                slope = find_slope_sc_branch(vb, isig, SC_trans_index)

                # Calculate TES resistance, subtract the SC slope (adjusting units as needed)
                r_tes = (R_COLD_SHUNT) * ((I_bias[:min_length] / I_tes) - 1) - (R_PARA)
                for i in range(len(r_tes)):
                    r_tes[i] = (r_tes[i] - slope)
                # r_tes -= slope  # Adjust slope units to MΩ before subtracting
                # r_tes *= 1e-6 # Megaohms

                if len(r_tes) != len(vb):
                    r_tes = r_tes[:len(vb)]  # Adjust shape to match vb

                # Calculate power through TES
                power_tes = (isig[:len(vb)] ** 2 * r_tes)  # Power through TES
                ####### MATH #######
                
                if ptype == "rv":  # Plot resistance through TES
                    # r_tes *= 1e-6  # Convert to MΩ
                    lower_y_lim = np.percentile(r_tes, 10)
                    upper_y_lim = np.percentile(r_tes, 90)
                    y_val = max(abs(lower_y_lim), abs(upper_y_lim))
                
                    ax.plot(vb[:-1], abs(r_tes[:-1]), '.', color=cs[tes], label=NAMES[tes])  # Plot R-V curve
                    ax.set_ylabel(r'R(M$\Omega$)')
                    ax.set_xlabel(r'TES bias (V)')
                    # ax.set_xlim(vb.min(), vb.max())  # Convert x-limits
                    # ax.set_ylim(0, y_val)
                
                if ptype == "pv":  # Power through TES
                    power_tes *= 1e6  # Convert to µW
                    lower_y_lim = np.percentile(power_tes, 5)
                    upper_y_lim = np.percentile(power_tes, 95)
                    y_val = max(abs(lower_y_lim), abs(upper_y_lim))
                
                    ax.plot(vb, power_tes, '.', color=cs[tes], label=NAMES[tes])
                    ax.set_ylabel(r'P (µW)')
                    ax.set_xlabel(r'TES bias (V)')
                    ax.set_xlim(vb.min(), vb.max())
                    ax.set_ylim(0, y_val)

            # Remove legend from plot if it exists
            for ax in axs:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
        
        else: 
            TES.append(NAMES[tes])
            SC_VB.append("N/A")

    # Plot the table, regardless of plot type
    TES.append("Stitch Type")
    SC_VB.append(stitch_type)
    table_data = [[TES[i], SC_VB[i]] for i in range(len(TES))]
    
    # Add the table to the first axis
    table = axs[0].table(cellText=table_data, colLabels=['TES', 'Transition (V)'],
                         cellLoc='center', loc='bottom', bbox=[-1, 0, .7, 1])  # Adjust bbox values
    
    # Color the table cells based on TES colors
    for i, key in enumerate(TES):
        if key == "Stitch Type":
            continue
        row_index = i
        cell = table[(row_index + 1, 0)]  # +1 to skip the header
        color = cs[i] if i < len(cs) else 'white'  # Ensure there's a color for each row
        cell.set_facecolor(color)  # Set cell background color
        cell.set_text_props(color='black')  # Ensure text is visible
    
    # Set title
    plt.suptitle(f"{runNumber}: Runs {start}-{end}")
    if num_plots != 1:
        plt.tight_layout(rect=[0, .01, 1, 0.99]) 
        plt.subplots_adjust(wspace=0.35) 
    plt.show()
    # return axs
