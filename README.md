# Programs created @ SLAC for HVeV analysis
StitchIVs is an importable program that handles the standard data analysis takes for analyzing the electronic behavior of different transition edge sensor (TES) devices under a bias voltage. The program corrects for quantum flux jumps due to the effects of the superconducting quantum interference devices (SQUIDs) in the circuit. The quantum jumps are corrected by interpolating outlier differences between data. 
Futhermore, the program has a main function, plot_sweep, to handle plotting current, resistance, and power through a TES as a function of bias voltage with quantum flux jump correction. 

PlottingIVRVPV is an example notebook of running StitchIVs on OLAF 11 run data.
