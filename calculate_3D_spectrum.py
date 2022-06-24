#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:53:07 2020

@author: fnal
"""

import inspect
import subprocess
import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
from scipy import interpolate
from scipy import stats
from scipy import optimize
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import re
import glob
from collections import defaultdict
import itertools
import math
import copy
from datetime import datetime
from IPython import get_ipython
import time

#%% class Spectrum
# class representing a measurement (shot + attenuators + contours) and
# its deconvoluted spectrum

class Spectrum:
  #%% __init__: Class initializer
  def __init__(self, shot_dir, measurements_dir, default_spectrum_file, max_PSL_on_Reconstruction_Input, max_PSL_on_Sampling_image):
    self.shot_dir = shot_dir
    self.measurements_dir = measurements_dir
    self.default_spectrum_filename = default_spectrum_file + ".txt"
    self.max_PSL_on_Reconstruction_Input = max_PSL_on_Reconstruction_Input
    self.max_PSL_on_Sampling_image = max_PSL_on_Sampling_image
    
    self.base_dir = "/Users/fnal/Documents/Research/LWFA/LWFA 4.0"
    self.all_shots_base_dir = self.base_dir + '/' + "analysis 4.0/preliminary x-ray analysis"
    self.analysis_dir = self.base_dir + '/' + "analysis 4.0/x-ray analysis/6. Python implementation"
    self.outputs_dir = self.analysis_dir + '/' + "Calculate 3D Spectrum/outputs"
    self.plots_dir = self.analysis_dir + '/' + "Calculate 3D Spectrum/plots"
    self.this_shot_base_dir = self.all_shots_base_dir + '/' + self.shot_dir
    self.measurements_base_dir = self.this_shot_base_dir + '/' + self.measurements_dir
    self.default_spectrum_file_path = self.this_shot_base_dir + '/' + self.default_spectrum_filename
    
    os.chdir(self.measurements_base_dir)
    contours_byte = subprocess.run(["ls | cut -d',' -f2 | sort --numeric-sort -u | grep -v stats"], capture_output=True, shell=True)

    # contours levels can be integer or float so keep track of them as string 
    self.contours = contours_byte.stdout.decode("utf-8").split('\n')
    # remove empty entry created by the last '\n' character returned by the Unix ls command
    self.contours.pop()
    self.display_statistics = False
    # unpooled attenuators not implemented
    self.pool_attenuators = True
    self.plot_inputs = False
    self.num_detectors = {}
    self.spectrum = {}
    self.default_spectrum_on_contour = {}
    self.debug =  False

  #%% calculate_3D_specrum(): Top-level function
  def calculate_3D_specrum(self):
    # Function hierarchy:
    # calculate_3D_spectrum() [single- or multi-contour]
    # 	calculate_response_functions() [always multi-contour]
    #   map_attenuator_to_response_function() [always multi-contour]
    #   read_measurements() (single- or multi-contour]
    #     pool_detector_data() [single- or multi-contour]
    # 
    # 	interpolate_default_spectrum() [always multi-contour]
    #   for contour in contours
    #     calculate_spectrum_on_contour() [single- or multi-contour]
    #   end for
    # 
    #   interpolate_deconvoluted_spectrum() [single- or multi-contour]
    #   do a comparison of the default and the deconvoluted spectrum
    print(inspect.currentframe().f_code.co_name + ": working out of the directory '" + self.measurements_dir + "'")

    self.calculate_response_functions()
    self.read_measurements()
    if (self.display_statistics):
      self.display_statistics()
    self.display_contour_errors()
    self.interpolate_default_spectrum()
    
    print(f'\nEnergies [keV]: {self.energies}\n')
#   for contour in self.measurements.keys():    
#   for contour in reversed(self.measurements.keys()):
#   for contour in ['30.63', '34.29']:
    for contour in ['11.11']:
#   for contour in ['12.33']:
#    for contour in ['231.3']:
      self.calculate_spectrum_on_contour(contour)
    
    energies_file = Path(self.outputs_dir + "/" + datetime.today().strftime('%Y%m%d') + " - energies.txt")
    energies_file.write_text(json.dumps(self.energies.tolist()))

    spectrum_file = Path(self.outputs_dir + "/" + datetime.today().strftime('%Y%m%d') + " - full spectrum (NN default spectrum).txt")
    spectrum_file.write_text(json.dumps(self.spectrum))
     
    self.interpolate_deconvoluted_spectrum()

  #%% calculate_response_functions(): Calculates all response functions
  # (called from the top level)
  def calculate_response_functions(self):
    # Steps to reproduce the response functions:
    # 
    # 0) Imports densities
    # 1) Imports the mass attenuations
    # 2) Hardcodes the thicknesses
    # 3) Construct the response functions using the data above and formulas in "all residual sensitivities and filter thicknesses.pdf". 
    # For the MS IP response function, includes extrapolation in Matlab generating it in Mathematica and then exporting to text file and reading that in Matlab.
    # To use the profile as an attenuator, use ResponseWakefieldOffFilterMS.
    # Constructs the response function using information from the work on 09/03/2020
    print(inspect.currentframe().f_code.co_name + ": working out of the directory '" + self.measurements_dir + "'")

    #%% 0) Import densities, either from existing file or from NIST web site
    
    densities_file = Path(self.analysis_dir + "/densities.txt")
    if densities_file.is_file():
      densities = json.loads(densities_file.read_text())
    else:
      # get it from the web site
      densities_url = "http://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html"
      densities_dfs = pd.read_html(densities_url)
      densities_df = densities_dfs[0]
      densities_df = pd.DataFrame(densities_df.iloc[3:95,[0,1,9]])
      elements_df = pd.DataFrame(densities_df[1][:] + densities_df[0][:])
      densities_df = elements_df.join(densities_df[9])
      densities_df.set_index(0, inplace=True)
      densities_str = densities_df.T.to_dict('records')[0]
      densities = dict([key, float(value)]  
                       for key, value in densities_str.items())  
      densities['PEEK'] = 1.32 # from Wikipedia 
      densities_file.write_text(json.dumps(densities))
      
    print(densities)

    #%% 1) Imports mass attenuations, either from existing files or from NIST web site, and interpolates over them 

    # Al 13, Ti 22, Cu 29, Zn 30, Zr 40, Mo 42, Ag 47, Sn 50, Gd 64, Pb 82
    materials = ['Al13', 'Ti22', 'Cu29', 'Zn30', 'Zr40', 'Mo42', 'Ag47', 'Sn50', 'Gd64', 'Pb82', 'C19H12O3 (PEEK)']
    attenuations = {} 
    for ii in range(len(materials)):
      attenuations_file = Path(self.analysis_dir + "/attenuations/attenuations_" + materials[ii] + ".txt")
      if attenuations_file.is_file():
        #attenuations[materials[ii]] = json.loads(attenuations_file.read_text())
        attenuations[materials[ii]] = pd.read_csv(attenuations_file, delimiter='\t')
      else:
        attenuations_base_url = "https://physics.nist.gov/cgi-bin/ffast/ffast.pl"
        if (materials[ii][0:8] == 'C19H12O3'):
          parameters = '?Formula=' + materials[ii][0:8]
        else:
          parameters = '?Z=' + materials[ii][2:]
        parameters = parameters + "&gtype=3&range=S&lower=0.200&upper=311&density=&frames=no&htmltable=1"
        attenuations_url = attenuations_base_url + parameters
        attenuations_dfs = pd.read_html(attenuations_url)
        attenuations_df = attenuations_dfs[0]
        attenuations_df.to_csv(attenuations_file, delimiter='\t', index=False)
        attenuations[materials[ii]] = attenuations_df
        print(attenuations_df)
      
    #%% 2) Get hardcoded thicknesses
    
    thickness = self.get_all_thicknesses()
    
    #%% 3) Read IP calibration data
    
    IP_calibration_filename = "IP calibration - x-ray - FLA 7000 - MS - Maddox.txt"
    IP_calibration_file = Path(self.analysis_dir + '/' + IP_calibration_filename)
    if IP_calibration_file.is_file():
      MScalibdataMaddox100um = pd.read_csv(IP_calibration_file, delimiter='\t', names=['E(keV)','Sensitivity mPSL/photon'])
      MaddoxStepSizeMultiplierMS100um = 1.38
      MScalibdataMaddox100um['Sensitivity mPSL/photon'] = MaddoxStepSizeMultiplierMS100um * MScalibdataMaddox100um['Sensitivity mPSL/photon']
    else:
      sys.exit("The file '" + self.analysis_dir + '/' + IP_calibration_filename + "' not found.")
    
    #%% 4) create the energy grid

    edge_energies = self.make_energy_grid()
    energies_up_to_100keV = np.concatenate((edge_energies, np.arange(np.ceil(max(edge_energies)), 101, 1)))
    energies_beyond_100keV = np.arange(101, 311, 1)
    energies = np.concatenate((energies_up_to_100keV, energies_beyond_100keV))
    self.energies = energies

    #%% 5) interpolate mass attenuation data
    
    MassAttenuationPEEKf = interpolate.interp1d(attenuations['C19H12O3 (PEEK)']['E(keV)'], attenuations['C19H12O3 (PEEK)']['[µ/ρ] Total cm2 g-1'], kind='linear', fill_value='extrapolate')
    MassAttenuationPEEK  = MassAttenuationPEEKf(energies)

    MassAttenuationAl13f = interpolate.interp1d(attenuations['Al13']['E(keV)'], attenuations['Al13']['[µ/ρ] Total cm2 g-1'], kind='linear', fill_value='extrapolate')
    MassAttenuationAl13  = MassAttenuationAl13f(energies)
    
    MassAttenuationCu29f = interpolate.interp1d(attenuations['Cu29']['E(keV)'], attenuations['Cu29']['[µ/ρ] Total cm2 g-1'], kind='linear', fill_value='extrapolate')
    MassAttenuationCu29  = MassAttenuationCu29f(energies)

    MassAttenuationTi22f = interpolate.interp1d(attenuations['Ti22']['E(keV)'], attenuations['Ti22']['[µ/ρ] Total cm2 g-1'], kind='linear', fill_value='extrapolate')
    MassAttenuationTi22  = MassAttenuationTi22f(energies)

    MassAttenuationZn30f = interpolate.interp1d(attenuations['Zn30']['E(keV)'], attenuations['Zn30']['[µ/ρ] Total cm2 g-1'], kind='linear', fill_value='extrapolate')
    MassAttenuationZn30  = MassAttenuationZn30f(energies)

    MassAttenuationZr40f = interpolate.interp1d(attenuations['Zr40']['E(keV)'], attenuations['Zr40']['[µ/ρ] Total cm2 g-1'], kind='linear', fill_value='extrapolate')
    MassAttenuationZr40  = MassAttenuationZr40f(energies)

    MassAttenuationMo42f = interpolate.interp1d(attenuations['Mo42']['E(keV)'], attenuations['Mo42']['[µ/ρ] Total cm2 g-1'], kind='linear', fill_value='extrapolate')
    MassAttenuationMo42  = MassAttenuationMo42f(energies)

    MassAttenuationAg47f = interpolate.interp1d(attenuations['Ag47']['E(keV)'], attenuations['Ag47']['[µ/ρ] Total cm2 g-1'], kind='linear', fill_value='extrapolate')
    MassAttenuationAg47  = MassAttenuationAg47f(energies)

    MassAttenuationSn50f = interpolate.interp1d(attenuations['Sn50']['E(keV)'], attenuations['Sn50']['[µ/ρ] Total cm2 g-1'], kind='linear', fill_value='extrapolate')
    MassAttenuationSn50  = MassAttenuationSn50f(energies)

    MassAttenuationGd64f = interpolate.interp1d(attenuations['Gd64']['E(keV)'], attenuations['Gd64']['[µ/ρ] Total cm2 g-1'], kind='linear', fill_value='extrapolate')
    MassAttenuationGd64  = MassAttenuationGd64f(energies)

    MassAttenuationPb82f = interpolate.interp1d(attenuations['Pb82']['E(keV)'], attenuations['Pb82']['[µ/ρ] Total cm2 g-1'], kind='linear', fill_value='extrapolate')
    MassAttenuationPb82  = MassAttenuationPb82f(energies)

    #%% 6) interpolate IP calibration data

    zero = pd.DataFrame([[0, 0]], columns=['E(keV)', 'Sensitivity mPSL/photon'])
    MScalibdataMaddox100um = zero.append(MScalibdataMaddox100um, ignore_index=True)    
    MScalibMaddox100umf = interpolate.interp1d(MScalibdataMaddox100um['E(keV)'], MScalibdataMaddox100um['Sensitivity mPSL/photon'], kind='linear', fill_value='extrapolate')
    MScalibMaddox100um = MScalibMaddox100umf(energies_up_to_100keV)
    extrapolated_sensitivities = self.extrapolation_handler(energies_beyond_100keV)
    # 0.001 factor converts from mPSL to PSL
    MScalibMaddox100um = 0.001 * np.concatenate((MScalibMaddox100um, extrapolated_sensitivities))

    #%% 7) Response functions

    # This is the profile region where contour crosses region between filters
    # (x-ray goes through full PEEK layer + Al)
    self.ResponseWakefieldOffFilterMS = MScalibMaddox100um * \
                                   np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                                   np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronFullPEEK'] * 0.0001)

    # This is for the thin PEEK layer when contour crosses empty hole on mask
    # (x-ray goes through thin PEEK thickness + Al)
    self.ResponseWakefieldNoFilterMS = MScalibMaddox100um * \
                                   np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                                   np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)

    # ResponseWakefieldOffFilterSR = SRcalibMaddox50um * \
    #                                np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
    #                                np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronFullPEEK'] * 0.0001)

    self.ResponseWakefield1 = MScalibMaddox100um * \
                         np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                         np.exp(-MassAttenuationTi22 * densities['Ti22'] * thickness['ThicknessMicronWakefield1'] * 0.0001) * \
                         np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield1Cu'] * 0.0001) * \
                         np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    self.ResponseWakefield2 = MScalibMaddox100um * \
                         np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                         np.exp(-MassAttenuationZn30 * densities['Zn30'] * thickness['ThicknessMicronWakefield2'] * 0.0001) * \
                         np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield2Cu'] * 0.0001) * \
                         np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)  
    ResSensitivityWakefield2to1 = self.ResponseWakefield2 - self.ResponseWakefield1
    ResSensitivityPerUnitBW2to1 = np.divide(ResSensitivityWakefield2to1, 0.001 * self.energies)
                                          
    self.ResponseWakefield3 = MScalibMaddox100um * \
                         np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                         np.exp(-MassAttenuationZn30 * densities['Zn30'] * thickness['ThicknessMicronWakefield3'] * 0.0001) * \
                         np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield3Cu'] * 0.0001) * \
                         np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    self.ResponseWakefield4 = MScalibMaddox100um * \
                         np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                         np.exp(-MassAttenuationZr40 * densities['Zr40'] * thickness['ThicknessMicronWakefield4'] * 0.0001) * \
                         np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield4Cu'] * 0.0001) * \
                         np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    ResSensitivityWakefield4to3 = self.ResponseWakefield4 - self.ResponseWakefield3
    ResSensitivityPerUnitBW4to3 = np.divide(ResSensitivityWakefield4to3, 0.001 * self.energies)

    self.ResponseWakefield5 = MScalibMaddox100um * \
                         np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                         np.exp(-MassAttenuationZr40 * densities['Zr40'] * thickness['ThicknessMicronWakefield5'] * 0.0001) * \
                         np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield5Cu'] * 0.0001) * \
                         np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    self.ResponseWakefield6 = MScalibMaddox100um * \
                         np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                         np.exp(-MassAttenuationMo42 * densities['Mo42'] * thickness['ThicknessMicronWakefield6'] * 0.0001) * \
                         np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield6Cu'] * 0.0001) * \
                         np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    ResSensitivityWakefield6to5 = self.ResponseWakefield6 - self.ResponseWakefield5
    ResSensitivityPerUnitBW6to5 = np.divide(ResSensitivityWakefield6to5, 0.001 * self.energies)

    self.ResponseWakefield7 = MScalibMaddox100um * \
                         np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                         np.exp(-MassAttenuationMo42 * densities['Mo42'] * thickness['ThicknessMicronWakefield7'] * 0.0001) * \
                         np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield7Cu'] * 0.0001) * \
                         np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    self.ResponseWakefield8 = MScalibMaddox100um * \
                         np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                         np.exp(-MassAttenuationAg47 * densities['Ag47'] * thickness['ThicknessMicronWakefield8'] * 0.0001) * \
                         np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield8Cu'] * 0.0001) * \
                         np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    ResSensitivityWakefield8to7 = self.ResponseWakefield8 - self.ResponseWakefield7
    ResSensitivityPerUnitBW8to7 = np.divide(ResSensitivityWakefield8to7, 0.001 * self.energies)

    self.ResponseWakefield9 = MScalibMaddox100um * \
                         np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                         np.exp(-MassAttenuationAg47 * densities['Ag47'] * thickness['ThicknessMicronWakefield9'] * 0.0001) * \
                         np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield9Cu'] * 0.0001) * \
                         np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    self.ResponseWakefield10 = MScalibMaddox100um * \
                          np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                          np.exp(-MassAttenuationSn50 * densities['Sn50'] * thickness['ThicknessMicronWakefield10'] * 0.0001) * \
                          np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield10Cu'] * 0.0001) * \
                          np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    ResSensitivityWakefield10to9 = self.ResponseWakefield10 - self.ResponseWakefield9
    ResSensitivityPerUnitBW10to9 = np.divide(ResSensitivityWakefield10to9, 0.001 * self.energies)

    self.ResponseWakefield11 = MScalibMaddox100um * \
                          np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                          np.exp(-MassAttenuationSn50 * densities['Sn50'] * thickness['ThicknessMicronWakefield11'] * 0.0001) * \
                          np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield11Cu'] * 0.0001) * \
                          np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001) 
    self.ResponseWakefield12 = MScalibMaddox100um * \
                          np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                          np.exp(-MassAttenuationGd64 * densities['Gd64'] * thickness['ThicknessMicronWakefield12'] * 0.0001) * \
                          np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield12Cu'] * 0.0001) * \
                          np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    ResSensitivityWakefield12to11 = self.ResponseWakefield12 - self.ResponseWakefield11
    ResSensitivityPerUnitBW12to11 = np.divide(ResSensitivityWakefield12to11, 0.001 * self.energies)

    self.ResponseWakefield13 = MScalibMaddox100um * \
                          np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                          np.exp(-MassAttenuationGd64 * densities['Gd64'] * thickness['ThicknessMicronWakefield13'] * 0.0001) * \
                          np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield13Cu'] * 0.0001) * \
                          np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    self.ResponseWakefield14 = MScalibMaddox100um * \
                          np.exp(-MassAttenuationAl13 * densities['Al13'] * thickness['ThicknessMicronTotalDefaultAlFoil'] * 0.0001) * \
                          np.exp(-MassAttenuationPb82 * densities['Pb82'] * thickness['ThicknessMicronWakefield14'] * 0.0001) * \
                          np.exp(-MassAttenuationCu29 * densities['Cu29'] * thickness['ThicknessMicronWakefield14Cu'] * 0.0001) * \
                          np.exp(-MassAttenuationPEEK * densities['PEEK'] * thickness['ThicknessMicronPEEK'] * 0.0001)
    ResSensitivityWakefield14to13 = self.ResponseWakefield14 - self.ResponseWakefield13
    ResSensitivityPerUnitBW14to13 = np.divide(ResSensitivityWakefield14to13, 0.001 * self.energies)

    if (self.plot_inputs):
      next_image = self.plot_attenuations(0, 'C19H12O3 (PEEK)', attenuations['C19H12O3 (PEEK)']['E(keV)'], attenuations['C19H12O3 (PEEK)'][ '[µ/ρ] Total cm2 g-1'], MassAttenuationPEEK)
      next_image = self.plot_attenuations(next_image, 'Al13', attenuations['Al13']['E(keV)'], attenuations['Al13'][ '[µ/ρ] Total cm2 g-1'], MassAttenuationAl13)      
      next_image = self.plot_attenuations(next_image, 'Cu29', attenuations['Cu29']['E(keV)'], attenuations['Cu29'][ '[µ/ρ] Total cm2 g-1'], MassAttenuationCu29)      
      next_image = self.plot_attenuations(next_image, 'Ti22', attenuations['Ti22']['E(keV)'], attenuations['Ti22'][ '[µ/ρ] Total cm2 g-1'], MassAttenuationTi22)      
      next_image = self.plot_attenuations(next_image, 'Zn30', attenuations['Zn30']['E(keV)'], attenuations['Zn30'][ '[µ/ρ] Total cm2 g-1'], MassAttenuationZn30)      
      next_image = self.plot_attenuations(next_image, 'Zr40', attenuations['Zr40']['E(keV)'], attenuations['Zr40'][ '[µ/ρ] Total cm2 g-1'], MassAttenuationZr40)      
      next_image = self.plot_attenuations(next_image, 'Mo42', attenuations['Mo42']['E(keV)'], attenuations['Mo42'][ '[µ/ρ] Total cm2 g-1'], MassAttenuationMo42)      
      next_image = self.plot_attenuations(next_image, 'Ag47', attenuations['Ag47']['E(keV)'], attenuations['Ag47'][ '[µ/ρ] Total cm2 g-1'], MassAttenuationAg47)      
      next_image = self.plot_attenuations(next_image, 'Sn50', attenuations['Sn50']['E(keV)'], attenuations['Sn50'][ '[µ/ρ] Total cm2 g-1'], MassAttenuationSn50)      
      next_image = self.plot_attenuations(next_image, 'Gd64', attenuations['Gd64']['E(keV)'], attenuations['Gd64'][ '[µ/ρ] Total cm2 g-1'], MassAttenuationGd64)      
      next_image = self.plot_attenuations(next_image, 'Pb82', attenuations['Pb82']['E(keV)'], attenuations['Pb82'][ '[µ/ρ] Total cm2 g-1'], MassAttenuationPb82)      
        
      next_image = self.plot_response_function(next_image, 'MS IP Sensitivity', [MScalibMaddox100um, MScalibdataMaddox100um['E(keV)'], MScalibdataMaddox100um['Sensitivity mPSL/photon']])      
    
      next_image = self.plot_response_function(next_image, 'Off Filter (MS)', [self.ResponseWakefieldOffFilterMS])
      next_image = self.plot_response_function(next_image, 'No Filter (MS)', [self.ResponseWakefieldNoFilterMS])

      next_image = self.plot_response_function(next_image, 'Filter 1 (Al13+Ti22+Cu29+PEEK)', [self.ResponseWakefield1])
      next_image = self.plot_response_function(next_image, 'Filter 2 (Al13+Zn30+Cu29+PEEK)', [self.ResponseWakefield2])
      next_image = self.plot_response_function(next_image, 'Filter 1 (Al13+Ti22+Cu29+PEEK), Filter 2 (Al13+Zn30+Cu29+PEEK)', [self.ResponseWakefield1, self.ResponseWakefield2])
      next_image = self.plot_response_function(next_image, 'Filter 2 minus Filter 1', [ResSensitivityWakefield2to1], y_label='Residual Sensitivity')
      next_image = self.plot_response_function(next_image, 'Filter 2 minus Filter 1 (PUBW)', [ResSensitivityPerUnitBW2to1], y_label='Residual Sensitivity PUBW')

      next_image = self.plot_response_function(next_image, 'Filter 3 (Al13+Zn30+Cu29+PEEK)', [self.ResponseWakefield3])
      next_image = self.plot_response_function(next_image, 'Filter 4 (Al13+Zr40+Cu29+PEEK)', [self.ResponseWakefield4])
      next_image = self.plot_response_function(next_image, 'Filter 3 (Al13+Zn30+Cu29+PEEK), Filter 4 (Al13+Zr40+Cu29+PEEK)', [self.ResponseWakefield3, self.ResponseWakefield4])
      next_image = self.plot_response_function(next_image, 'Filter 4 minus Filter 3', [ResSensitivityWakefield4to3], y_label='Residual Sensitivity')
      next_image = self.plot_response_function(next_image, 'Filter 4 minus Filter 3 (PUBW)', [ResSensitivityPerUnitBW4to3], y_label='Residual Sensitivity PUBW')

      next_image = self.plot_response_function(next_image, 'Filter 5 (Al13+Zr40+Cu29+PEEK)', [self.ResponseWakefield5])
      next_image = self.plot_response_function(next_image, 'Filter 6 (Al13+Mo42+Cu29+PEEK)', [self.ResponseWakefield6])
      next_image = self.plot_response_function(next_image, 'Filter 5 (Al13+Zr40+Cu29+PEEK), Filter 6 (Al13+Mo42+Cu29+PEEK)', [self.ResponseWakefield5, self.ResponseWakefield6])
      next_image = self.plot_response_function(next_image, 'Filter 6 minus Filter 5', [ResSensitivityWakefield6to5], y_label='Residual Sensitivity')
      next_image = self.plot_response_function(next_image, 'Filter 6 minus Filter 5 (PUBW)', [ResSensitivityPerUnitBW6to5], y_label='Residual Sensitivity PUBW')

      next_image = self.plot_response_function(next_image, 'Filter 7 (Al13+Mo42+Cu29+PEEK)', [self.ResponseWakefield7])
      next_image = self.plot_response_function(next_image, 'Filter 8 (Al13+Ag47+Cu29+PEEK)', [self.ResponseWakefield8])
      next_image = self.plot_response_function(next_image, 'Filter 7 (Al13+Mo42+Cu29+PEEK), Filter 8 (Al13+Ag47+Cu29+PEEK)', [self.ResponseWakefield7, self.ResponseWakefield8])
      next_image = self.plot_response_function(next_image, 'Filter 8 minus Filter 7', [ResSensitivityWakefield8to7], y_label='Residual Sensitivity')
      next_image = self.plot_response_function(next_image, 'Filter 8 minus Filter 8 (PUBW)', [ResSensitivityPerUnitBW8to7], y_label='Residual Sensitivity PUBW')

      next_image = self.plot_response_function(next_image, 'Filter 9 (Al13+Ag47+Cu29+PEEK)', [self.ResponseWakefield9])
      next_image = self.plot_response_function(next_image, 'Filter 10 (Al13+Sn50+Cu29+PEEK)', [self.ResponseWakefield10])
      next_image = self.plot_response_function(next_image, 'Filter 9 (Al13+Ag47+Cu29+PEEK), Filter 10 (Al13+Sn50+Cu29+PEEK)', [self.ResponseWakefield9, self.ResponseWakefield10])
      next_image = self.plot_response_function(next_image, 'Filter 10 minus Filter 9', [ResSensitivityWakefield10to9], y_label='Residual Sensitivity')
      next_image = self.plot_response_function(next_image, 'Filter 10 minus Filter 9 (PUBW)', [ResSensitivityPerUnitBW10to9], y_label='Residual Sensitivity PUBW')

      next_image = self.plot_response_function(next_image, 'Filter 11 (Al13+Sn50+Cu29+PEEK)', [self.ResponseWakefield11])
      next_image = self.plot_response_function(next_image, 'Filter 12 (Al13+Gd64+Cu29+PEEK)', [self.ResponseWakefield12])
      next_image = self.plot_response_function(next_image, 'Filter 11 (Al13+Sn50+Cu29+PEEK), Filter 12 (Al13+Gd64+Cu29+PEEK)', [self.ResponseWakefield11, self.ResponseWakefield12])
      next_image = self.plot_response_function(next_image, 'Filter 12 minus Filter 11', [ResSensitivityWakefield12to11], y_label='Residual Sensitivity')
      next_image = self.plot_response_function(next_image, 'Filter 12 minus Filter 11 (PUBW)', [ResSensitivityPerUnitBW12to11], y_label='Residual Sensitivity PUBW')

      next_image = self.plot_response_function(next_image, 'Filter 13 (Al13+Gd64+Cu29+PEEK)', [self.ResponseWakefield13])
      next_image = self.plot_response_function(next_image, 'Filter 14 (Al13+Pb82+Cu29+PEEK)', [self.ResponseWakefield14])
      next_image = self.plot_response_function(next_image, 'Filter 13 (Al13+Gd64+Cu29+PEEK), Filter 14 (Al13+Pb82+Cu29+PEEK)', [self.ResponseWakefield13, self.ResponseWakefield14])
      next_image = self.plot_response_function(next_image, 'Filter 14 minus Filter 13', [ResSensitivityWakefield14to13], y_label='Residual Sensitivity')
      next_image = self.plot_response_function(next_image, 'Filter 14 minus Filter 13 (PUBW)', [ResSensitivityPerUnitBW14to13], y_label='Residual Sensitivity PUBW')

      next_image = self.plot_response_function(next_image, 'All Filters', \
                                               [ResSensitivityWakefield2to1, ResSensitivityWakefield4to3, \
                                                ResSensitivityWakefield6to5, ResSensitivityWakefield8to7, \
                                                ResSensitivityWakefield10to9, ResSensitivityWakefield12to11, \
                                                ResSensitivityWakefield14to13], y_label='Residual Sensitivity')

      next_image = self.plot_response_function(next_image, 'All Filters (PUBW)', \
                                               [ResSensitivityPerUnitBW2to1, ResSensitivityPerUnitBW4to3, \
                                                ResSensitivityPerUnitBW6to5, ResSensitivityPerUnitBW8to7, \
                                                ResSensitivityPerUnitBW10to9, ResSensitivityPerUnitBW12to11, \
                                                ResSensitivityPerUnitBW14to13], y_label='Residual Sensitivity PUBW')

      next_image = self.plot_response_function(next_image, 'All Filters (up to 100 keV)', \
                                               [ResSensitivityWakefield2to1, ResSensitivityWakefield4to3, \
                                                ResSensitivityWakefield6to5, ResSensitivityWakefield8to7, \
                                                ResSensitivityWakefield10to9, ResSensitivityWakefield12to11, \
                                                ResSensitivityWakefield14to13], y_label='Residual Sensitivity', max_x=100)

      next_image = self.plot_response_function(next_image, 'All Filters (PUBW, up to 100 keV)', \
                                               [ResSensitivityPerUnitBW2to1, ResSensitivityPerUnitBW4to3, \
                                                ResSensitivityPerUnitBW6to5, ResSensitivityPerUnitBW8to7, \
                                                ResSensitivityPerUnitBW10to9, ResSensitivityPerUnitBW12to11, \
                                                ResSensitivityPerUnitBW14to13], y_label='Residual Sensitivity PUBW', max_x=100)
    
    self.map_attenuator_to_response_function()

  #%% get_all_thicknesses(): Hardcoded thicknesses (in microns) needed to calculate the response functions
  # (used by calculate_response_functions())  
  def get_all_thicknesses(self):
    
    # create a dictionary
    thicknesses = {}
    
    # 1 mil = 25 microns
    thicknesses['ThicknessMicronPEEK'] = 10 * 25 # 10 mils
    thicknesses['ThicknessMicronFullPEEK'] =  122.65 * 25 # PEEK spec thickness was 1/8" = 125 mil; using Mitutoyo micrometer measurement
    thicknesses['ThicknessMicronDeflector'] = 2.1025 * 25 # wakefield 4.0 deflector angle = 28.4 deg. Deflector thickness: 1 mil/sin(28.4) = 2.1025 mil
    thicknesses['ThicknessMicronXrayFilterCover'] = 1 * 25 # 1 mil Al foil covering the x-ray filters
    thicknesses['ThicknessMicronTotalDefaultAlFoil'] = thicknesses['ThicknessMicronDeflector'] + thicknesses['ThicknessMicronXrayFilterCover']

    # Ross filter pairs (K edges)
    # 1) 5 - 10 keV : Ti (4.9664) & Zn (9.6586),
    # 2) 10 - 15 keV : Zn (9.6586) & Zr (17.9976),
    # 3) 15 - 20 keV : Zr (17.9976) & Mo (19.9995),
    # 4) 20 - 25 keV : Mo (19.9995) & Ag (25.5140),
    # 5) 25 - 30 keV : Ag (25.5140) & Sn (29.2001),
    # 6) 30 - 50 keV : Sn (29.2001) & Gd (50.2391), 
    # 7) 50 - 90 keV : Gd (50.2391) & Pb (88.0045)

    # pair 1: Ti-Zn
    thicknesses['ThicknessMicronWakefield1'] = 75
    thicknesses['ThicknessMicronWakefield1Cu'] = 0
    thicknesses['ThicknessMicronWakefield2'] = 20
    thicknesses['ThicknessMicronWakefield2Cu'] = 0
    # pair 2: Zn-Zr
    thicknesses['ThicknessMicronWakefield3'] = 200
    thicknesses['ThicknessMicronWakefield3Cu'] = 0
    thicknesses['ThicknessMicronWakefield4'] = 100
    thicknesses['ThicknessMicronWakefield4Cu'] = 0
    # pair 3: Zr-Mo
    thicknesses['ThicknessMicronWakefield5'] = 100
    thicknesses['ThicknessMicronWakefield5Cu'] = 0
    thicknesses['ThicknessMicronWakefield6'] = 55
    thicknesses['ThicknessMicronWakefield6Cu'] = 0
    # pair 4: Mo-Ag
    thicknesses['ThicknessMicronWakefield7'] = 100
    thicknesses['ThicknessMicronWakefield7Cu'] = 40
    thicknesses['ThicknessMicronWakefield8'] = 75
    thicknesses['ThicknessMicronWakefield8Cu'] = 34
    # pair 5: Ag-Sn
    thicknesses['ThicknessMicronWakefield9'] = 75
    thicknesses['ThicknessMicronWakefield9Cu'] = 34
    thicknesses['ThicknessMicronWakefield10'] = 100
    thicknesses['ThicknessMicronWakefield10Cu'] = 30
    # pair 6: Sn-Gd
    thicknesses['ThicknessMicronWakefield11'] = 100
    thicknesses['ThicknessMicronWakefield11Cu'] = 30
    thicknesses['ThicknessMicronWakefield12'] = 50
    thicknesses['ThicknessMicronWakefield12Cu'] = 25
    # pair 7: Gd-Pb
    thicknesses['ThicknessMicronWakefield13'] = 50
    thicknesses['ThicknessMicronWakefield13Cu'] = 20
    thicknesses['ThicknessMicronWakefield14'] = 15
    thicknesses['ThicknessMicronWakefield14Cu'] = 34
    
    return thicknesses
  
  #%% extrapolation_handler(): from Mathematica code '18. Ross filter analysis - updated.nb')
  # (used by calculate_response_functions())
  def extrapolation_handler(self, energies):
    sensitivities = 11.7545 * np.exp(-0.0137733 * (-16.3876 + energies))
    return sensitivities
 
  #%% make_energy_grid(): create global energy grid
  # (used by calculate_response_functions())
  def make_energy_grid(self):
    # o Mass attenuation data is for E from 0.2013709 to 310.1304 keV
    # o MS IP sensitivity data is from 7.731 to 100 with a 0 prepended to the beginning
    # o Curves are interpolated using the full set of data and then evaluated
    #   on the range 0.200-310.2 keV to fully include mass sensitivity data range
    # o Note that in order to capture the K-edge, the energy step size can't be > 0.001

    # (all energies in keV)
    #
    # Ross filter bin pairs (K edges):
    #   1) 5 - 10 keV : Ti (4.9664) & Zn (9.6586),
    #   2) 10 - 15 keV : Zn (9.6586) & Zr (17.9976),
    #   3) 15 - 20 keV : Zr (17.9976) & Mo (19.9995),
    #   4) 20 - 25 keV : Mo (19.9995) & Ag (25.5140),
    #   5) 25 - 30 keV : Ag (25.5140) & Sn (29.2001),
    #   6) 30 - 50 keV : Sn (29.2001) & Gd (50.2391), 
    #   7) 50 - 90 keV : Gd (50.2391) & Pb (88.0045)
    #
    # Ross filter non-bin attenuators:
    #   Al K: 1.5596
    #   Cu K: 8.9789  
    #   Cu L I: 1.0961, Cu L II: 0.9510, Cu L III: 0.9311

    # PEEK:
    #   C K: 0.2838
    #   O K: 0.5320
    #
    # MS/SR IP:
    #   Ba L I: 5.9888, Ba K: 37.4406
    #   Br K: 13.4737
    #   I K: 33.1694
    
    print("\nCreating fine energy grid around edges (lower bound to edge to upper bound):")
    if (True):
      edge_energies = np.arange(0.200, 10.001, 0.001)
    else:
      M_V_edge_Mo = 0.2270
      edge_energies = self.make_local_energy_grid(np.arange(0.2, M_V_edge_Mo, 1), M_V_edge_Mo)
  
      M_IV_edge_Mo = 0.2303
      edge_energies = self.make_local_energy_grid(edge_energies, M_IV_edge_Mo)

      N_III_edge_Gd = 0.2709
      edge_energies = self.make_local_energy_grid(edge_energies, N_III_edge_Gd)
  
      K_edge_PEEK_C = 0.2838
      edge_energies = self.make_local_energy_grid(edge_energies, K_edge_PEEK_C)
  
      N_II_edge_Gd = 0.2885
      edge_energies = self.make_local_energy_grid(edge_energies, N_II_edge_Gd)
  
      M_III_edge_Zr = 0.3305
      edge_energies = self.make_local_energy_grid(edge_energies, M_III_edge_Zr)
  
      M_II_edge_Zr = 0.3442
      edge_energies = self.make_local_energy_grid(edge_energies, M_II_edge_Zr)
  
      M_V_edge_Ag = 0.3667
      edge_energies = self.make_local_energy_grid(edge_energies, M_V_edge_Ag)
  
      M_IV_edge_Ag = 0.3728
      edge_energies = self.make_local_energy_grid(edge_energies, M_IV_edge_Ag)
  
      N_I_edge_Gd = 0.3758
      edge_energies = self.make_local_energy_grid(edge_energies, N_I_edge_Gd)
  
      M_III_edge_Mo = 0.3923
      edge_energies = self.make_local_energy_grid(edge_energies, M_III_edge_Mo)
  
      M_II_edge_Mo = 0.4097
      edge_energies = self.make_local_energy_grid(edge_energies, M_II_edge_Mo)
  
      N_V_edge_Pb = 0.4129
      edge_energies = self.make_local_energy_grid(edge_energies, N_V_edge_Pb)
  
      M_I_edge_Zr = 0.4303
      edge_energies = self.make_local_energy_grid(edge_energies, M_I_edge_Zr)
  
      N_IV_edge_Pb = 0.4352
      edge_energies = self.make_local_energy_grid(edge_energies, N_IV_edge_Pb)
  
      L_III_edge_Ti = 0.4555
      edge_energies = self.make_local_energy_grid(edge_energies, L_III_edge_Ti)
  
      L_II_edge_Ti = 0.4615
      edge_energies = self.make_local_energy_grid(edge_energies, L_II_edge_Ti)
  
      M_V_edge_Sn = 0.4848
      edge_energies = self.make_local_energy_grid(edge_energies, M_V_edge_Sn)
  
      M_IV_edge_Sn = 0.4933
      edge_energies = self.make_local_energy_grid(edge_energies, M_IV_edge_Sn)
  
      M_I_edge_Mo = 0.5046
      edge_energies = self.make_local_energy_grid(edge_energies, M_I_edge_Mo)
  
      K_edge_PEEK_O = 0.5320
      edge_energies = self.make_local_energy_grid(edge_energies, K_edge_PEEK_O)
  
      L_I_edge_Ti = 0.5637
      edge_energies = self.make_local_energy_grid(edge_energies, L_I_edge_Ti)

      M_III_edge_Ag = 0.5714
      edge_energies = self.make_local_energy_grid(edge_energies, M_III_edge_Ag)

      M_II_edge_Ag = 0.6024
      edge_energies = self.make_local_energy_grid(edge_energies, M_II_edge_Ag)
	
      N_III_edge_Pb = 0.6445
      edge_energies = self.make_local_energy_grid(edge_energies, N_III_edge_Pb)

      M_III_edge_Sn = 0.7144
      edge_energies = self.make_local_energy_grid(edge_energies, M_III_edge_Sn)

      M_I_edge_Ag = 0.7175
      edge_energies = self.make_local_energy_grid(edge_energies, M_I_edge_Ag)

      M_II_edge_Sn = 0.7564
      edge_energies = self.make_local_energy_grid(edge_energies, M_II_edge_Sn)

      N_II_edge_Pb = 0.7639
      edge_energies = self.make_local_energy_grid(edge_energies, N_II_edge_Pb)

      M_I_edge_Sn = 0.8838
      edge_energies = self.make_local_energy_grid(edge_energies, M_I_edge_Sn)

      N_I_edge_Pb = 0.8936
      edge_energies = self.make_local_energy_grid(edge_energies, N_I_edge_Pb)

      L_III_edge_Cu = 0.9311
      edge_energies = self.make_local_energy_grid(edge_energies, L_III_edge_Cu)

      L_II_edge_Cu = 0.9510
      edge_energies = self.make_local_energy_grid(edge_energies, L_II_edge_Cu)

      L_III_edge_Zn = 1.0197
      edge_energies = self.make_local_energy_grid(edge_energies, L_III_edge_Zn)

      L_II_edge_Zn = 1.0428
      edge_energies = self.make_local_energy_grid(edge_energies, L_II_edge_Zn)

      L_I_edge_Cu = 1.0961
      edge_energies = self.make_local_energy_grid(edge_energies, L_I_edge_Cu)

      M_V_edge_Gd = 1.1852
      edge_energies = self.make_local_energy_grid(edge_energies, M_V_edge_Gd)

      L_I_edge_Zn = 1.1936
      edge_energies = self.make_local_energy_grid(edge_energies, L_I_edge_Zn)

      M_IV_edge_Gd = 1.2172
      edge_energies = self.make_local_energy_grid(edge_energies, M_IV_edge_Gd)

      M_III_edge_Gd = 1.5440
      edge_energies = self.make_local_energy_grid(edge_energies, M_III_edge_Gd)

      # Al K = 1.5596 keV
      K_edge_Al_start = 1.55804
      edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Al_start)

      K_edge_Al_end = 1.5674
      edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Al_end)

      M_II_edge_Gd = 1.6883
      edge_energies = self.make_local_energy_grid(edge_energies, M_II_edge_Gd)

      M_I_edge_Gd = 1.8808
      edge_energies = self.make_local_energy_grid(edge_energies, M_I_edge_Gd)

      L_III_edge_Zr = 2.2223
      edge_energies = self.make_local_energy_grid(edge_energies, L_III_edge_Zr)

      L_II_edge_Zr = 2.3067
      edge_energies = self.make_local_energy_grid(edge_energies, L_II_edge_Zr)

      M_V_edge_Pb = 2.4840
      edge_energies = self.make_local_energy_grid(edge_energies, M_V_edge_Pb)

      L_III_edge_Mo = 2.5202
      edge_energies = self.make_local_energy_grid(edge_energies, L_III_edge_Mo)

      L_I_edge_Zr = 2.5316
      edge_energies = self.make_local_energy_grid(edge_energies, L_I_edge_Zr)

      M_IV_edge_Pb = 2.5856
      edge_energies = self.make_local_energy_grid(edge_energies, M_IV_edge_Pb)

      L_II_edge_Mo = 2.6251
      edge_energies = self.make_local_energy_grid(edge_energies, L_II_edge_Mo)

      L_I_edge_Mo = 2.8655
      edge_energies = self.make_local_energy_grid(edge_energies, L_I_edge_Mo)
	
      M_III_edge_Pb = 3.0664
      edge_energies = self.make_local_energy_grid(edge_energies, M_III_edge_Pb)

      L_III_edge_Ag = 3.3511
      edge_energies = self.make_local_energy_grid(edge_energies, L_III_edge_Ag)

      L_II_edge_Ag = 3.5237
      edge_energies = self.make_local_energy_grid(edge_energies, L_II_edge_Ag)

      M_II_edge_Pb = 3.5542
      edge_energies = self.make_local_energy_grid(edge_energies, M_II_edge_Pb)

      L_I_edge_Ag = 3.8058
      edge_energies = self.make_local_energy_grid(edge_energies, L_I_edge_Ag)

      M_I_edge_Pb = 3.8507
      edge_energies = self.make_local_energy_grid(edge_energies, M_I_edge_Pb)

      L_III_edge_Sn = 3.9288
      edge_energies = self.make_local_energy_grid(edge_energies, L_III_edge_Sn)

      L_II_edge_Sn = 4.1561
      edge_energies = self.make_local_energy_grid(edge_energies, L_II_edge_Sn)

      L_I_edge_Sn = 4.4647
      edge_energies = self.make_local_energy_grid(edge_energies, L_I_edge_Sn)

      # Ti K = 4.9664 keV
      K_edge_Ti_start = 4.96143
      edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Ti_start)

      K_edge_Ti_end = 4.99123
      edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Ti_end)
    
      L_I_edge_Ba = 5.9888
      edge_energies = self.make_local_energy_grid(edge_energies, L_I_edge_Ba)

      L_III_edge_Gd = 7.2428
      edge_energies = self.make_local_energy_grid(edge_energies, L_III_edge_Gd)

      L_II_edge_Gd = 7.9303
      edge_energies = self.make_local_energy_grid(edge_energies, L_II_edge_Gd)

      L_I_edge_Gd = 8.3756
      edge_energies = self.make_local_energy_grid(edge_energies, L_I_edge_Gd)

      # Cu K = 8.9789
      K_edge_Cu_start = 8.96992
      edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Cu_start)

      K_edge_Cu_end = 9.03079
      edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Cu_end)

      K_edge_Zn = 9.6586
      edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Zn)

    # Pb L III: 13.0352
    L_III_edge_Pb_start = 13.0222
    edge_energies = self.make_local_energy_grid(edge_energies, L_III_edge_Pb_start)

    L_III_edge_Pb_end = 13.1004
    edge_energies = self.make_local_energy_grid(edge_energies, L_III_edge_Pb_end)

    K_edge_Br = 13.4737
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Br)

    # Pb L II: 15.2000 keV
    L_II_edge_Pb_start = 15.1848
    edge_energies = self.make_local_energy_grid(edge_energies, L_II_edge_Pb_start)

    L_II_edge_Pb_end = 15.276
    edge_energies = self.make_local_energy_grid(edge_energies, L_II_edge_Pb_end)

    # Pb L I: 15.8608
    L_I_edge_Pb_start = 15.8449
    edge_energies = self.make_local_energy_grid(edge_energies, L_I_edge_Pb_start)

    L_I_edge_Pb_end = 15.9401
    edge_energies = self.make_local_energy_grid(edge_energies, L_I_edge_Pb_end)

    # Zr K: 17.9976 keV
    K_edge_Zr_start = 17.9796
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Zr_start)

    K_edge_Zr_end = 18.0876
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Zr_end)

    # Mo K: 19.9995
    K_edge_Mo_start = 19.9795
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Mo_start)

    K_edge_Mo_end = 20.1121
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Mo_end)

    # Ag K: 25.5140
    K_edge_Ag_start = 25.4885
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Ag_start)

    K_edge_Ag_end = 25.6416
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Ag_end)

    # Sn K: 29.2001
    K_edge_Sn_start = 29.1709
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Sn_start)

    K_edge_Sn_end = 29.3461
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Sn_end)

    # I K: 33.1694 keV
    K_edge_I_start = 32.875
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_I_start)

    K_edge_I_end = 33.1694
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_I_end)

    K_edge_Ba = 37.4406
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Ba)

    # Gd K: 50.2391 keV
    K_edge_Gd = 50.1889
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Gd)

    K_edge_Gd = 50.4903
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Gd)

    # Pb K: 88.0045
    K_edge_Pb_start = 87.9165
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Pb_start)
    
    K_edge_Pb_end = 88.4445
    edge_energies = self.make_local_energy_grid(edge_energies, K_edge_Pb_end)
        
    edge_energies = np.unique(np.sort(edge_energies.round(decimals=4)))
    
    return edge_energies

    
  #%% make_local_energy_grid(): creates a granular energy grid on either side of the edge energy
  # (used by calculate_response_functions() via make_energy_grid())
  def make_local_energy_grid(self, energies, edge_E):
    half_interval = 0.0050
    fine_step_size = 0.0001
    coarse_step_size = 1

    fine_energies_low = edge_E - half_interval
    fine_energies_high = edge_E + half_interval
    print(f'{np.around(fine_energies_low, decimals=4)} to {edge_E} to {np.around(fine_energies_high, decimals=4)}')
    prev_ceil = np.ceil(max(energies))
    next_floor = np.floor(fine_energies_low)
    num_steps_coarse = round((next_floor - prev_ceil) / coarse_step_size)
    num_steps_fine = round((fine_energies_high - fine_energies_low) / fine_step_size)
    fine_grid = np.linspace(fine_energies_low, fine_energies_high, num_steps_fine + 1)

    if (num_steps_coarse > 0):
      coarse_grid = np.linspace(prev_ceil, next_floor, num_steps_coarse + 1)
      energies = np.concatenate((energies, coarse_grid, fine_grid))
    else:
      energies = np.concatenate((energies, fine_grid))

    return energies


  #%% map_attenuator_to_response_function()
  # (called by calculate_response_functions())
  
  # self.measurements['11.11'].keys()
  # dict_keys(['Mo100', 'Mo55', 'PEEK10mil', 'Pb15', 'Ti75', 'Zn20', 'Zn200', 'Zr100', 'PEEK125mil'])
  #
  # for ii in range(len(self.measurements['11.11']['Mo55'])):
  #   self.measurements['11.11']['Mo55'][ii]['ID']
  # 'Mo55,15,1'
  # 'Mo55,9,3'
  def map_attenuator_to_response_function(self):
    print(inspect.currentframe().f_code.co_name + ": working out of the directory '" + self.measurements_dir + "'")

    new_dict = {}
    new_dict['PEEK125mil'] = self.ResponseWakefieldOffFilterMS
    new_dict['PEEK10mil'] = self.ResponseWakefieldNoFilterMS
    new_dict['Ti75'] = self.ResponseWakefield1
    new_dict['Zn20'] = self.ResponseWakefield2
    new_dict['Zn200'] = self.ResponseWakefield3
    new_dict['Zr100'] = self.ResponseWakefield4
    new_dict['Mo55'] = self.ResponseWakefield6
    new_dict['Mo100'] = self.ResponseWakefield7   
    new_dict['Ag75'] = self.ResponseWakefield8
    new_dict['Sn100'] = self.ResponseWakefield10
    new_dict['Gd50'] = self.ResponseWakefield12
    new_dict['Pb15'] = self.ResponseWakefield14

    self.attenuator_to_response_function_dict = new_dict

  #%% plot_attenuations()
  def plot_attenuations(self, image_no, title, data_x, data_y, interpolated_y, is_small=False):
    if (is_small):
      if (image_no % 2 == 0):
        plt.subplot(121)
      else:
        plt.subplot(121)

    plt.loglog(data_x, data_y, 'x', self.energies, interpolated_y, markersize=2, linewidth=1)
    plt.xlabel('E [keV]')
    plt.ylabel('µ/ρ  [cm2 g-1]')
    plt.title(title)
    plt.grid(True, which='both')
    if (image_no % 2 == 1 or not is_small):
      plt.savefig(self.plots_dir + "/" + str(image_no) + ") " + title + ".pdf", dpi=1200)
      plt.show()
      image_no += 1
      
    return image_no
      
  #%% plot_response_function()
  def plot_response_function(self, image_no, title, y_data, is_small=False, y_label='Response', max_x=310):
    if (is_small):
      if (image_no % 2 == 0):
        plt.subplot(121)
      else:
        plt.subplot(121)

    num_plots = len(y_data)
    if (num_plots == 1):
      plt.plot(self.energies, y_data[0], linewidth=1)
    elif (num_plots == 2):
      plt.plot(self.energies, y_data[0], self.energies, y_data[1], linewidth=1)
    elif (num_plots == 3):
      # y_data[1] is really energy data (i.e. x data)
      plt.plot(self.energies, y_data[0], y_data[1], y_data[2], 'x', markersize=2, linewidth=1)
    elif (num_plots == 7):
      plt.plot(self.energies, y_data[0], \
               self.energies, y_data[1], \
               self.energies, y_data[2], \
               self.energies, y_data[3], \
               self.energies, y_data[4], \
               self.energies, y_data[5], \
               self.energies, y_data[6], linewidth=1)
    else:
      print("Don't know how to overlay " + str(num_plots) + " plots to create " + title + ".")
      return image_no
      
    plt.xlabel('E [keV]')
    if (title.find("PUBW") == -1):
      plt.ylabel(y_label +  ' [mPSL/photon]')
    else:
      plt.ylabel(y_label +  ' [mPSL/photon/keV]')
        
    plt.title(title)
    plt.grid(True, which='both')
    ax = plt.gca()
    if (num_plots == 7):
      ax.set_xlim([0, max_x])
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if (image_no % 2 == 1 or not is_small):
      plt.savefig(self.plots_dir + "/" + str(image_no) + ") " + title + ".pdf", dpi=1200)
      plt.show()
      image_no += 1
      
    return image_no
      
  #%% read_measurements(): read the measurement data: one file per intersection of contour and filter
  def read_measurements(self):
    print(inspect.currentframe().f_code.co_name + ": working out of the directory '" + self.measurements_dir + "'")

    self.unpooled_measurements = {}
    file_counter = 0
    other_counter = 0
    self.good_detectors = []
    self.failed_detectors_no_data = []
    self.failed_detectors_few_data = []
    for contour in self.contours:
      self.unpooled_measurements[contour] = defaultdict(list)    
      self.sorted_filenames = self.sort_by_contour_then_filter(contour)

      for filename in self.sorted_filenames:
        # skip PEEK measurements (to see if convergence improves)
        #if (filename.find("PEEK") != -1 or filename.find("profile") != -1):
        #if (filename.find("PEEK") != -1):
        #  print("Skipped " + filename)
        #  continue

        file_counter += 1
        #print(filename)
        if (filename.find("other") != -1):
          print("Skipped " + filename)
          other_counter += 1
          continue

        fields = re.split(',| ', filename)
        filtertype = fields[6]
        if (filtertype != "profile"):
          ID = fields[6] + ',' + fields[7] + ',' + fields[8]
        else:
          ID = "PEEK125mil"
          filtertype = "PEEK125mil"
        filedata_df = pd.read_csv(self.measurements_base_dir + '/' + filename, delimiter=' ', names=['X', 'Y', 'PSL'])
        filedata_notNaN_df = filedata_df[filedata_df.PSL.notna()]
        sample_size = len(filedata_notNaN_df.PSL)
        if (sample_size == 0):
          print("Detector " + filtertype + " on contour " + contour + \
                " has no measurements and was excluded.")
          self.failed_detectors_no_data.append(filename)
          continue
        elif (sample_size == 1):
          print("Detector " + filtertype + " on contour " + contour + \
                " has " + str(sample_size) + " measurement and was excluded.")
          self.failed_detectors_few_data.append(filename)
          continue
        
        mean = stats.tmean(filedata_notNaN_df.PSL)
        std = 18 * stats.tstd(filedata_notNaN_df.PSL)

        measurement_dict = {}
        measurement_dict['ID'] = ID
        measurement_dict['file_name'] = filename
        measurement_dict['file_df'] = filedata_df
        measurement_dict['sample_size'] = sample_size
        measurement_dict['mean'] = mean
        measurement_dict['std'] = std
        self.unpooled_measurements[contour][filtertype].append(measurement_dict)

        self.good_detectors.append(filename)
        
        print("Added " + filename)
        
    print("Numbers of measurement files read: " + str(file_counter))
    print("Numbers of good detectors: " + str(len(self.good_detectors)))
    print("Numbers of failed detectors (no measurements): " + str(len(self.failed_detectors_no_data)))
    print("Numbers of failed detectors (insufficient measurements): " + str(len(self.failed_detectors_no_data)))
    print("Numbers of 'other' detectors: " + str(other_counter))
      
    if (self.pool_attenuators):
      self.measurements = self.pool_attenuator_data()
    else:
      self.measurements = self.unpooled_measurements

        
  #%% sort_by_contour_then_filter()
  # (helper function used by read_measurements())
  def sort_by_contour_then_filter(self, contour):
    filenames = glob.glob('*,' + contour + ',*.txt')
    sorted_filenames = sorted(filenames, key=lambda fn: int(re.split(',', fn)[1]) if (float(re.split(',', fn)[1]) == int(float(re.split(',', fn)[1]))) else float(re.split(',', fn)[1]))
    resorted_filenames = sorted(sorted_filenames, key=lambda fn: re.split(' ', fn)[4])
    return resorted_filenames

  #%% pool_attenuator_data()
  # (called by read_measurements())
  def pool_attenuator_data(self):
    print(inspect.currentframe().f_code.co_name + ": working out of the directory '" + self.measurements_dir + "'")

    ### add entry to data structure for pooled attenuators
    self.pooled_measurements = copy.deepcopy(self.unpooled_measurements)
    for contour in self.unpooled_measurements.keys():
      # can't modify a dictionary while iterating over it
#     dictcopy = copy.deepcopy(self.measurements[contour])
#     for attenuator in dictcopy.keys():
      for attenuator in self.unpooled_measurements[contour].keys():
        num_samples = len(self.unpooled_measurements[contour][attenuator])
        if (num_samples == 1):
          continue
        else:
          pooled_measurements_dict = {}
          delimiter = "\n"
          # ID can refer to a filter or PEEK (10 mil [no filter] or 125 mil spec [off-filter or full thickness]
          IDs = list(map(lambda idx: self.unpooled_measurements[contour][attenuator][idx]['ID'], range(num_samples)))
          pooled_ID = delimiter.join(IDs)
          pooled_measurements_dict['ID'] = pooled_ID

          #delimiter = "\n"
          file_names = list(map(lambda idx: self.unpooled_measurements[contour][attenuator][idx]['file_name'], range(num_samples)))
          pooled_measurements_dict['file_name'] = delimiter.join(file_names)

          file_dataframes = list(map(lambda idx: self.unpooled_measurements[contour][attenuator][idx]['file_df'], range(num_samples)))
          file_dfs = pd.concat(file_dataframes)
          pooled_measurements_dict['file_df'] = file_dfs

          file_dfs_notNaN = file_dfs[file_dfs.PSL.notna()]

          pooled_measurements_dict['sample_size'] = len(file_dfs_notNaN.PSL)
          pooled_measurements_dict['mean'] = stats.tmean(file_dfs_notNaN.PSL)
          pooled_measurements_dict['std'] = 16 * stats.tstd(file_dfs_notNaN.PSL)
          
#         self.measurements[contour][pooled_ID].append(pooled_measurements_dict)
          self.pooled_measurements[contour][attenuator].clear()
          self.pooled_measurements[contour][attenuator].append(pooled_measurements_dict)
          
    return self.pooled_measurements

  #%% display_statistics(): run K-S and A-D tests to help decide when to pool measurements or exclude them
  # (called from the top level)
  def display_statistics(self):
    print(inspect.currentframe().f_code.co_name + "...")

    ### calculate K-S and A-D test results and print all statistics 
    
    for contour in self.measurements.keys():
      off_filter = self.measurements[contour]['PEEK125mil'][0]['mean']
      contour_error = (float(contour)/off_filter - 1) * 100
      print(f'- contour {contour}: off-filter = {off_filter:.3f} ({contour_error:.1f}% error)')  
        
      for attenuator in sorted(self.measurements[contour].keys()):
        
        num_samples = len(self.measurements[contour][attenuator])
        if (num_samples == 1):
          # attenuator either appears on contour once or has been pooled
          ID = self.measurements[contour][attenuator][0]['ID']
          df = self.measurements[contour][attenuator][0]['file_df']['PSL']
          df = df[df.notna()]
          sample_size = self.measurements[contour][attenuator][0]['sample_size']
          mean = self.measurements[contour][attenuator][0]['mean']
          std = self.measurements[contour][attenuator][0]['std']
          # compare distribution of single-sample or pooled-sample 
          # measurements to normal distributio using K-S and A-D tests
          KS_test_N = stats.ks_1samp(df, stats.norm.cdf)
          KS_test_N2 = stats.kstest(df, 'norm')
          #KS_test_T = stats.ks_1samp(df, stats.t.cdf(sample_size - 1))
          AD_test = stats.anderson(df, 'norm')

#         if (attenuator.count('\n') == 0):
          if (not self.pool_attenuators):
            # attenuator appears once on contour
            print(f'\tAttenuator ID: {ID}')
          else:
            # attenuator has been pooled
            num_samples = ID.count('\n') + 1
            attenuator = attenuator.split(',')[0]
            attenuators = ID.replace('\n', ' and ')
            print(f'\tPooled attenuators: {attenuators}')
            ID = ID.replace('\n', ' U ')
            attenuator_data = list(map(lambda idx: self.measurements[contour][attenuator][idx]['file_df']['PSL'], range(num_samples)))
            attenuator_data = list(map(lambda idx: attenuator_data[idx][attenuator_data[idx].notna()],range(num_samples)))
            AD_test = stats.anderson_ksamp(attenuator_data)
            print(f'\t\t{num_samples}-sample scipy.stats.anderson_ksamp test statistic = {AD_test[0]:.3f}, p-value = {AD_test[2]:.3f}')

          print(f'\t\tscipy.stats.ks_1samp test statistic (N) = {KS_test_N[0]:.3f}, p-value = {KS_test_N[1]:.3f}')
          print(f'\t\tscipy.stats.kstest test statistic (N) = {KS_test_N2[0]:.3f}, p-value = {KS_test_N2[1]:.3f}')
          #print(f'\t\tscipy.stats.ks_1samp test statistic (T) = {KS_test_T[0]:.3f}, p-value = {KS_test_T[1]:.3f}')
          print(f'\t\tscipy.stats.anderson test statistic = {AD_test[0]:.3f}, crit values: {AD_test[1]}, sig level: {AD_test[2]}')
          print(f'\t\t{ID}: sample_size={sample_size}, mean={mean:.3f}, std={std:.3f}\n')

        else:
          # happens when attenuator appears more once on contour and there is no pooling
          if (contour == '13.55'):
            print('here')

          all_filters = " and "
          filters = list(map(lambda idx: self.measurements[contour][attenuator][idx]['ID'], range(num_samples)))
          filters = all_filters.join(filters)
          print(f'\tRepeated filter: {attenuator} ({num_samples}) ({filters})')

          indices = [*range(0, num_samples)]
          dicts = list(map(lambda idx: self.measurements[contour][attenuator][idx], indices))
          
          for elements in itertools.combinations(dicts, 2):

            ID_0 = elements[0]['ID']            
            ID_1 = elements[1]['ID']
            
            if (elements[0]['ID'] == elements[1]['ID']):
              continue
              
            sample_size_0 = elements[0]['sample_size']
            mean_0 = elements[0]['mean']
            std_0 = elements[0]['std']
            sample_size_1 = elements[1]['sample_size']
            mean_1 = elements[1]['mean']
            std_1 = elements[1]['std']
            
            # filters with no data have already been labeled as 'failed' and excluded
            # these filters provide a single measurement and are also excluded
            if (math.isnan(std_0)):
              print(f'{ID_0} has std=NaN; skipped')
              continue
            
            if (math.isnan(std_1)):
              print(f'{ID_1} has std=NaN; skipped')
              continue
            
            df_0 = elements[0]['file_df']['PSL']
            df_0 = df_0[df_0.notna()]
            df_1 = elements[1]['file_df']['PSL']
            df_1 = df_1[df_1.notna()]
             
            print(f'\tPairwise statistics for {attenuator} on contour {contour}: {ID_0} and {ID_1}')
            KS_test = stats.ks_2samp(df_0, df_1)
            AD_test = stats.anderson_ksamp([df_0, df_1])
            print(f'\t\tscipy.stats.ks_2samp test statistic = {KS_test[0]:.3f}, p-value = {KS_test[1]:.3f}')
            print(f'\t\tscipy.stats.anderson_ksamp test statistic = {AD_test[0]:.3f}, p-value = {AD_test[2]:.3f}')
            print(f'\t\t{ID_0}: sample_size={sample_size_0}, mean={mean_0:.3f}, std={std_0:.3f}')
            print(f'\t\t{ID_1}: sample_size={sample_size_1}, mean={mean_1:.3f}, std={std_1:.3f}')
            
  #%% display_contour_errors(): display contour value, average off-filter value, and the relative error
  # (called from the top level)
  def display_contour_errors(self):
#    print(inspect.currentframe().f_code.co_name + ": working out of the directory '" + self.measurements_dir + "'")

    print("Contour\t\tOff-filter\tError [%]\tPixels\n")
 
    MAPE = 0
    weighted_MAPE = 0
    denominator = 0
    for contour in self.measurements.keys():
      sample_size = self.measurements[contour]['PEEK125mil'][0]['sample_size']
      off_filter = self.measurements[contour]['PEEK125mil'][0]['mean']
      contour_value = float(contour)
      
      error =(contour_value / off_filter - 1) * 100
      MAPE = MAPE + abs(error)
      weighted_MAPE = weighted_MAPE + sample_size * abs(error)
      denominator = denominator + sample_size      
    
      print(f'{contour_value:.2f}\t\t{off_filter:.2f}\t\t{error:.1f}\t\t\t{sample_size}')
      
    MAPE = MAPE / len(self.measurements.keys())
    weighted_MAPE = weighted_MAPE / denominator  
    print(f'MAPE is {MAPE:.3f}%. Weighted MAPE is {weighted_MAPE:.3f}%.')
  
  #%% interpolate_default_spectrum(): run bi-harmonic spline on the spectrum generated using Ross filter data
  # (called from the top level)
  def interpolate_default_spectrum(self):
    print(inspect.currentframe().f_code.co_name + ": working out of the directory '" + self.measurements_dir + "'")

    # read file generated by GenerateSpectrumFromRossFilterData.ijm
    default_spectrum_df = pd.read_csv(self.default_spectrum_file_path, delimiter=' ', names=['CL', 'EE', 'ZZ', 'deltaE', 'deltaZ'])
    # convert contour level to PSL (from percent of max PSL on Reconstruction Image Input)
    default_spectrum_df['CL']=default_spectrum_df['CL'] * self.max_PSL_on_Reconstruction_Input / 100
    # keep 2 decimal points to match measurement file contour levels
    default_spectrum_df['CL']=default_spectrum_df['CL'].round(decimals=2)
    # remove rows where the spectrum is NA
    default_spectrum_df = default_spectrum_df[default_spectrum_df['ZZ'].notna()]
    # remove rows where the spectrum is negative
    default_spectrum_df = default_spectrum_df[default_spectrum_df['ZZ'] >= 0]

    # add rows to force the spectrum to vanish at E=0 on all contours

    # for CL in np.unique(default_spectrum_df['CL']):
    #   insert_loc = default_spectrum_df[default_spectrum_df['CL']==CL].index[0]
    #   line = pd.DataFrame([[CL, 0, 0, 0, 0]], columns=['CL', 'EE', 'ZZ', 'deltaE', 'deltaZ'], index=[insert_loc])
    #   start_df = default_spectrum_df.iloc[:insert_loc]
    #   end_df = default_spectrum_df.iloc[insert_loc:]
    #   default_spectrum_df = pd.concat([start_df, line, end_df]).reset_index(drop=True)

    # add rows to force the spectrum to vanish at E=0 on contours at regular intervals

    # replaced with less granular interval to speed up
    # for CL in np.arange(250, -0.5, -0.5):
      
    for CL in np.arange(289, 1, -1):
      line = pd.DataFrame([[CL, 0, 0, 0, 0]], columns=['CL', 'EE', 'ZZ', 'deltaE', 'deltaZ'], index=[0])
      default_spectrum_df = pd.concat([line, default_spectrum_df])
    default_spectrum_df.reset_index(drop=True)

    # add rows to force the spectrum to vanish for all energies on contour level=0
    energies_df = pd.DataFrame(data=self.energies.flatten(), columns=['EE'])      
    zerosCL = pd.DataFrame([[0]]*len(self.energies), columns=['CL'])
    zeroRest = pd.DataFrame([[0, 0, 0]]*len(self.energies), columns=['ZZ', 'deltaE', 'deltaZ'])
    zerosCL = zerosCL.join(energies_df)
    zeroRest = zerosCL.join(zeroRest)
    zero_line = pd.DataFrame([[0, 0, 0, 0, 0]], columns=['CL', 'EE', 'ZZ', 'deltaE', 'deltaZ'], index=[0])
    default_spectrum_df = pd.concat([zero_line, zeroRest, default_spectrum_df]).reset_index(drop=True)

#   new_row = {'CL':289, 'EE':16, 'ZZ':4.20597E11, 'deltaE':0, 'deltaZ':0}
#   new_row = {'CL':257.5, 'EE':16, 'ZZ': 6.359109E11, 'deltaE':0, 'deltaZ':0}    
    new_row = {'CL':257.5, 'EE':16, 'ZZ': 6.359109E11, 'deltaE':8, 'deltaZ':6.359109E10}    
    default_spectrum_df = default_spectrum_df.append(new_row, ignore_index=True)
    
    # conversion from per stradian (output of GenerateSpectrumFromRossFilterData.ijm) to per pixel
    solidAnglePerPixel=pow(100*0.000001,2)/pow(2.725,2)
    # default_spectrum_Z = default_spectrum_Z * solidAnglePerPixel
    default_spectrum_df['ZZ_per_pixel'] = default_spectrum_df['ZZ'] * solidAnglePerPixel
    
    default_spectrum_X=default_spectrum_df['EE']
    default_spectrum_Y=default_spectrum_df['CL']
    default_spectrum_Z=default_spectrum_df['ZZ_per_pixel']
    
    # raw data
    fig = plt.figure()
    # plt.ioff()
    plt.plot(default_spectrum_X, default_spectrum_Y, 'xk', label="input point", markersize=2) # 'xk' means black x markers
    plt.legend(loc="lower right")
    plt.xlabel('E [keV]')
    plt.ylabel('Contour Level [PSL]')
    plt.title('Raw data (Ross filter spectrum + boundary conditions)\nplt.plot(default_spectrum_X, default_spectrum_Y)')
    ax = plt.gca()
    ax.set_xlim([0, 100])
    ax.set_ylim([25 * np.ceil(self.max_PSL_on_Sampling_image/25), 0])
    plt.show()

    fig = plt.figure()
    # plt.ion()
    # ax = plt.axes(projection='3d')
    ax = Axes3D(fig, auto_add_to_figure=False) # recommended
    fig.add_axes(ax)
    ax.scatter(default_spectrum_X, default_spectrum_Y, default_spectrum_Z, color='k', s=10, marker='x', label="input point")
    sc = ax.plot_trisurf(default_spectrum_X, default_spectrum_Y, default_spectrum_Z, cmap=cm.jet, edgecolor='none', antialiased=False)
    plt.legend(loc="lower right")
    plt.xlabel('E [keV]', fontsize=7)
    plt.ylabel('CL [PSL]', fontsize=7)
    ax.set_zlabel('Spectrum [photons/keV/pixel]', fontsize=7)
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(5)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(5)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(5)
    # can't set title
    # plt.title('Raw data (Ross filter spectrum + boundary conditions)')
    # ax.set_title('Raw data (Ross filter spectrum + boundary conditions)')
    # "Placement 0, 0 would be the bottom left, 1, 1 would be the top right."
    title = "Raw data (Ross filter spectrum + boundary conditions)"
    title = title + "\nax.scatter(default_spectrum_X, default_spectrum_Y, default_spectrum_Z)"
    title = title + "\nax.plot_trisurf(default_spectrum_X, default_spectrum_Y, default_spectrum_Z)"
    ax.text2D(0.1, 0.9, title, fontsize='small', transform=ax.transAxes)        
    # ax.set_xlim([0, 100])
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_ylim([25 * np.ceil(self.max_PSL_on_Sampling_image/25), 0])
    plt.colorbar(sc, shrink=0.50)
    ax.view_init(elev=0, azim=0)
    plt.savefig(self.plots_dir + "/default spectrum 3D 0 0.pdf", dpi=1200)
    print("saved 'default spectrum 3D 0 0.pdf'")
    plt.show()
    # plt.pause(1)
    # fig = plt.figure()
    ax.view_init(elev=0, azim=-90)
    # plt.draw()
    plt.savefig(self.plots_dir + "/default spectrum 3D -90 0.pdf", dpi=1200)
    print("saved default spectrum 3D -90 0.pdf")
    plt.show()
    # sys.exit("exitting...")
    
    # interpolation grid (matches the existing energy and contour grid)
    
    # not using this more granular (along the contour level axis) grid
    # Y, X = np.meshgrid(np.unique(default_spectrum_df['CL']), self.energies)
    self.contours_float = list(map(lambda contour: float(contour), self.contours))
    Y, X = np.meshgrid(self.contours_float, self.energies)

    # interpolate.interp2d()   
    if (False):
    # fails due data being sparse
      interpf = interpolate.interp2d(default_spectrum_Y, default_spectrum_X, default_spectrum_Z)

    # interpolate.CloughTocher2DInterpolator()
    if (False):
      interpf = interpolate.CloughTocher2DInterpolator(list(zip(default_spectrum_X, default_spectrum_Y)), default_spectrum_Z)
      Z = interpf(X, Y)
      plt.pcolormesh(X, Y, Z, shading='auto', cmap=cm.jet)
      plt.plot(default_spectrum_X, default_spectrum_Y, 'xk', label="input point", markersize=2) # 'xk' means black x markers
      plt.legend(loc="lower right")
      plt.xlabel('E [keV]')
      plt.ylabel('Contour Level [PSL]')
      plt.title('CloughTocher2DInterpolator')
      ax = plt.gca()
      ax.set_xlim([0, 100])
      ax.set_ylim([25 * np.ceil(self.max_PSL_on_Sampling_image/25), 0])
      plt.colorbar()
      plt.show()
        
    # interpolate.griddata(method='nearest')
    grid_z0 = interpolate.griddata(list(zip(default_spectrum_X, default_spectrum_Y)), default_spectrum_Z, (X, Y), method='nearest')
    
    fig = plt.figure()
    plt.pcolormesh(X, Y, grid_z0, shading='auto', cmap=cm.jet)
    plt.plot(default_spectrum_X, default_spectrum_Y, 'xk', label="input point", markersize=2)
    plt.legend(loc="lower right")
    plt.xlabel('E [keV]')
    plt.ylabel('CL [PSL]')
    title = 'Raw data + griddata (Nearest) interpolation'
    title = title + '\nplt.pcolormesh(X, Y, grid_z0)'
    title = title + '\nplt.plot(default_spectrum_X, default_spectrum_Y)'
    plt.title(title, fontsize = 'small')
    ax = plt.gca()
    ax.set_xlim([0, 100]) # use for pcolormesh
    ax.set_ylim([25 * np.ceil(self.max_PSL_on_Sampling_image/25), 0])
    plt.colorbar() # use for pcolormesh
    plt.show()
        
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False) # recommended
    fig.add_axes(ax)
    ax.plot_surface(X, Y, grid_z0, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.scatter(default_spectrum_X, default_spectrum_Y, default_spectrum_Z, color='k', s=10, marker='x', label="input point")
    # plt.plot(default_spectrum_X, default_spectrum_Y, 'xk', label="input point", markersize=2)
    plt.legend(loc="lower right")
    plt.xlabel('E [keV]')
    plt.ylabel('CL [PSL]')
    # doesn't display
    # plt.title('griddata: Nearest')
    title = "Raw data + griddata (Nearest) interpolation"
    title = title + "\nax.plot_surface(X, Y, grid_z0)"
    title = title + "\nax.scatter(default_spectrum_X, default_spectrum_Y, default_spectrum_Z)"
    ax.text2D(0.05, 0.9, title, transform=ax.transAxes)        
    # ax = plt.gca()
    ax.set_ylim([25 * np.ceil(self.max_PSL_on_Sampling_image/25), 0])
    plt.show()

    # sys.exit("exitting...")

    # interpolate.griddata(method='linear')
    if (False):
      grid_z1 = interpolate.griddata(list(zip(default_spectrum_X, default_spectrum_Y)), default_spectrum_Z, (X, Y), method='linear')
  
      plt.pcolormesh(X, Y, grid_z1, shading='auto', cmap=cm.jet)
      plt.plot(default_spectrum_X, default_spectrum_Y, 'xk', label="input point", markersize=2)
      plt.legend(loc="lower right")
      plt.xlabel('E [keV]')
      plt.ylabel('Contour Level [PSL]')
      plt.title('griddata: Linear')
      ax = plt.gca()
      ax.set_xlim([0, 100])
      ax.set_ylim([25 * np.ceil(self.max_PSL_on_Sampling_image/25), 0])
      plt.colorbar()
      plt.show()
    
    # interpolate.griddata(method='cubic')
    if (False):
      grid_z2 = interpolate.griddata(list(zip(default_spectrum_X, default_spectrum_Y)), default_spectrum_Z, (X, Y), method='cubic')
  
      plt.pcolormesh(X, Y, grid_z2, shading='auto', cmap=cm.jet)
      plt.plot(default_spectrum_X, default_spectrum_Y, 'xk', label="input point", markersize=2)
      plt.legend(loc="lower right")
      plt.xlabel('E [keV]')
      plt.ylabel('Contour Level [PSL]')
      plt.title('griddata: Cubic')
      ax = plt.gca()
      ax.set_xlim([0, 100])
      ax.set_ylim([25 * np.ceil(self.max_PSL_on_Sampling_image/25), 0])
      plt.colorbar()
      plt.show()

    # interpolate.Rbf()
    if (False):
      rbfi = interpolate.Rbf(default_spectrum_X, default_spectrum_Y, default_spectrum_Z)  # radial basis function interpolator instance
      Z = rbfi(X, Y)
      
      plt.pcolormesh(X, Y, Z, shading='auto', cmap=cm.jet)
      plt.plot(default_spectrum_X, default_spectrum_Y, 'xk', label="input point", markersize=2) # 'xk' means black x markers
      plt.legend(loc="lower right")
      plt.xlabel('E [keV]')
      plt.ylabel('Contour Level [PSL]')
      plt.title('CloughTocher2DInterpolator')
      ax = plt.gca()
      ax.set_xlim([0, 100])
      ax.set_ylim([25 * np.ceil(self.max_PSL_on_Sampling_image/25), 0])
      plt.colorbar()
      plt.show()

    # interpolated values
    # create a uniform spectrum
    # df = pd.DataFrame(1, columns=self.measurements.keys(), index = range(len(self.energies)))
    # df = pd.DataFrame(1/max(self.energies), columns=self.measurements.keys(), index = range(len(self.energies)))
    # self.default_spectrum  = df
    # self.default_spectrum = grid_z0
    # self.default_spectrum = grid_z0 / 1E10
    # self.default_spectrum = grid_z0 * solidAnglePerPixel
    self.default_spectrum = grid_z0

  #%% calculate_spectrum_on_contour(): calculate contour-level spectrum
  # (called from the top level)
  def calculate_spectrum_on_contour(self, contour):
    if (self.debug):
      print(inspect.currentframe().f_code.co_name + ": processing contour " + contour + "...")
    lengths = list(map(lambda attenuator: len(self.measurements[contour][attenuator]), self.measurements[contour].keys()))
    pooled_lengths = list(filter(lambda length: length == 1, lengths))
    self.num_detectors[contour] = sum(pooled_lengths)
    parameters = (contour, self.default_spectrum)
    x0 = [0] * self.num_detectors[contour]
#   lw = [-0.1] * self.num_detectors[contour]
#   up = [0.1] * self.num_detectors[contour] 
#   optimize.dual_annealing(self.Z_f, bounds=list(zip(lw, up)), args=parameters, seed=1234)
    np.set_printoptions(precision=6, linewidth=150)
    print(f'Contour {contour}: {list(self.measurements[contour].keys())}')
#   OptimizeResult = optimize.minimize(self.Z_f, x0, args=parameters, method='L-BFGS-B', jac=self.dZ_f, options={'disp': True})
    minimizer_args = {"args": parameters, "method": "L-BFGS-B", "jac": True, "options": {'disp': False}}
    OptimizeResult = optimize.basinhopping(self.func, x0, minimizer_kwargs=minimizer_args, disp=True)
    print(f'Contour {contour}: {list(self.measurements[contour].keys())}')
#   print(OptimizeResult)
    
    cidx = self.contours.index(contour)
    attenuators = self.measurements[contour].keys()
    indices = range(len(self.measurements[contour].keys()))
    x = OptimizeResult['x']

    default_spectrum_on_this_contour = []
    spectrum_on_this_contour = []
    for eidx in range(len(self.energies)):  
      # f_def = self.default_spectrum[contour][eidx]
      f_def = self.default_spectrum[eidx,cidx]
      default_spectrum_on_this_contour.append(f_def)
      
      sum_over_attenuators = sum(list(map(lambda idx, attenuator: x[idx] * self.attenuator_to_response_function_dict[attenuator][eidx], indices, attenuators)))
      spectrum_on_this_contour.append(f_def * np.exp(-sum_over_attenuators))

    self.spectrum[contour] = spectrum_on_this_contour
    self.default_spectrum_on_contour[contour] = default_spectrum_on_this_contour

#   print(f'\nSpectrum on contour {contour}:\n{spectrum_on_this_contour}\n')
    print(f'\nSpectrum on contour {contour} is: (skipped)\n')
    
    default_spectrum_file = Path(self.outputs_dir + "/" + datetime.today().strftime('%Y%m%d') + " - default spectrum on contour " + contour + " (NN default spectrum).txt")
    default_spectrum_file.write_text(json.dumps(self.default_spectrum_on_contour[contour]))

    spectrum_file = Path(self.outputs_dir + "/" + datetime.today().strftime('%Y%m%d') + " - spectrum on contour " + contour + " (NN default spectrum).txt")
    spectrum_file.write_text(json.dumps(self.spectrum[contour]))

    print(f'\nSaved to {spectrum_file}\n')
#   time.sleep(60)
    print(OptimizeResult)
#   sys.exit("done")
#   input("Press Enter to continue...")


  #%% Z_f(): construct -Z, the objective function to minimize (Z is the potential function to maximize)
  # (this function is called by scipy.optimize.minimize())
  def Z_f(self, x, contour, default_spectrum):
    if (self.debug):
      print("\n" + inspect.currentframe().f_code.co_name + ": processing contour " + contour + "...")
      print(f'\tx={x}')
        
#   if contour in ['30.63', '34.29']:
#     print("On " + contour + " contour")
  
    cidx = self.contours.index(contour)
    attenuators = self.measurements[contour].keys()
    indices = range(len(self.measurements[contour].keys()))
    
    Z_f1 = 0
    for eidx in range(len(self.energies)):  
      # f_def = default_spectrum[contour][eidx]
      f_def = self.default_spectrum[eidx,cidx]
      sum_over_attenuators = sum(list(map(lambda idx, attenuator: x[idx] * self.attenuator_to_response_function_dict[attenuator][eidx], indices, attenuators)))
      e_to_minus_sum_over_attenuators = np.exp(-sum_over_attenuators)
#     print(f'E={self.energies[eidx]:.4f} (exponent exponential): {-sum_over_attenuators:.6f} {e_to_minus_sum_over_attenuators}')
      if (np.isinf(e_to_minus_sum_over_attenuators)):
        print(f'\ninf encountered: sum_over_attenuators is {sum_over_attenuators} at E={self.energies[eidx]}')
        for idx, attenuator in zip(indices, attenuators):
          print(f'At E={self.energies[eidx]}, attenuator={attenuator}, Lagrange multiplier={x[idx]}, response={self.attenuator_to_response_function_dict[attenuator][eidx]}')
        # to pause the program when there's an overflow
        input("Press Enter to continue...")
#     else:
#       print(f'inf not encountered: sum_over_attenuators is {sum_over_attenuators} at {self.energies[eidx]}')

#     if (self.energies[eidx] == 33.1744 or self.energies[eidx] == 34 or self.energies[eidx] == 35 or self.energies[eidx] == 60 or self.energies[eidx] == 61 or self.energies[eidx] == 62):
#      for idx, attenuator in zip(indices, attenuators):
#        print(f'At E={self.energies[eidx]}, attenuator={attenuator}, Lagrange multiplier={x[idx]}, response={self.attenuator_to_response_function_dict[attenuator][eidx]}')
        
      Z_f1 = Z_f1 + f_def * e_to_minus_sum_over_attenuators

    omega = self.num_detectors[contour]
    sum_over_attenuators = sum(list(map(lambda idx, attenuator: math.pow(x[idx] * self.measurements[contour][attenuator][0]['std'], 2), indices, attenuators)))
    Z_f2 = math.sqrt(omega * sum_over_attenuators)

    Z_f3 = sum(list(map(lambda idx, attenuator: x[idx] * self.measurements[contour][attenuator][0]['mean'], indices, attenuators)))
    
      
#   print(f'\nZ_f1 = {Z_f1}')
#   print(f'Z_f2 = {Z_f2}')
#   print(f'Z_f3 = {Z_f3}')
#   print(f'Z_f1 + Z_f2 + Z_f3 = {Z_f1+Z_f2+Z_f3}')

    return Z_f1 + Z_f2 + Z_f3

  #%% dZ_f(): construct -dZ/dx[idx], the Jacobian of the objective function to minimize (Z is the potential function to maximize)
  # (this function is called by scipy.optimize.minimize())
  def dZ_f(self, x, contour, default_spectrum):
    if (self.debug):
      print("\n" + inspect.currentframe().f_code.co_name + ": processing contour " + contour + "...")
      print(f'\tx={x}')

    cidx = self.contours.index(contour)
    attenuators = self.measurements[contour].keys()
    indices = range(len(self.measurements[contour].keys()))

    df = []
    # loop over gradient vector components
    for attenuator in self.measurements[contour].keys():
      
      idx = [*attenuators].index(attenuator)
      dZ_f1 = 0
      for eidx in range(len(self.energies)):  
        # f_def = default_spectrum[contour][eidx]
        f_def = self.default_spectrum[eidx,cidx]
        sum_over_attenuators = sum(list(map(lambda idx, attenuator: x[idx] * self.attenuator_to_response_function_dict[attenuator][eidx], indices, attenuators)))
        #print(f'\tsum_over_attenuators={sum_over_attenuators}')
        dZ_f1 = dZ_f1 + self.attenuator_to_response_function_dict[attenuator][eidx] * f_def * np.exp(-sum_over_attenuators)
  
      omega = self.num_detectors[contour]
      sum_over_attenuators = sum(list(map(lambda idx, attenuator: math.pow(x[idx] * self.measurements[contour][attenuator][0]['std'], 2), indices, attenuators)))

      if (sum_over_attenuators != 0):
        dZ_f2 = math.sqrt(omega) * (x[idx] * math.pow(self.measurements[contour][attenuator][0]['std'], 2)) / math.sqrt(sum_over_attenuators)
      else:
        dZ_f2 = math.sqrt(omega) * self.measurements[contour][attenuator][0]['std']
  
      dZ_f3 = self.measurements[contour][attenuator][0]['mean']

      # print the 3 terms in the Jacobian
      if (False):      
        print(f'\ndZ_f1 (contour={contour}, attenuator={attenuator}) = {dZ_f1}')
        print(f'dZ_f2 (contour={contour}, attenuator={attenuator}) = {dZ_f2}')
        print(f'dZ_f3 (contour={contour}, attenuator={attenuator}) = {dZ_f3}')
        print(f'dZ_f1 + dZ_f2 + dZ_f3 (contour={contour}, attenuator={attenuator}) = {dZ_f1+dZ_f2+dZ_f3}')
    
      df.append(-dZ_f1+dZ_f2+dZ_f3)
    
    if (math.isnan(-dZ_f1+dZ_f2+dZ_f3)):
      print("\nnan found\n")
    
    if (self.debug):      
      print(f'\tdf(x)={np.around(np.array(df),3)}')
  
    return (df)

  def func(self, x, contour, default_spectrum):
    return self.Z_f(x, contour, default_spectrum), self.dZ_f(x, contour, default_spectrum) 
  

  #%% interpolate_deconvoluted_spectrum(): interpolate contour-level spectra
  # (called from the top level)
  def interpolate_deconvoluted_spectrum(self):
    print(inspect.currentframe().f_code.co_name + ": working out of the directory '" + self.measurements_dir + "'")
    
#%% Start here

print("Start Time =", datetime.now().strftime("%H:%M:%S"))
spectrum_8351 = Spectrum("8351 - Ross filter - set 2", "8351 - measurements - 2,10 - avg", "8351 - spectrum - 2,10 - avg,TT5pc,PUE", 270.789, 289.941)
spectrum_8351.calculate_3D_specrum()
print("End Time =", datetime.now().strftime("%H:%M:%S"))


    