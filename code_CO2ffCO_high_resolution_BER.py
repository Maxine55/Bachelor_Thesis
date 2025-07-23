# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:54:14 2025
@author: maxine
"""
from math import *
import time as tt
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from datetime import datetime
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import linregress

#Daten JFJ:
DatenICOS_ganz = np.genfromtxt("ICOS_ATC_L2_L2-2024.1_JFJ_13.9_CTS.CO",skip_header=45,delimiter=";") #MEASUREMENT UNIT: nmol.mol-¹ (=10^-9=ppb)
DatenICOS_ganz = DatenICOS_ganz[DatenICOS_ganz[:,8] >= 0] # Negative Werte entfernen
DatenICOS = DatenICOS_ganz[:,[2,3,4,5,8,9]] 
DatenICOS = pd.DataFrame(DatenICOS)
DatenICOS = DatenICOS.rename(columns={0: 'Jahr', 1: 'Monat', 2: "Tag", 3: "Stunde", 4:"CO_jfj", 5:"CO_jfj_err"}) #D14C_JFJ_err ist weighted std_err
DatenICOS['Datum'] = pd.to_datetime(dict(year=DatenICOS['Jahr'], month=DatenICOS['Monat'], day=DatenICOS['Tag'], hour=DatenICOS["Stunde"]))
DatenICOS = DatenICOS.drop(columns=['Jahr',"Monat","Tag","Stunde"])
DatenICOS["CO_jfj"]=DatenICOS["CO_jfj"]*10**(-3) #in ppm
DatenICOS["CO_jfj_err"]=DatenICOS["CO_jfj_err"]*10**(-3) #in ppm


#Daten BER:
DatenBern = pd.read_csv("D14CO2_CO2_BER_2021_2023.csv",delimiter=";")
DatenBern = DatenBern.iloc[:69,3:8] 
DatenBern['SampDate'] = pd.to_datetime(DatenBern['SampDate'])  
DatenBern = DatenBern.sort_values(by='SampDate').reset_index(drop=True)

#attach overlapping data:
Daten = pd.merge_asof(DatenBern, DatenICOS, left_on='SampDate', right_on='Datum', direction="nearest") #zeiten die am nächsten beieinander sind mergen


# =============================================================================
# RCO selber berechnen: 
# =============================================================================
CO2ff_final = pd.read_csv("CO2ff_finalfinal_BER.csv",header=1, encoding='latin1',delimiter=",",names=["time","CO2ff","s_CO2ff"]) 

CO2ff_final['time'] = pd.to_datetime(CO2ff_final['time'])
Daten = Daten.rename(columns={'SampDate': 'time'})
Daten['time'] = pd.to_datetime(Daten['time'])


Daten["del_CO"] = (Daten["CO"]-Daten["CO_jfj"])

Daten_RCO = pd.merge_asof(CO2ff_final, Daten[['time', 'del_CO']], on='time', direction="nearest") #zeiten die am nächsten beieinander sind mergen

Daten_RCO["RCO"] = Daten_RCO["del_CO"]/Daten_RCO["CO2ff"]

#Median von RCO:
RCO_Median = Daten_RCO["RCO"].median()



#### RCO: normaler linearer Fit: y = m*x + b
mask = np.isfinite(Daten_RCO["CO2ff"]) & np.isfinite(Daten_RCO["del_CO"])
x = Daten_RCO["CO2ff"][mask]
y = Daten_RCO["del_CO"][mask]
slope_lin, b = np.polyfit(x, y, 1)
#plot linearer fit:
plt.title(f"slope={slope_lin:.3f}")
plt.scatter(x,y, label='Daten')
plt.plot(x, slope_lin*x+ b, color='red', label='Linearer Fit')
plt.legend()
plt.xlabel("ΔCO2ff [ppm]")
plt.ylabel("ΔCO [ppm]")
plt.show()


#### RCO: geometric mean regression:
r, _ = pearsonr(x, y)
sx = np.std(x, ddof=1)
sy = np.std(y, ddof=1)
slope_geom = np.sign(r) * (sy / sx)
intercept = np.mean(y) - slope_geom * np.mean(x)
#plot:
plt.title(f"slope={slope_geom:.3f}")
plt.scatter(x, y, label='Daten')
plt.plot(x, slope_geom * x + intercept, 'r-', label='Geometric Mean Regression')
plt.legend()
plt.xlabel("ΔCO2ff [ppm]")
plt.ylabel("ΔCO [ppm]")
plt.show()


#Berechnung CO2ff_CO aus RCO_median:
CO2ff_CO_RCOmedian = Daten["del_CO"]/RCO_Median
Daten["CO2ff_CO_RCOmedian"]= CO2ff_CO_RCOmedian

#Berechnung CO2ff_CO aus RCO_linfit:
CO2ff_CO_RCOlinfit = Daten["del_CO"]/slope_lin

#Berechnung CO2ff_CO aus RCO_geofit:
CO2ff_CO_RCOgeofit = Daten["del_CO"]/slope_geom

#plot mit RCO_median:
plt.figure(figsize=(10, 5))
plt.xlabel("time")
plt.ylabel("$\mathrm{CO}_{2\mathrm{ff}}^{\mathrm{CO}}$ [ppm]")
plt.plot(CO2ff_final['time'],CO2ff_final["CO2ff"],".",color="plum",label="$\mathrm{CO}_{2\mathrm{ff}}$ from $^{14}\mathrm{C}$ measurements")
plt.plot(CO2ff_final['time'],CO2ff_final["CO2ff"],color="plum")
plt.plot(Daten["Datum"],CO2ff_CO_RCOmedian,color="cyan")
plt.plot(Daten["Datum"],CO2ff_CO_RCOmedian,".",color="blue",label="$\mathrm{CO}_{2\mathrm{ff}}$ from CO measurements (RCO=40.787)")
plt.legend()
plt.grid()
plt.show()



#plot mit RCO aus lin fit:
plt.figure(figsize=(10, 5))
plt.title(f"CO2ff_CO, RCO from linear fit={slope_lin:.3f}")
plt.xlabel("time")
plt.ylabel("CO2ff_CO [ppm]")
plt.plot(Daten["Datum"],CO2ff_CO_RCOlinfit,color="cyan")
plt.plot(Daten["Datum"],CO2ff_CO_RCOlinfit,",",color="blue")
plt.legend()
plt.grid()
plt.show()

#plot mit RCO aus geom fit:
plt.figure(figsize=(10, 5))
plt.title(f"CO2ff_CO, RCO from geometric mean regression={slope_geom:.3f}")
plt.xlabel("time")
plt.ylabel("CO2ff_CO [ppm]")
plt.plot(Daten["Datum"],CO2ff_CO_RCOgeofit,color="cyan")
plt.plot(Daten["Datum"],CO2ff_CO_RCOgeofit,",",color="blue")
plt.legend()
plt.grid()
plt.show()




# =============================================================================
# CO vs 14C: 1:1 fit
# =============================================================================

CO2ff_final_clean = CO2ff_final.dropna()
Daten_clean = Daten.dropna(subset=['CO2ff_CO_RCOmedian'])


times_A = CO2ff_final_clean['time'].astype(np.int64)  
times_B = Daten_clean["Datum"].astype(np.int64)

#interpolate von B zu A:
interp_values_B = np.interp(times_A, times_B, Daten_clean["CO2ff_CO_RCOmedian"])

#interpoliert:
A = CO2ff_final_clean["CO2ff"].values
B_interp = interp_values_B

#lin reg:
slope, intercept, r_value, p_value, std_err = linregress(A, B_interp)


# Plot
plt.figure(figsize=(6,6))
plt.scatter(A, B_interp, alpha=0.5, label='interpolated data points')
plt.plot(A, slope*A + intercept, color='red', label=f'linear fit with slope={slope:.3f}')
plt.plot(A, A, linewidth=0.5, label='1:1 fit')
plt.xlabel('$\mathrm{CO}_{2\mathrm{ff}}$ from $^{14}\mathrm{C}$ measurements')
plt.ylabel('$\mathrm{CO}_{2\mathrm{ff}}^{\mathrm{CO}}$ (interpolated)')
plt.legend()
plt.grid(True)
plt.show()





































