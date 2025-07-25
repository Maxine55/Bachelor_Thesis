# -*- coding: utf-8 -*-
"""
Created on Sun May  4 12:34:04 2025
@author: maxine
"""
from math import *
import time as tt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

DatenBern = pd.read_csv("D14CO2_CO2_BER_2021_2023.csv",delimiter=";")
DatenBern = DatenBern.iloc[:69,3:8] 
DatenBern['SampDate'] = pd.to_datetime(DatenBern['SampDate'])  
DatenBern = DatenBern.sort_values(by='SampDate').reset_index(drop=True)

t = DatenBern["SampDate"]
D14C_meas = DatenBern["D14CO2"]
D14C_meas_err = DatenBern["D14CO2_err"]
CO2_meas = DatenBern["CO2"]
CO_meas = DatenBern["CO"]


# =============================================================================
#  Vergleich mit JFJ (ICOS):
# =============================================================================
    
#JFJ D14C:
DatenJFJ_ganz = np.genfromtxt("ICOS_ATC_L2_L2-2024.1_JFJ_6.0_779.14C.txt",skip_header=42,delimiter=";") 
DatenJFJ = DatenJFJ_ganz[103:,[2,3,4,10,11]] #2021-feb bis 2023-dez 
DatenJFJ = pd.DataFrame(DatenJFJ)
DatenJFJ = DatenJFJ.rename(columns={0: 'Jahr', 1: 'Monat', 2: "Tag", 3: "D14C_JFJ", 4: "D14C_JFJ_err"}) #D14C_JFJ_err ist weighted std_err
DatenJFJ['Datum'] = pd.to_datetime(dict(year=DatenJFJ['Jahr'], month=DatenJFJ['Monat'], day=DatenJFJ['Tag']))

t_jfj = DatenJFJ['Datum']
D14C_jfj = DatenJFJ["D14C_JFJ"]
D14C_jfj_err = DatenJFJ["D14C_JFJ_err"]

#JFJ CO2:
DatenCO2_JFJ = np.genfromtxt("ICOS_ATC_L2_L2-2024.1_JFJ_13.9_CTS.CO2",delimiter=";") 
DatenCO2_JFJ = DatenCO2_JFJ[36096:61632,[2,3,4,8,9]] #2021-feb bis 2023-dez
DatenCO2_JFJ = pd.DataFrame(DatenCO2_JFJ)
DatenCO2_JFJ = DatenCO2_JFJ[DatenCO2_JFJ[3] >= 0] # Negative Werte entfernen
DatenCO2_JFJ = DatenCO2_JFJ.rename(columns={0: 'Jahr', 1: 'Monat', 2: "Tag", 3: "CO2_JFJ", 4: "CO2_JFJ_err"}) 
DatenCO2_JFJ['Datum'] = pd.to_datetime(dict(year=DatenCO2_JFJ['Jahr'], month=DatenCO2_JFJ['Monat'], day=DatenCO2_JFJ['Tag']))



#plot D14C BER&JFJ:
plt.figure(figsize=(10, 5))
plt.title("Δ14C: Bern vs Jungfraujoch")
plt.xlabel("time [year]")
plt.ylabel("Δ14C [‰]")
plt.plot(t,D14C_meas,".",color="blue")
plt.plot(t,D14C_meas,color="grey")
plt.fill_between(t, D14C_meas - D14C_meas_err, D14C_meas + D14C_meas_err, color="blue",alpha=0.2,label="error width")
plt.plot(t_jfj,D14C_jfj,".",color="purple")
plt.plot(t_jfj,D14C_jfj,color="red")
plt.fill_between(t_jfj, D14C_jfj - D14C_jfj_err, D14C_jfj + D14C_jfj_err, color="red",alpha=0.2,label="error width")
plt.legend()
plt.grid()
plt.show()



# =============================================================================
#  monatliche mittelwerte:
# =============================================================================

#BERN:
DatenBern["Monat"] = DatenBern["SampDate"].dt.to_period("M")    
DatenBern_mm = DatenBern.groupby("Monat")[["D14CO2","D14CO2_err","CO2","CO"]].mean()
DatenBern_mm = DatenBern_mm.reset_index()
DatenBern_mm["Monat"]=DatenBern_mm["Monat"].dt.to_timestamp()

t_mm = DatenBern_mm["Monat"]
D14C_meas_mm = DatenBern_mm["D14CO2"]
D14C_meas_err_mm = DatenBern_mm["D14CO2_err"]
CO2_meas_mm = DatenBern_mm["CO2"]
CO_meas_mm = DatenBern_mm["CO"]



#JFJ:
#D14C
DatenJFJ["Monat"] = DatenJFJ["Datum"].dt.to_period("M")    
DatenJFJ_mm = DatenJFJ.groupby("Monat")[["D14C_JFJ","D14C_JFJ_err"]].mean()
DatenJFJ_mm = DatenJFJ_mm.reset_index()
DatenJFJ_mm["Monat"]=DatenJFJ_mm["Monat"].dt.to_timestamp()

t_jfj_mm = DatenJFJ_mm["Monat"]
D14C_jfj_mm = DatenJFJ_mm["D14C_JFJ"]
D14C_jfj_err_mm = DatenJFJ_mm["D14C_JFJ_err"]
#CO2
DatenCO2_JFJ["Monat"] = DatenCO2_JFJ["Datum"].dt.to_period("M")    
DatenCO2_JFJ_mm = DatenCO2_JFJ.groupby("Monat")[["CO2_JFJ","CO2_JFJ_err"]].mean()
DatenCO2_JFJ_mm = DatenCO2_JFJ_mm.reset_index()
DatenCO2_JFJ_mm["Monat"]=DatenCO2_JFJ_mm["Monat"].dt.to_timestamp()

CO2_jfj_mm = DatenCO2_JFJ_mm["CO2_JFJ"]
CO2_jfj_err_mm = DatenCO2_JFJ_mm["CO2_JFJ_err"]


#merged mm:
DATEN_merged = pd.merge(DatenBern_mm, DatenJFJ_mm, left_on='Monat', right_on='Monat', how='inner')
DATEN_merged = pd.merge(DATEN_merged, DatenCO2_JFJ_mm, left_on='Monat', right_on='Monat', how='inner')


#Plot D14C BER&JFJ monthly means:
plt.figure(figsize=(10, 5))
plt.title("Δ14C Bern (monthly averages)")
plt.xlabel("time [year]")
plt.ylabel("Δ14C [‰]")
plt.plot(DATEN_merged["Monat"], DATEN_merged["D14CO2"], ".",color="blue")
plt.plot(DATEN_merged["Monat"], DATEN_merged["D14CO2"], color="grey")
plt.fill_between(DATEN_merged["Monat"], DATEN_merged["D14CO2"] - DATEN_merged["D14CO2_err"], DATEN_merged["D14CO2"] + DATEN_merged["D14CO2_err"], color="blue",alpha=0.2,label="error width")
plt.plot(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"], ".",color="purple")
plt.plot(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"], color="red")
plt.fill_between(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"] - DATEN_merged["D14C_JFJ_err"], DATEN_merged["D14C_JFJ"] + DATEN_merged["D14C_JFJ_err"], color="red",alpha=0.2,label="error width")
plt.legend()
plt.grid()
plt.xticks(rotation=60)
plt.show()



# =============================================================================
#  Fossil fuel component CO2ff: 
# =============================================================================

# In Formel einsetzen:
CO2ff = DATEN_merged["CO2"]*(DATEN_merged["D14C_JFJ"]-DATEN_merged["D14CO2"])/(DATEN_merged["D14C_JFJ"]+1000)
t_CO2ff = DATEN_merged["Monat"]   

#plot zsm mit D14C Beromünster:
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14CO2"], ".",color="blue")
axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14CO2"], color="grey")
axs[0].fill_between(DATEN_merged["Monat"], DATEN_merged["D14CO2"] - DATEN_merged["D14CO2_err"], DATEN_merged["D14CO2"] + DATEN_merged["D14CO2_err"], color="blue",alpha=0.2,label="error width")
axs[0].set_ylabel("Δ14C [‰]")
axs[0].legend()
axs[0].grid()

axs[1].plot(t_CO2ff, CO2ff, ".",color="purple", label="calculated CO2ff ")
axs[1].plot(t_CO2ff, CO2ff, color="cyan")
axs[1].set_xlabel("time [year]")
axs[1].set_ylabel("CO2ff [ppm]")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()



# =============================================================================
#  Biospheric component CO2bio:
# =============================================================================

# In Formel einsetzen:
CO2bio = DATEN_merged["CO2"] - DATEN_merged["CO2_JFJ"] - CO2ff 


#plot D14C & CO2ff & CO2bio:
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))

axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14CO2"], ".",color="blue", label="$\Delta^{14}\mathrm{C}$ Bern ")
axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14CO2"], color="grey")
axs[0].fill_between(DATEN_merged["Monat"], DATEN_merged["D14CO2"] - DATEN_merged["D14CO2_err"], DATEN_merged["D14CO2"] + DATEN_merged["D14CO2_err"], color="blue",alpha=0.2)
axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"], ".",color="purple", label="$\Delta^{14}\mathrm{C}$ Jungfraujoch ")
axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"], color="red")
axs[0].fill_between(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"] - DATEN_merged["D14C_JFJ_err"], DATEN_merged["D14C_JFJ"] + DATEN_merged["D14C_JFJ_err"], color="red",alpha=0.2)
axs[0].set_ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
axs[0].legend()
axs[0].grid()

axs[1].plot(t_CO2ff, CO2ff, ".",color="darkblue", label="calculated $\mathrm{CO}_{2\mathrm{ff}}$ without correction")
axs[1].plot(t_CO2ff, CO2ff, color="cyan")
axs[1].set_ylabel("$\mathrm{CO}_{2\mathrm{ff}}$ [ppm]")
axs[1].legend()
axs[1].grid()

axs[2].plot(t_CO2ff, CO2bio, ".",color="green", label="calculated $\mathrm{CO}_{2\mathrm{bio}}$ without correction")
axs[2].plot(t_CO2ff, CO2bio, color="lightgreen")
axs[2].set_xlabel("time [year]")
axs[2].set_ylabel("$\mathrm{CO}_{2\mathrm{bio}}$ [ppm]")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()




# =============================================================================
# Correction NPP Simulation: (2021-01 bis 2021-12) 
# =============================================================================

DatenNPP_ber = pd.read_csv("SIM_NUC_D14CO2_BER.csv",delimiter=";")
DatenNPP_ber['SampDate'] = pd.to_datetime(DatenNPP_ber['SampDate'])  
DatenNPP_ber = DatenNPP_ber.sort_values(by='SampDate').reset_index(drop=True)

NPP_merged = pd.merge_asof(DatenBern, DatenNPP_ber, on='SampDate', direction='nearest') #zeiten die am nächsten beieinander sind mergen

# ######################## von D14C_meas D14C_NPP_simuliert abziehen (die, die gleiches datum haben):

# # Subtraktion:
NPP_merged['D14C_korrigiert'] = NPP_merged['D14CO2'] - NPP_merged['D_D14CO2_NPP_all']
D14C_meas_korrigiert = NPP_merged[['SampDate', 'D14C_korrigiert']]


# ########################## monatlicher mittelwert von D14C_meas_korrigiert:
    
D14C_meas_korrigiert["Monat"] = D14C_meas_korrigiert["SampDate"].dt.to_period("M")    
mm_D14C_meas_korr = D14C_meas_korrigiert.groupby("Monat")[["D14C_korrigiert"]].mean()
mm_D14C_meas_korr = mm_D14C_meas_korr.reset_index()
    
    

# ########################## CO2 fossil fuel mit NPP-correction berechnen:

CO2ff_corr_sim = DATEN_merged["CO2"]*(DATEN_merged["D14C_JFJ"]-mm_D14C_meas_korr["D14C_korrigiert"])/(DATEN_merged["D14C_JFJ"]+1000)
CO2ff_corr_sim = pd.DataFrame(np.array(CO2ff_corr_sim),t_mm)
CO2ff_corr_sim = pd.merge(CO2ff_corr_sim, t_CO2ff, left_on='Monat', right_on='Monat', how='inner')



# =============================================================================
# Δ14C_bio = 80‰, -3‰, Δ14C_bg
# =============================================================================

CO2ff_bio80 = (DATEN_merged["CO2_JFJ"]*(DATEN_merged["D14C_JFJ"]-80)-DATEN_merged["CO2"]*(DATEN_merged["D14CO2"]-80))/(80+1000)
CO2ff_bio3 = (DATEN_merged["CO2_JFJ"]*(DATEN_merged["D14C_JFJ"]-(-3))-DATEN_merged["CO2"]*(DATEN_merged["D14CO2"]-(-3)))/((-3)+1000)


# =============================================================================
# nuclear corr und biospheric corr in einem:
# =============================================================================
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 5))

axs[0].plot(t_CO2ff,CO2ff,".",color="darkblue",alpha=0.5,label="without nuclear correction")
axs[0].plot(t_CO2ff,CO2ff,color="cyan")
axs[0].plot(t_CO2ff,CO2ff_corr_sim[0],".",color="red",alpha=0.5,label="with nuclear correction (simulation)")
axs[0].plot(t_CO2ff,CO2ff_corr_sim[0],color="pink")
axs[0].set_ylabel("$\mathrm{CO}_{2\mathrm{ff}}$ [ppm]")
axs[0].legend()
axs[0].grid()

axs[1].plot(t_CO2ff,CO2ff_bio80,color="green")
axs[1].plot(t_CO2ff,CO2ff_bio3,color="red")
axs[1].plot(t_CO2ff,CO2ff,color="cyan")
axs[1].plot(t_CO2ff,CO2ff,".",color="cyan",label="$\Delta^{14}\mathrm{C}_\mathrm{bio}$ = $\Delta^{14}\mathrm{C}_\mathrm{bg}$ (no correction)")
axs[1].plot(t_CO2ff,CO2ff_bio80,".",color="green",label="$\Delta^{14}\mathrm{C}_\mathrm{bio}$ = 80‰")
axs[1].plot(t_CO2ff,CO2ff_bio3,".",color="red",label="$\Delta^{14}\mathrm{C}_\mathrm{bio}$ = -3‰")
axs[1].set_xlabel("time [year]")
axs[1].set_ylabel("$\mathrm{CO}_{2\mathrm{ff}}$ [ppm]")
axs[1].legend()
axs[1].grid()


plt.tight_layout()
plt.show()


# =============================================================================
# BEIDE KORREKTUREN:
# =============================================================================

NPP_merged['SampDate'] = pd.to_datetime(NPP_merged['SampDate'])
monthly_avg = NPP_merged.groupby(NPP_merged['SampDate'].dt.to_period('M'))['D_D14CO2_NPP_all'].mean().reset_index()
monthly_avg['SampDate'] = monthly_avg['SampDate'].dt.to_timestamp()
monthly_avg.columns = ['Monat', 'D_D14CO2_NPP_all_MW']
monthly_avg=monthly_avg[:29]
simulierte_daten = np.array(monthly_avg["D_D14CO2_NPP_all_MW"])

CO2ff_beidecorr = (DATEN_merged["CO2_JFJ"]*(DATEN_merged["D14C_JFJ"]-(-3))-DATEN_merged["CO2"]*(DATEN_merged["D14CO2"]-(-3)-simulierte_daten))/((-3)+1000)
CO2bio_beidecorr = DATEN_merged["CO2"] - DATEN_merged["CO2_JFJ"] - CO2ff_beidecorr


fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))

axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14CO2"], ".",color="blue", label="$\Delta^{14}\mathrm{C}$ Bern ")
axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14CO2"], color="grey")
axs[0].fill_between(DATEN_merged["Monat"], DATEN_merged["D14CO2"] - DATEN_merged["D14CO2_err"], DATEN_merged["D14CO2"] + DATEN_merged["D14CO2_err"], color="blue",alpha=0.2)
axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"], ".",color="purple", label="$\Delta^{14}\mathrm{C}$ Jungfraujoch ")
axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"], color="red")
axs[0].fill_between(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"] - DATEN_merged["D14C_JFJ_err"], DATEN_merged["D14C_JFJ"] + DATEN_merged["D14C_JFJ_err"], color="red",alpha=0.2)
axs[0].set_ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
axs[0].legend()
axs[0].grid()

axs[1].plot(t_CO2ff,CO2ff,".",color="darkgrey",alpha=0.5,label="without correction")
axs[1].plot(t_CO2ff,CO2ff,color="grey")
axs[1].plot(t_CO2ff,CO2ff_beidecorr,".",color="darkblue",alpha=0.5,label="with correction")
axs[1].plot(t_CO2ff,CO2ff_beidecorr,color="cyan")
axs[1].set_ylabel("$\mathrm{CO}_{2\mathrm{ff}}$ [ppm]")
axs[1].legend()
axs[1].grid()

axs[2].plot(t_CO2ff, CO2bio, ".",color="darkgrey", label="withoout correction")
axs[2].plot(t_CO2ff, CO2bio, color="grey")
axs[2].plot(t_CO2ff,CO2bio_beidecorr,".",color="darkgreen",label="with correction")
axs[2].plot(t_CO2ff,CO2bio_beidecorr,color="green")
axs[2].set_xlabel("time [year]")
axs[2].set_ylabel("$\mathrm{CO}_{2\mathrm{bio}}$ [ppm]")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()



# =============================================================================
# Fehlerrechnung für CO2ff:
# =============================================================================
DATEN_merged["CO2_err"]=0.1

s_D14Cmeas = DATEN_merged["D14CO2_err"]
s_D14Cbg = DATEN_merged["D14C_JFJ_err"]
s_CO2meas = DATEN_merged["CO2_err"]
s_CO2bg = DATEN_merged["CO2_JFJ_err"]

D14C_sim = CO2ff_corr_sim[0] - CO2ff[0]

f1= (DATEN_merged["D14C_JFJ"]-(-3))/((-3)+1000)
f2= DATEN_merged["CO2_JFJ"]/((-3)+1000)
f3= (DATEN_merged["D14CO2"]-D14C_sim-(-3))/((-3)+1000)
f4= DATEN_merged["CO2"]/((-3)+1000)

s_CO2ff = np.sqrt(f1**2*s_CO2bg**2+f2**2*s_D14Cbg**2+f3**2*s_CO2meas**2+f4**2*s_D14Cmeas**2)


# =============================================================================
# Fehlerrechnung für CO2bio: 
# =============================================================================

s_CO2bio = np.sqrt(s_CO2meas**2 + s_CO2bg**2 + s_CO2ff**2)


# =============================================================================
# Letzter Plot, beste Näherungen:
# =============================================================================

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))

axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14CO2"], color="grey")
axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14CO2"], ".",color="blue", label="$\Delta^{14}\mathrm{C}$ Bern ")
axs[0].fill_between(DATEN_merged["Monat"], DATEN_merged["D14CO2"] - DATEN_merged["D14CO2_err"], DATEN_merged["D14CO2"] + DATEN_merged["D14CO2_err"], color="blue",alpha=0.2)
axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"], color="red")
axs[0].plot(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"], ".",color="purple",label="$\Delta^{14}\mathrm{C}$ Jungfraujoch")
axs[0].fill_between(DATEN_merged["Monat"], DATEN_merged["D14C_JFJ"] - DATEN_merged["D14C_JFJ_err"], DATEN_merged["D14C_JFJ"] + DATEN_merged["D14C_JFJ_err"], color="red",alpha=0.2)
axs[0].set_ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
axs[0].legend()
axs[0].grid()

axs[1].plot(t_CO2ff,CO2ff_beidecorr,color="cyan")
axs[1].plot(t_CO2ff,CO2ff_beidecorr,".",color="blue")
axs[1].fill_between(t_CO2ff,CO2ff_beidecorr-s_CO2ff,CO2ff_beidecorr+s_CO2ff,color="blue",alpha=0.2,label="error width")
axs[1].set_ylabel("$\mathrm{CO}_{2\mathrm{ff}}$ [ppm]")
axs[1].legend()
axs[1].grid()

axs[2].plot(t_CO2ff,CO2bio_beidecorr,color="lime")
axs[2].plot(t_CO2ff,CO2bio_beidecorr,".",color="green")
axs[2].fill_between(t_CO2ff,CO2bio_beidecorr-s_CO2bio,CO2bio_beidecorr+s_CO2bio,color="green",alpha=0.2,label="error width")
axs[2].set_xlabel("time [year]")
axs[2].set_ylabel("$\mathrm{CO}_{2\mathrm{bio}}$ [ppm]")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()




# =============================================================================
# CO2ff abspeichern für high resolution:
# =============================================================================

CO2ff_final = pd.DataFrame({
    't_CO2ff': t_CO2ff,
    'CO2ff_corr': CO2ff_beidecorr,
    's_CO2ff': s_CO2ff
})

#CO2ff_final.to_csv("C:/Users/maxin/OneDrive/Desktop/Bachelorarbeit/14C/Daten/CO2ff_finalfinal_BER.csv")







