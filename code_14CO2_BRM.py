# -*- coding: utf-8 -*-
"""
Created on Sat May  3 16:12:13 2025
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
from scipy.stats import linregress

#Sampleliste:
xls = pd.ExcelFile("14C-Sampleliste_2201_Kopie.xls")
Sampleliste_komplett = xls.parse(xls.sheet_names[0],header=2)
Sampleliste = Sampleliste_komplett[184:]

#Beromünster-Daten:
xls = pd.ExcelFile("CC-CH-14CO2_report_250225.xlsx")
DatenBeromünster_komplett = xls.parse(xls.sheet_names[0],header=0)

#Data Bero&Sampleliste merged:
xls = pd.ExcelFile("Beromünster_Daten_merged2.xlsx")
DatenBeromünster = xls.parse(xls.sheet_names[0],header=0)


# =============================================================================
#  Average: same sample & same date:
# =============================================================================

def f(d):
    d["Probendatum"] = pd.to_datetime(d["Probendatum"], errors='coerce')
    d["d13_IRMS"] = pd.to_numeric(d["d13_IRMS"], errors='coerce')
    d_mean = d.groupby("sample_code", as_index=False)[["F14C","u(F14C)", "d13C_LARA"]].mean()
    d_merged = d_mean.merge(d[["Proben-Id", "Probendatum", "d13_IRMS"]].drop_duplicates(), 
                                      left_on="sample_code", right_on="Proben-Id", 
                                      how="left")
    d_merged = d_merged.drop(columns=["Proben-Id"]).drop_duplicates()
    d_final = d_merged.groupby("Probendatum", as_index=False)[["F14C","u(F14C)", "d13C_LARA", "d13_IRMS"]].mean()
    d_final = d_final.sort_values(by="Probendatum")
    
    return d_final
d_result2= f(DatenBeromünster)

t_datum_bero = pd.to_datetime(d_result2["Probendatum"])
F14C = np.array(d_result2["F14C"])
F14C_err = np.array(d_result2["u(F14C)"])
d13C_AMS = np.array(d_result2["d13C_LARA"])
d13C_AMS_err = 1.2 # ‰
d13C_IRMS = np.array(d_result2["d13_IRMS"])
d13C_IRMS_err = 0.1 # ‰

# =============================================================================
# Quality control with d13C:
# =============================================================================

#Plot d13C AMS&IRMS:
plt.figure(figsize=(10, 5))
plt.xlabel("time [year]")
plt.ylabel("$\delta^{13}\mathrm{C}$ [‰]")
plt.plot(t_datum_bero,d13C_AMS,".",color="blue",label="AMS")
plt.plot(t_datum_bero,d13C_AMS,color="green")
plt.fill_between(t_datum_bero,d13C_AMS-d13C_AMS_err,d13C_AMS+d13C_AMS_err,color="lightgreen",alpha=0.4,label="error width:1.2‰")
plt.plot(t_datum_bero,d13C_IRMS,".",color="purple",label="IRMS")
plt.plot(t_datum_bero,d13C_IRMS,color="red")
plt.fill_between(t_datum_bero,d13C_IRMS-d13C_IRMS_err,d13C_IRMS+d13C_IRMS_err,color="red",alpha=0.4,label="error width:0.1‰")
plt.legend()
plt.ylim(-19,-5)
plt.show()


#### mean value von abs(difference):
diff_AMS_IRMS = abs(d13C_IRMS[:157]-d13C_AMS[:157]) #bis 157, da ab dort nan werte
median = np.median(diff_AMS_IRMS)
std = np.std(diff_AMS_IRMS)

indexe=[]
for i in range(len(diff_AMS_IRMS)):
    if diff_AMS_IRMS[i]>median+(3*std): 
        indexe.append(i)
indexe = np.array(indexe)

d13C_AMS_bereinigt = np.array([x for i, x in enumerate(d13C_AMS[:157]) if i not in indexe])
t_d13C_bereinigt = np.array([x for i, x in enumerate(t_datum_bero[:157]) if i not in indexe])

#Plot d13C AMS&IRMS bereinigt:
plt.figure(figsize=(10, 5))
plt.xlabel("time [year]")
plt.ylabel("$\delta^{13}\mathrm{C}$ [‰]")
plt.plot(t_datum_bero,d13C_AMS,".",color="black",label="AMS removed data")
plt.plot(t_d13C_bereinigt,d13C_AMS_bereinigt,".",color="blue",label="AMS cleaned")
plt.plot(t_d13C_bereinigt,d13C_AMS_bereinigt,color="green")
plt.fill_between(t_d13C_bereinigt,d13C_AMS_bereinigt-d13C_AMS_err,d13C_AMS_bereinigt+d13C_AMS_err,color="lightgreen",alpha=0.4,label="error width:1.2‰")
plt.plot(t_datum_bero,d13C_IRMS,".",color="purple",label="IRMS")
plt.plot(t_datum_bero,d13C_IRMS,color="red")
plt.plot(t_datum_bero,d13C_IRMS+median+(3*std),color="grey",linewidth=0.5)
plt.plot(t_datum_bero,d13C_IRMS-(median+(3*std)),color="grey",linewidth=0.5,label="threshold for median+3sd")
plt.fill_between(t_datum_bero,d13C_IRMS-d13C_IRMS_err,d13C_IRMS+d13C_IRMS_err,color="red",alpha=0.4,label="error width:0.1‰")
plt.legend()
plt.ylim(-19,-3)
plt.show()


# =============================================================================
#  F14C in D14C:   
# =============================================================================

Jahre = np.array([t.year for t in d_result2["Probendatum"]])

x=np.linspace(1950,1950,len(Jahre))

D14C_bero = (F14C*np.exp((1950-Jahre)/8267)-1)*1000
D14C_bero_err = (np.exp((1950-Jahre)/8267))*1000*F14C_err #(berechnet mit Gauss-Fehlerfortpflanzung)

#Plot Δ14C: 
plt.figure(figsize=(10, 5))
plt.xlabel("time [year]")
plt.ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
plt.plot(t_datum_bero,D14C_bero,".",color="blue")
plt.plot(t_datum_bero,D14C_bero,color="grey")
plt.fill_between(t_datum_bero,D14C_bero-D14C_bero_err,D14C_bero+D14C_bero_err,color="blue",alpha=0.2,label="error width")
plt.legend()
plt.ylim(-45,25)
plt.show()


#Δ14C bereinigt:
D14C_bero_bereinigt = np.array([x for i, x in enumerate(D14C_bero) if i not in indexe])
D14C_bero_err_bereinigt = np.array([x for i, x in enumerate(D14C_bero_err) if i not in indexe])
t_D14C_bereinigt = np.array([x for i, x in enumerate(t_datum_bero) if i not in indexe])

plt.figure(figsize=(10, 5))
plt.xlabel("time [year]")
plt.ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
plt.plot(t_datum_bero,D14C_bero,".",color="red",label="removed data")
plt.plot(t_D14C_bereinigt,D14C_bero_bereinigt,".",color="blue")
plt.plot(t_D14C_bereinigt,D14C_bero_bereinigt,color="grey")
plt.fill_between(t_D14C_bereinigt,D14C_bero_bereinigt-D14C_bero_err_bereinigt,D14C_bero_bereinigt+D14C_bero_err_bereinigt,color="blue",alpha=0.2,label="error width")
plt.legend()
plt.ylim(-45,25)
plt.show()




# =============================================================================
#  Comparison with JFJ (ICOS):
# =============================================================================
    
DatenJFJ_ganz = np.genfromtxt("ICOS_ATC_L2_L2-2024.1_JFJ_6.0_779.14C.txt",skip_header=42,delimiter=";") 
DatenJFJ = DatenJFJ_ganz[4:127,:] #dez15 to märz22
WeightedStdErr = DatenJFJ[:,11]
D14C_JFJ = DatenJFJ[:,10]
t_decimal_JFJ = DatenJFJ[:,7] #decimal time in years

d_result2_bereinigt = d_result2.drop(indexe)

def date_to_decimal_year(date_series):
    years = date_series.dt.year
    start_of_year = pd.to_datetime(years.astype(str) + "-01-01")
    end_of_year = pd.to_datetime((years + 1).astype(str) + "-01-01")
    days_in_year = (end_of_year - start_of_year).dt.days
    day_of_year = (date_series - start_of_year).dt.total_seconds() / 86400
    decimal_year = years + day_of_year / days_in_year
    return decimal_year

t_decimal_bero = date_to_decimal_year(d_result2_bereinigt["Probendatum"])

plt.figure(figsize=(10, 5))
plt.xlabel("time [year]")
plt.ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
plt.plot(t_decimal_bero,D14C_bero_bereinigt,".",color="blue",label="Beromünster")
plt.plot(t_decimal_bero,D14C_bero_bereinigt,color="grey")
plt.fill_between(t_decimal_bero,D14C_bero_bereinigt-D14C_bero_err_bereinigt,D14C_bero_bereinigt+D14C_bero_err_bereinigt,color="blue",alpha=0.2,label="error width")
plt.plot(t_decimal_JFJ,D14C_JFJ,".",color="purple",label="Jungfraujoch")
plt.plot(t_decimal_JFJ,D14C_JFJ,color="red")
plt.fill_between(t_decimal_JFJ,D14C_JFJ-WeightedStdErr,D14C_JFJ+WeightedStdErr,color="red",alpha=0.2,label="error width")
plt.xlim(2016,2022)
plt.legend()
plt.show()


# =============================================================================
#  monthly mean:
# =============================================================================

###Beromünster:

# Monatliche Mittelwerte berechnen
d_result2_bereinigt["Datum"] = d_result2_bereinigt["Probendatum"].dt.to_period("M")
monthly_means_bero = d_result2_bereinigt.groupby("Datum")[["F14C","u(F14C)", "d13C_LARA", "d13_IRMS"]].mean()
monthly_means_bero = monthly_means_bero.reset_index()

monthly_means_F14C_bero = np.array(monthly_means_bero["F14C"])
Jahre = np.array([t.year for t in monthly_means_bero["Datum"]]) 

monthly_means_bero["D14C_bero_monthlymean"] = (np.array(monthly_means_bero["F14C"])*np.exp((1950-Jahre)/8267)-1)*1000
monthly_means_bero["D14C_bero_err_monthlymean"] = (np.exp((1950-Jahre)/8267))*1000*np.array(monthly_means_bero["u(F14C)"])

monthly_means_D14C_bero = (monthly_means_F14C_bero*np.exp((1950-Jahre)/8267)-1)*1000
monthly_means_bero['datetime'] = monthly_means_bero['Datum'].dt.to_timestamp()




###JFJ:

# DataFrame erstellen
Daten_gefiltert = [[row[2], row[3], row[4], row[10],row[11]] for row in DatenJFJ]
df = pd.DataFrame(Daten_gefiltert, columns=["Jahr", "Monat", "Tag", "D14C", "u(D14C)"])

# Monatliche Mittelwerte berechnen
monthly_means_JFJ = df.groupby(["Jahr", "Monat"])[["D14C","u(D14C)"]].mean()
monthly_means_JFJ = monthly_means_JFJ.reset_index()

#croppen, sd gleiche Grösse wie bero: (zb bei sigma=2: 2017-02 aus jfj rausnehmen)
monthly_means_JFJ['Datum'] = pd.to_datetime({
    'year': monthly_means_JFJ['Jahr'],
    'month': monthly_means_JFJ['Monat'],
    'day': 1  # Füge einen Standard-Tag hinzu
}).dt.to_period('M')
gemeinsame_daten = pd.Series(sorted(set(monthly_means_JFJ['Datum']) & set(monthly_means_bero['Datum'])))
monthly_means_JFJ_filtered = monthly_means_JFJ[monthly_means_JFJ['Datum'].isin(gemeinsame_daten)].reset_index(drop=True)
monthly_means_bero_filtered = monthly_means_bero[monthly_means_bero['Datum'].isin(gemeinsame_daten)].reset_index(drop=True)



###plot monthly means:
t = monthly_means_bero_filtered["datetime"]
D14C_bero_mm = monthly_means_bero_filtered["D14C_bero_monthlymean"]
D14C_bero_mm_err = monthly_means_bero_filtered["D14C_bero_err_monthlymean"]
D14C_jfj_mm = monthly_means_JFJ_filtered["D14C"]
D14C_jfj_mm_err = monthly_means_JFJ_filtered["u(D14C)"]


plt.figure(figsize=(10, 5))
plt.xlabel("time [year]")
plt.ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
plt.plot(t,D14C_bero_mm,".",color="blue",label="Beromünster")
plt.plot(t,D14C_bero_mm,color="grey")
plt.fill_between(t,D14C_bero_mm-D14C_bero_mm_err,D14C_bero_mm+D14C_bero_mm_err,color="blue",alpha=0.2,label="error width")
plt.plot(t,D14C_jfj_mm,".",color="purple",label="Jungfraujoch") 
plt.plot(t,D14C_jfj_mm,color="red") 
plt.fill_between(t,D14C_jfj_mm-D14C_jfj_mm_err,D14C_jfj_mm+D14C_jfj_mm_err,color="red",alpha=0.2,label="error width")
plt.legend()
plt.show()




# =============================================================================
#  Fossil fuel component CO2ff: 
# =============================================================================

#CO2meas monatlicher MW berechnen:
Sampleliste = Sampleliste.drop(index=[258, 259,382,385]) #diese sehen nach fehlerhaften messwerten aus
Sampleliste["Probendatum"] = pd.to_datetime(Sampleliste["Probendatum"], errors="coerce")
Sampleliste = Sampleliste.dropna(subset=["Probendatum", "CO2_kalibriert"]) #CO2_dry_sync 212.5m vorher benutzt
Sampleliste["Probendatum"] = pd.to_datetime(Sampleliste["Probendatum"])
Sampleliste["Monat"] = Sampleliste["Probendatum"].dt.to_period("M")  # Jahr-Monat extrahieren
monthly_means_CO2meas = Sampleliste.groupby("Monat")["CO2_kalibriert"].mean().reset_index()


#croppen, sd gleiche Grösse wie bero&jfj: (zb bei sigma=2: 2017-02 aus jfj rausnehmen)
gemeinsame_daten = pd.Series(sorted(set(monthly_means_JFJ_filtered['Datum']) & set(monthly_means_CO2meas['Monat'])))
monthly_means_CO2meas_filtered = monthly_means_CO2meas[monthly_means_CO2meas['Monat'].isin(gemeinsame_daten)].reset_index(drop=True)
monthly_means_JFJ_filtered = monthly_means_JFJ[monthly_means_JFJ['Datum'].isin(gemeinsame_daten)].reset_index(drop=True)
monthly_means_bero_filtered = monthly_means_bero[monthly_means_bero['Datum'].isin(gemeinsame_daten)].reset_index(drop=True)


# In Formel einsetzen:
CO2ff = monthly_means_CO2meas_filtered["CO2_kalibriert"]*(monthly_means_JFJ_filtered["D14C"]-monthly_means_bero_filtered["D14C_bero_monthlymean"])/(monthly_means_JFJ_filtered["D14C"]+1000)
t_CO2ff = monthly_means_bero_filtered["datetime"]
  

#plot zsm mit D14C Beromünster:
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

axs[0].plot(t,D14C_bero_mm,".", color="blue", label="$\Delta^{14}\mathrm{C}$ Beromünster (filtered, monthly average)")
axs[0].plot(t,D14C_bero_mm,color="grey")
axs[0].fill_between(t,D14C_bero_mm-D14C_bero_mm_err,D14C_bero_mm+D14C_bero_mm_err,color="blue",alpha=0.2,label="error width")
axs[0].set_ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
axs[0].legend()
axs[0].grid()

axs[1].plot(t_CO2ff, CO2ff, ".",color="purple", label="calculated CO2ff (filtered)")
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

DatenCO2_JFJ = np.genfromtxt("ICOS_ATC_OBSPACK-Europe-L2-2024_JFJ_13.9_CTS.CO2",delimiter=";") 
DatenCO2_JFJ = pd.DataFrame(DatenCO2_JFJ[:,2:10])

# Negative Werte entfernen:
DatenCO2_JFJ = DatenCO2_JFJ[DatenCO2_JFJ[6] >= 0]

# Mittelwerte pro Monat berechnen:
monthly_means_CO2_JFJ = DatenCO2_JFJ.groupby([0,1])[6,7].mean()
monthly_means_CO2_JFJ = monthly_means_CO2_JFJ.reset_index()

#crop:  
monthly_means_CO2_JFJ['Datum'] = pd.to_datetime({
    'year': monthly_means_CO2_JFJ[0],
    'month': monthly_means_CO2_JFJ[1],
    'day': 1  # Füge einen Standard-Tag hinzu
}).dt.to_period('M')
gemeinsame_daten = pd.Series(sorted(set(monthly_means_JFJ_filtered['Datum']) & set(monthly_means_CO2_JFJ["Datum"])))
monthly_means_CO2_JFJ_filtered = monthly_means_CO2_JFJ[monthly_means_CO2_JFJ['Datum'].isin(gemeinsame_daten)].reset_index(drop=True)


# In Formel einsetzen:
CO2bio = monthly_means_CO2meas_filtered["CO2_kalibriert"] - monthly_means_CO2_JFJ_filtered[6] - CO2ff  


#plot D14C & CO2ff & CO2bio:
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))
axs[0].plot(t,D14C_bero_mm,".", color="blue", label="$\Delta^{14}\mathrm{C}$ Beromünster ")
axs[0].plot(t,D14C_bero_mm,color="grey")
axs[0].fill_between(t,D14C_bero_mm-D14C_bero_mm_err,D14C_bero_mm+D14C_bero_mm_err,color="blue",alpha=0.2)
axs[0].plot(t,D14C_jfj_mm,".",color="purple",label="$\Delta^{14}\mathrm{C}$ Jungfraujoch") 
axs[0].plot(t,D14C_jfj_mm,color="red") 
axs[0].fill_between(t,D14C_jfj_mm-D14C_jfj_mm_err,D14C_jfj_mm+D14C_jfj_mm_err,color="red",alpha=0.2)
axs[0].set_ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
axs[0].legend()
axs[0].grid()

axs[1].plot(t_CO2ff, CO2ff, ".",color="darkblue", label="calculated $\mathrm{CO}_{2\mathrm{ff}}$ without correction")
axs[1].plot(t_CO2ff, CO2ff, color="cyan")
axs[1].set_ylabel("$\mathrm{CO}_{2\mathrm{ff}}$ [ppm]")
axs[1].legend()
axs[1].grid()

axs[2].plot(t_CO2ff, CO2bio, ".",color="green", label="calculated $\mathrm{CO}_{2\mathrm{bio}}$ without corrections")
axs[2].plot(t_CO2ff, CO2bio, color="lightgreen")
axs[2].set_xlabel("time [year]")
axs[2].set_ylabel("$\mathrm{CO}_{2\mathrm{bio}}$ [ppm]")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()




# =============================================================================
# Correction NPP Simulation: (2021-01 to 2021-12) 
# =============================================================================

DatenNPP_brm = pd.read_csv("SIM_NUC_D14CO2_BRM.csv",delimiter=";")
DatenNPP_brm['SampDate'] = pd.to_datetime(DatenNPP_brm['SampDate'])  
DatenNPP_brm = DatenNPP_brm.sort_values(by='SampDate').reset_index(drop=True)

D14C_meas = pd.DataFrame({'Probedatum': d_result2_bereinigt['Probendatum'],'D14C_bero': D14C_bero_bereinigt})



######################## von D14C_meas D14C_NPP_simuliert abziehen (die, die gleiches datum haben):

# DataFrames zusammenführen (inner join, um nur übereinstimmende Daten zu behalten)
merged_df = pd.merge(D14C_meas, DatenNPP_brm, left_on='Probedatum', right_on='SampDate', how='inner')

# Subtraktion:
merged_df['D14C_korrigiert'] = merged_df['D14C_bero'] - merged_df['D_D14CO2_NPP_all']
D14C_meas_korrigiert = merged_df[['Probedatum', 'D14C_korrigiert']]


########################## monatlicher mittelwert von D14C_meas_korrigiert:
    
D14C_meas_korrigiert["Probedatum"] = D14C_meas_korrigiert["Probedatum"].dt.to_period("M")    
mm_D14C_meas_korr = D14C_meas_korrigiert.groupby("Probedatum")[["D14C_korrigiert"]].mean()
mm_D14C_meas_korr = mm_D14C_meas_korr.reset_index()
    
    

########################## CO2 fossil fuel mit NPP-correction berechnen:

gemeinsame_daten = pd.Series(sorted(set(monthly_means_JFJ_filtered['Datum']) & set(mm_D14C_meas_korr["Probedatum"])))
monthly_means_JFJ_filtered_npp = monthly_means_JFJ_filtered[monthly_means_JFJ_filtered['Datum'].isin(gemeinsame_daten)].reset_index(drop=True)
monthly_means_CO2meas_filtered_npp = monthly_means_CO2meas_filtered[monthly_means_CO2meas_filtered["Monat"].isin(gemeinsame_daten)].reset_index(drop=True)
mm_D14C_meas_korr = mm_D14C_meas_korr[mm_D14C_meas_korr["Probedatum"].isin(gemeinsame_daten)].reset_index(drop=True)  

CO2ff_corr_sim = monthly_means_CO2meas_filtered_npp["CO2_kalibriert"]*(monthly_means_JFJ_filtered_npp["D14C"]-mm_D14C_meas_korr["D14C_korrigiert"])/(monthly_means_JFJ_filtered_npp["D14C"]+1000)
t3 = mm_D14C_meas_korr["Probedatum"].dt.to_timestamp()


#plot zsm (mit und ohne corr)
t3_CO2ff = t_CO2ff[(t_CO2ff >= pd.to_datetime('2021-01-01')) & 
            (t_CO2ff <= pd.to_datetime('2021-12-01'))].copy()
index3 = t3_CO2ff.index[0]



# =============================================================================
# Correction NPP mean value of simulation (2020-01-2020-12)
# =============================================================================

# MW der Differenz von CO2ff zu CO2ff_corr_sim:
differenz = np.array(CO2ff_corr_sim) -np.array(CO2ff[index3:])
MW_differenz = np.mean(differenz)

t2_CO2ff = t_CO2ff[(t_CO2ff >= pd.to_datetime('2020-01-01')) & 
            (t_CO2ff <= pd.to_datetime('2020-12-01'))].copy()
index2 = t2_CO2ff.index[0]

CO2ff_corr_sim_MW = CO2ff[index2:index3] + MW_differenz

t2 = t_CO2ff[index2:index3]


# =============================================================================
# Correction NPP Berhanu mean value: 1.6 promille (2015-12 to 2019-12)
# =============================================================================

t1 = t_CO2ff[:index2]

a = monthly_means_CO2meas_filtered["CO2_kalibriert"][:index2]
b_ = np.array(monthly_means_bero_filtered["D14C_bero_monthlymean"][:index2])
b=[]
for i in range(len(b_)):
    b.append(b_[i]-1.6)
b=np.array(b)

c = monthly_means_JFJ_filtered["D14C"][:index2]


CO2ff_corr_berhanu = a*(c-b)/(c+1000)


# =============================================================================
# Plot merged: berhanu mean value & simulation mean value & simulation:
# =============================================================================
plt.figure(figsize=(10, 5))
plt.xlabel("time [year]")
plt.ylabel("$\mathrm{CO}_{2\mathrm{ff}}$ [ppm]")
plt.plot(t_CO2ff,CO2ff,color="cyan")
plt.plot(t_CO2ff,CO2ff,".",color="blue",alpha=0.5,label="without nuclear correction")
plt.plot(t1,CO2ff_corr_berhanu,color="violet",label="$\Delta^{14}\mathrm{C}_\mathrm{nucl}$=1.6‰")
plt.plot(t1,CO2ff_corr_berhanu,".",color="red",alpha=0.5,label="with nuclear correction")
plt.plot(t2,CO2ff_corr_sim_MW,color="lightgreen",label="simulation mean value")
plt.plot(t2,CO2ff_corr_sim_MW,".",color="red",alpha=0.5)
plt.plot(t3,CO2ff_corr_sim,color="sienna",label="simulation")
plt.plot(t3,CO2ff_corr_sim,".",color="red",alpha=0.5)
plt.grid()
plt.legend()
plt.show()


CO2ff_corr_gesamt = pd.concat([CO2ff_corr_berhanu, CO2ff_corr_sim_MW, CO2ff_corr_sim], ignore_index=True)
durchschnittliche_korr_nukl= (CO2ff_corr_gesamt - CO2ff).mean()


# =============================================================================
# Δ14C_bio = 80‰, -3‰, Δ14C_bg
# =============================================================================

CO2ff_bio80 = (monthly_means_CO2_JFJ_filtered[6]*(monthly_means_JFJ_filtered["D14C"]-80)-monthly_means_CO2meas_filtered["CO2_kalibriert"]*(monthly_means_bero_filtered["D14C_bero_monthlymean"]-80))/(80+1000)
CO2ff_bio3 = (monthly_means_CO2_JFJ_filtered[6]*(monthly_means_JFJ_filtered["D14C"]-(-3))-monthly_means_CO2meas_filtered["CO2_kalibriert"]*(monthly_means_bero_filtered["D14C_bero_monthlymean"]-(-3)))/((-3)+1000)



#Plot von alles drei varianten (gleich bg, 80‰, -3‰)
plt.figure(figsize=(10, 5))
plt.xlabel("time [year]")
plt.ylabel("$\mathrm{CO}_{2\mathrm{ff}}$ [ppm]")
plt.plot(t_CO2ff,CO2ff_bio80,color="green")
plt.plot(t_CO2ff,CO2ff_bio3,color="red")
plt.plot(t_CO2ff,CO2ff,color="cyan")
plt.plot(t_CO2ff,CO2ff,".",color="cyan",label="$\Delta^{14}\mathrm{C}_\mathrm{bio}$ = $\Delta^{14}\mathrm{C}_\mathrm{bg}$ (no correction)")
plt.plot(t_CO2ff,CO2ff_bio80,".",color="green",label="$\Delta^{14}\mathrm{C}_\mathrm{bio}$ = 80‰")
plt.plot(t_CO2ff,CO2ff_bio3,".",color="red",label="$\Delta^{14}\mathrm{C}_\mathrm{bio}$ = -3‰")
plt.grid()
plt.legend()
plt.show()

# =============================================================================
# BOTH CORRECTIONS:
# =============================================================================

merged_df['Probedatum'] = pd.to_datetime(merged_df['Probedatum'])
monthly_avg = merged_df.groupby(merged_df['Probedatum'].dt.to_period('M'))['D_D14CO2_NPP_all'].mean().reset_index()
monthly_avg['Probedatum'] = monthly_avg['Probedatum'].dt.to_timestamp()
monthly_avg.columns = ['Monat', 'D_D14CO2_NPP_all_MW']
monthly_avg_sim = monthly_avg[:12]

meanvalue_sim= monthly_avg_sim["D_D14CO2_NPP_all_MW"].mean()
mv_sim=[]
for i in range(12):
    mv_sim.append(meanvalue_sim)
mv_sim=np.array(mv_sim)

mv_berhanu=[]
for i in range(len(b_)):
    mv_berhanu.append(1.6)
mv_berhanu=np.array(mv_berhanu)

simulation_values= np.array(monthly_avg_sim["D_D14CO2_NPP_all_MW"])
simulierte_daten= np.concatenate((mv_berhanu, mv_sim, simulation_values)) 

CO2ff_beidecorr = (monthly_means_CO2_JFJ_filtered[6]*(monthly_means_JFJ_filtered["D14C"]-(-3))-monthly_means_CO2meas_filtered["CO2_kalibriert"]*(monthly_means_bero_filtered["D14C_bero_monthlymean"]-simulierte_daten-(-3)))/((-3)+1000)
CO2bio_beidecorr = monthly_means_CO2meas_filtered["CO2_kalibriert"] - monthly_means_CO2_JFJ_filtered[6] - CO2ff_beidecorr

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))

axs[0].plot(t,D14C_bero_mm,".", color="blue", label="$\Delta^{14}\mathrm{C}$ Beromünster ")
axs[0].plot(t,D14C_bero_mm,color="grey")
axs[0].fill_between(t,D14C_bero_mm-D14C_bero_mm_err,D14C_bero_mm+D14C_bero_mm_err,color="blue",alpha=0.2)
axs[0].plot(t,D14C_jfj_mm,".",color="purple",label="$\Delta^{14}\mathrm{C}$ Jungfraujoch") 
axs[0].plot(t,D14C_jfj_mm,color="red") 
axs[0].fill_between(t,D14C_jfj_mm-D14C_jfj_mm_err,D14C_jfj_mm+D14C_jfj_mm_err,color="red",alpha=0.2)
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
# Error for CO2ff: 
# =============================================================================

s_D14Cmeas = monthly_means_bero_filtered["D14C_bero_err_monthlymean"]
s_D14Cbg = monthly_means_JFJ_filtered["u(D14C)"]
s_CO2meas = Sampleliste["CO2_kalibr_err"].iloc[0:len(s_D14Cmeas)].reset_index(drop=True)
s_CO2bg = monthly_means_CO2_JFJ_filtered[7]

D14C_sim = CO2ff_corr_gesamt - CO2ff

f1= (monthly_means_JFJ_filtered["D14C"]-(-3))/((-3)+1000)
f2= monthly_means_CO2_JFJ_filtered[6]/((-3)+1000)
f3= (monthly_means_bero_filtered["D14C_bero_monthlymean"]-D14C_sim-(-3))/((-3)+1000)
f4= monthly_means_CO2meas["CO2_kalibriert"]/((-3)+1000)

s_CO2ff = np.sqrt(f1**2*s_CO2bg**2+f2**2*s_D14Cbg**2+f3**2*s_CO2meas**2+f4**2*s_D14Cmeas**2)



# =============================================================================
# Error for CO2bio: 
# =============================================================================

s_CO2bio = np.sqrt(s_CO2meas**2 + s_CO2bg**2 + s_CO2ff**2)


# =============================================================================
# SLOPE of CO2ff&CO2bio:
# =============================================================================
t_num = (t_CO2ff - t_CO2ff.min()) / pd.Timedelta(days=365.25)
t_num = t_num.astype(float)

slope_CO2ff, intercept_CO2ff, r_value_CO2ff, p_value_CO2ff, std_err_CO2ff = linregress(t_num, CO2ff_beidecorr)
slope_CO2bio, intercept_CO2bio, r_value_CO2bio, p_value_CO2bio, std_err_CO2bio = linregress(t_num, CO2bio_beidecorr)
lin_regr_bio = intercept_CO2bio + slope_CO2bio*t_num


overall_increase_bio=lin_regr_bio.iloc[-1]-lin_regr_bio.iloc[0]


# =============================================================================
# Final plot, best approx:
# =============================================================================

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))

axs[0].plot(t,D14C_bero_mm,color="grey")
axs[0].plot(t,D14C_bero_mm,".", color="blue", label="$\Delta^{14}\mathrm{C}$ Beromünster ")
axs[0].fill_between(t,D14C_bero_mm-D14C_bero_mm_err,D14C_bero_mm+D14C_bero_mm_err,color="blue",alpha=0.2)
axs[0].plot(t,D14C_jfj_mm,color="red") 
axs[0].plot(t,D14C_jfj_mm,".",color="purple",label="$\Delta^{14}\mathrm{C}$ Jungfraujoch") 
axs[0].fill_between(t,D14C_jfj_mm-D14C_jfj_mm_err,D14C_jfj_mm+D14C_jfj_mm_err,color="red",alpha=0.2)
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
plt.plot(t_CO2ff,lin_regr_bio,'r--',label="linear regression")
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

#CO2ff_final.to_csv("C:/Users/maxin/OneDrive/Desktop/Bachelorarbeit/14C/Daten/CO2ff_finalfinal_BRM.csv")




