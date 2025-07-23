# -*- coding: utf-8 -*-
"""
Created on Tue May  6 10:41:48 2025
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
import matplotlib.dates as mdates

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


#Daten BRM:
DatenBRM = pd.read_csv("BRM_CO_MM10.csv",header=4, encoding='latin1',delimiter=";",names=["time","CO","CO_212"]) #Einheit: ppb (=10^-9)
DatenBRM['time'] = pd.to_datetime(DatenBRM['time'])
DatenBRM["CO_212"]=DatenBRM["CO_212"]*10**(-3) #ppm
DatenBRM["CO"]=DatenBRM["CO"]*10**(-3) #ppm
DatenBRM["CO_err"]=2*10**(-3) #error: 2 ppb


#attach overlapping data:
Daten = pd.merge(DatenICOS, DatenBRM, left_on='Datum', right_on='time', how='inner')


# =============================================================================
# with RCO from Berhanu paper:
# =============================================================================

#RCO: Berhanu-Wert über alle Jahreszeiten
RCO_berhanu  = 13.4*10**(-3)
RCO_berhanu_err = 1.3*10**(-3)


#Berechnung CO2ff_CO:
Daten["del_CO"] = (Daten["CO_212"]-Daten["CO_jfj"])
Daten["CO2ff_CO"] = Daten["del_CO"]/RCO_berhanu


# =============================================================================
# RCO selber berechnen: 
# =============================================================================
CO2ff_final = pd.read_csv("CO2ff_finalfinal_BRM.csv",header=1, encoding='latin1',delimiter=",",names=["time","CO2ff","s_CO2ff"]) 

CO2ff_final['time'] = pd.to_datetime(CO2ff_final['time'])
Daten['time'] = pd.to_datetime(Daten['time'])

Daten_RCO = pd.merge(CO2ff_final, Daten[['time', 'del_CO']], on='time', how='inner')

Daten_RCO["RCO"] = Daten_RCO["del_CO"]/Daten_RCO["CO2ff"]

#### Median von RCO:
RCO_Median = Daten_RCO["RCO"].median()


#### RCO: linearer Fit: y = m*x + b
mask = np.isfinite(Daten_RCO["CO2ff"]) & np.isfinite(Daten_RCO["del_CO"])
x = Daten_RCO["CO2ff"][mask]
y = Daten_RCO["del_CO"][mask]
m, b = np.polyfit(x, y, 1)

#plot linearer fit:
plt.scatter(x,y, label='Daten')
plt.plot(x, m*x+ b, color='red', label='Linearer Fit')
plt.legend()
plt.xlabel("ΔCO2ff [ppm]")
plt.ylabel("ΔCO [ppm]")
plt.show()


#### RCO: geometric mean regression:
r, _ = pearsonr(x, y)
sx = np.std(x, ddof=1)
sy = np.std(y, ddof=1)
slope = np.sign(r) * (sy / sx)
intercept = np.mean(y) - slope * np.mean(x)
#plot:
plt.scatter(x, y, label='Daten')
plt.plot(x, slope * x + intercept, 'r-', label='Geometric Mean Regression')
plt.legend()
plt.xlabel("ΔCO2ff [ppm]")
plt.ylabel("ΔCO [ppm]")
plt.show()


# =============================================================================
# ################# CO2ff_CO mit RCO_Median:
# =============================================================================
CO2ff_CO_selber = Daten["del_CO"]/RCO_Median
Daten["CO2ff_CO_selber"]=CO2ff_CO_selber

###outliers:
argmax=Daten['CO2ff_CO_selber'].idxmax()
argmin=Daten['CO2ff_CO_selber'].idxmin()
Daten = Daten.drop([argmax, argmin])
CO2ff_CO_selber = CO2ff_CO_selber.drop([argmax, argmin])

###only winter/summer:
Daten['Monat'] = Daten['Datum'].dt.month
winter = Daten[Daten['Monat'].isin([12, 1, 2])].copy() #dez,jan,feb
sommer = Daten[Daten['Monat'].isin([6, 7, 8])].copy() #juni,juli,aug


#whole time series:
plt.figure(figsize=(10, 5))
plt.xlabel("time")
plt.ylabel("$\mathrm{CO}_{2\mathrm{ff}}^{\mathrm{CO}}$ [ppm]")
plt.plot(Daten["Datum"],CO2ff_CO_selber,color="cyan",label="high-resolution time series with CO measurements (RCO=0.017)")
plt.plot(Daten["Datum"],CO2ff_CO_selber,",",color="blue")
plt.plot(CO2ff_final['time'],CO2ff_final["CO2ff"],".",color="purple",label="$\mathrm{CO}_{2\mathrm{ff}}$ from $^{14}\mathrm{C}$ measurements")
plt.legend()
plt.xlim(pd.to_datetime('2016-12-13'), pd.to_datetime('2024-03-31'))
plt.show()    

#visualization summer/winter:
plt.figure(figsize=(10, 5))
plt.xlabel("time")
plt.ylabel("$\mathrm{CO}_{2\mathrm{ff}}^{\mathrm{CO}}$ [ppm]")
plt.plot(Daten["Datum"],CO2ff_CO_selber,color="cyan")
plt.plot(Daten["Datum"],CO2ff_CO_selber,",",color="blue")
plt.plot(winter["Datum"],winter["CO2ff_CO_selber"],".",color="darkgrey",label="december, january, february")
plt.plot(sommer["Datum"],sommer["CO2ff_CO_selber"],".",color="pink",label="june, july, august")
plt.plot([], [], ' ', label=f"RCO={RCO_Median:.3f}")  # dummyplot
plt.legend()
plt.show()    



### plots of 2017:
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharey=True)
ylabel = r"$\mathrm{CO}_{2\mathrm{ff}}^{\mathrm{CO}}$ [ppm]"

# Plot 1: whole Jahr
axs[0].plot(Daten["Datum"], CO2ff_CO_selber, color="cyan",label="$\mathrm{CO}_{2\mathrm{ff}}^{\mathrm{CO}}$ with RCO=0.017")
axs[0].plot(Daten["Datum"], CO2ff_CO_selber, ",", color="blue")
axs[0].plot(CO2ff_final['time'],CO2ff_final["CO2ff"],".",color="purple",label="$\mathrm{CO}_{2\mathrm{ff}}$ from $^{14}\mathrm{C}$ measurements")
axs[0].set_xlim(pd.to_datetime('2017-01-01'), pd.to_datetime('2017-12-31'))
axs[0].set_ylim(-10, 25)
axs[0].set_ylabel(ylabel)
axs[0].legend()
axs[0].set_title("2017")
# Plot 2: Januar
axs[1].plot(Daten["Datum"], CO2ff_CO_selber, color="cyan")
axs[1].plot(Daten["Datum"], CO2ff_CO_selber, ".", color="blue")
axs[1].set_xlim(pd.to_datetime('2017-01-01'), pd.to_datetime('2017-01-31'))
axs[1].set_ylim(-10, 25)
axs[1].set_ylabel(ylabel)
axs[1].set_title("january 2017")
# Plot 3: Juli
axs[2].plot(Daten["Datum"], CO2ff_CO_selber, color="cyan")
axs[2].plot(Daten["Datum"], CO2ff_CO_selber, ".", color="blue")
axs[2].set_xlim(pd.to_datetime('2017-07-01'), pd.to_datetime('2017-07-31'))
axs[2].set_ylim(-10, 25)
axs[2].set_xlabel("Time")
axs[2].set_ylabel(ylabel)
axs[2].set_title("july 2017")

plt.tight_layout()
plt.show()


# day of 2017:
fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
dates = [
    ('2016-12-23', '2016-12-24'),
    ('2016-12-24', '2016-12-25'),
    ('2016-12-25', '2016-12-26'),
    ('2016-12-26', '2016-12-27'),
]
time_formatter = mdates.DateFormatter('%H:%M')

for i, (start, end) in enumerate(dates):
    ax = axs[i]
    ax.set_xlabel("time")
    if i == 0:
        ax.set_ylabel(r"$\mathrm{CO}_{2\mathrm{ff}}^{\mathrm{CO}}$ [ppm]")
    ax.plot(Daten["Datum"], CO2ff_CO_selber, color="cyan")
    ax.plot(Daten["Datum"], CO2ff_CO_selber, ".", color="blue")
    ax.plot([], [], ' ', label=f"RCO={RCO_Median:.3f}")
    ax.legend()
    ax.set_xlim(pd.to_datetime(start), pd.to_datetime(end))
    ax.set_title(f"{start} – {end}")
    ax.xaxis.set_major_formatter(time_formatter) 
    ax.tick_params(axis='x', rotation=45)  

plt.tight_layout()
plt.show()

print("winter average=",winter["CO2ff_CO_selber"].mean()) #ppm
print("summer average=",sommer["CO2ff_CO_selber"].mean()) #ppm



# =============================================================================
# SEASONAL cycle:
# =============================================================================

#high resolution months:
Daten['Datum'] = pd.to_datetime(Daten['Datum'])
Daten['Monat'] = Daten['Datum'].dt.month
monatliche_mittelwerte = Daten.groupby('Monat')['CO2ff_CO_selber'].mean()
monatliche_mittelwerte = monatliche_mittelwerte.sort_index()

#high resolution weeks:
Daten['Woche'] = Daten['Datum'].dt.isocalendar().week
woechentliche_mittelwerte = Daten.groupby('Woche')['CO2ff_CO_selber'].mean()
woechentliche_mittelwerte = woechentliche_mittelwerte.sort_index()

#C14 months:
CO2ff_final['time'] = pd.to_datetime(CO2ff_final['time'])
CO2ff_final['Monat'] = CO2ff_final['time'].dt.month
monatliche_mittelwerte_c14 = CO2ff_final.groupby('Monat')['CO2ff'].mean()
monatliche_mittelwerte_c14 = monatliche_mittelwerte_c14.sort_index()



###PLOT:
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(
    woechentliche_mittelwerte.index, 
    woechentliche_mittelwerte.values, 
    label='weekly average of $\mathrm{CO}_{2\mathrm{ff}}^{\mathrm{CO}}$', 
    color='gray', linestyle='--', marker='.', alpha=0.7
)
ax1.plot(
    monatliche_mittelwerte.index * 4.3, 
    monatliche_mittelwerte.values, 
    label='monthly average of $\mathrm{CO}_{2\mathrm{ff}}^{\mathrm{CO}}$', 
    color='blue', linewidth=2, marker='o'
)
ax1.plot(
    monatliche_mittelwerte_c14.index * 4.3, 
    monatliche_mittelwerte_c14.values, 
    label='monthly average of $^{14}C$ measurements', 
    color='purple', linewidth=2, marker='o'
)

ax1.set_xlabel("calendar week")
ax1.set_ylabel("$\mathrm{CO}_{2\mathrm{ff}}$ [ppm]")
ax1.legend()
ax1.grid(True)
ax1.set_xticks(range(0, 55, 4))

ax2 = ax1.twiny()
ax2.spines["bottom"].set_position(("outward", 40))
ax2.xaxis.set_ticks_position("bottom")
ax2.xaxis.set_label_position("bottom")
ax2.spines["top"].set_visible(False)

monate = list(range(1, 13))
wochen_ticks = [monat * 4.3 for monat in monate]
ax2.set_xticks(wochen_ticks)
ax2.set_xticklabels(monate)
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel("month")

plt.tight_layout()
plt.show()




# =============================================================================
# DIURNAL cycle:
# =============================================================================

#alles zusammen (average over all seasons):
Daten['Stunde'] = Daten['Datum'].dt.hour
stunden_mittelwerte = Daten.groupby('Stunde')['CO2ff_CO_selber'].mean()

plt.figure(figsize=(10, 4))
plt.plot(stunden_mittelwerte.index, stunden_mittelwerte.values, marker='o')
plt.xticks(range(0, 24))
plt.xlabel("Stunde des Tages")
plt.ylabel("CO2ff_CO_selber [ppm]")
plt.title("Stündlicher Mittelwert über alle Tage")
plt.grid(True)
plt.tight_layout()
plt.show()


#average over different seasons:
monatsgruppen = {
    'Dec-Jan-Feb': [12, 1, 2],
    'Mar-Apr-May': [3, 4, 5],
    'Jun-Jul-Aug': [6, 7, 8],
    'Sep-Oct-Nov': [9, 10, 11]
}

plt.figure(figsize=(14, 10))

for i, (gruppe, monate) in enumerate(monatsgruppen.items(), 1):
    datagruppe = Daten[Daten['Monat'].isin(monate)]
    stunden_mw = datagruppe.groupby('Stunde')['CO2ff_CO_selber'].mean()
    
    plt.subplot(2, 2, i)
    plt.plot(stunden_mw.index, stunden_mw.values, marker='o')
    plt.title(f'{gruppe}')
    plt.xlabel('hour of a day')
    plt.ylabel('$\mathrm{CO}_{2\mathrm{ff}}$ [ppm]')
    plt.xticks(range(0, 24))
    plt.grid(True)

plt.tight_layout()
plt.show()


#separate plots:













