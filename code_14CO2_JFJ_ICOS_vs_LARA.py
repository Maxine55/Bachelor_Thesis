# -*- coding: utf-8 -*-
"""Created on Tue Feb 25 16:38:15 2025@author: maxine"""

from math import *
import time as tt
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from datetime import datetime
import scipy.stats as stats

# =============================================================================
# ICOS-Daten:
# =============================================================================

DatenICOS=np.genfromtxt("ICOS_ATC_L2_L2-2024.1_JFJ_6.0_779.14C.txt",skip_header=42,delimiter=";") 

WeightedStdErr=DatenICOS[:,11]
AnalyticalStdev=DatenICOS[:,14]

D14C_ICOS=DatenICOS[:,10]
time_ICOS=DatenICOS[:,7] #decimal time in years

Err_up_ICOS = D14C_ICOS + WeightedStdErr
Err_down_ICOS = D14C_ICOS - WeightedStdErr

#plot ICOS JFJ
plt.figure(figsize=(10, 5))
plt.title("ICOS: $\Delta^{14}\mathrm{C}$ Jungfraujoch")
plt.xlabel("time [year]")
plt.ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
plt.plot(time_ICOS,D14C_ICOS,color="blue")
plt.plot(time_ICOS,D14C_ICOS,".",color="green",label="ICOS data: continuous measurement")
plt.fill_between(time_ICOS,Err_down_ICOS,Err_up_ICOS,color="lightblue",alpha=0.7,label="error width")
plt.legend()
plt.grid()
plt.show()



# =============================================================================
# LARA-Daten:
# =============================================================================

Daten_LARA=np.genfromtxt("D14CO2_JFJ_LARA_since_April_2023.txt",skip_header=1,dtype=None,encoding="utf-8",delimiter=";") 

D14C_LARA = Daten_LARA["f3"]
D14C_LARA_err = Daten_LARA["f4"]

time_LARA = np.char.split(Daten_LARA["f2"], " ")
time_LARA = np.array([d[0] for d in time_LARA]) 

Err_up_LARA = D14C_LARA + D14C_LARA_err
Err_down_LARA = D14C_LARA - D14C_LARA_err

plt.figure(figsize=(10, 5))
plt.title("LARA: $\Delta^{14}\mathrm{C}$ Jungfraujoch: 2023-2025")
plt.xlabel("time")
plt.ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
plt.plot(time_LARA,D14C_LARA,color="blue")
plt.plot(time_LARA,D14C_LARA,".",color="green",label="LARA data: night-only measurement")
plt.fill_between(time_LARA,Err_down_LARA,Err_up_LARA,color="lightblue",alpha=0.7,label="error width")
plt.xticks(rotation=60)
plt.legend()
plt.grid()
plt.show()



# =============================================================================
# ####################### JFJ: ICOS&LARA whole time series: ####################
# =============================================================================
dataT=np.genfromtxt("D14CO2_JFJ_LARA_since_April_2023.txt",skip_header=1,dtype=None,encoding="utf-8",delimiter=";") 

SampDate=dataT["f2"]
D14CO2=dataT["f3"]
D14CO2_err=dataT["f4"]


###############(chatgpt)datum in dezimalzahl jahr:
def datetime_to_decimal_year(datum_str):
    dt = datetime.strptime(datum_str, "%d.%m.%Y %H:%M")
    year = dt.year
    start_of_year = datetime(year, 1, 1)
    start_of_next_year = datetime(year + 1, 1, 1)
    seconds_since_start = (dt - start_of_year).total_seconds()
    total_seconds_in_year = (start_of_next_year - start_of_year).total_seconds()
    return year + (seconds_since_start / total_seconds_in_year)

decimal_year = np.vectorize(datetime_to_decimal_year)(SampDate)
################

#ICOS und S&T Daten überlappend: Achtung: nur 7 messungen haben gleiche zeit. Erklärung offset? - nicht gross, evtl in messungenauigkeit drin)
plt.figure(figsize=(10, 5))
plt.title("$\Delta^{14}\mathrm{C}$ Jungfraujoch: 2016-2025")
plt.xlabel("time [year]")
plt.ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
plt.plot(time_ICOS,D14C_ICOS,color="blue")
plt.plot(time_ICOS,D14C_ICOS,".",color="green",label="ICOS data")
plt.plot(decimal_year,D14CO2,color="red")
plt.plot(decimal_year,D14CO2,".",color="purple",label="LARA data")
plt.legend()
plt.grid()
plt.show()


# =============================================================================
# ################# ICOS new: #################
# =============================================================================
xls = pd.ExcelFile("JFJ_14CO2_2021_to_2024_for_Eliza.xlsx")
Daten_SH = xls.parse(xls.sheet_names[0])

def monthly_mean(Dataframe,columnname_time,columnname1_value,columnname2_value): 
    Dataframe[columnname_time] = pd.to_datetime(Dataframe[columnname_time])  
    Dataframe = Dataframe.sort_values(by=columnname_time).reset_index(drop=True)
    Dataframe = Dataframe[Dataframe[columnname1_value] != '\\N']
    Dataframe = Dataframe[Dataframe[columnname2_value] != '\\N']
    Dataframe['year_month'] = Dataframe[columnname_time].dt.to_period('M')
    Dataframe_mean = Dataframe.groupby('year_month')[columnname1_value,columnname2_value].mean().reset_index()
    
    return Dataframe_mean


Daten_SH_mean=monthly_mean(Daten_SH,"samplingmiddate","d14cmeanvalue","d14cmeanerror")

Daten_T=pd.DataFrame(dataT).reset_index()
Daten_T_mean=monthly_mean(Daten_T, "f2", "f3","f4")
Daten_SH_mean["year_month"] = Daten_SH_mean["year_month"].astype(str)
Daten_T_mean["year_month"] = Daten_T_mean["year_month"].astype(str)

Daten_T_mean = Daten_T_mean.drop(index=8)

#plot ICOS&LARA monthly mean:
plt.figure(figsize=(10, 5))
plt.title("$\Delta^{14}\mathrm{C}$ JFJ: 2021-2024, continuous vs night, monthly averages")
plt.xlabel("time")
plt.ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
plt.plot(Daten_SH_mean["year_month"],Daten_SH_mean["d14cmeanvalue"],color="lightblue")
plt.plot(Daten_SH_mean["year_month"],Daten_SH_mean["d14cmeanvalue"],".",color="blue",label="continuous (ICOS)")
plt.plot(Daten_T_mean["year_month"],Daten_T_mean["f3"],color="lightgreen")
plt.plot(Daten_T_mean["year_month"],Daten_T_mean["f3"],".",color="green",label="night-only (LARA)")
plt.xticks(rotation=60)
plt.errorbar(Daten_SH_mean["year_month"],Daten_SH_mean["d14cmeanvalue"], yerr=Daten_SH_mean["d14cmeanerror"], fmt=" ", color="grey")
plt.errorbar(Daten_T_mean["year_month"],Daten_T_mean["f3"], yerr=Daten_T_mean["f4"], fmt=" ", color="grey")
plt.grid()
plt.legend()
plt.show()

#plot ICOS&LARA monthly mean: (only overlap) 
plt.figure(figsize=(10, 5))
plt.xlabel("time")
plt.ylabel("$\Delta^{14}\mathrm{C}$ [‰]")
plt.plot(Daten_SH_mean["year_month"].iloc[25:],Daten_SH_mean["d14cmeanvalue"].iloc[25:],color="lightblue")
plt.plot(Daten_SH_mean["year_month"].iloc[25:],Daten_SH_mean["d14cmeanvalue"].iloc[25:],".",color="blue",label="continuous measurement (ICOS)")
plt.plot(Daten_T_mean["year_month"],Daten_T_mean["f3"],color="lightgreen")
plt.plot(Daten_T_mean["year_month"],Daten_T_mean["f3"],".",color="green",label="night-only measurement (LARA)")
plt.errorbar(Daten_SH_mean["year_month"].iloc[25:],Daten_SH_mean["d14cmeanvalue"].iloc[25:], yerr=Daten_SH_mean["d14cmeanerror"].iloc[25:], fmt=" ", color="grey")
plt.errorbar(Daten_T_mean["year_month"],Daten_T_mean["f3"], yerr=Daten_T_mean["f4"], fmt=" ", color="grey")
plt.xticks(rotation=60)
plt.grid()
plt.legend()
plt.show()


# =============================================================================
# Difference: ICOS - LARA:
# =============================================================================

diff_ICOS_LARA = np.array(Daten_SH_mean["d14cmeanvalue"].iloc[25:]) - np.array(Daten_T_mean["f3"])

std = np.std(diff_ICOS_LARA)
median = np.median(diff_ICOS_LARA)

#plot difference
plt.figure(figsize=(10, 5))
plt.title("$\Delta^{14}\mathrm{C}_\mathrm{ICOS}$ - $\Delta^{14}\mathrm{C}_\mathrm{LARA}$ ")
plt.xlabel("time")
plt.ylabel("$\Delta^{14}\mathrm{C}_\mathrm{ICOS}$ - $\Delta^{14}\mathrm{C}_\mathrm{LARA}$  [‰]")
plt.plot(Daten_T_mean["year_month"],diff_ICOS_LARA,"o",color="red")
plt.xticks(rotation=60)
plt.axhline(y=0,color="black")
plt.axhline(y=median,color="blue",label="median")
plt.axhspan(median-1*std,median+1*std,color="lightblue",alpha=0.4,label="median +/- 1*std")
plt.grid()
plt.legend()
plt.show()


# =============================================================================
# weighted t-Test:
# =============================================================================

#gewichtete Mittelwerte von ICOS und LARA:
Icos = np.array(Daten_SH_mean["d14cmeanvalue"].iloc[25:])
Icos_err = np.array(Daten_SH_mean["d14cmeanerror"].iloc[25:])
Lara = np.array(Daten_T_mean["f3"])
Lara_err = np.array(Daten_T_mean["f4"])

a=0
b=0
for i in range(len(Icos)):
    a=a+Icos[i]/(Icos_err[i])**2
    b=b+1/(Icos_err[i])**2
    
x_gew_ICOS = a/b
sigm_gew_ICOS = np.sqrt(b)
print("gewichteter MW ICOS:",x_gew_ICOS,"gewichteter Fehler ICOS:",sigm_gew_ICOS)

c=0
d=0
for i in range(len(Lara)):
    c=c+Lara[i]/(Lara_err[i])**2
    d=d+1/(Lara_err[i])**2
    
x_gew_LARA = c/d
sigm_gew_LARA = np.sqrt(d)
print("gewichteter MW LARA:",x_gew_LARA,"gewichteter Fehler LARA:",sigm_gew_LARA)

#t-Wert:
t = (x_gew_ICOS-x_gew_LARA)/(np.sqrt(sigm_gew_ICOS**2+sigm_gew_LARA**2))
print("t-Wert = ",t)



#############Plot of twosided t-Test: 
f = len(Icos)   # Freiheitsgrade
alpha = 0.05  # Signifikanzniveau
t_crit = stats.t.ppf(1 - alpha/2, f)

x = np.linspace(-5, 5, 500)
y = stats.t.pdf(x, f)

plt.figure(figsize=(10,6))
plt.plot(x, y, label=f"t-distribution (dof={f})", color='black') #dof=degrees of freedom
plt.fill_between(x, 0, y, where=(x <= -t_crit) | (x >= t_crit), color='red', alpha=0.3, label="rejection area (α=0.05)")
plt.axvline(t, color='blue', linestyle='--', label=f"t = {t:.3f}")
plt.xlabel('t-value')
plt.ylabel('density')
plt.legend()
plt.grid(True)
plt.show()

#alpha sd t_crit=t:
alpha_zweiseitig = 2 * (1 - stats.t.cdf(abs(t),f))



#############Plot of onesided t-Test: 
f = len(Icos)   # Freiheitsgrade 
alpha = 0.05  # Signifikanzniveau
t_crit_rechts = stats.t.ppf(1 - alpha, f)
t_crit_links = -t_crit_rechts  # (stats.t.ppf(alpha, f))

x = np.linspace(-5, 5, 1000)
y = stats.t.pdf(x, d)

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
# Rechtsseitiger Test (ICOS<LARA)
axs[0].plot(x, y, color='black', label=f"t-Verteilung (dof={f})")
axs[0].fill_between(x, 0, y, where=(x >= t_crit_rechts), color='red', alpha=0.3, label="Ablehnungsbereich")
axs[0].axvline(t, color='blue', linestyle='--', label=f"t berechnet = {t:.3f}")
axs[0].set_title(f'einseitiger t-Test: H₀: μ_ICOS ≤ μ_LARA (rechtsseitig), α={alpha}')
axs[0].set_xlabel('t [no units]')
axs[0].set_ylabel('Dichte')
axs[0].legend()
axs[0].grid(True)
# Linksseitiger Test (ICOS>LARA)
axs[1].plot(x, y, color='black', label=f"t-Verteilung (dof={f})")
axs[1].fill_between(x, 0, y, where=(x <= t_crit_links), color='red', alpha=0.3, label="Ablehnungsbereich")
axs[1].axvline(t, color='blue', linestyle='--', label=f"t berechnet = {t:.3f}")
axs[1].set_title(f'einseitiger t-Test: H₀: μ_ICOS ≥ μ_LARA (linksseitig), α={alpha}')
axs[1].set_xlabel('t [no units]')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


#alpha sd t_crit=t:
# Einseitiger Test – rechtsseitig
alpha_rechts = 1 - stats.t.cdf(t,f)
# Einseitiger Test – linksseitig
alpha_links = stats.t.cdf(t,f)

print("###")
print("rechtseitige Nullhypothese μ_ICOS ≤ μ_LARA ablehnen, sd gilt: Daten_ICOS > Daten_LARA. Zu 36.9% wurde H₀ fälschlicherweise abgelehnt, wäre also eigentlich richtig.")
print("###")
print("zweiseitige Nullhypothese μ_ICOS = μ_LARA ablehnen, sd gilt: Daten_ICOS ≠ Daten_LARA. Zu 73.9% wurde H₀ fälschlicherweise abgelehnt, wäre also eigentlich richtig.")
print("###")
print("Es gibt keinen statistisch signifikanten Unterschied zwischen den beiden Mittelwerten auf dem 5 %-Niveau. Die Nullhypothese bleibt bestehen (muss aber nicht richtig sein).")



# =============================================================================
# PAIRED t-test:
# =============================================================================

D_mean = diff_ICOS_LARA.mean()
D_std = diff_ICOS_LARA.std(ddof=1)
n = len(diff_ICOS_LARA)

t_paired = (D_mean*np.sqrt(n))/D_std
t_crit_paired = 2.831 #literature value



######BOTH t-tests in one plot:
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

axs[0].plot(x, y, label=f"t-distribution (dof={f})", color='black') #dof=degrees of freedom
axs[0].fill_between(x, 0, y, where=(x <= -t_crit) | (x >= t_crit), color='red', alpha=0.3, label="rejection area (α=0.05)")
axs[0].axvline(t, color='blue', linestyle='--', label=f"t = {t:.3f} (weighted t-test)")
axs[0].set_xlabel('t-value')
axs[0].set_ylabel('density')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(x, y, label=f"t-distribution (dof={f})", color='black') #dof=degrees of freedom
axs[1].fill_between(x, 0, y, where=(x <= -t_crit_paired) | (x >= t_crit_paired), color='red', alpha=0.3, label="rejection area (α=0.01)")
axs[1].axvline(t_paired, color='blue', linestyle='--', label=f"t = {t_paired:.3f} (paired t-test)")
axs[1].set_xlabel('t-value')
axs[1].set_ylabel('density')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()






























