import streamlit as st
import ezc3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import tempfile

st.set_page_config(page_title="Paramètres Spatio-temporaux", layout="centered")
st.title("🦿 Paramètres Spatio-temporaux - Interface interactive")

# 1. Upload des fichiers .c3d
st.header("1. Importer un ou plusieurs fichiers .c3d dont au moins un fichier d'essai statique et un d'essai dynamique")
uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers .c3d", type="c3d", accept_multiple_files=True)

if uploaded_files:
    selected_file_statique = st.selectbox("Choisissez un fichier statique pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique = st.selectbox("Choisissez un fichier dynamique pour l'analyse", uploaded_files, format_func=lambda x: x.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_statique.read())
        statique_path = tmp.name

    statique = ezc3d.c3d(statique_path)  # acquisition statique
    labelsStat = statique['parameters']['POINT']['LABELS']['value']
    freqStat = statique['header']['points']['frame_rate']
    first_frameStat = statique['header']['points']['first_frame']
    n_framesStat = statique['data']['points'].shape[2]
    time_offsetStat = first_frameStat / freqStat
    timeStat = np.arange(n_framesStat) / freqStat + time_offsetStat

    markersStat  = statique['data']['points']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique.read())
        dynamique_path = tmp.name

    acq1 = ezc3d.c3d(dynamique_path)  # acquisition dynamique
    labels = acq1['parameters']['POINT']['LABELS']['value']
    freq = acq1['header']['points']['frame_rate']
    first_frame = acq1['header']['points']['first_frame']
    n_frames = acq1['data']['points'].shape[2]
    time_offset = first_frame / freq
    time = np.arange(n_frames) / freq + time_offset
    
    markers1 = acq1['data']['points']
    data1 = acq1['data']['points']

if st.button("Lancer le calcul des paramètres spatio-temporaux"):
    try:
        # Extraction des coordonnées
        a1, a2, b1, b2, c1, c2 = markersStat[:,labelsStat.index('LASI'),:][0, 0], markersStat[:,labelsStat.index('LANK'),:][0, 0], markersStat[:,labelsStat.index('LASI'),:][1, 0], markersStat[:,labelsStat.index('LANK'),:][1, 0], markersStat[:,labelsStat.index('LASI'),:][2, 0], markersStat[:,labelsStat.index('LANK'),:][2, 0]
        LgJambeL = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))

        d1, d2, e1, e2, f1, f2 = markersStat[:,labelsStat.index('RASI'),:][0, 0], markersStat[:,labelsStat.index('RANK'),:][0, 0], markersStat[:,labelsStat.index('RASI'),:][1, 0], markersStat[:,labelsStat.index('RANK'),:][1, 0], markersStat[:,labelsStat.index('RASI'),:][2, 0], markersStat[:,labelsStat.index('RANK'),:][2, 0]
        LgJambeR = np.sqrt((d2-d1)*(d2-d1)+(e2-e1)*(e2-e1)+(f2-f1)*(f2-f1))

        LargeurPelvis = np.abs(markersStat[:,labelsStat.index('RASI'),:][1, 0] - markersStat[:,labelsStat.index('LASI'),:][1, 0])
        
        # Détection event gauche
        # Détection des cycles à partir du marqueur LHEE (talon gauche)
        points = acq1['data']['points']
        if "LHEE" in labels:
            idx_lhee = labels.index("LHEE")
            z_lhee = points[2, idx_lhee, :]
          
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z = -z_lhee
            min_distance = int(freq * 0.8)
          
            # Détection pics
            peaks, _ = find_peaks(inverted_z, distance = min_distance, prominence = 1)
          
            # Début et fin des cycles = entre chaque pic
            lhee_cycle_start_indices = peaks[:-1]
            lhee_cycle_end_indices = peaks[1:]
            min_lhee_cycle_duration = int(0.5 * freq)
            lhee_valid_cycles = [
            (start, end) for start, end in zip(lhee_cycle_start_indices, lhee_cycle_end_indices) if (end - start) >= min_lhee_cycle_duration]
            lhee_n_cycles = len(lhee_valid_cycles)
    
        # Event toe off gauche 
        # Détection des cycles à partir du marqueur LTOE (orteil gauche )
        points = acq1['data']['points']
        if "LTOE" in labels:
            idx_ltoe = labels.index("LTOE")
            z_ltoe = points[2, idx_ltoe, :]
          
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z = -z_ltoe
            min_distance = int(freq * 0.8)
          
            # Détection pics
            peaks, _ = find_peaks(inverted_z, distance = min_distance, prominence = 1)
          
            # Début et fin des cycles = entre chaque pic
            ltoe_cycle_start_indices = peaks[:-1]
            ltoe_cycle_end_indices = peaks[1:]
            min_ltoe_cycle_duration = int(0.5 * freq)
            ltoe_valid_cycles = [(start, end) for start, end in zip(ltoe_cycle_start_indices, ltoe_cycle_end_indices) if (end - start) >= min_ltoe_cycle_duration]
            ltoe_n_cycles = len(ltoe_valid_cycles)
    
        # Détection event droite
        # Détection des cycles à partir du marqueur RHEE (talon droite)
        points = acq1['data']['points']
        if "RHEE" in labels:
            idx_rhee = labels.index("RHEE")
            z_rhee = points[2, idx_rhee, :]
          
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z = -z_rhee
            min_distance = int(freq * 0.8)
          
            # Détection pics
            peaks, _ = find_peaks(inverted_z, distance = min_distance, prominence = 1)
          
            # Début et fin des cycles = entre chaque pic
            rhee_cycle_start_indices = peaks[:-1]
            rhee_cycle_end_indices = peaks[1:]
            min_rhee_cycle_duration = int(0.5 * freq)
            rhee_valid_cycles = [(start, end) for start, end in zip(rhee_cycle_start_indices, rhee_cycle_end_indices) if (end - start) >= min_rhee_cycle_duration]
            rhee_n_cycles = len(rhee_valid_cycles)
    
        # Event toe off droit 
        # Détection des cycles à partir du marqueur RTOE (orteil gauche )
        points = acq1['data']['points']
        if "RTOE" in labels:
            idx_rtoe = labels.index("RTOE")
            z_rtoe = points[2, idx_rtoe, :]
          
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z = -z_rtoe
            min_distance = int(freq * 0.8)
          
            # Détection pics
            peaks, _ = find_peaks(inverted_z, distance = min_distance, prominence = 1)
          
            # Début et fin des cycles = entre chaque pic
            rtoe_cycle_start_indices = peaks[:-1]
            rtoe_cycle_end_indices = peaks[1:]
            min_rtoe_cycle_duration = int(0.5 * freq)
            rtoe_valid_cycles = [(start, end) for start, end in zip(rtoe_cycle_start_indices, rtoe_cycle_end_indices) if (end - start) >= min_rtoe_cycle_duration]
            rtoe_n_cycles = len(rtoe_valid_cycles)
    
        # Synchroniser Toe off et Attaque talon 
        # Droite 
        if  rhee_cycle_start_indices[0] > rtoe_cycle_start_indices [0] : 
            rtoe_cycle_start_indices = np.delete( rtoe_cycle_start_indices,0)
          
        while len(rhee_cycle_start_indices) > len(rtoe_cycle_start_indices) : 
            rhee_cycle_start_indices = np.delete (rhee_cycle_start_indices, -1)
          
          
        # Gauche 
        if  lhee_cycle_start_indices[0] > ltoe_cycle_start_indices [0] : 
            ltoe_cycle_start_indices = np.delete(ltoe_cycle_start_indices,0)
          
        while len(lhee_cycle_start_indices) > len(ltoe_cycle_start_indices) : 
            lhee_cycle_start_indices = np.delete (lhee_cycle_start_indices, -1)
    
        # Longueur pas à droite
        LgPasR = []
        for i,j in rhee_valid_cycles:
            a1, a2, b1, b2, c1, c2 = markers1[:,labels.index('RANK'),:][0,i], markers1[:,labels.index('RANK'),:][0,j], markers1[:,labels.index('RANK'),:][1,i], markers1[:,labels.index('RANK'),:][1,j], markers1[:,labels.index('RANK'),:][2,i], markers1[:,labels.index('RANK'),:][2,j]
            z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
            LgPasR.append(z)
        LgPasRmoy = np.mean(LgPasR)
        VarLgPr = np.std(LgPasR)
        LgPasRnorma = round(LgPasRmoy / LgJambeR *100,2)
        stdLgPasRnorma = round(VarLgPr / LgJambeR *100,2)
          
          
        # Longueur pas à gauche
        LgPasG = []
        for i,j in lhee_valid_cycles:
            a1, a2, b1, b2, c1, c2 = markers1[:,labels.index('LANK'),:][0,i], markers1[:,labels.index('LANK'),:][0,j], markers1[:,labels.index('LANK'),:][1,i], markers1[:,labels.index('LANK'),:][1,j], markers1[:,labels.index('LANK'),:][2,i], markers1[:,labels.index('LANK'),:][2,j]
            z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
            LgPasG.append(z)
        LgPasLmoy = np.mean(LgPasG)
        VarLgPl = np.std(LgPasG)
        LgPasLnorma = round(LgPasLmoy / LgJambeL *100,2)
        stdLgPasLnorma = round(VarLgPl / LgJambeL *100,2)
          
          
        # Vitesse de marche
        Vmarche = round(((markers1[:,labels.index('STRN'),:][0,-1]-markers1[:,labels.index('STRN'),:][0,0]) / (len(markers1[:,labels.index('STRN'),:][0,:]) / 100)) / 1000,2)
    
        # Cadence 
        Cadence_D = []
        for i,j in rhee_valid_cycles : 
            a = 1/((j-i)/freq)
            Cadence_D.append(a)
          
        Cadence_D_m = np.mean(Cadence_D)
        stdCadence_D_m = np.std(Cadence_D)
          
        Cadence_G = []
        for i,j in lhee_valid_cycles : 
            a = 1/((j-i)/freq)
            Cadence_G.append(a)
          
        Cadence_G_m = np.mean(Cadence_G)
        stdCadence_G_m = np.std(Cadence_G)
          
        Cadence_M = round(np.mean([Cadence_D_m,Cadence_G_m]),2)
        stdCadence_M = round(np.mean([stdCadence_D_m,stdCadence_G_m]),2)
    
        # Durée d'un cycle 
        # Droit : 
        DureeCycleD = []
        for i,j in rhee_valid_cycles : 
            a = (j-i)/freq
            DureeCycleD.append(a)
          
        DureeCycleD_m = round(np.mean(DureeCycleD),2)
        stdDureeCycleD_m = round(np.std(DureeCycleD),2)
          
        # Gauche : 
        DureeCycleG = []
        for i,j in lhee_valid_cycles : 
            a = (j-i)/freq
            DureeCycleG.append(a)
          
        DureeCycleG_m = round(np.mean(DureeCycleG),2)
        stdDureeCycleG_m = round(np.std(DureeCycleG),2)
          
          
        # Totale : 
        DureeCycle_M = round(np.mean([DureeCycleD_m,DureeCycleG_m]),2)
        stdDureeCycle_M = round(np.std([stdDureeCycleD_m,stdDureeCycleG_m]),2)
    
        
    
        # Phase de simple appui et de double appui 
        # Simple appui droit 
        SimpleAppui_D = []
        for i in range (0, len(lhee_cycle_start_indices)-1) : 
            a = lhee_cycle_start_indices[i+1] - ltoe_cycle_start_indices [i]
            b = a / freq / DureeCycleD_m *100
            SimpleAppui_D.append(b)
          
        SimpleAppui_D_m = round(np.mean(SimpleAppui_D),2)
        stdSimpleAppui_D_m = round(np.std(SimpleAppui_D),2)
          
        # Simple appui gauche
        SimpleAppui_G = []
        for i in range (0, len(rhee_cycle_start_indices)-1) : 
            a = rhee_cycle_start_indices[i+1] - rtoe_cycle_start_indices [i]
            b = a / freq / DureeCycleD_m *100
            SimpleAppui_G.append(b)
          
        SimpleAppui_G_m = round(np.mean(SimpleAppui_G),2)
        stdSimpleAppui_G_m = round(np.std(SimpleAppui_G),2)
          
        # Phase de double appui : 
        DoubleAppui = []
          
        if rhee_cycle_start_indices [0] > lhee_cycle_start_indices [0] : 
            for i in range (0, len(rhee_cycle_start_indices)) : 
              a = ltoe_cycle_start_indices[i] - lhee_cycle_start_indices[i] - (rtoe_cycle_start_indices[i] - rhee_cycle_start_indices[i])
              b = abs(a / freq / DureeCycleD_m *100)
              DoubleAppui.append(b)
        elif rhee_cycle_start_indices [0] < lhee_cycle_start_indices [0] : 
            for i in range (0, len(lhee_cycle_start_indices)) :
              a = rtoe_cycle_start_indices[i] - rhee_cycle_start_indices[i] - (ltoe_cycle_start_indices[i] - lhee_cycle_start_indices[i])
              b = abs(a / freq / DureeCycleD_m *100)
              DoubleAppui.append(b)
          
        DoubleAppui_m = round(np.mean(DoubleAppui),2)
        stdDoubleAppui_m = round(np.std(DoubleAppui),2)

        # Phase d'appui 
        # Droit :
        Appui_D = []
        for i in SimpleAppui_D : 
          a = i + DoubleAppui_m
          Appui_D.append(a)
        
        Appui_D_m = round(np.mean(Appui_D),2)
        stdAppui_D_m = round(np.std(Appui_D),2)

        # Gauche 
        Appui_G = []
        for i in SimpleAppui_G : 
          a = i + DoubleAppui
          Appui_G.append(a)
        
        Appui_G_m = round(np.mean(Appui_G),2)
        stdAppui_G_m = round(np.std(Appui_G),2)

        # Phase d'oscillation 
        # Droit : 
        Oscillation_D = []
        for i in  Appui_D : 
            b = 100 - i
            Oscillation_D.append(b)
          
        Oscillation_D_m = round(np.mean(Oscillation_D),2)
        stdOscillation_D_m = round(np.std(Oscillation_D),2)
          
        # Gauche : 
        Oscillation_G = []
        for i in  Appui_G : 
            b = 100 - i
            Oscillation_G.append(b)
          
        Oscillation_G_m = round(np.mean(Oscillation_G),2)
        stdOscillation_G_m = round(np.std(Oscillation_G),2)

        
        # Format tableau
        DATA = pd.DataFrame(({"Moyenne":[LgPasRnorma, LgPasLnorma, Vmarche, Cadence_M, DureeCycle_M, Appui_D_m, Appui_G_m, Oscillation_D_m, Oscillation_G_m, SimpleAppui_D_m, SimpleAppui_G_m, DoubleAppui_m], "Ecart-type":[stdLgPasRnorma, stdLgPasLnorma, 0, stdCadence_M, stdDureeCycle_M, stdAppui_D_m, stdAppui_G_m, stdOscillation_D_m, stdOscillation_G_m, stdSimpleAppui_D_m, stdSimpleAppui_G_m, stdDoubleAppui_m]}), index=["LgPasRnorma (% de Lg jambe)", "LgPasLnorma (% de Lg jambe)", "Vmarche (m/s)", "Cadence_M (pas/s)", "DureeCycle_M (s)", "Appui_D_m (% du cycle de marche)", "Appui_G_m (% du cycle de marche)", "Oscillation_D_m (% du cycle de marche)", "Oscillation_G_m (% du cycle de marche)", "SimpleAppui_D_m (% du cycle de marche)", "SimpleAppui_G_m (% du cycle de marche)", "DoubleAppui_m (% du cycle de marche)"])
    
    
        st.markdown("### 📊 Paramètres Spatio-temporaux")
        st.table(DATA)
        st.write(f"**Longueur Jambe gauche** : {LgJambeL:.2f} mm")
        st.write(f"**Longueur Jambe droite** : {LgJambeR:.2f} mm")

    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")
