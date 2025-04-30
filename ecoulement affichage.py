#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:15:35 2025

@author: marcello
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from math import sqrt
import tkinter as tk
from tkinter import messagebox


def solve_fouille_relax(L, Zsol, Zfouille, Zbp, Zfond, Zn1, Zn2, couches,
                        max_iter=100000, tol=1e-6,
                        output_excel=True, excel_filename="resultats_h.xlsx"):
    """
    Calcule l'écoulement souterrain sous palplanche par différences finies,
    avec j=0 en surface (Zn1) et j=nz-1 au fond (Zbp - 3).
    Retourne : x, z, h, ip, j_substrat, j_bp, dz, dx, nx
    """

    # --- 1) Grille
    dx = dz = 1   
    
    # 2) Hauteur totale du domaine
    height = Zsol - Zfond 
    
    # 3) Nombre de nœuds verticaux
    nz = int(height / dz) + 1

    # 4) Coordonnées réelles
    z = Zsol - np.arange(nz) * dz  
    width = 2 * L
    nx = int(width / dx) 
    x = np.linspace(0, width, nx)    
    ip_1 = nx // 4                # palplanche 1 en i=ip_1
    ip_2 = 3 * nx //4 - 1            # palplanche 2 en i=ip_2

    # Construire k_z 1D selon les couches
    k_z = np.zeros(nz)
    for j, zj in enumerate(z):
        for Z_top, Z_bot, kval in couches:
            if Z_top >= zj > Z_bot:
                k_z[j] = kval
                break
    # Matrice perméabilité 2D
    K = np.repeat(k_z[:, None], nx, axis=1)
    
    # Champ de charge hydraulique
    h = np.zeros((nz, nx))
    for i in range(nx):
        if i <= ip_1 or i >= ip_2:
            h[:, i] = Zn1
        else:
            h[:, i] = Zn2

    # --- 2) Indices clés
    j_sol = 0
    j_Zn1      =  int((Zsol - Zn1)     / dz)  # nappe aval 
    j_fouille = int((Zsol - Zfouille) / dz) # fouille 
    j_Zn2      = int((Zsol - Zn2)     / dz)  # nappe entre palplanches
    j_bp       = int((Zsol - Zbp)     / dz)  # bas de palplanche
    j_substrat = nz - 1                 # fond imperméable
    print(f"Indices calculés: j_fouille={j_fouille}, j_Zn2={j_Zn2}, j_bp={j_bp}, j_substrat = {j_substrat}, ip_1 = {ip_1}, ip_2 = {ip_2}")
    print(K)
    


    # --- 3) Conditions limites initiales
    # a) Surfaces amont/aval
    h[j_sol,    :ip_1]  = Zsol
    h[j_sol,    ip_2+1:]  = Zsol
    h[j_Zn2,    ip_1:ip_2+1]  = Zn2
    # b) Côtés
    h[:, 0]           = Zsol
    h[:, -1]     = Zsol
    
    # c) Fouille 
    for j in range(j_sol, j_fouille+1):
        for i in range(ip_1, ip_2 + 1):
            # Pour la fouille, on peut imposer la charge = altitude (pression = 0)
            h[j, i] = z[j]

    # d) Neumann à Zn1
    #h[j_Zn1, :] = h[j_Zn1 + 1, :]


    # --- 4) Boucle de calcul différence finie
    for it in range(max_iter):
        h_old = h.copy()
        for j in range(1, nz):
            for i in range(1, nx-1):
                # bord supérieur en j=0 n’est pas recalculé
                if j == nz-1:
                    # fond
                    h[j, i] = (h[j, i-1] + h[j, i+1]) / 4 + h[j-1, i] / 2
                elif i == ip_1 and j == j_bp + 1:
                    # point juste sous la palplanche 1
                    h[j, i] = (
                        h[j, i-1]/4 + h[j, i+1]/4 + h[j+1, i]/4 +
                        h[j-1, i-1]/8 + h[j-1, i+1]/8
                    )
                elif i == ip_1 and j <= j_bp:
                    # palplanche 1
                    h[j, i] = (h[j, i-1]/2 + h[j+1, i]/4 + h[j-1, i]/4)
                elif i == ip_1 + 1 and j <= j_bp and j > j_Zn2:
                    # juste à droite de la palplanche 1
                    h[j, i] = (h[j-1, i]/4 + h[j+1, i]/4 + h[j, i+1]/2)
                elif i == ip_1 - 1 and j <= j_Zn2:
                    # Juste à gauche
                    h[j, i] = (h[j, i-1]/2 + h[j+1, i]/4 + h[j-1, i]/4)
                elif i == ip_1 and j > j_bp:
                    # Dessous
                    h[j, i] = 0.25 * (
                        h[j+1, i] + h[j-1, i] + h[j, i+1] + h[j, i-1]
                    )
                elif i == ip_2 and j == j_bp + 1:
                    # point juste sous la palplanche 2
                    h[j, i] = (
                        h[j, i-1]/4 + h[j, i+1]/4 + h[j+1, i]/4 +
                        h[j-1, i-1]/8 + h[j-1, i+1]/8
                    )
                elif i == ip_2 and j <= j_bp:
                    # palplanche 2
                    h[j, i] = (h[j, i+1]/2 + h[j+1, i]/4 + h[j-1, i]/4)
                elif i == ip_2 - 1 and j <= j_bp and j > j_Zn2:
                    # juste à gauche de la palplanche 2
                    h[j, i] = (h[j-1, i]/4 + h[j+1, i]/4 + h[j, i-1]/2)
                elif i == ip_2 + 1 and j <= j_Zn2:
                    # Juste à droite
                    h[j, i] = (h[j, i+1]/2 + h[j+1, i]/4 + h[j-1, i]/4)
                elif i == ip_2 and j > j_bp:
                    # Dessous
                    h[j, i] = 0.25 * (
                        h[j+1, i] + h[j-1, i] + h[j, i+1] + h[j, i-1]
                    )
                else:
                    # Moyennes harmoniques k
                    k_e = 2 / (1/K[j, i]   + 1/K[j, i+1])
                    k_w = 2 / (1/K[j, i]   + 1/K[j, i-1])
                    k_n = 2 / (1/K[j-1, i] + 1/K[j, i])
                    k_s = 2 / (1/K[j+1, i] + 1/K[j, i])
                    # Schéma 5 points généralisé
                    num = (k_e * h[j, i+1] + k_w * h[j, i-1]
                         + k_n * h[j-1, i] + k_s * h[j+1, i])
                    den = (k_e + k_w + k_n + k_s)
                    h[j, i] = num / den

        # --- 5) Ré-imposer les mêmes CL après update
        h[j_sol,    :ip_1]  = Zsol
        h[j_sol,    ip_2+1:]  = Zsol
        h[j_Zn2,    ip_1:ip_2+1]  = Zn2
        h[:, 0]           = Zsol
        h[:, -1]     = Zsol
        for j in range(j_sol, j_fouille+1):
            for i in range(ip_1, ip_2 + 1):
                # Pour la fouille, on peut imposer la charge = altitude (pression = 0)
                h[j, i] = z[j]
        #h[j_Zn1, :] = h[j_Zn1 + 1, :]

        # Convergence ?
        if np.max(np.abs(h - h_old)) < tol:
            print(f"Convergence atteinte après {it} itérations.")
            break
    else:
        print(f"NON convergé après {max_iter} itérations")

    # --- 6) Export Excel
    if output_excel:
        pd.DataFrame(h).to_excel(excel_filename, index=False, header=False)
        print(f"Matrice h exportée vers {excel_filename}")
        

    return x, z, h, ip_1, ip_2, j_substrat, j_bp, dz, dx, nx, nz, K


def compute_flux(h, K, dx, dz):
    excel_filename_1 = 'resultats_ux.xlsx'
    excel_filename_2 = 'resultats_uz.xlsx'
    """Calcule ux et uz (vitesses de Darcy centrées)."""
    nz, nx = h.shape
    ux = np.zeros_like(h)
    uz = np.zeros_like(h)
    for j in range(nz):
        for i in range(1, nx-1):
            # moyenne harmonique sur i-1 et i+1
            k_centre = 2 / (1/K[j, i-1] + 1/K[j, i+1])
            ux[j, i] = -k_centre * (h[j, i+1] - h[j, i-1]) / (2 * dx)
    # uz au centre (j=1..nz-2)
    for j in range(1, nz-1):
        for i in range(nx):
            k_centre = 2 / (1/K[j-1, i] + 1/K[j+1, i])
            uz[j, i] = -k_centre * (h[j-1, i] - h[j+1, i]) / (2 * dz)
    pd.DataFrame(ux).to_excel(excel_filename_1, index=False, header=False)
    pd.DataFrame(uz).to_excel(excel_filename_2, index=False, header=False)
    print(f"Matrice ux exportée vers {excel_filename_1}")
    print(f"Matrice uz exportée vers {excel_filename_2}")
    return ux, uz


def compute_total_flow_zone(h, K, L, dx, dz, ip_1, ip_2, Zn1, Zn2, Zbp, Zfouille, Zfond, Zsol, nz, j_bp, nx, uz):
    
    """
    Calcule le débit avec différentes approches améliorées:
    1. Flux vertical à la base de la fouille (méthode directe)
    2. Flux vertical à la base de la fouille (à partir de uz)
    3. Flux horizontal sous les palplanches
    """
    j_fouille = int((Zsol - Zfouille) / dz)
    
    # Méthode 1: Flux vertical à j_fouille - différence finie ordre 1
    q_vertical = 0
    cells_counted = 0
    for i in range(ip_1+1, ip_2):  # Exclure les cellules des palplanches
        # perméabilité verticale moyenne
        k_v = 2/(1/K[j_fouille,i] + 1/K[j_fouille+1,i])
        # Gradient vertical au niveau du fond de fouille
        qz = -k_v * (h[j_fouille, i] - h[j_fouille + 1, i]) / dz
        q_vertical += qz * dx  # Ajouter tous les flux, pas seulement positifs
        cells_counted += 1
    
    # Méthode 2: Flux vertical à partir du champ uz préalablement calculé
    q_vertical2 = 0
    for i in range(ip_1+1, ip_2):  # Exclure les cellules des palplanches
        qz = uz[j_fouille, i]
        q_vertical2 += qz * dx  # Ajouter tous les flux
    
    # Méthode 3: Flux horizontal sous les palplanches
    q_sous_palplanche1 = 0  # Sous palplanche 1
    q_sous_palplanche2 = 0  # Sous palplanche 2
    for j in range(j_fouille, j_bp+1):
        k_h1 = 2/(1/K[j,ip_1-1] + 1/K[j,ip_1+1])
        qx1 = -k_h1 * (h[j, ip_1-1] - h[j, ip_1+1]) / (2*dx)
        q_sous_palplanche1 += qx1 * dz
        
        # Flux horizontal traversant palplanche 2
        k_h2 = 2/(1/K[j,ip_2-1] + 1/K[j,ip_2+1])
        qx2 = -k_h2 * (h[j, ip_2 - 1] - h[j, ip_2 + 1]) / (2*dx)
        q_sous_palplanche2 += qx2 * dz
    
    # Total des deux palplanches
    q_sous_palplanche = q_sous_palplanche1 + q_sous_palplanche2
    
    # Conversion en m³/h
    q_vertical *= 3600
    q_vertical2 *= 3600
    q_sous_palplanche1 *= 3600
    q_sous_palplanche2 *= 3600
    q_sous_palplanche *= 3600
    
    # Affichage des résultats
    print("j_fouille = " + str(j_fouille))
    print("nx = " + str(nx))
    print("dz = " + str(dz))
    print("dx = " + str(dx))
    print("cells counted = " + str(cells_counted))
    print(f"Débit vertical entrant (méthode différence finie): {q_vertical:.4f} m³/h/ml")
    print(f"Débit vertical entrant (à partir de uz): {q_vertical2:.4f} m³/h/ml")
    print(f"Débit sous palplanche 1: {q_sous_palplanche1:.4f} m³/h")
    print(f"Débit sous palplanche 2: {q_sous_palplanche2:.4f} m³/h")
    print(f"Débit total sous palplanches: {q_sous_palplanche:.4f} m³/h")
    
    return {
        "q_vertical": q_vertical,
        "q_vertical2": q_vertical2,
        "q_sous_palplanche1": q_sous_palplanche1,
        "q_sous_palplanche2": q_sous_palplanche2,
        "q_sous_palplanche": q_sous_palplanche,
        "cells_counted": cells_counted
    }
    

def theoretical_flow_estimate(k, L, Zn1, Zn2, Zbp, Zfouille, Zfond):
    
    """Estimation du débit théorique selon la formule de dupuit."""
    
    delta_h = 0 - Zn2    # Différence de charge
    # Loi de Darcy pour écoulement confiné
    h_sous_palplanche = Zfouille - Zbp
    gradient_hydraulique = delta_h / h_sous_palplanche
    q_darcy = k * gradient_hydraulique * L * 3600  # m³/h/ml
    
    # Affichage des résultats
    print("Zn1 = " + str(Zn1))
    print("Zn2 = " + str(Zn2))
    print("Zbp = " + str(Zbp))
    print("Delta H = " + str(delta_h))
    print("l = " + str(h_sous_palplanche))
    print("i = " + str(gradient_hydraulique))
    print("Zfouille = " + str(Zfouille))
    print(f"Delta H: {delta_h:.2f} m")
    print(f"Hauteur sous palplanche: {h_sous_palplanche:.2f} m")
    print(f"Gradient hydraulique: {gradient_hydraulique:.6f}")
    print(f"Estimation Darcy (écoulement confiné): {q_darcy:.4f} m³/h/ml")
    
    return {
        "q_darcy": q_darcy,
    }    
    
def plot_results(x, z, h, ux, uz,
                 ip_1, ip_2, j_bp, j_sol,
                 Zn1, Zn2, Zfond, Zsol, Zfouille, Zbp,
                 L, k):
    """
    Trace :
     - contours de h
     - lignes de courant (streamplot) en rouge
     - palplanche et nappes
    """
    # 1) Inverser z pour que y soit strictement croissant
    y = z[::-1]         # de Zfond (bas) à Zn1 (haut)
    # 2) Renverser les champs pour qu’ils correspondent à y
    h_r  = h[::-1, :]
    ux_r = ux[::-1, :]
    uz_r = uz[::-1, :]

    # 3) Maillage
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(14,7))
    # Charge hydraulique
    cs = ax.contourf(X, Y, h_r, levels=30, cmap='viridis', alpha=0.7)
    plt.colorbar(cs, ax=ax, label='Charge hydraulique (m)')
    # Equipotentielles
    ax.contour(X, Y, h_r, levels=10, colors='green', linewidths=0.5)

    # 4) Masquage des zones hors fouille (pour le courant)
    mask = np.ones_like(ux_r, dtype=bool)
    nz = len(z)
    for jj in range(nz):
        orig_j = nz - 1 - jj
        for i in range(len(x)):
            # côté aval hors fouille
            if x[i] >= x[ip_1] and x[i] <= x[ip_2] and z[orig_j] > Zfouille:
                mask[jj, i] = False
            # palplanche 1
            if i == ip_1 and z[orig_j] > Zbp:
                mask[jj, i] = False
            # palplanche 2
            if i == ip_2 and z[orig_j] > Zbp:
                mask[jj, i] = False                
    ux_m = np.ma.masked_where(~mask, ux_r)
    uz_m = np.ma.masked_where(~mask, uz_r)

    # 5) Lignes de courant
    ax.streamplot(x, y, ux_m, uz_m,
                  color='red', density=1.2,
                  linewidth=1, arrowsize=1)

    # 6) Palplanches et nappes
    ax.plot([x[ip_1], x[ip_1]], [Zbp, Zsol],
            'k-', lw=4, solid_capstyle='butt')
    ax.plot([x[ip_2], x[ip_2]], [Zbp, Zsol],
            'k-', lw=4, solid_capstyle='butt')
    ax.hlines(Zn1, 0, x[ip_1],
              colors='blue', linestyles='-', lw=3,
              label='Nappe amont')
    ax.hlines(Zn1, x[ip_2], x[-1],
              colors='blue', linestyles='-', lw=3)
    ax.hlines(Zn2, x[ip_1], x[ip_2],
              colors='red', linestyles='-', lw=3,
              label='Nappe aval')
    ax.hlines(Zfouille, x[ip_1], x[ip_2],
              colors='brown', linestyles='--', lw=2,
              label='Fouille aval')

    ax.set_title("Écoulement sous palplanche")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_xlim(0, 2*L)
    ax.set_ylim(Zfond, Zsol)
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show(block=True)

def run_test():
    """Script de test sans GUI pour valider l'affichage."""
    # Exemple de paramètres
    Zfouille = -18.6  # altitude du fond fouille
    Zbp = -25       # base palplanche
    Zn1 = -12      # nappe amont
    Zn2 = -18.6      # nappe aval
    k1 = 1      # perméabilité couche 1 (m/s)
    k2 = 1e-5      # perméabilité couche 2 (m/s)
    Zsol = 0      # altitude du sol
    L = 10             # Distance entre les deux palplanches
    Zfond = Zbp - (Zfouille - Zbp)
    couches = [(Zsol, Zn1, k1),
               (Zn1, Zfond, k2)]
    x, z, h, ip_1, ip_2, j_sub, j_bp, dz, dx, nx, nz, K = solve_fouille_relax(
        L, Zsol, Zfouille, Zbp, Zfond, Zn1, Zn2, couches)
    ux, uz = compute_flux(h, K, dx, dz)
    q_tot = compute_total_flow_zone(h, K, L, dx, dz, ip_1, ip_2, Zn1, Zn2, Zbp, Zfouille, Zfond, Zsol, nz, j_bp, nx, uz)
    q_theorique = theoretical_flow_estimate(k2, L, Zn1, Zn2, Zbp, Zfouille, Zfond)
    plot_results(x, z, h, ux, uz, ip_1, ip_2, j_bp, j_sub, Zn1, Zn2, Zfond, Zsol, Zfouille, Zbp, L, k1)
    print("q_tot = " )
    print(q_tot)
    print("q_theorique = " + str(q_theorique))


if __name__ == '__main__':
    run_test()


