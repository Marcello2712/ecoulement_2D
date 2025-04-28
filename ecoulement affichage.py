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


def solve_fouille_relax(L, Zsol, Zfouille, Zbp, Zimp, Zn1, Zn2, k,
                        max_iter=10000, tol=1e-6,
                        output_excel=True, excel_filename="resultats_h.xlsx"):
    """
    Solve l'écoulement souterrain sous palplanche par différences finies,
    avec j=0 en surface (Zn1) et j=nz-1 au fond (Zimp).
    Retourne : x, z, h, ip, j_substrat, j_bp, dz, dx, nx
    """

    # --- 1) Grille
    dx = dz = 1.0
    # 1) Fond de grille = Zimp (pas Zfouille-2)
    Zfond = Zimp  

    # 2) Hauteur totale du domaine
    height = Zn1 - Zfond      # Zn1 – Zimp

    # 3) Nombre de nœuds verticaux
    nz = int(height / dz) + 1

    # 4) Coordonnées réelles
    z = Zn1 - np.arange(nz) * dz   # z[0]=Zn1, z[-1]=Zimp
    
    width = 2 * L
    Zfond = Zfouille - 2
    nx = int(L / dx) + 1
    x = np.linspace(0, width, nx)

    # Champ de charge hydraulique
    h = np.zeros((nz, nx))

    # --- 2) Indices clés
    ip         = nx // 2                # palplanche en i=ip
    j_Zn1      = 0                      # nappe aval en surface
    j_fouille  = int((Zn1 - Zfouille) / dz)
    j_Zn2      = int((Zn1 - Zn2)     / dz)
    j_bp       = int((Zn1 - Zbp)     / dz)
    j_substrat = nz - 1                 # fond imperméable

    # --- 3) Conditions limites initiales
    # a) Surfaces amont/aval
    h[j_Zn1,    :ip]  = Zn1
    h[j_Zn2,    ip:]  = Zn2
    # b) Côtés
    h[:, 0]           = Zn1
    h[j_Zn2:, -1]     = Zn2
    # c) Fouille (tête=0 entre surface et j_fouille, à droite de la palplanche)
    h[ j_Zn1 : j_fouille+1 , ip: ] = 0
    # d) bug sous palplanche :
    for j in range(j_bp, nz-1):
        h[j, ip] = 0.25 * (
            h[j+1, ip] + h[j-1, ip] + h[j, ip+1] + h[j, ip-1]
        )

    # --- 4) Boucle de relaxation SOR
    for it in range(max_iter):
        h_old = h.copy()

        for j in range(1, nz):
            for i in range(1, nx-1):
                # bord supérieur en j=0 n’est pas recalculé
                if j == nz-1:
                    # fond
                    h[j, i] = (h[j, i-1] + h[j, i+1]) / 4 + h[j-1, i] / 2
                elif i == ip and j == j_bp + 1:
                    # point juste sous la palplanche
                    h[j, i] = (
                        h[j, i-1]/4 + h[j, i+1]/4 + h[j+1, i]/4 +
                        h[j-1, i-1]/8 + h[j-1, i+1]/8
                    )
                elif i == ip and j <= j_bp:
                    # palplanche
                    h[j, i] = (h[j, i-1]/2 + h[j+1, i]/4 + h[j-1, i]/4)
                elif i == ip + 1 and j <= j_bp and j > j_Zn2:
                    # juste à droite de la palplanche
                    h[j, i] = (h[j-1, i]/4 + h[j+1, i]/4 + h[j, i-1]/2)
                elif i == ip - 1 and j <= j_Zn2:
                    # Juste à gauche
                    h[j, i] = (h[j, i-1]/2 + h[j+1, i]/4 + h[j-1, i]/4)
                elif i == ip and j > j_bp:
                    h[j, i] = 0.25 * (
                        h[j+1, i] + h[j-1, i] + h[j, i+1] + h[j, i-1]
                    )
                else:
                    # cas général
                    h[j, i] = 0.25 * (
                        h[j+1, i] + h[j-1, i] + h[j, i+1] + h[j, i-1]
                    )

        # --- 5) Ré-imposer les mêmes CL après update
        h[j_Zn1,    :ip]  = Zn1
        h[j_Zn2,    ip:]  = Zn2
        h[:, 0]           = Zn1
        h[j_Zn2:, -1]     = Zn2
        h[ j_Zn1 : j_fouille+1 , ip: ] = 0
        for j in range(j_bp, nz-1):
            h[j, ip] = 0.25 * (
                h[j+1, ip] + h[j-1, ip] + h[j, ip+1] + h[j, ip-1]
            )
            print("h[j, ip] = " + str(h[j, ip]))

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

    return x, z, h, ip, j_substrat, j_bp, dz, dx, nx


def compute_flux(h, k, dx, dz):
    """Calcule ux et uz (vitesses de Darcy centrées)."""
    ux = np.zeros_like(h)
    uz = np.zeros_like(h)
    ux[:, 1:-1] = -k * (h[:, 2:] - h[:, :-2]) / (2 * dx)
    uz[1:-1, :] = -k * (h[:-2, :] - h[2:, :]) / (2 * dz)
    return ux, uz

def compute_total_flow_zone(h, k, dx, dz, ip, Zn1, Zfouille, nx):
    """Débit vertical entrant dans la fouille, à j_fouille."""
    j_fouille = int((Zn1 - Zfouille) / dz)
    q_total = 0
    for i in range(ip + 1, nx - 1):  # jusqu’à l’avant-dernier point
        qz = -k * (h[j_fouille, i] - h[j_fouille + 1, i]) / dz
        q_total += qz * dz
    print(f"Débit total vertical entrant dans la fouille : {q_total:.4f} m³/s")
    print("Largeur considérée :", nx - 1 - (ip + 1), "cellules")
    return q_total

def plot_results(x, z, h, ux, uz,
                 ip, j_bp, j_sol,
                 Zn1, Zn2, Zimp, Zsol, Zfouille, Zbp,
                 L, k):
    """
    Trace :
     - contours de h
     - lignes de courant (streamplot) en rouge
     - palplanche et nappes
    """
    # 1) Inverser z pour que y soit strictement croissant
    y = z[::-1]         # de Zimp (bas) à Zn1 (haut)
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
            if x[i] >= x[ip] and z[orig_j] > Zfouille:
                mask[jj, i] = False
            # palplanche
            if i == ip and z[orig_j] > Zbp:
                mask[jj, i] = False
    ux_m = np.ma.masked_where(~mask, ux_r)
    uz_m = np.ma.masked_where(~mask, uz_r)

    # 5) Lignes de courant
    ax.streamplot(x, y, ux_m, uz_m,
                  color='red', density=1.2,
                  linewidth=1, arrowsize=1)

    # 6) Palplanche et nappes
    ax.plot([x[ip], x[ip]], [Zbp, Zsol],
            'k-', lw=4, solid_capstyle='butt')
    ax.hlines(Zn1, 0, x[ip],
              colors='blue', linestyles='-', lw=3,
              label='Nappe amont')
    ax.hlines(Zn2, x[ip], L,
              colors='red', linestyles='-', lw=3,
              label='Nappe aval')
    ax.hlines(Zfouille, x[ip], x[-1],
              colors='brown', linestyles='--', lw=2,
              label='Fouille aval')

    ax.set_title("Écoulement sous palplanche")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_xlim(L/2, 3*L/2)
    ax.set_ylim(Zimp, Zsol)
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show(block=True)



def run_test():
    """Script de test sans GUI pour valider l'affichage."""
    # Exemple de paramètres
    Zsol = 29      # altitude du sol
    Zfouille = 17.5  # altitude du fond fouille
    Zbp = 3       # base palplanche
    Zimp = 0       # imperméable en fond
    Zn1 = 25      # nappe amont
    Zn2 = 17      # nappe aval
    k = 1e-5      # perméabilité (m/s)
    L = 50
    x, z, h, ip, j_sub, j_bp, dz, dx, nx = solve_fouille_relax(
        L, Zsol, Zfouille, Zbp, Zimp, Zn1, Zn2, k)
    ux, uz = compute_flux(h, k, dx, dz)
    q_tot = compute_total_flow_zone(h, k, dx, dz, ip, Zn1, Zfouille, nx)
    plot_results(x, z, h, ux, uz, ip, j_bp, j_sub, Zn1, Zn2, Zimp, Zsol, Zfouille, Zbp, L, k)

def run_simulation(entries):
    try:
        Zsol = float(entries['Zsol'].get())
        Zfouille = float(entries['Zfouille'].get())
        Zbp = float(entries['Zbp'].get())
        Zimp = float(entries['Zimp'].get())
        Zn1 = float(entries['Zn1'].get())
        Zn2 = float(entries['Zn2'].get())
        k = float(entries['k'].get())
        S = float(entries['S'].get())
    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides.")
        return

    if not (Zimp <= Zbp <= Zfouille <= Zsol):
        messagebox.showerror("Erreur", "Zimp ≤ Zbp ≤ Zfouille ≤ Zsol est requis.")
        return

    L = sqrt(S)
    x, z, h, ip, j_substrat, j_bp, dz, dx, nx = solve_fouille_relax(L, Zsol, Zfouille, Zbp, Zimp, Zn1, Zn2, k)
    ux, uz = compute_flux(h, k, dx, dz)
    Q_total = compute_total_flow_zone(h, k, dx, dz, ip, Zn1, Zfouille, nx)
    delta_h = Zn1 - Zn2
    l = Zfouille - Zbp
    i = delta_h / l
    Q_ref = k * S * i

    plot_results(x, z, h, ux, uz, ip, j_bp, j_substrat, Zn1, Zn2, Zimp, Zsol, L)

    print(f"Débit de référence (Dupuit) : {Q_ref:.3e} m³/s")
    print(f"Débit total estimé (numérique) : {Q_total:.3e} m³/s")
    print(f"Écart relatif : {abs((Q_total - Q_ref)) / Q_ref * 100:.2f}%")
    
def build_gui():
    fields = ['Zsol', 'Zfouille', 'Zbp', 'Zimp', 'Zn1', 'Zn2', 'k', 'S']
    root = tk.Tk()
    root.title("Simulation d'écoulement sous palplanche")
    entries = {}
    for i, f in enumerate(fields):
        tk.Label(root, text=f).grid(row=i, column=0, padx=5, pady=2)
        e = tk.Entry(root); e.grid(row=i, column=1, padx=5, pady=2)
        entries[f] = e
    tk.Button(root, text="Lancer la simulation", command=lambda: run_simulation(entries)).grid(
        row=len(fields), column=0, columnspan=2, pady=10)
    root.mainloop()

if __name__ == '__main__':
    # Pour tester sans interface graphique
    run_test()
    # Si tout s'affiche correctement, décommenter la ligne suivante :
    # build_gui()
    

