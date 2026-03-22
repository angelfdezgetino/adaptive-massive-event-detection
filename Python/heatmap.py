import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ==================================Parámetros============================================
csv_path     = "../data/detection_events/events_detected_all.csv"

station      = "Ciudad Universitaria"
value_col    = "resid"
use_pos_only = True  
day_lbls_full = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
month_lbls_en = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# Tamaños compactos
FIGSIZE = (3.6, 5.2)
F_DAY   = 7
F_Y     = 7
F_TIT   = 11
F_XB    = 6
BAR_H   = 0.55
GRID_ALPHA = 0.05

# ==================================Lectura y obtención de datos============================================
# df = pd.read_parquet(parquet_path)
df = pd.read_csv(csv_path)

df = df[df["station_name"] == station].copy()
df["date"] = pd.to_datetime(df["date"])
if use_pos_only:
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").clip(lower=0.0)
years = np.sort(df["date"].dt.year.unique())

# Escala de color tipo dayplot: vmin=0, vcenter=0.2, vmax=p99 (global estación)
vmin, vcenter = 0.0, 0.2
vmax = np.nanpercentile(df[value_col], 99) if df[value_col].notna().any() else 1.0
vmax = max(vmax, vcenter + 1e-9)
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

monday = pd.offsets.Week(weekday=0)

def week_grid_for_year(df_year: pd.DataFrame, year: int):
    d0, d1 = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31) #se establece el primer y el ultimo dia del año

    s = (df_year.set_index("date")[value_col]
         .loc[d0:d1]
         .resample("D").sum()
         .reindex(pd.date_range(d0, d1, freq="D"), fill_value=0.0)) # para recorar el año

    week0 = (d0 - monday)  # semana 0
    last_monday_after = (d1 + pd.Timedelta(days=1)) + monday # lunes siguiente al 31-dic
    n_weeks = ((last_monday_after - week0).days // 7) # número de semanas completas (filas)

    M = np.zeros((n_weeks, 7), dtype=float)  # Matriz de las semanas para el heatmap filas=semanas, cols=Mon..Sun
    for day, val in s.items():
        w = (day - week0).days // 7
        d = day.weekday()
        if 0 <= w < n_weeks:
            M[w, d] += float(val)

    weekly_totals  = M.sum(axis=1) # suma valores por semanas (barra derecha)
    weekday_totals = M.sum(axis=0) # suma valores por días de la semana (barra invertida abajo)
    week_mondays = [week0 + pd.Timedelta(days=7*w) for w in range(n_weeks)] # fechas de los lunes de cada semana
    return M, weekly_totals, weekday_totals, week_mondays, d0, d1, week0

for y in years:
    df_y = df[df["date"].dt.year == y].copy()
    M, weekly_totals, weekday_totals, week_mondays, d0, d1, week0 = week_grid_for_year(df_y, y)
    n_weeks = M.shape[0]

    fig = plt.figure(figsize=FIGSIZE, dpi=220)
    fig.suptitle(f"Station {station} — {y}", fontsize=10, ha="center", y=0.95)
    gs = GridSpec(2, 2,
                  width_ratios=[0.76, 0.24],
                  height_ratios=[0.80, 0.20],
                  wspace=0.08, hspace=0.08)

    axM = fig.add_subplot(gs[0, 0])             # matriz
    axR = fig.add_subplot(gs[0, 1], sharey=axM) # barras por semana (derecha)
    axB = fig.add_subplot(gs[1, 0])             # barras por día (abajo invertidas)

    # ==================================Heatmap girado (semanaXdias)============================================
    axM.imshow(M, origin="upper", aspect="auto",
               cmap="Blues", norm=norm, interpolation="nearest")

    # Rejilla 
    axM.set_xticks(np.arange(-.5, 7, 1), minor=True)
    axM.set_yticks(np.arange(-.5, n_weeks, 1), minor=True)
    axM.grid(which="minor", color="#000", alpha=GRID_ALPHA, linewidth=0.4)
    axM.tick_params(which="minor", bottom=False, left=False)

    # --- Cabecera eje X días de la semana (Mon..Sun) ---
    axM.set_xticks(np.arange(7))
    axM.set_xticklabels(day_lbls_full, fontsize=F_DAY)
    axM.xaxis.tick_top()
    axM.tick_params(axis="x", pad=3, length=0)

    # --- Meses en eje Y + marca delimitado de cada mes ---
    #  Marca del inicio de mes: desde el borde del dia uno hasta el final de la semana
    def row_of_date(d):  # fila (semana) con respecto a week0
        return ((pd.Timestamp(d) - week0).days) // 7

    def col_of_date(d):  # columna (weekday) 0=Lunes..6=Domingo
        return pd.Timestamp(d).weekday()

    month_starts = pd.date_range(d0, d1, freq="MS")

    line_color = "#000000"
    line_width = 1.2

    # (opcional) que el spine izq no tape la barra vertical
    axM.spines["left"].set_zorder(0)

    for d in month_starts:
        r = row_of_date(d)         # semana del día 1
        c = col_of_date(d)         # weekday del día 1

        y_top  = r - 0.5           # borde superior de la semana (alineado a rejilla)
        y_bot  = r + 0.5           # borde inferior de la semana
        x_edge = c - 0.5           # borde izq. exacto de la celda del día 1
        x_end  = 6.5               # borde dcho. (domingo)
        x_w0   = -0.5              # inicio de semana (lunes)

        # ─ arriba: desde el borde de la celda del día 1 hasta el domingo
        axM.plot([x_edge, x_end], [y_top, y_top],
                color=line_color, lw=line_width, solid_capstyle="butt",
                zorder=4, clip_on=False)

        # │ vertical: borde exacto de la celda del día 1
        axM.plot([x_edge, x_edge], [y_top, y_bot],
                color=line_color, lw=line_width, solid_capstyle="butt",
                zorder=4, clip_on=False)

        # ─ abajo: desde el borde inferior de la celda del día 1 hasta el lunes
        axM.plot([x_w0, x_edge], [y_bot, y_bot],
                color=line_color, lw=line_width, solid_capstyle="butt",
                zorder=4, clip_on=False)
    #
    # Repetir el patrón de 3 trazos SOLO para el 31 de diciembre 
    d_last = pd.Timestamp(y, 12, 31)

    r = row_of_date(d_last)        # fila (semana) del 31-dic
    c = col_of_date(d_last)        # columna (weekday) del 31-dic  (0=L..6=D)

    y_top  = r - 0.5               # borde superior de esa semana
    y_bot  = r + 0.5               # borde inferior de esa semana
    x_edge = c - 0.5               # borde izq. exacto de la celda del 31-dic
    x_end  = 6.5                   # borde derecho (domingo)
    x_w0   = -0.5                  # inicio de semana (lunes)

    # ─ arriba: desde el borde de la celda del 31-dic hasta el domingo
    axM.plot([x_edge, x_end], [y_top, y_top],
            color=line_color, lw=line_width, solid_capstyle="butt",
            zorder=4, clip_on=False)

    # │ vertical: borde exacto de la celda del 31-dic
    axM.plot([x_edge, x_edge], [y_top, y_bot],
            color=line_color, lw=line_width, solid_capstyle="butt",
            zorder=4, clip_on=False)

    # ─ abajo: desde el borde inferior de la celda del 31-dic hasta el lunes
    axM.plot([x_w0, x_edge], [y_bot, y_bot],
            color=line_color, lw=line_width, solid_capstyle="butt",
            zorder=4, clip_on=False)
    
    # --- Meses en el lado izquierdo (centrados en su bloque) ---
    month_starts = pd.date_range(d0, d1, freq="MS")
    month_ends   = list(month_starts[1:] - pd.Timedelta(days=1)) + [d1]
    row_start = [row_of_date(d) for d in month_starts]   # 1ª semana del mes (contiene el día 1)
    row_end   = [row_of_date(d) for d in month_ends]     # última semana del mes
    centers   = [(a + b) / 2 for a, b in zip(row_start, row_end)]
    axM.set_yticks([])

    # Posición del texto: -0.55 = fuera del panel; -0.48 = dentro del primer lunes
    x_text = -0.55
    for lab, yc in zip(month_lbls_en, centers):
        axM.text(x_text, yc, lab, ha="right", va="center",
                fontsize=F_Y, color="#000000", zorder=5, clip_on=False)

    # ==================================Barras DERECHA (por semana)============================================
    y_pos = np.arange(n_weeks)
    axR.barh(y_pos, weekly_totals, height=BAR_H, color="#2171b5")
    axR.set_ylim(axM.get_ylim())
    axR.set_yticks([])                 # sin etiquetas (las de mes están en axM)
    axR.spines["left"].set_visible(False)
    axR.set_xlim(0, weekly_totals.max()*1.05 if weekly_totals.max()>0 else 1)
    axR.tick_params(axis="x", labelsize=5, pad=1, length=1.2, width=0.5)
    axR.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axR.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    for sp in axR.spines.values():
        sp.set_linewidth(0.6)
    axR.spines["top"].set_visible(True)
    axR.spines["right"].set_visible(False)
    axR.spines["bottom"].set_visible(True)
    axR.spines["left"].set_visible(True)
    
    # =================================Barras ABAJO invertidas (por día)=================================
    x_pos = np.arange(7)
    axB.bar(x_pos, weekday_totals, color="#2171b5", width=0.65)

    # Y: 0 arriba, valores hacia abajo y visibles
    ymax = float(np.nanmax(weekday_totals)) if np.isfinite(np.nanmax(weekday_totals)) else 1.0
    axB.set_ylim(ymax * 1.08 if ymax > 0 else 1.0, 0)
    axB.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axB.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    axB.tick_params(axis="y", labelsize=6)

    # X ARRIBA: ticks SOLO en los BORDES de las celdas; sin etiquetas
    axB.set_xlim(-0.5, 6.5)
    border_ticks = np.arange(-0.5, 7.0, 1.0)
    axB.set_xticks(border_ticks)
    axB.set_xticklabels([])
    axB.xaxis.set_ticks_position("top")
    axB.xaxis.set_label_position("top")
    axB.tick_params(axis="x", which="major",
                    top=True, bottom=False, length=4, width=0.8, pad=2)
    axB.minorticks_off()

    # Eje Y (raya) visible y alineado con el inicio de semana (x = -0.5)
    axB.spines["left"].set_visible(True)
    axB.spines["left"].set_position(("data", -0.5))  # pega la raya al borde de la 1ª celda
    axB.spines["left"].set_linewidth(0.9)
    axB.tick_params(axis="y", left=True, right=False, length=3, width=0.8)

    # Resto de spines
    axB.spines["top"].set_visible(True)
    axB.spines["right"].set_visible(True)
    axB.spines["bottom"].set_visible(False)



    plt.tight_layout(pad=0.4)
    plt.savefig(f"Fig6_heatmap_{station}_year_{y}.svg", format="svg", dpi=300, bbox_inches="tight")
    plt.show()
