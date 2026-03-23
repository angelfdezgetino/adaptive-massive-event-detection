import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ========Params
csv_path     = "../data/detection_events/events_detected_all.csv"

value_col    = "resid"
use_pos_only = True  
day_lbls_full = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
month_lbls_en = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


#Size
FIGSIZE = (3.6, 5.2)
F_DAY   = 7
F_Y     = 7
F_TIT   = 11
F_XB    = 6
BAR_H   = 0.55
GRID_ALPHA = 0.05

station = input("Station (exact name or All): ").strip()

df = pd.read_csv(csv_path)

df = df[df["station_name"] == station].copy()
df["date"] = pd.to_datetime(df["date"])
if use_pos_only:
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").clip(lower=0.0)
years = np.sort(df["date"].dt.year.unique())

# Colour Scale
vmin, vcenter = 0.0, 0.2
vmax = np.nanpercentile(df[value_col], 99) if df[value_col].notna().any() else 1.0
vmax = max(vmax, vcenter + 1e-9)
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

monday = pd.offsets.Week(weekday=0)

def week_grid_for_year(df_year: pd.DataFrame, year: int):
    d0, d1 = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31) 

    s = (df_year.set_index("date")[value_col]
         .loc[d0:d1]
         .resample("D").sum()
         .reindex(pd.date_range(d0, d1, freq="D"), fill_value=0.0)) 

    week0 = (d0 - monday) 
    last_monday_after = (d1 + pd.Timedelta(days=1)) + monday 
    n_weeks = ((last_monday_after - week0).days // 7) 

    M = np.zeros((n_weeks, 7), dtype=float) 
    for day, val in s.items():
        w = (day - week0).days // 7
        d = day.weekday()
        if 0 <= w < n_weeks:
            M[w, d] += float(val)

    weekly_totals  = M.sum(axis=1) 
    weekday_totals = M.sum(axis=0) 
    week_mondays = [week0 + pd.Timedelta(days=7*w) for w in range(n_weeks)] 
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

    axM = fig.add_subplot(gs[0, 0])             # heatmap
    axR = fig.add_subplot(gs[0, 1], sharey=axM) # week bar chart (right)
    axB = fig.add_subplot(gs[1, 0])             # bar chart by day (bottom)

    # ======== Heatmap
    axM.imshow(M, origin="upper", aspect="auto",
               cmap="Blues", norm=norm, interpolation="nearest")

    # Rejilla 
    axM.set_xticks(np.arange(-.5, 7, 1), minor=True)
    axM.set_yticks(np.arange(-.5, n_weeks, 1), minor=True)
    axM.grid(which="minor", color="#000", alpha=GRID_ALPHA, linewidth=0.4)
    axM.tick_params(which="minor", bottom=False, left=False)

    # ====  Header
    axM.set_xticks(np.arange(7))
    axM.set_xticklabels(day_lbls_full, fontsize=F_DAY)
    axM.xaxis.tick_top()
    axM.tick_params(axis="x", pad=3, length=0)

    def row_of_date(d):  
        return ((pd.Timestamp(d) - week0).days) // 7

    def col_of_date(d):  
        return pd.Timestamp(d).weekday()

    month_starts = pd.date_range(d0, d1, freq="MS")

    line_color = "#000000"
    line_width = 1.2

    axM.spines["left"].set_zorder(0)

    for d in month_starts:
        r = row_of_date(d)         # week day 1
        c = col_of_date(d)         # weekday day 1

        y_top  = r - 0.5           
        y_bot  = r + 0.5           
        x_edge = c - 0.5           
        x_end  = 6.5              
        x_w0   = -0.5              

        # ─ top
        axM.plot([x_edge, x_end], [y_top, y_top],
                color=line_color, lw=line_width, solid_capstyle="butt",
                zorder=4, clip_on=False)

        # | vertical
        axM.plot([x_edge, x_edge], [y_top, y_bot],
                color=line_color, lw=line_width, solid_capstyle="butt",
                zorder=4, clip_on=False)

        # ─ bottom
        axM.plot([x_w0, x_edge], [y_bot, y_bot],
                color=line_color, lw=line_width, solid_capstyle="butt",
                zorder=4, clip_on=False)
    
    d_last = pd.Timestamp(y, 12, 31)

    r = row_of_date(d_last)        # week day 31 of december
    c = col_of_date(d_last)        # weekday 31 of december  (0=L..6=D)

    y_top  = r - 0.5               
    y_bot  = r + 0.5               
    x_edge = c - 0.5               
    x_end  = 6.5                   
    x_w0   = -0.5                  

    # ─ top
    axM.plot([x_edge, x_end], [y_top, y_top],
            color=line_color, lw=line_width, solid_capstyle="butt",
            zorder=4, clip_on=False)

    # │ vertical
    axM.plot([x_edge, x_edge], [y_top, y_bot],
            color=line_color, lw=line_width, solid_capstyle="butt",
            zorder=4, clip_on=False)

    # ─ bottom
    axM.plot([x_w0, x_edge], [y_bot, y_bot],
            color=line_color, lw=line_width, solid_capstyle="butt",
            zorder=4, clip_on=False)
    
    
    month_starts = pd.date_range(d0, d1, freq="MS")
    month_ends   = list(month_starts[1:] - pd.Timedelta(days=1)) + [d1]
    row_start = [row_of_date(d) for d in month_starts]   
    row_end   = [row_of_date(d) for d in month_ends]     
    centers   = [(a + b) / 2 for a, b in zip(row_start, row_end)]
    axM.set_yticks([])

    # Position of the text
    x_text = -0.55
    for lab, yc in zip(month_lbls_en, centers):
        axM.text(x_text, yc, lab, ha="right", va="center",
                fontsize=F_Y, color="#000000", zorder=5, clip_on=False)

    # ======== week bar chart (right)
    y_pos = np.arange(n_weeks)
    axR.barh(y_pos, weekly_totals, height=BAR_H, color="#2171b5")
    axR.set_ylim(axM.get_ylim())
    axR.set_yticks([])                 
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
    
    # ======= bar chart by day (bottom)
    x_pos = np.arange(7)
    axB.bar(x_pos, weekday_totals, color="#2171b5", width=0.65)

    
    ymax = float(np.nanmax(weekday_totals)) if np.isfinite(np.nanmax(weekday_totals)) else 1.0
    axB.set_ylim(ymax * 1.08 if ymax > 0 else 1.0, 0)
    axB.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axB.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
    axB.tick_params(axis="y", labelsize=6)

    
    axB.set_xlim(-0.5, 6.5)
    border_ticks = np.arange(-0.5, 7.0, 1.0)
    axB.set_xticks(border_ticks)
    axB.set_xticklabels([])
    axB.xaxis.set_ticks_position("top")
    axB.xaxis.set_label_position("top")
    axB.tick_params(axis="x", which="major",
                    top=True, bottom=False, length=4, width=0.8, pad=2)
    axB.minorticks_off()

    
    axB.spines["left"].set_visible(True)
    axB.spines["left"].set_position(("data", -0.5))  
    axB.spines["left"].set_linewidth(0.9)
    axB.tick_params(axis="y", left=True, right=False, length=3, width=0.8)

    axB.spines["top"].set_visible(True)
    axB.spines["right"].set_visible(True)
    axB.spines["bottom"].set_visible(False)


    plt.tight_layout(pad=0.4)
    plt.savefig(f"Fig6_heatmap_{station}_year_{y}.svg", format="svg", dpi=300, bbox_inches="tight")
    plt.show()
