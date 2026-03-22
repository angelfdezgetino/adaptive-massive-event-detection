import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, MSTL, STL
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import os

# ========= Params
CSV_PATH       = '../data/metro_madrid/daily_records_since_2017.csv'

# ==== PostCovid
START_DATE = "2021-01-01"
END_DATE   = "2024-09-30"
# ==== Precovid
# START_DATE = "2017-01-01"
# END_DATE   = "2019-12-31"

# ==== Hampel+STL  
W_WEEKS = 7              # +-3 weeks
STL_PERIOD = 7
STL_SEASONAL = 13        # odd
LEVEL_WINDOW = "29D"     
LEVEL_MIN_PERIODS = 14

# ==== IQR 
K_IQR          = 1.5
WINDOW_DAYS_IQR = 90
MIN_PERIODS    = 14

# ==== MAD (point)
K_MAD          = 3
WINDOW_DAYS_MAD = 42          
C_CONST        = 1.4826

# ==== Type mode
CEI_THRESHOLD  = 0.50
PEI_THRESHOLD  = 0.33
T_EXCESS_PCT   = 0.10        

SHOW_PLOTS     = False        # True si quieres ver figuras

# ==== Ground truth 
EVENTS_BY_STATION = {
    'Ventas': '../data/eventos_ventas_completo.csv',
    'Santiago Bernabéu': '../data/eventos_santiago_bernabeu.csv',
    'Estadio Metropolitano': '../data/eventos_estadio_metropolitano.csv',
}

# ======== Preprocess, CEI/PEI and Hampel Filter
# Nombre => Name of station
# Orden => Order of station in each line
def preprocess_station_names(df):
    df = df.copy()
    if "ORDEN" in df.columns:
        df["ORDEN"] = pd.to_numeric(df["ORDEN"], errors="coerce")
        df.loc[(df["NOMBRE"] == "Atocha") & (df["ORDEN"] == 18), "NOMBRE"] = "Estación del Arte"
    df.loc[df["NOMBRE"] == "Atocha-Renfe", "NOMBRE"] = "Atocha"
    df.loc[df["NOMBRE"] == "Metropolitano", "NOMBRE"] = "Vicente Aleixandre"
    df.loc[df["NOMBRE"] == "Estadio Olímpico", "NOMBRE"] = "Estadio Metropolitano"
    df.loc[df["NOMBRE"] == "Campo de las Naciones", "NOMBRE"] = "Feria de Madrid"
    return df

def extract_runs_from_mask(mask):
    runs = []
    in_run = False
    start = None
    for i, val in enumerate(mask):
        if val and not in_run:
            in_run = True
            start = i
        elif (not val) and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs

def compute_cei_pei(resid_7d, threshold_pct=T_EXCESS_PCT, min_len=3, short_len=2):
    """"
    Compute CEI/PEI over the smoothed 7d residual log.
    """
    T_log = np.log1p(threshold_pct)
    mask = (resid_7d > T_log).fillna(False).to_numpy()
    runs = extract_runs_from_mask(mask)
    if len(runs) == 0:
        return 0.0, 0.0

    vals = resid_7d.fillna(0).to_numpy()
    masses, lengths = [], []
    for s, e in runs:
        length = e - s + 1
        mass = np.sum(vals[s:e+1] - T_log)
        masses.append(max(0.0, mass))
        lengths.append(length)

    total_mass = float(np.sum(masses))
    cei = (np.sum([m for m, l in zip(masses, lengths) if l >= min_len]) / total_mass) if total_mass > 0 else 0.0
    pei = float(np.sum([1 for l in lengths if l <= short_len]) / len(runs))
    return float(cei), float(pei)

def filtro_hample(series, W_weeks=W_WEEKS):
    """
      1) Hampel filter per day of week
      2) STL  to have clean trend and seasonality
      3) expected = trend + seasonal
      4) resid_final = temporal_serie - expected
      5) level_min
    """
    ts = series.copy()
    ts = ts.loc[START_DATE:END_DATE].copy()

    ts = pd.Series(ts).squeeze().astype(float).sort_index()

    ts_star = ts.groupby(ts.index.dayofweek).transform(
        lambda x: x.rolling(window=W_weeks, center=True, min_periods=3).median()
    )
    ts_star = ts_star.interpolate("time").bfill().ffill()

    stl = STL(ts_star, seasonal=STL_SEASONAL, period=STL_PERIOD, robust=True)
    res_stl = stl.fit()
    seasonality = res_stl.seasonal
    trend = res_stl.trend
    residual = res_stl.resid

    expected = trend + seasonality
    resid_final = ts - expected

    level_min = ts.rolling(LEVEL_WINDOW, center=True, min_periods=LEVEL_MIN_PERIODS).median()
    level_min = level_min.interpolate("time").bfill().ffill()

    return resid_final, ts_star, trend, seasonality, expected, level_min, residual

# ======== Detectors
def detect_iqr_simple(residues, k=K_IQR, window_days=WINDOW_DAYS_IQR, min_periods=MIN_PERIODS):
    """
    IQR: threshold(t) = median(t) + k * IQR(t)
    - median = Q2
    - k = 1.5 
    - IQR = Q3 - Q1
    """
    r = pd.Series(residues).astype(float).sort_index()

    Q1 = r.rolling(window_days, min_periods=min_periods).quantile(0.25)
    Q2 = r.rolling(window_days, min_periods=min_periods).quantile(0.50)
    Q3 = r.rolling(window_days, min_periods=min_periods).quantile(0.75)
    IQR = Q3 - Q1

    thresh = (Q2 + k * IQR)

    thresh = thresh.shift(1).ffill().bfill()

    mask_event = r > thresh
    df_clean = pd.DataFrame({"residuo": r, "threshold": thresh})
    df_events_clean = df_clean[mask_event]
    return mask_event, thresh, df_events_clean

def detect_mad_simple(residues, k=K_MAD, window_days=WINDOW_DAYS_MAD, min_periods=MIN_PERIODS, c_const=C_CONST):
    """
    MAD: roll_med + k * (c_const * roll_mad)
    - roll_med = rolling median
    - k = 3
    - c_const = 1.4826 
    - roll_mad = rolling median of absolute deviations from median
    """
    r = pd.Series(residues).astype(float)

    roll_med = r.rolling(window_days, min_periods=min_periods).median().shift(1).ffill()
    roll_mad = r.rolling(window_days, min_periods=min_periods).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    ).shift(1).ffill()

    sigma = c_const * roll_mad
    thresh = roll_med + k * sigma
    mask = r > thresh

    df_clean = pd.DataFrame({"residuo": r, "threshold": thresh})
    df_events_clean = df_clean[mask]
    return mask, thresh, df_events_clean

# ======== Pipeline
def pipeline_combinado(df_all, station_name, YEAR,
                       show_plots=False):
    """
    Steps B, C, D, E from the Workflow diagram:
    B) Time Series Decomposition
    C) Compute CEI/PEI and behaviour mode 
    D) Hampel filter
    E) Algorithms for event detection (IQR/MAD)
    *** Note: at the end a few plots are presented
    """
    df_est = df_all[df_all["NOMBRE"] == station_name]
    if df_est.empty:
        raise ValueError(f"No data for the station: {station_name}")
    df_daily = df_est.groupby(df_est["FECHA"].dt.date)["VALOR"].sum().reset_index(name="VALOR")
    df_daily["FECHA"] = pd.to_datetime(df_daily["FECHA"])

    # ==== Step B)
    s = df_daily.set_index("FECHA")["VALOR"].asfreq("D").interpolate()
    if (s <= 0).any():
        s += 1
    s_log = np.log(s)
    res_base = seasonal_decompose(s_log, model="additive", period=365)
    residues_base0 = res_base.resid.dropna()
    resid_7d = residues_base0.rolling(7, center=True, min_periods=4).mean()

    # ==== Step C)
    cei, pei = compute_cei_pei(resid_7d, threshold_pct=T_EXCESS_PCT)

    # ==== Step D) 
    # Clean Precovid or postCovid data
    df_daily = df_daily[(df_daily["FECHA"] >= START_DATE) & (df_daily["FECHA"] <= END_DATE)].copy()
    s = df_daily.set_index("FECHA")["VALOR"].asfreq("D").interpolate()
    residues_base, ts_star_plot, trend_plot, seas_plot, expected_plot, level_min, res_stl  = filtro_hample(s)
    

    # ==== Step E)
    if (cei >= CEI_THRESHOLD) and (pei <= PEI_THRESHOLD):
        modo = "collective"
        res = residues_base
        residues = res
        mask_event, thresh, df_events_clean = detect_iqr_simple(residues)
    else:
        modo = "punctual"
        residues = residues_base
        mask_event, thresh, df_events_clean = detect_mad_simple(residues)

    # Post-filter cleaning events under level_min
    df_events_clean = df_events_clean.copy()
    df_events_clean["observed"]  = s.reindex(df_events_clean.index)
    df_events_clean["level_min"] = level_min.reindex(df_events_clean.index)
    keep = df_events_clean["observed"] >= df_events_clean["level_min"]
    df_events_clean = df_events_clean[keep]

    # Events detected
    df_events_year = (
        df_events_clean[df_events_clean.index.year == YEAR]
        .reset_index()
        .rename(columns={"index": "FECHA"})
    )
    # ==== Note: this section is for evaluation over the ground truth generated
    if station_name in EVENTS_BY_STATION:
        df_gt = pd.read_csv(EVENTS_BY_STATION[station_name], parse_dates=["FECHA"]).dropna(subset=["FECHA"])
        df_gt = df_gt[pd.to_datetime(df_gt["FECHA"]).dt.year == YEAR]
        evt_days = set(pd.to_datetime(df_gt["FECHA"]).dt.normalize().unique())
        has_ground_truth = True
    else:
        evt_days = set()
        has_ground_truth = False

    residues_year = residues[residues.index.year == YEAR]
    df_daily_year = pd.DataFrame({
        "FECHA": pd.to_datetime(residues_year.index).normalize(),
        "residuo": residues_year.values
    })

    df_daily_year["evento_real"] = df_daily_year["FECHA"].isin(evt_days).astype("int8")
    df_daily_year["pred"] = df_daily_year["FECHA"].isin(
        pd.to_datetime(df_events_year["FECHA"]).dt.normalize()
    ).astype("int8")

    # Confusion matrix components
    tp = int(((df_daily_year.evento_real == 1) & (df_daily_year.pred == 1)).sum())
    fp = int(((df_daily_year.evento_real == 0) & (df_daily_year.pred == 1)).sum())
    fn = int(((df_daily_year.evento_real == 1) & (df_daily_year.pred == 0)).sum())
    tn = int(((df_daily_year.evento_real == 0) & (df_daily_year.pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy  = accuracy_score(df_daily_year["evento_real"].values, df_daily_year["pred"].values) if len(df_daily_year) else 0.0

    # AUC score
    thresh_year = thresh[thresh.index.year == YEAR]
    df_auc = df_daily_year.set_index("FECHA").copy()
    df_auc["threshold"] = thresh_year.reindex(df_auc.index)
    df_auc["score"] = (df_auc["residuo"] - df_auc["threshold"]).astype(float).fillna(0.0)

    auc_value = np.nan
    fpr = tpr = None
    y_true = df_auc["evento_real"].astype(int).values
    if has_ground_truth and pd.Series(y_true).nunique() == 2:
        auc_value = roc_auc_score(y_true, df_auc["score"].values)
        fpr, tpr, _ = roc_curve(y_true, df_auc["score"].values)

    # ==== Plots (ONLY IF YOU ARE ANALIZING ONE STATION)
    if show_plots:

        idx = s.index[s.index.year == YEAR]
        s_year = s.reindex(idx)
        expected_y = expected_plot.reindex(idx)
        ts_star_y = ts_star_plot.reindex(idx)
        resid_y = residues_base.reindex(idx)
        level_min_y = level_min.reindex(idx)
        thresh_y = thresh.reindex(idx)
        upper_band = expected_y + thresh_y

        det_dates = pd.to_datetime(df_events_year["FECHA"]) if not df_events_year.empty else pd.DatetimeIndex([])
        print("Detected event dates (raw):", det_dates)
        det_dates = det_dates[det_dates.dt.year == YEAR]

        # ==== FIG 1) Original Serie + level_min (28D) + OUTLIERS
        plt.figure(figsize=(14, 5))
        plt.plot(s_year, label="Original Serie (observed)", alpha=0.85)
        plt.plot(level_min_y, color="orange", linewidth=2, label="Moving Median Filter (28D) - level_min")
        if len(det_dates) > 0:
            vals = s_year.reindex(det_dates)
            plt.scatter(det_dates, vals, s=35, zorder=5, label="Outliers detected", color="red")
        plt.title(f"{station_name} ({YEAR}) - Outliers over original serie (post-filter level_min)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Fig4a_serie_{station_name}.svg", format="svg", dpi=300, bbox_inches="tight")
        plt.show()

        # ==== FIG 2) OBSERVED vs EXPECTED + Threshold (IQR/MAD)
        plt.figure(figsize=(14, 5))
        plt.plot(s_year, label="Observed (daily series)", alpha=0.85)
        plt.plot(expected_y, linestyle="--", label="Expected = Trend + Seasonal (STL on ts_star)", alpha=0.95)
        plt.plot(upper_band, linestyle=":", label="Upper band = Expected + Threshold", alpha=0.95)
        if len(det_dates) > 0:
            vals = s_year.reindex(det_dates)
            plt.scatter(det_dates, vals, s=35, zorder=5, label="Events detected", color="red")
        plt.title(f"{station_name} ({YEAR}) - Observed vs Expected + Upper band ({modo})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Fig4b_serie_{station_name}_2_thresholds.svg", format="svg", dpi=300, bbox_inches="tight")
        plt.show()

        # ==== FIG 3) Residual vs Threshold 
        plt.figure(figsize=(14, 5))
        plt.plot(resid_y, label="Residual = Observed - Expected", alpha=0.85)
        plt.plot(thresh_y, "--", label="Threshold (IQR/MAD)", alpha=0.95)
        plt.axhline(0, color="k", linewidth=0.8)
        if len(det_dates) > 0:
            vals_r = resid_y.reindex(det_dates)
            plt.scatter(det_dates, vals_r, s=35, zorder=5, label="Detected (residual)", color="red")
        plt.title(f"{station_name} ({YEAR}) - Residuals vs Threshold ({modo})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Fig4c_residual_{station_name}.svg", format="svg", dpi=300, bbox_inches="tight")
        plt.show()

        # ==== FIG 4) ts_star (smoothed serie) +  STL 
        plt.figure(figsize=(14, 4))
        plt.plot(ts_star_y, label="ts_star (Hampel: median per day of week)", alpha=0.9)
        plt.title(f"{station_name} ({YEAR}) - ts_star used by STL")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 4))
        plt.plot(trend_plot.reindex(idx), label="Trend (STL on ts_star)", alpha=0.9)
        plt.title(f"{station_name} ({YEAR}) - Trend component")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 4))
        plt.plot(seas_plot.reindex(idx), label="Seasonal (STL on ts_star)", alpha=0.9)
        plt.title(f"{station_name} ({YEAR}) - Seasonal component")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Fig4d_seasonal_{station_name}.svg", format="svg", dpi=300, bbox_inches="tight")
        plt.show()

        # ==== FIG 5) ROC/AUC curve
        if fpr is not None and tpr is not None and np.isfinite(auc_value):
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f"AUC={auc_value:.3f}")
            plt.plot([0, 1], [0, 1], "--", label="Azar")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC - {station_name} ({YEAR})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"Fig5_roc_{station_name}.svg", format="svg", dpi=300, bbox_inches="tight")
            plt.show()

    result = {
        'station': station_name,
        'year': YEAR,
        'CEI': float(cei),
        'PEI': float(pei),
        'mode': modo,
        'n_eventos_detectados_total': int(len(df_events_clean)),
        'n_eventos_detectados_year': int(len(df_events_year)),
        'df_events_year': df_events_year,
        'confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall)
        },
        'roc': {'fpr': fpr, 'tpr': tpr, 'auc': auc_value},
        'thresh_series': thresh,
        'has_ground_truth': has_ground_truth
    }
    return result


if __name__ == "__main__":
    
    station_in = input("Station (exact name or All): ").strip()
    year_in = input("Year (e.g., 2023 or All): ").strip()
    year_in = year_in.strip()
    station_in = station_in.strip()
    
    # Step A) from workflow
    df = pd.read_csv(CSV_PATH, parse_dates=["FECHA"]).dropna(subset=["FECHA", "VALOR", "NOMBRE"])
    df = preprocess_station_names(df)
    all_stations = sorted(df["NOMBRE"].dropna().unique())
    mask = (df["FECHA"] >= pd.to_datetime(START_DATE)) & (df["FECHA"] <= pd.to_datetime(END_DATE))
    all_years = sorted(df.loc[mask, "FECHA"].dt.year.dropna().unique())
    #  ==== Visualize all or one station/year
    if station_in == "" or station_in.lower() == "all":
        stations_to_run = all_stations
    else:
        if station_in not in all_stations:
            raise ValueError("The station '{}' does not exist in the CSV.".format(station_in))
        stations_to_run = [station_in]

    if year_in == "" or year_in.lower() == "all":
        years_to_run = all_years
    else:
        try:
            y = int(year_in)
        except ValueError:
            raise ValueError("YEAR must to be an integer (e.g., 2023) or 'All'.")
        if y not in all_years:
            raise ValueError("The year {} does not exist in the CSV.".format(y))
        years_to_run = [y]

    # Visualize plots only for a specific station
    station_is_specific = (station_in != "") and (station_in.lower() != "all")
    run_plots = station_is_specific  

    for station_name in stations_to_run:
        for YEAR in years_to_run:
            print("\n===============================")
            print("Station: {} | Year: {}".format(station_name, YEAR))
            print("===============================")
            
            # Step B,C,D,E) from workflow
            out = pipeline_combinado(
                df_all=df,
                station_name=station_name,
                YEAR=YEAR,
                show_plots=run_plots
            )

            print("\n--- Summary ---")
            print("Station: {}   Year: {}".format(out["station"], out["year"]))
            print("CEI = {:.3f}   PEI = {:.3f}   => mode = {}".format(out["CEI"], out["PEI"], out["mode"]))
            print("Events detected (full history): {}".format(out["n_eventos_detectados_total"]))
            print("Events detected (year {}): {}".format(YEAR, out["n_eventos_detectados_year"]))
            print("Confusion matrix (TP, FP, FN, TN):", out["confusion"])
            print("Metrics (precision, recall, accuracy):", out["metrics"])
            print("Are known events available?:", out["has_ground_truth"])

            if not out["df_events_year"].empty:
                print("\nEvents detected (year):")
                print(out["df_events_year"][["FECHA", "residuo", "threshold"]].head(100))
            else:
                print("\nNo events detected for the specified year according to the chosen method.")

            # Step F) from workflow

            # ==== Save detected events to CSV
            # route_csv = f"../data/detection_events/events_detected_all_17_19.csv" #Events precovid between 2017 and 2019
            route_csv = f"../data/detection_events/events_detected_all.csv"       #Events postcovid between 2021 and 2024
            # route_csv = f"../data/detection_events/events_detected_{station_name}.csv" #Events for a specific station (all years)

            df_station_clean_year = out['df_events_year']
            df_station_clean_year['year'] = df_station_clean_year['FECHA'].dt.year
            df_station_clean_year['station_name'] = station_name
            df_station_clean_year['mode'] = out['mode']
            df_station_clean_year = df_station_clean_year[['station_name','year','FECHA','residuo','threshold', 'mode']]
            df_station_clean_year.rename(columns={'FECHA':'date', 'residuo':'resid'}, inplace=True)
            
            header = not os.path.exists(route_csv)  
            df_station_clean_year.to_csv(route_csv, mode="a", index=False, header=header)
            print(f"Events detected from station: {station_name} and year: {YEAR} saved to CSV in the following format:")
            print(df_station_clean_year.head(5))
            print(f"Events saved in ../data/detection_events/events_detected_{station_name.replace(' ','_').lower()}_{YEAR}.csv")
