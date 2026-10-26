"""
Shared helpers for 'Racing With the Machine' (MDPI Merits).

Racing With the Machine: AI Exposure, Skill, and the Future of the Professions
Authors: Sarthak Pattnaik, Eugene Pinsky, Kathleen Park (Boston University)

Every notebook starts with `from rwm_common import *`.
Nothing in this module prints or logs API keys.
"""
from __future__ import annotations

import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INT = ROOT / "data" / "interim"
DATA_FIN = ROOT / "data" / "final"
OUT_FIG = ROOT / "output" / "figures"
OUT_TAB = ROOT / "output" / "tables"
for _p in (DATA_RAW, DATA_INT, DATA_FIN, OUT_FIG, OUT_TAB, DATA_RAW / "api"):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Study configuration (edit here, not in the notebooks)
# --------------------------------------------------------------------------
OEWS_YEARS = list(range(2019, 2026))          # May 2019 ... May 2025 estimates
MAIN_WINDOW = (2021, 2025)                    # consistent SOC 2018 coding
REF_YEAR = 2022                               # last pre-ChatGPT estimate (May 2022)
BASE_YEAR_WEIGHTS = 2022                      # employment weights
ONET_VERSIONS = ["30_1", "30_0", "29_3", "29_2", "29_1", "29_0", "28_3"]
EXPOSURE_URL = ("https://raw.githubusercontent.com/openai/GPTs-are-GPTs/"
                "main/data/occ_level.csv")
BTOS_URLS = {
    "national": "https://www.census.gov/hfp/btos/downloads/National.xlsx",
    "sector": "https://www.census.gov/hfp/btos/downloads/Sector.xlsx",
}
FRED_SERIES = {
    "PRS85006173": "Labor share, nonfarm business sector (index 2017=100)",
    "OPHNFB": "Output per hour, nonfarm business sector (index 2017=100)",
    "CGBD2024": "Unemployment rate, bachelor's degree, ages 20-24",
    "LNS14027662": "Unemployment rate, bachelor's degree and higher, 25+",
}
CONTACT_EMAIL = "spattna1@bu.edu"   # BLS asks automated clients to identify themselves
HEADERS = {
    "User-Agent": f"racing-with-the-machine academic research ({CONTACT_EMAIL})",
    "Accept": "*/*",
}

# --------------------------------------------------------------------------
# .env loading (accepts "KEY: value", "KEY=value", "KEY = value", spaces in KEY)
# --------------------------------------------------------------------------
def _norm_key(k: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", k.strip().upper()).strip("_")


def load_env() -> dict:
    """Find the nearest .env (project folder or a parent) and parse it."""
    for folder in [ROOT, *ROOT.parents]:
        f = folder / ".env"
        if f.exists():
            env = {}
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"^([^:=]+?)\s*[:=]\s*(.+)$", line)
                if m:
                    env[_norm_key(m.group(1))] = m.group(2).strip().strip('"').strip("'")
            print(f".env found at {f} with keys: {sorted(env)}")   # names only
            return env
    raise FileNotFoundError("No .env file found in the project folder or its parents.")


def get_key(env: dict, *candidates: str) -> str:
    """Return the first key present among candidate names, e.g. get_key(env, 'BEA_API_KEY', 'BEA_KEY')."""
    for c in candidates:
        if _norm_key(c) in env:
            return env[_norm_key(c)]
    raise KeyError(f"None of {candidates} found in .env")


# --------------------------------------------------------------------------
# HTTP with caching
# --------------------------------------------------------------------------
def download(url: str, dest: Path, force: bool = False, retries: int = 3, timeout: int = 180) -> Path:
    """Download url to dest unless it is already cached. Raises on HTTP errors."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as e:      # noqa: BLE001
            last = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"Download failed for {url}: {last}")


def download_first(urls: list[str], dest: Path, **kw) -> tuple[Path, str]:
    """Try several candidate URLs; return (path, url_used)."""
    errors = []
    for u in urls:
        try:
            return download(u, dest, **kw), u
        except Exception as e:      # noqa: BLE001
            errors.append(f"{u}: {e}")
    raise RuntimeError("All candidate URLs failed:\n  " + "\n  ".join(errors))


def get_json_cached(url: str, params: dict, cache_name: str, force: bool = False,
                    secret_params: tuple = ("UserID", "api_key")) -> dict:
    """GET JSON and cache it. Secret params are excluded from the cache file name and content."""
    f = DATA_RAW / "api" / f"{cache_name}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    secrets = [str(params[k]) for k in secret_params if k in params]

    def redact(msg: str) -> str:
        for s in secrets:
            msg = msg.replace(s, "<KEY>")
        return msg

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=180)
        r.raise_for_status()
        js = r.json()
    except Exception as e:      # noqa: BLE001  (requests puts the full URL, key included, in errors)
        raise RuntimeError(redact(f"{type(e).__name__}: {e}")) from None
    f.write_text(json.dumps(js))
    return js


# --------------------------------------------------------------------------
# BEA API
# --------------------------------------------------------------------------
BEA_URL = "https://apps.bea.gov/api/data"


def bea_results(js: dict):
    res = js.get("BEAAPI", {}).get("Results")
    if res is None:
        raise RuntimeError(f"Unexpected BEA response: {str(js)[:500]}")
    if isinstance(res, list):
        res = res[0]
    if "Error" in res:
        raise RuntimeError(f"BEA error: {res['Error']}")
    return res


def bea_param_values(key: str, dataset: str, param: str) -> pd.DataFrame:
    js = get_json_cached(BEA_URL, {"UserID": key, "method": "GetParameterValues",
                                   "datasetname": dataset, "ParameterName": param,
                                   "ResultFormat": "json"},
                         f"bea_params_{dataset}_{param}")
    return pd.DataFrame(bea_results(js)["ParamValue"])


def bea_get(key: str, cache_name: str, **params) -> pd.DataFrame:
    p = {"UserID": key, "method": "GetData", "ResultFormat": "json", **params}
    js = get_json_cached(BEA_URL, p, cache_name)
    return pd.DataFrame(bea_results(js)["Data"])


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False)
                         .str.replace("$", "", regex=False).str.strip(),
                         errors="coerce")


# --------------------------------------------------------------------------
# FRED API
# --------------------------------------------------------------------------
def fred_series(key: str, series_id: str) -> pd.DataFrame:
    js = get_json_cached("https://api.stlouisfed.org/fred/series/observations",
                         {"series_id": series_id, "api_key": key, "file_type": "json"},
                         f"fred_{series_id}")
    df = pd.DataFrame(js["observations"])[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series_id"] = series_id
    return df


# --------------------------------------------------------------------------
# Small data helpers
# --------------------------------------------------------------------------
def soc6(code: str) -> str:
    """'11-1011.03' -> '11-1011'."""
    return str(code).strip()[:7]


def zscore(s: pd.Series, w: pd.Series | None = None) -> pd.Series:
    s = s.astype(float)
    if w is None:
        return (s - s.mean()) / s.std(ddof=0)
    m = np.average(s.dropna(), weights=w[s.notna()])
    v = np.average((s.dropna() - m) ** 2, weights=w[s.notna()])
    return (s - m) / np.sqrt(v)


def weighted_quantile_bins(x: pd.Series, w: pd.Series, q: int = 10) -> pd.Series:
    """Employment-weighted quantile bins (1..q)."""
    d = pd.DataFrame({"x": x, "w": w}).dropna()
    d = d.sort_values("x")
    cw = d["w"].cumsum() / d["w"].sum()
    bins = np.minimum((cw * q).apply(np.ceil).astype(int), q).clip(lower=1)
    return bins.reindex(x.index)


def save_table(df: pd.DataFrame, name: str, caption: str = "", float_fmt: str = "{:.3f}") -> None:
    """Write CSV and a LaTeX table to output/tables."""
    df.to_csv(OUT_TAB / f"{name}.csv")
    try:
        latex = df.to_latex(float_format=lambda x: float_fmt.format(x), caption=caption,
                            label=f"tab:{name}", escape=True)
        (OUT_TAB / f"{name}.tex").write_text(latex)
    except Exception as e:      # noqa: BLE001  (jinja2 missing etc.)
        print(f"LaTeX export skipped for {name}: {e}")
    print(f"Saved table {name}")


# --------------------------------------------------------------------------
# Regression helpers (statsmodels)
# --------------------------------------------------------------------------
def fe_ols(df: pd.DataFrame, formula: str, cluster: str, weights: str | None = None):
    """OLS / WLS with fixed effects as dummies and cluster-robust SEs."""
    import statsmodels.formula.api as smf
    d = df.copy()
    if weights:
        d = d[d[weights].notna() & (d[weights] > 0)]
    # Let patsy decide which rows are complete, then align clusters and weights to them.
    rows = smf.ols(formula, data=d).data.row_labels
    d = d.loc[rows]
    groups = pd.factorize(d[cluster])[0]
    if weights:
        mod = smf.wls(formula, data=d, weights=d[weights].to_numpy())
    else:
        mod = smf.ols(formula, data=d)
    return mod.fit(cov_type="cluster", cov_kwds={"groups": groups})


def coef_table(res, keep_prefixes: tuple, label: str = "") -> pd.DataFrame:
    """Tidy coefficient table for the terms whose names start with any prefix."""
    names = [n for n in res.params.index if n.startswith(keep_prefixes)]
    ci = res.conf_int().loc[names]
    out = pd.DataFrame({
        "coef": res.params[names], "se": res.bse[names],
        "p": res.pvalues[names], "ci_low": ci[0], "ci_high": ci[1],
    })
    out["n_obs"] = int(res.nobs)
    if label:
        out.insert(0, "model", label)
    return out


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def short_name(s: str, n: int = 26) -> str:
    """Shorten an industry or occupation name at a word boundary."""
    s = str(s).split(",")[0].split("(")[0].strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# --------------------------------------------------------------------------
# Plot style
# --------------------------------------------------------------------------
PALETTE = {"main": "#1f4e79", "alt": "#c55a11", "grey": "#7f7f7f", "light": "#9dc3e6"}


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
        "font.family": "serif",
    })


def save_fig(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"{name}.{ext}", bbox_inches="tight")
    print(f"Saved figure {name}")


# --------------------------------------------------------------------------
# Occupation-panel analysis helpers (used by notebooks 04 and 06)
# --------------------------------------------------------------------------
def prepare_occ_panel(window: tuple = MAIN_WINDOW, exposure_col: str = "beta_gpt",
                      balanced: bool = True, drop_years: tuple = ()) -> tuple[pd.DataFrame, list]:
    """Load the occupation-year panel, restrict to a window, standardise exposure.

    Returns (panel, years). Exposure is z-scored across occupations (one value per SOC).
    """
    p = pd.read_csv(DATA_FIN / "occ_panel.csv", dtype={"soc": str})
    p = p[(p["year"] >= window[0]) & (p["year"] <= window[1])]
    p = p[~p["year"].isin(drop_years)].copy()
    years = sorted(p["year"].unique().tolist())
    if REF_YEAR not in years:
        raise ValueError(f"Reference year {REF_YEAR} not in panel years {years}")
    p = p.dropna(subset=[exposure_col, "tot_emp"])
    if balanced:
        n = p.groupby("soc")["year"].nunique()
        p = p[p["soc"].isin(n[n == len(years)].index)].copy()
    cross = p.drop_duplicates("soc").set_index("soc")
    p["ez"] = p["soc"].map(zscore(cross[exposure_col]))
    p["post"] = (p["year"] > REF_YEAR).astype(int)
    p["lemp"] = np.log(p["tot_emp"])
    p["lmed"] = np.log(p["a_median"])
    p["l9010"] = np.log(p["a_pct90"] / p["a_pct10"])
    p["l7525"] = np.log(p["a_pct75"] / p["a_pct25"])
    base = p[p["year"] == BASE_YEAR_WEIGHTS].set_index("soc")["tot_emp"]
    p["w"] = p["soc"].map(base)
    p["major"] = p["soc"].str[:2]
    p["major_year"] = p["major"] + "_" + p["year"].astype(str)
    return p.reset_index(drop=True), years


def demean(df: pd.DataFrame, cols: list, fes: list, w: np.ndarray | None = None,
           tol: float = 1e-10, maxit: int = 5000) -> np.ndarray:
    """Absorb one or more sets of fixed effects by alternating (weighted) demeaning."""
    X = df[cols].to_numpy(dtype=float).copy()
    codes = [pd.factorize(df[f])[0] for f in fes]
    wv = np.ones(len(df)) if w is None else np.asarray(w, dtype=float)
    sums = [np.bincount(c, weights=wv) for c in codes]
    for _ in range(maxit):
        old = X.copy()
        for c, sw in zip(codes, sums):
            means = np.vstack([np.bincount(c, weights=wv * X[:, j]) for j in range(X.shape[1])]).T / sw[:, None]
            X -= means[c]
        if len(fes) == 1 or np.max(np.abs(X - old)) < tol:
            break
    return X


def fe_reg(df: pd.DataFrame, y: str, xvars: list, fes: list, cluster: str = "soc",
           weights: str | None = None):
    """OLS/WLS of y on xvars with absorbed fixed effects and cluster-robust SEs.

    Equivalent to reghdfe y xvars [aw=weights], absorb(fes) vce(cluster cluster).
    """
    import statsmodels.api as sm
    d = df.dropna(subset=[y, *xvars, *fes, cluster] + ([weights] if weights else [])).copy()
    if weights:
        d = d[d[weights] > 0]
    w = d[weights].to_numpy(float) if weights else None
    Z = demean(d, [y, *xvars], fes, w)
    Y, X = Z[:, 0], pd.DataFrame(Z[:, 1:], columns=xvars, index=d.index)
    mod = sm.WLS(Y, X, weights=w) if weights else sm.OLS(Y, X)
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    res.n_clusters = int(d[cluster].nunique())
    return res


def _fe_list(fe: str) -> list:
    """Accept 'year', 'C(year)', 'major_year' or 'C(major_year)'; occupation FE is always included."""
    return ["soc", fe.replace("C(", "").replace(")", "").strip()]


def event_study(p: pd.DataFrame, y: str, years: list, var: str = "ez",
                fe: str = "year", weights: str | None = None):
    """y_ot = soc FE + `fe` FE + sum_{t != REF_YEAR} beta_t (var x 1[year = t])."""
    p = p.copy()
    cols = []
    for t in years:
        if t == REF_YEAR:
            continue
        c = f"{var}_y{t}"
        p[c] = p[var] * (p["year"] == t)
        cols.append(c)
    res = fe_reg(p, y, cols, _fe_list(fe), "soc", weights)
    tab = coef_table(res, (f"{var}_y",))
    tab["year"] = [int(n.split("_y")[-1]) for n in tab.index]
    ref = pd.DataFrame({"coef": [0.0], "se": [0.0], "p": [np.nan], "ci_low": [0.0],
                        "ci_high": [0.0], "n_obs": [int(res.nobs)], "year": [REF_YEAR]},
                       index=[f"{var}_y{REF_YEAR}"])
    tab = pd.concat([tab, ref]).sort_values("year")
    pre = [c for c in cols if int(c.split("_y")[-1]) < REF_YEAR]
    tab.attrs["pretrend_p"] = (float(res.wald_test(", ".join(f"{c} = 0" for c in pre),
                                                   scalar=True).pvalue) if pre else np.nan)
    tab.attrs["n_soc"] = res.n_clusters
    return tab, res


def pooled(p: pd.DataFrame, y: str, terms: dict, fe: str = "year",
           weights: str | None = None, label: str = ""):
    """Pooled DiD. `terms` maps new column name -> Series, e.g. {'ez_post': p.ez * p.post}."""
    p = p.copy()
    for k, v in terms.items():
        p[k] = v
    res = fe_reg(p, y, list(terms), _fe_list(fe), "soc", weights)
    return coef_table(res, tuple(terms), label=label or y), res


def plot_event(tabs: dict, title: str, ylabel: str, name: str) -> None:
    """tabs: {label: event-study table}. Draws point estimates with 95% CIs."""
    import matplotlib.pyplot as plt
    set_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    colors = [PALETTE["main"], PALETTE["alt"], PALETTE["grey"], PALETTE["light"]]
    k = len(tabs)
    for i, (lab, t) in enumerate(tabs.items()):
        off = (i - (k - 1) / 2) * 0.12
        ax.errorbar(t["year"] + off, t["coef"],
                    yerr=[t["coef"] - t["ci_low"], t["ci_high"] - t["coef"]],
                    fmt="o", capsize=3, color=colors[i % 4], label=lab)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(REF_YEAR + 0.5, color=PALETTE["grey"], ls="--", lw=0.8)
    ax.text(REF_YEAR + 0.55, ax.get_ylim()[1] * 0.9, "ChatGPT\n(Nov 2022)", fontsize=8, va="top")
    ax.set_xlabel("OEWS reference year (May)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if k > 1:
        ax.legend(loc="lower left", fontsize=8)
    save_fig(fig, name)
    plt.show()


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha for a set of item columns (rows = occupations)."""
    x = items.dropna()
    k = x.shape[1]
    return float(k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1)))
