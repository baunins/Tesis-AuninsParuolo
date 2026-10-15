import os
import gc
import sys
import hashlib
import datetime as dt
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*singleton fixed effect.*")

PANEL_CSV = "panel_eis.csv"
OUTDIR    = "ResultadosCorrecciones"

Y    = "ret_d_w"

EXPOSICION = {
    "aiie": ("AIIE", "naics4", "resultados"),
    "eis":  ("EIS",  "naics3", "resultados_eis"),
}
AIIE = "AIIE"
CL1  = "naics4"
CL2  = "naics4 + date_i"

MUESTRA_COL = "naics4"

ESTANDARIZAR = True

Z = ["fixed_assets_to_assets", "intangibles_to_assets",
     "book_to_market", "leverage"]

Z_FUERA = ["beta", "size", "duration"]

SH_LVL = ["FOMC", "GPR", "EPU"]
SH_ABS = ["FOMC_abs", "GPR_abs", "EPU_abs"]
TODOS  = SH_LVL + SH_ABS

SH_EVENTO = ["FOMC"]

POST_DESDE = "2022-11-30"

FOMC_FIN   = "auto"
FOMC_TOPE  = None
HMAX       = 20

FE = "permaticker + sector_day"

def huella(path):
    st = os.stat(path)
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if st.st_size > (2 << 20):
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return (f"{os.path.basename(path)} | {st.st_size/1e6:,.1f} MB | "
            f"modificado {dt.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M} | "
            f"huella {h.hexdigest()[:12]}")

def cargar(muestra_extra=(), etiqueta=""):
    pedidas = (["permaticker", "date", "closeadj", Y, AIIE, CL1, MUESTRA_COL,
                "sector", "in_sp500"] + Z + Z_FUERA + TODOS + list(muestra_extra))
    usecols = list(dict.fromkeys(pedidas))
    head = pd.read_csv(PANEL_CSV, nrows=0).columns
    falta = [c for c in usecols if c not in head]
    if falta:
        sys.exit(f"[error] a {PANEL_CSV} le faltan columnas: {falta}\n"
                 f"        si son EIS_z, AIIE_z o naics3, corre antes "
                 f"agregar_eis_al_panel.py")
    dtypes = {c: "float32" for c in [Y, AIIE, "closeadj"] + Z + Z_FUERA + TODOS
              + list(muestra_extra)}
    dtypes.update({CL1: "float32", MUESTRA_COL: "float32",
                   "permaticker": "int32",
                   "in_sp500": "int8", "sector": "category"})

    print(f"cargando {PANEL_CSV} ...")
    df = pd.read_csv(PANEL_CSV, usecols=usecols, dtype=dtypes, parse_dates=["date"])

    df = df[df["in_sp500"] == 1].drop(columns="in_sp500")

    df["date_i"] = (df["date"].values.astype("datetime64[D]")
                    .astype("int32"))
    df["sector_day"] = pd.factorize(
        df["sector"].cat.codes.astype("int64") * 1_000_000 + df["date_i"])[0]
    df["Post"] = (df["date"] >= pd.Timestamp(POST_DESDE)).astype("int8")

    n0 = len(df)
    sin_ind = df[MUESTRA_COL].isna() & df[AIIE].notna()
    if sin_ind.any():
        print(f"  [ojo] {int(sin_ind.sum()):,} filas "
              f"({df.loc[sin_ind, 'permaticker'].nunique()} firmas) tienen {AIIE} "
              f"pero no {MUESTRA_COL}: quedan fuera de la muestra, tambien con "
              f"--cluster firma (si no, las dos corridas no serian comparables).")
    df = df.dropna(subset=[MUESTRA_COL, "sector"])
    if len(df) < n0:
        print(f"  {n0-len(df):,} filas descartadas por {MUESTRA_COL}/sector vacio")
    df[MUESTRA_COL] = df[MUESTRA_COL].astype("int32")
    if CL1 != MUESTRA_COL and CL1 != "permaticker":
        df = df.dropna(subset=[CL1])
        df[CL1] = df[CL1].astype("int32")

    n1 = len(df)
    df = df.dropna(subset=[AIIE])
    if len(df) < n1:
        print(f"  {n1-len(df):,} filas sin {AIIE} descartadas "
              f"({df['permaticker'].nunique()} firmas quedan)")

    if muestra_extra:
        n2 = len(df)
        df = df.dropna(subset=list(muestra_extra))
        print(f"  [{etiqueta or 'muestra'}] exige {list(muestra_extra)} no faltantes: "
              f"{n2:,} -> {len(df):,} filas ({n2-len(df):,} descartadas, "
              f"{df['permaticker'].nunique()} firmas quedan)")

    nv = df.groupby("permaticker", observed=True)[AIIE].nunique()
    if (nv > 1).any():
        malas = nv[nv > 1]
        sys.exit(f"[error] {AIIE} NO es constante dentro de la firma en "
                 f"{len(malas)} firmas (por ejemplo {list(malas.index[:5])}).\n"
                 f"        El diseño supone exposicion invariante en el tiempo: el "
                 f"efecto fijo de firma la absorbe y solo se estima la interaccion.\n"
                 f"        Revisa como se armo la columna en el panel.")

    vals = df.drop_duplicates("permaticker")[AIIE]
    n_val = int(vals.nunique())
    print(f"  {AIIE}: {n_val:,} valores distintos entre {len(vals):,} firmas")
    if AIIE == "EIS" and n_val < 0.5 * len(vals):
        aviso = (f"el EIS toma {n_val} valores distintos entre {len(vals)} firmas: esto "
                 f"parece la medida POR INDUSTRIA, no la de Eisfeldt por firma.\n"
                 f"        Reconstrui el panel con empresas_unicas_EIS_firma_detalle.csv "
                 f"(python eis_por_firma.py).")
        if CL1 == "permaticker":
            sys.exit(f"[error] {aviso}\n        Con una medida por industria, clusterizar "
                     f"por firma no tiene sentido.")
        print(f"  [OJO] {aviso}")

    if ESTANDARIZAR:
        f1 = df.drop_duplicates("permaticker")[AIIE]
        mu, sd = float(f1.mean()), float(f1.std())
        df[AIIE] = ((df[AIIE] - mu) / sd).astype("float32")
        print(f"  {AIIE} estandarizado en el corte de firmas: "
              f"media={mu:.4f}  sd={sd:.4f}  (beta = efecto de 1 desvio)")

    df = df.sort_values(["permaticker", "date"]).reset_index(drop=True)
    gc.collect()

    print(f"panel: {len(df):,} filas | {df['permaticker'].nunique()} firmas | "
          f"{df[MUESTRA_COL].nunique()} {MUESTRA_COL} | "
          f"{df[CL1].nunique()} {CL1} (clusters) | "
          f"{df['sector_day'].nunique():,} sector-dia | "
          f"{df['date_i'].nunique():,} fechas")
    return df

def resolver_tope_fomc(df):
    global FOMC_TOPE
    if str(FOMC_FIN).lower() == "auto":
        con = df.loc[df["FOMC"].notna() & (df["FOMC"] != 0), "date"]
        FOMC_TOPE = con.max() if len(con) else df["date"].max()
        print(f"  tope de muestra para FOMC y FOMC_abs (auto): {FOMC_TOPE.date()} "
              f"= ultima fecha con sorpresa distinta de cero. Las fechas "
              f"posteriores tienen shock cero: no identifican nada y solo "
              f"inflarian el N.")
    else:
        FOMC_TOPE = pd.Timestamp(FOMC_FIN)
        print(f"  tope de muestra para FOMC y FOMC_abs (fijado a mano): "
              f"{FOMC_TOPE.date()}")
    return FOMC_TOPE

def muestra(df, x):
    if x.startswith("FOMC"):
        if FOMC_TOPE is None:
            sys.exit("[error] interno: el tope del FOMC no fue resuelto")
        return df[df["date"] <= FOMC_TOPE]
    return df

PERIODOS = [
    (1, "P1 placebo (hasta 2011)",          None,         "2011-12-31"),
    (2, "P2 aprendizaje profundo (2012-nov 2022)", "2012-01-01", "2022-11-29"),
    (3, "P3 ChatGPT (desde 30/11/2022)",     "2022-11-30", None),
]
ETQ = {p: e for p, e, _, _ in PERIODOS}
OUTDIR_BASE = "ResultadosCorrecciones"
FE_P = "permaticker^periodo + sector_day"
PCT_MIN   = 99.0
HUECO_MAX = 10
MIN_DIAS  = 15

def buscar_coef(t, partes):
    for i in t.index:
        if sorted(str(i).split(":")) == sorted(partes):
            return i
    return None

def asignar_periodo(df):
    p = np.zeros(len(df), dtype="int8")
    for cod, _, desde, hasta in PERIODOS:
        m = np.ones(len(df), dtype=bool)
        if desde:
            m &= (df["date"] >= pd.Timestamp(desde)).to_numpy()
        if hasta:
            m &= (df["date"] <= pd.Timestamp(hasta)).to_numpy()
        if (p[m] != 0).any():
            sys.exit("[error] los periodos se superponen: revisa PERIODOS")
        p[m] = cod
    if (p == 0).any():
        sys.exit("[error] hay fechas que no caen en ningun periodo: revisa PERIODOS")
    df["periodo"] = p
    return df

def firmas_permanentes(df, definicion):
    fechas = np.sort(df["date_i"].unique())
    T = len(fechas)
    pos = pd.Series(np.arange(T), index=fechas)
    g = df[["permaticker", "date_i"]].drop_duplicates()
    g["k"] = pos.reindex(g["date_i"]).to_numpy()
    g = g.sort_values(["permaticker", "k"])
    g["gap"] = g.groupby("permaticker")["k"].diff().fillna(1) - 1
    r = g.groupby("permaticker").agg(ruedas=("k", "size"), primera=("k", "min"),
                                     ultima=("k", "max"), hueco_int=("gap", "max"))
    r["hueco_max"] = np.maximum.reduce([r["hueco_int"].to_numpy(),
                                        r["primera"].to_numpy(),
                                        (T - 1 - r["ultima"]).to_numpy()])
    r["pct"] = 100 * r["ruedas"] / T
    r["estricta"] = r["ruedas"] == T
    r["tolerante"] = (r["pct"] >= PCT_MIN) & (r["hueco_max"] <= HUECO_MAX)
    print("\n" + "=" * 72 + "\nFIRMAS PERMANENTES\n" + "=" * 72)
    print(f"  ventana: {pd.Timestamp(fechas[0], unit='D').date()} -> "
          f"{pd.Timestamp(fechas[-1], unit='D').date()}  ({T:,} ruedas)")
    print(f"  estricta  (100 % de las ruedas)                  : {int(r['estricta'].sum()):,}")
    print(f"  tolerante (>= {PCT_MIN:g} % y hueco <= {HUECO_MAX} ruedas)    : "
          f"{int(r['tolerante'].sum()):,}")
    print(f"  se usa: {definicion}")
    return r[r[definicion]].reset_index()[["permaticker", "ruedas", "pct", "hueco_max"]]

def sd_shock(df, x):
    con = df[x].notna() & (df[x] != 0)
    return float(df.loc[con, ["date", x]].drop_duplicates("date")[x].std())

def _extraer(m, nombres):
    cn = list(m._coefnames)
    idx = [cn.index(n) for n in nombres]
    b = np.asarray(m.coef(), dtype=float)[idx]
    V = np.asarray(m._vcov, dtype=float)[np.ix_(idx, idx)]
    return b, V

def _tests(b, V, cods, G):
    from scipy import stats
    out = {}
    k = len(b)
    if k >= 2:
        R = np.zeros((k - 1, k))
        R[:, 0] = -1
        R[np.arange(k - 1), np.arange(1, k)] = 1
        rb = R @ b
        W = float(rb @ np.linalg.pinv(R @ V @ R.T) @ rb)
        F = W / (k - 1)
        out["igualdad"] = (np.nan, np.nan, F, f"F({k-1},{G-1})",
                           float(stats.f.sf(F, k - 1, G - 1)))
    for i in range(k):
        for j in range(i + 1, k):
            dif = b[j] - b[i]
            se = float(np.sqrt(max(V[i, i] + V[j, j] - 2 * V[i, j], 0)))
            t = dif / se if se > 0 else np.nan
            out[f"P{cods[j]}-P{cods[i]}"] = (dif, se, t, f"t({G-1})",
                                             float(2 * stats.t.sf(abs(t), G - 1)))
    return out

def estimar_periodos(pf, df, x, sdx, etq_muestra, verificar=False):
    d = muestra(df, x)
    cols = list(dict.fromkeys([Y, AIIE, x, CL1, "permaticker", "sector_day",
                               "date_i", "date", "periodo"] + Z))
    d = d[cols].dropna(subset=cols).reset_index(drop=True)

    con = d[x] != 0
    dias = d.loc[con].groupby("periodo")["date_i"].nunique()
    activos = [p for p, *_ in PERIODOS if dias.get(p, 0) > 0]
    for p, *_ in PERIODOS:
        n = int(dias.get(p, 0))
        if n == 0:
            print(f"    [ojo] {x}: ningun dia con shock en {ETQ[p]} -> ese periodo "
                  f"no se estima y sus filas salen de la muestra")
        elif n < MIN_DIAS:
            print(f"    [ojo] {x}: solo {n} dias con shock en {ETQ[p]}: "
                  f"el beta de ese periodo va a ser poco preciso")
    if not activos:
        return [], {}, None, []
    d = d[d["periodo"].isin(activos)].reset_index(drop=True)

    terminos, nombres = [], []
    for p in activos:
        c = f"{x}_P{p}"
        d[c] = (d[x] * (d["periodo"] == p)).astype("float32")
        terminos += [f"{AIIE}:{c}"] + [f"{z}:{c}" for z in Z]
    formula = f"{Y} ~ {' + '.join(terminos)} | {FE_P}"

    m = pf.feols(formula, data=d, vcov={"CRV1": CL1})
    t1 = m.tidy()
    claves = [buscar_coef(t1, [AIIE, f"{x}_P{p}"]) for p in activos]
    if any(c is None for c in claves):
        print(f"    [error] no encontre los coeficientes de {x} en la salida")
        return [], {}, None, []
    b, V1 = _extraer(m, claves)
    G1 = int(d[CL1].nunique())
    tests1 = _tests(b, V1, activos, G1)

    t2, V2, tests2 = None, None, {}
    try:
        try:
            m.vcov({"CRV1": CL2})
        except Exception:
            m = pf.feols(formula, data=d, vcov={"CRV1": CL2})
        t2 = m.tidy()
        _, V2 = _extraer(m, claves)
        G2 = int(min(d[c].nunique() for c in CL2.replace(" ", "").split("+")))
        tests2 = _tests(b, V2, activos, G2)
    except Exception as e:
        print(f"    (cluster de dos vias fallo: {e})")
    n_est = int(getattr(m, "_N", len(d)))
    del m
    gc.collect()

    filas = []
    for p, c in zip(activos, claves):
        sub = d[d["periodo"] == p]
        f = {"muestra": etq_muestra, "shock": x, "periodo": f"P{p}",
             "etiqueta": ETQ[p], "desde": sub["date"].min().date(),
             "hasta": sub["date"].max().date(),
             "dias_con_shock": int(dias.get(p, 0)), "N": len(sub),
             "firmas": int(sub["permaticker"].nunique()),
             "beta": t1.loc[c, "Estimate"],
             "se_1via": t1.loc[c, "Std. Error"], "p_1via": t1.loc[c, "Pr(>|t|)"]}
        if t2 is not None:
            f["se_2vias"] = t2.loc[c, "Std. Error"]
            f["p_2vias"] = t2.loc[c, "Pr(>|t|)"]
        f["efecto_1sd"] = f["beta"] * sdx if sdx else np.nan
        f["N_modelo"] = n_est
        filas.append(f)

    tests = []
    for nombre, (dif, se, st, dist, pv) in tests1.items():
        r = {"muestra": etq_muestra, "shock": x,
             "test": "H0: beta igual en todos los periodos" if nombre == "igualdad"
             else f"beta_{nombre.split('-')[0]} - beta_{nombre.split('-')[1]}",
             "diferencia": dif, "se_1via": se, "estadistico_1via": st,
             "dist_1via": dist, "p_1via": pv}
        if nombre in tests2:
            _, se2, st2, dist2, pv2 = tests2[nombre]
            r.update({"se_2vias": se2, "estadistico_2vias": st2,
                      "dist_2vias": dist2, "p_2vias": pv2})
        tests.append(r)

    verif = []
    if verificar:
        for p in activos:
            s = d[d["periodo"] == p]
            ms = pf.feols(f"{Y} ~ {AIIE}:{x} + " + " + ".join(f"{z}:{x}" for z in Z)
                          + " | permaticker + sector_day", data=s, vcov={"CRV1": CL1})
            ts = ms.tidy()
            k = buscar_coef(ts, [AIIE, x])
            bc = float(t1.loc[claves[activos.index(p)], "Estimate"])
            verif.append({"muestra": etq_muestra, "shock": x, "periodo": f"P{p}",
                          "beta_conjunto": bc, "beta_separado": float(ts.loc[k, "Estimate"]),
                          "dif_abs": abs(bc - float(ts.loc[k, "Estimate"])),
                          "se_conjunto": float(t1.loc[claves[activos.index(p)], "Std. Error"]),
                          "se_separado": float(ts.loc[k, "Std. Error"])})
            del ms
            gc.collect()

    tabla = t1.reset_index()
    if t2 is not None:
        tabla = tabla.merge(t2.reset_index()[[tabla.columns[0], "Std. Error", "Pr(>|t|)"]]
                            .rename(columns={"Std. Error": "se_2vias",
                                             "Pr(>|t|)": "p_2vias"}),
                            on=tabla.columns[0], how="left")
    return filas, {"tests": tests}, tabla, verif

def _acumular(path, nuevo, claves):
    if os.path.exists(path):
        viejo = pd.read_csv(path)
        llaves = set(map(tuple, nuevo[claves].astype(str).values))
        viejo = viejo[[tuple(v) not in llaves
                       for v in viejo[claves].astype(str).values]]
        nuevo = pd.concat([viejo, nuevo], ignore_index=True)
    nuevo.to_csv(path, index=False)
    return nuevo

def graficar(coef, outdir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (no pude graficar: {e})")
        return
    se = "se_2vias" if "se_2vias" in coef.columns else "se_1via"
    for x, g in coef.groupby("shock"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(0, color="black", lw=.8)
        muestras = list(g["muestra"].unique())
        for i, (mu, h) in enumerate(g.groupby("muestra", sort=False)):
            pos = h["periodo"].str[1:].astype(int) + (i - (len(muestras) - 1) / 2) * 0.15
            ax.errorbar(pos, h["beta"], yerr=1.96 * h[se], fmt="o", capsize=4,
                        label=mu)
        ax.set_xticks([p for p, *_ in PERIODOS])
        ax.set_xticklabels(["P1 placebo\n(<=2011)", "P2 deep learning\n(2012-nov 2022)",
                            "P3 ChatGPT\n(>=30/11/2022)"])
        ax.set_ylabel(f"beta (exposicion x {x})")
        ax.set_title(f"{x}: efecto por periodo, IC 95 % ({'cluster 2 vias' if se == 'se_2vias' else 'cluster 1 via'})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{outdir}/periodos_{x}.png", dpi=130)
        plt.close(fig)

def main():
    import pyfixest as pf
    global AIIE, CL1, CL2, OUTDIR, PANEL_CSV, MUESTRA_COL

    args = sys.argv[1:]
    def flag(n): return n in args
    def valor(n, default=None):
        if n in args:
            i = args.index(n) + 1
            if i < len(args) and not args[i].startswith("--"):
                return args[i]
        return default

    modo = "eis" if flag("--eis") else "aiie"
    AIIE, CL1, _ = EXPOSICION[modo]
    MUESTRA_COL = CL1
    sub = modo
    PANEL_CSV = valor("--panel", PANEL_CSV)
    if not os.path.exists(PANEL_CSV):
        sys.exit(f"[error] no existe {PANEL_CSV}. Es el mismo panel del script "
                 f"principal (lo crea agregar_eis_al_panel.py).")
    cl_pedido = valor("--cluster", "industria").lower()
    if cl_pedido not in ("industria", "firma"):
        sys.exit("[error] --cluster: valores validos 'industria' y 'firma'.")
    if cl_pedido == "firma":
        if modo == "aiie":
            sys.exit("[error] --cluster firma no va con el AIIE (Moulton). Solo con --eis.")
        CL1 = "permaticker"
        sub += "_cluster_firma"
    extra, etiqueta = [], ""
    if flag("--muestra-comun") and modo == "aiie":
        extra, etiqueta = ["EIS", "naics3"], "--muestra-comun"
        sub += "_muestra_comun"
    CL2 = f"{CL1} + date_i"

    definicion = valor("--definicion", "tolerante").lower()
    if definicion not in ("tolerante", "estricta"):
        sys.exit("[error] --definicion: 'tolerante' o 'estricta'.")
    que = valor("--muestra", "ambas").lower()
    if que not in ("completa", "permanentes", "ambas"):
        sys.exit("[error] --muestra: 'completa', 'permanentes' o 'ambas'.")
    shocks = [s.strip() for s in valor("--shocks", ",".join(TODOS)).split(",") if s.strip()]
    malos = [s for s in shocks if s not in TODOS]
    if malos:
        sys.exit(f"[error] --shocks {malos}: validos {TODOS}")
    perm_csv = valor("--permanentes-csv")
    if perm_csv and not os.path.exists(perm_csv):
        sys.exit(f"[error] no existe {perm_csv}")

    OUTDIR = os.path.join(OUTDIR_BASE, sub)
    os.makedirs(os.path.join(OUTDIR, "tablas"), exist_ok=True)
    print(f"MODO: {modo.upper()} | exposicion = {AIIE} | cluster = {CL1} (y {CL1} + fecha)"
          f" | FE = {FE_P} | salida = ./{OUTDIR}/")
    hp = huella(PANEL_CSV)
    print(f"PANEL: {hp}")
    with open(os.path.join(OUTDIR, "panel_usado.txt"), "a") as f:
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M}  "
                f"{' '.join(args) if args else '(base)'}  ->  {hp}\n")

    df = cargar(muestra_extra=extra, etiqueta=etiqueta)
    resolver_tope_fomc(df)
    df = asignar_periodo(df)

    cortes = []
    for p, e, desde, hasta in PERIODOS:
        s = df[df["periodo"] == p]
        fila = {"periodo": f"P{p}", "etiqueta": e, "desde_def": desde or "inicio",
                "hasta_def": hasta or "fin", "primera_fecha": s["date"].min().date(),
                "ultima_fecha": s["date"].max().date(), "ruedas": s["date_i"].nunique(),
                "firmas": s["permaticker"].nunique()}
        for x in TODOS:
            sx = muestra(s, x)
            fila[f"dias_{x}"] = int(sx.loc[sx[x].notna() & (sx[x] != 0), "date_i"].nunique())
        cortes.append(fila)
    cortes = pd.DataFrame(cortes)
    cortes.to_csv(f"{OUTDIR}/CORTES.csv", index=False)
    print("\n" + "=" * 72 + "\nCORTES (fijados antes de estimar)\n" + "=" * 72)
    print(cortes.to_string(index=False))

    muestras = []
    if que in ("completa", "ambas"):
        muestras.append(("completa", df))
    if que in ("permanentes", "ambas"):
        if perm_csv:
            lista = pd.read_csv(perm_csv)
            perm = lista[["permaticker"]].drop_duplicates()
            print(f"\n  permanentes tomadas de {perm_csv}: {len(perm):,} firmas")
        else:
            perm = firmas_permanentes(df, definicion)
        perm.to_csv(f"{OUTDIR}/PERMANENTES.csv", index=False)
        dp = df[df["permaticker"].isin(set(perm["permaticker"].astype("int64")))]
        print(f"  muestra de permanentes: {dp['permaticker'].nunique():,} firmas, "
              f"{len(dp):,} filas")
        muestras.append(("permanentes", dp))

    sds = {x: sd_shock(df, x) for x in TODOS}

    coef, tests, verif = [], [], []
    for etq, d in muestras:
        print("\n" + "=" * 72 +
              f"\nMODELO CONJUNTO  r ~ sum_p beta_p (AIIE x shock x D_p) + "
              f"sum_p gamma_p'(Z x shock x D_p)\nmuestra: {etq}\n" + "=" * 72)
        for x in shocks:
            print(f"  {x} ...")
            fl, ex, tabla, vf = estimar_periodos(pf, d, x, sds.get(x), etq,
                                                 verificar=flag("--verificar"))
            if not fl:
                continue
            for f in fl:
                print(f"    {f['periodo']}  beta={f['beta']: .5g}  "
                      f"p1={f['p_1via']:.4f}  p2={f.get('p_2vias', float('nan')):.4f}  "
                      f"dias={f['dias_con_shock']}  N={f['N']:,}")
            for t in ex["tests"]:
                print(f"    {t['test']:40s} p1={t['p_1via']:.4f}  "
                      f"p2={t.get('p_2vias', float('nan')):.4f}")
            for v in vf:
                print(f"    [verificar] {v['periodo']} conjunto={v['beta_conjunto']:.6g} "
                      f"separado={v['beta_separado']:.6g}  |dif|={v['dif_abs']:.2e}")
            coef += fl
            tests += ex["tests"]
            verif += vf
            tabla.to_csv(f"{OUTDIR}/tablas/{etq}_{x}.csv", index=False)
            _acumular(f"{OUTDIR}/PERIODOS_COEF.csv", pd.DataFrame(fl), ["muestra", "shock"])
            _acumular(f"{OUTDIR}/PERIODOS_TESTS.csv", pd.DataFrame(ex["tests"]),
                      ["muestra", "shock"])
            if vf:
                _acumular(f"{OUTDIR}/VERIFICACION.csv", pd.DataFrame(vf),
                          ["muestra", "shock"])
            gc.collect()

    if not coef:
        print("\nno se estimo nada")
        return

    C = pd.read_csv(f"{OUTDIR}/PERIODOS_COEF.csv")
    T = pd.read_csv(f"{OUTDIR}/PERIODOS_TESTS.csv")
    for tab in (C, T):
        for c, n in [("p_1via", "signif_1via"), ("p_2vias", "signif_2vias")]:
            if c in tab.columns:
                tab[n] = pd.cut(tab[c], [0, .01, .05, .10, 1],
                                labels=["***", "**", "*", "ns"], include_lowest=True)
    C.to_csv(f"{OUTDIR}/PERIODOS_COEF.csv", index=False)
    T.to_csv(f"{OUTDIR}/PERIODOS_TESTS.csv", index=False)

    comp = None
    if set(C["muestra"]) >= {"completa", "permanentes"}:
        keep = ["shock", "periodo", "beta", "se_1via", "p_1via", "N", "firmas"]
        keep += [c for c in ("se_2vias", "p_2vias") if c in C.columns]
        a = C[C["muestra"] == "completa"][keep]
        b = C[C["muestra"] == "permanentes"][keep]
        comp = a.merge(b, on=["shock", "periodo"], suffixes=("_completa", "_perm"))
        comp["dif_beta"] = comp["beta_perm"] - comp["beta_completa"]
        comp.to_csv(f"{OUTDIR}/PERIODOS_COMPOSICION.csv", index=False)
        print("\n" + "=" * 72 + "\nEFECTO COMPOSICION (completa vs permanentes)\n" + "=" * 72)
        print(comp[["shock", "periodo", "beta_completa", "beta_perm", "dif_beta"]]
              .to_string(index=False))

    print("\n" + "=" * 72 + "\nRESUMEN\n" + "=" * 72)
    cols = [c for c in ["muestra", "shock", "periodo", "beta", "efecto_1sd", "p_1via",
                        "signif_1via", "p_2vias", "signif_2vias", "dias_con_shock", "N"]
            if c in C.columns]
    print(C[cols].to_string(index=False))
    print()
    cols = [c for c in ["muestra", "shock", "test", "diferencia", "p_1via",
                        "signif_1via", "p_2vias", "signif_2vias"] if c in T.columns]
    print(T[cols].to_string(index=False))

    graficar(C, OUTDIR)

    xlsx = f"{OUTDIR}/resultados.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
            C.to_excel(w, sheet_name="PERIODOS_COEF", index=False)
            T.to_excel(w, sheet_name="PERIODOS_TESTS", index=False)
            if comp is not None:
                comp.to_excel(w, sheet_name="PERIODOS_COMPOSICION", index=False)
            cortes.to_excel(w, sheet_name="CORTES", index=False)
            if os.path.exists(f"{OUTDIR}/VERIFICACION.csv"):
                pd.read_csv(f"{OUTDIR}/VERIFICACION.csv").to_excel(
                    w, sheet_name="VERIFICACION", index=False)
            for fn in sorted(os.listdir(f"{OUTDIR}/tablas")):
                pd.read_csv(f"{OUTDIR}/tablas/{fn}").to_excel(
                    w, sheet_name=fn[:-4][:31], index=False)
        print(f"\n-> {xlsx}")
    except Exception as e:
        print(f"\n(no pude escribir el Excel: {e}; quedan los CSV)")
    print(f"-> todo en ./{OUTDIR}/")
    print("Data from Sharadar. No redistribuir el crudo.")

if __name__ == "__main__":
    main()
