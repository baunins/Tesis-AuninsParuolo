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
OUTDIR    = "resultados"

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

OUTDIR_BASE = "ResultadosCorrecciones"
BLOQUES = [("con signo", SH_LVL), ("valor absoluto", SH_ABS)]

def buscar_coef(t, partes):
    for i in t.index:
        if sorted(str(i).split(":")) == sorted(partes):
            return i
    return None

def zx(x):
    return " + ".join(f"{z}:{x}" for z in Z)

def sd_shock(df, x):
    con = df[x].notna() & (df[x] != 0)
    return float(df.loc[con, ["date", x]].drop_duplicates("date")[x].std())

def _es_evento(x):
    return any(x == e or x == f"{e}_abs" for e in SH_EVENTO)

def correlaciones(df):
    s0 = df.drop_duplicates("date")[["date"] + TODOS].sort_values("date")
    filas = []
    print("\n" + "=" * 72 + "\nCORRELACIONES ENTRE SHOCKS (nivel fecha)\n" + "=" * 72)
    for version, cols in BLOQUES:
        s = s0[cols].copy()
        for x in cols:
            if _es_evento(x):
                s[x] = s[x].where(s[x] != 0)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                d = s[[a, b]].dropna()
                ok = len(d) >= 3 and d[a].nunique() > 1 and d[b].nunique() > 1
                filas.append({
                    "version": version, "par": f"{a}-{b}", "N": len(d),
                    "muestra": ("solo reuniones del FOMC"
                                if _es_evento(a) or _es_evento(b)
                                else "dias con dato en ambas series"),
                    "pearson": float(d[a].corr(d[b])) if ok else np.nan,
                    "spearman": float(d[a].corr(d[b], method="spearman")) if ok else np.nan})
    out = pd.DataFrame(filas)
    print(out.round(4).to_string(index=False))
    return out

def _fit(pf, formula, d):
    m = pf.feols(formula, data=d, vcov={"CRV1": CL1})
    t1 = m.tidy()
    V1 = np.array(m._vcov, dtype=float, copy=True)
    t2 = V2 = None
    try:
        try:
            m.vcov({"CRV1": CL2})
        except Exception:
            m = pf.feols(formula, data=d, vcov={"CRV1": CL2})
        t2 = m.tidy()
        V2 = np.array(m._vcov, dtype=float, copy=True)
    except Exception as e:
        print(f"    (dos vias fallo: {e})")
    return t1, t2, m, V1, V2

def _wald_cero(m, V, nombres, G):
    from scipy import stats
    cn = list(m._coefnames)
    idx = [cn.index(n) for n in nombres]
    b = np.asarray(m.coef(), dtype=float)[idx]
    V = V[np.ix_(idx, idx)]
    F = float(b @ np.linalg.pinv(V) @ b) / len(b)
    return F, float(stats.f.sf(F, len(b), G - 1))

def _filas(t1, t2, shocks, ejercicio, version, estimacion, n, sds, G):
    from scipy import stats
    tc = stats.t.ppf(.975, G - 1)
    out = []
    for x in shocks:
        k = buscar_coef(t1, [AIIE, x])
        if k is None:
            continue
        esc = sds[x] * 1e4
        b, s1 = t1.loc[k, "Estimate"], t1.loc[k, "Std. Error"]
        f = {"ejercicio": ejercicio, "version": version, "shock": x,
             "estimacion": estimacion, "N": n,
             "efecto_pb": b * esc, "se1_pb": s1 * esc,
             "ic1_lo": (b - tc * s1) * esc, "ic1_hi": (b + tc * s1) * esc,
             "p_1via": t1.loc[k, "Pr(>|t|)"], "beta": b}
        if t2 is not None:
            s2 = t2.loc[k, "Std. Error"]
            f.update({"se2_pb": s2 * esc, "ic2_lo": (b - tc * s2) * esc,
                      "ic2_hi": (b + tc * s2) * esc, "p_2vias": t2.loc[k, "Pr(>|t|)"]})
        out.append(f)
    return out

def muestra_comun(df, shocks, extra=()):
    d = df[df["date"] <= FOMC_TOPE] if any(_es_evento(x) for x in shocks) else df
    cols = list(dict.fromkeys([Y, AIIE, CL1, "permaticker", "sector_day", "date_i",
                               "date"] + Z + list(shocks) + list(extra)))
    return d[cols].dropna(subset=cols).reset_index(drop=True)

def base_simultaneo(pf, df, sds, tablas, wald):
    filas = []
    for version, shocks in BLOQUES:
        print(f"\n  BASE, {version}: {', '.join(shocks)}")
        d = muestra_comun(df, shocks)
        G = int(d[CL1].nunique())
        rhs = " + ".join(f"{AIIE}:{x} + {zx(x)}" for x in shocks)
        t1, t2, m, V1, V2 = _fit(pf, f"{Y} ~ {rhs} | {FE}", d)
        n = int(getattr(m, "_N", len(d)))
        claves = [buscar_coef(t1, [AIIE, x]) for x in shocks]
        F, p = _wald_cero(m, V1, claves, G)
        wald.append({"ejercicio": "base", "version": version,
                     "gl": f"F({len(shocks)},{G-1})", "F_1via": F, "p_1via": p})
        if V2 is not None:
            F2, p2 = _wald_cero(m, V2, claves, G)
            wald[-1].update({"F_2vias": F2, "p_2vias": p2})
        del m; gc.collect()
        tablas[f"base_conj_{version.split()[0]}"] = t1.reset_index()
        filas += _filas(t1, t2, shocks, "base", version, "conjunta", n, sds, G)
        for x in shocks:
            s1, s2, ms, _, _ = _fit(pf, f"{Y} ~ {AIIE}:{x} + {zx(x)} | {FE}", d)
            filas += _filas(s1, s2, [x], "base", version, "separada",
                            int(getattr(ms, "_N", len(d))), sds, G)
            del ms; gc.collect()
        for x in shocks:
            a = [f for f in filas if f["shock"] == x and f["ejercicio"] == "base"]
            sep = [f for f in a if f["estimacion"] == "separada"][-1]
            con = [f for f in a if f["estimacion"] == "conjunta"][-1]
            print(f"    {x:9s} separada {sep['efecto_pb']:+.3f} pb (p {sep['p_1via']:.3f})"
                  f"  | conjunta {con['efecto_pb']:+.3f} pb (p {con['p_1via']:.3f})")
        print(f"    Wald tres beta = 0 (1 via): F = {F:.3f}, p = {p:.3f}")
    return filas

def _residuo_abs(pf, d, rhs):
    m1 = pf.feols(f"{Y} ~ {rhs} | {FE}", data=d, vcov={"CRV1": CL1}, fixef_rm="none")
    r = np.asarray(m1.resid()).ravel()
    del m1; gc.collect()
    if len(r) != len(d):
        raise RuntimeError(f"residuos {len(r):,} vs filas {len(d):,}")
    return np.abs(r).astype("float32")

def dispersion_simultaneo(pf, df, sds, tablas, wald):
    shocks = SH_ABS
    print(f"\n  DISPERSION: {', '.join(shocks)}")
    d = muestra_comun(df, shocks)
    G = int(d[CL1].nunique())
    filas = []
    d["_absres"] = _residuo_abs(pf, d, " + ".join(zx(x) for x in shocks))
    rhs = " + ".join(f"{AIIE}:{x} + {zx(x)}" for x in shocks)
    t1, t2, m, V1, V2 = _fit(pf, f"_absres ~ {rhs} | {FE}", d)
    claves = [buscar_coef(t1, [AIIE, x]) for x in shocks]
    F, p = _wald_cero(m, V1, claves, G)
    wald.append({"ejercicio": "dispersion", "version": "valor absoluto",
                 "gl": f"F({len(shocks)},{G-1})", "F_1via": F, "p_1via": p})
    if V2 is not None:
        F2, p2 = _wald_cero(m, V2, claves, G)
        wald[-1].update({"F_2vias": F2, "p_2vias": p2})
    n = int(getattr(m, "_N", len(d)))
    del m; gc.collect()
    tablas["disp_conj"] = t1.reset_index()
    filas += _filas(t1, t2, shocks, "dispersion", "valor absoluto", "conjunta", n, sds, G)
    for x in shocks:
        d["_absres"] = _residuo_abs(pf, d, zx(x))
        s1, s2, ms, _, _ = _fit(pf, f"_absres ~ {AIIE}:{x} + {zx(x)} | {FE}", d)
        filas += _filas(s1, s2, [x], "dispersion", "valor absoluto", "separada",
                        int(getattr(ms, "_N", len(d))), sds, G)
        del ms; gc.collect()
    for x in shocks:
        a = [f for f in filas if f["shock"] == x]
        sep = [f for f in a if f["estimacion"] == "separada"][-1]
        con = [f for f in a if f["estimacion"] == "conjunta"][-1]
        print(f"    {x:9s} separada {sep['efecto_pb']:+.3f} pb (p {sep['p_1via']:.3f})"
              f"  | conjunta {con['efecto_pb']:+.3f} pb (p {con['p_1via']:.3f})")
    print(f"    Wald tres beta = 0 (1 via): F = {F:.3f}, p = {p:.3f}")
    return filas

def main():
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
    CL2 = f"{CL1} + date_i"
    PANEL_CSV = valor("--panel", PANEL_CSV)
    if not os.path.exists(PANEL_CSV):
        sys.exit(f"[error] no existe {PANEL_CSV}.")
    OUTDIR = os.path.join(OUTDIR_BASE, "simultaneo" + ("_eis" if modo == "eis" else ""))
    os.makedirs(OUTDIR, exist_ok=True)
    hp = huella(PANEL_CSV)
    print(f"MODO: {modo.upper()} | exposicion = {AIIE} | cluster = {CL1} | salida = ./{OUTDIR}/")
    print(f"PANEL: {hp}")
    with open(os.path.join(OUTDIR, "panel_usado.txt"), "a") as f:
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M}  "
                f"{' '.join(args) if args else '(todo)'}  ->  {hp}\n")

    df = cargar()
    resolver_tope_fomc(df)

    corr = correlaciones(df)
    corr.to_csv(f"{OUTDIR}/CORRELACIONES_SHOCKS.csv", index=False)
    hojas = {"CORRELACIONES_SHOCKS": corr}
    if flag("--solo-correlaciones"):
        print(f"\n-> {OUTDIR}/CORRELACIONES_SHOCKS.csv")
        return

    import pyfixest as pf
    sds = {x: sd_shock(df, x) for x in TODOS}
    tablas, wald = {}, []
    print("\n" + "=" * 72 + "\nSHOCKS EN SIMULTANEO\n" + "=" * 72)
    print(f"  muestra comun hasta {FOMC_TOPE.date()} (ultima reunion del FOMC)")
    base = pd.DataFrame(base_simultaneo(pf, df, sds, tablas, wald))
    base.to_csv(f"{OUTDIR}/SIMULTANEO_BASE.csv", index=False)
    hojas["SIMULTANEO_BASE"] = base
    pd.DataFrame(wald).to_csv(f"{OUTDIR}/SIMULTANEO_WALD.csv", index=False)

    if not flag("--sin-dispersion"):
        disp = pd.DataFrame(dispersion_simultaneo(pf, df, sds, tablas, wald))
        disp.to_csv(f"{OUTDIR}/SIMULTANEO_DISPERSION.csv", index=False)
        hojas["SIMULTANEO_DISPERSION"] = disp
        pd.DataFrame(wald).to_csv(f"{OUTDIR}/SIMULTANEO_WALD.csv", index=False)
    hojas["SIMULTANEO_WALD"] = pd.DataFrame(wald)

    os.makedirs(f"{OUTDIR}/tablas", exist_ok=True)
    for k, t in tablas.items():
        t.to_csv(f"{OUTDIR}/tablas/{k}.csv", index=False)
    try:
        with pd.ExcelWriter(f"{OUTDIR}/resultados.xlsx", engine="openpyxl") as w:
            for k, v in {**hojas, **tablas}.items():
                v.to_excel(w, sheet_name=k[:31], index=False)
        print(f"\n-> {OUTDIR}/resultados.xlsx")
    except Exception as e:
        print(f"(no pude escribir el Excel: {e}; quedan los CSV)")
    print(f"-> todo en ./{OUTDIR}/")
    print("Data from Sharadar. No redistribuir el crudo.")

if __name__ == "__main__":
    main()
