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

def diagnostico(df):
    filas = []
    print("\n" + "=" * 72 + "\nDIAGNOSTICO DE COBERTURA\n" + "=" * 72)
    for x in TODOS:
        con_dato = df[x].notna() & (df[x] != 0)
        nz = df.loc[con_dato, ["date", x, "Post"]].drop_duplicates("date")
        if nz.empty:
            print(f"  {x:9s} sin ningun dia con dato: se saltea")
            continue
        post = nz[nz["Post"] == 1]
        filas.append({
            "shock": x,
            "dias_con_dato": len(nz),
            "dias_pre": len(nz) - len(post),
            "dias_post": len(post),
            "desde": nz["date"].min().date(),
            "hasta": nz["date"].max().date(),
            "sd": float(nz[x].std()),
        })
        print(f"  {x:9s} dias={len(nz):5d}  pre={len(nz)-len(post):5d}  "
              f"post={len(post):4d}  {nz['date'].min().date()} -> "
              f"{nz['date'].max().date()}  sd={nz[x].std():.5g}")

    cob = pd.DataFrame(filas, columns=["shock", "dias_con_dato", "dias_pre",
                                       "dias_post", "desde", "hasta", "sd"])
    fp = df.loc[df["FOMC"].notna() & (df["FOMC"] != 0) & (df["Post"] == 1),
                "date"].drop_duplicates()
    if len(fp):
        print(f"\n  fechas FOMC posteriores al {POST_DESDE} ({len(fp)}), solo informativo:")
        print("   ", [d.date().isoformat() for d in sorted(fp)])
    print(f"\n  {AIIE}: valores distintos =", df[AIIE].nunique(),
          "| firmas =", df["permaticker"].nunique(),
          f"| clusters ({CL1}) =", df[CL1].nunique())
    f1 = df.drop_duplicates("permaticker")
    v = f1.groupby("sector", observed=True)[AIIE].agg(firmas="size", sd="std")
    v["sd"] = v["sd"].fillna(0)
    v["peso_%"] = 100 * v["firmas"] * v["sd"] ** 2 / (v["firmas"] * v["sd"] ** 2).sum()
    print("\n  variacion de la exposicion DENTRO de cada sector "
          "(peso aproximado de cada sector en beta):")
    print(v.sort_values("peso_%", ascending=False).round(3).to_string())
    return cob

def correlaciones(df):
    cols = [c for c in Z + Z_FUERA if c in df.columns]
    c = df[cols].corr()
    print("\n" + "=" * 72 + "\nCORRELACIONES DEL VECTOR Z  (incluye las excluidas)\n"
          + "=" * 72)
    print(c.round(3).to_string())
    fuera = [z for z in Z_FUERA if z in c.columns]
    if fuera:
        print("\n  correlacion de cada variable EXCLUIDA con las que quedan en Z:")
        for z in fuera:
            col = c.loc[[x for x in Z if x in c.index], z].abs()
            if not col.notna().any():
                print(f"    {z:8s} sin correlacion calculable (sin datos o constante)")
                continue
            peor = col.idxmax()
            print(f"    {z:8s} max |rho| = {abs(c.loc[peor, z]):.3f}  (con {peor})")
    return c.reset_index().rename(columns={"index": "variable"})

def correlaciones_shocks(df):
    cols = [c for c in SH_LVL if c in df.columns]
    if len(cols) < 2:
        print("\n(menos de dos series de shocks en el panel: se saltea el cuadro)")
        return pd.DataFrame(columns=["par", "muestra", "N", "pearson", "spearman"])

    s = df.drop_duplicates("date")[["date"] + cols].sort_values("date")

    enmascarados = {}
    for x in cols:
        if x in SH_EVENTO:
            ceros = int((s[x] == 0).sum())
            s[x] = s[x].where(s[x] != 0)
            enmascarados[x] = ceros

    print("\n" + "=" * 72 + "\nCORRELACIONES ENTRE LAS SERIES DE SHOCKS\n" + "=" * 72)
    print(f"  nivel = fecha ({len(s):,} fechas unicas en el panel)")
    for x, n in enmascarados.items():
        print(f"  {x}: {n:,} dias sin reunion tratados como faltantes "
              f"(quedan {int(s[x].notna().sum()):,})")

    filas = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            d = s[[a, b]].dropna()
            if len(d) < 3 or d[a].nunique() < 2 or d[b].nunique() < 2:
                filas.append({"par": f"{a}-{b}", "muestra": _etiqueta(a, b),
                              "N": len(d), "pearson": np.nan, "spearman": np.nan})
                continue
            filas.append({
                "par": f"{a}-{b}",
                "muestra": _etiqueta(a, b),
                "N": len(d),
                "pearson": float(d[a].corr(d[b])),
                "spearman": float(d[a].corr(d[b], method="spearman")),
            })

    out = pd.DataFrame(filas)
    print()
    print(out.assign(pearson=out["pearson"].round(4),
                     spearman=out["spearman"].round(4)).to_string(index=False))

    val = out.dropna(subset=["pearson"])
    if len(val):
        peor = val.loc[val["pearson"].abs().idxmax()]
        print(f"\n  par mas correlacionado: {peor['par']}  "
              f"pearson = {peor['pearson']:.4f}  (N = {int(peor['N']):,})")
        disc = val[(val["pearson"] - val["spearman"]).abs() > 0.10]
        if len(disc):
            print("  OJO: Pearson y Spearman difieren en mas de 0,10 en "
                  + ", ".join(disc["par"]) + ".")
            print("       La correlacion lineal esta influida por dias extremos; "
                  "conviene reportar las dos.")
    return out

def _etiqueta(a, b):
    ev = [x for x in (a, b) if x in SH_EVENTO]
    if not ev:
        return "dias con dato en ambas series"
    return "solo dias con " + " y ".join(ev)

def zx(x):
    return " + ".join(f"{z}:{x}" for z in Z)

def f_base(x, dep=None):
    return f"{dep or Y} ~ {AIIE}:{x} + {zx(x)} | {FE}"

def buscar_coef(t, partes):
    for i in t.index:
        if sorted(str(i).split(":")) == sorted(partes):
            return i
    return None

def estimar_doble(pf, formula, datos, partes_list, etiquetas, bloque, shock, sd_shock):
    m1 = pf.feols(formula, data=datos, vcov={"CRV1": CL1})
    t1 = m1.tidy()
    n_est = int(getattr(m1, "_N", len(datos)))
    try:
        m2 = pf.feols(formula, data=datos, vcov={"CRV1": CL2})
        t2 = m2.tidy()
    except Exception as e:
        print(f"    (cluster de dos vias fallo: {e})")
        t2 = None

    filas = []
    for partes, etq in zip(partes_list, etiquetas):
        n = buscar_coef(t1, partes)
        if n is None:
            continue
        r1 = t1.loc[n]
        fila = {"bloque": bloque, "shock": shock, "parametro": etq,
                "coeficiente": n, "N": n_est, "N_filas_entrada": int(len(datos)),
                "estimate": r1["Estimate"],
                "se_1via": r1["Std. Error"], "p_1via": r1["Pr(>|t|)"]}
        if t2 is not None and n in t2.index:
            fila["se_2vias"] = t2.loc[n, "Std. Error"]
            fila["p_2vias"] = t2.loc[n, "Pr(>|t|)"]
        fila["efecto_1sd"] = r1["Estimate"] * sd_shock if sd_shock else np.nan
        filas.append(fila)

    del m1
    if t2 is not None:
        del m2
    gc.collect()
    return filas, t1.reset_index()

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
    AIIE, CL1, OUTDIR = EXPOSICION[modo]
    MUESTRA_COL = CL1
    PANEL_CSV = valor("--panel", PANEL_CSV)
    if not os.path.exists(PANEL_CSV):
        sys.exit(f"[error] no existe {PANEL_CSV}.\n"
                 f"        Las tres corridas tienen que salir del mismo panel: "
                 f"corre antes\n"
                 f"        python agregar_eis_al_panel.py --eis "
                 f"empresas_unicas_EIS_firma_detalle.csv")
    cl_pedido = valor("--cluster", "industria").lower()
    if cl_pedido not in ("industria", "firma"):
        sys.exit(f"[error] --cluster {cl_pedido}: valores validos son 'industria' "
                 f"(el default) y 'firma'.")
    if cl_pedido == "firma":
        if modo == "aiie":
            sys.exit("[error] --cluster firma no va con el AIIE: ese indice es constante "
                     "dentro de cada NAICS-4, asi que agrupar por firma da errores "
                     "estandar espuriamente bajos (problema de Moulton).\n"
                     "        Es una robustez de la corrida con --eis, que es la que "
                     "tiene exposicion por firma.")
        CL1 = "permaticker"
        OUTDIR = f"{OUTDIR}_cluster_firma"
    head = list(pd.read_csv(PANEL_CSV, nrows=0).columns)
    if AIIE not in head:
        sys.exit(f"[error] {PANEL_CSV} no tiene la columna {AIIE}.\n"
                 f"        Corre primero: python agregar_eis_al_panel.py")
    if MUESTRA_COL not in head:
        sys.exit(f"[error] {PANEL_CSV} no tiene la columna {MUESTRA_COL}, que es la "
                 f"industria con la que se define la muestra y se agrupan los "
                 f"errores estandar en este modo.\n"
                 f"        Corre antes agregar_eis_al_panel.py, que la agrega "
                 f"junto con el EIS.")
    if flag("--jorda"):
        pedido = [s.strip() for s in valor("--jorda", "").split(",") if s.strip()]
        malos = [s for s in pedido if s not in TODOS]
        if malos:
            sys.exit(f"[error] --jorda {','.join(malos)}: no son shocks del panel.\n"
                     f"        Validos: {', '.join(TODOS)}")
    extra, etiqueta = [], ""
    if flag("--muestra-comun"):
        if modo == "eis":
            print("[aviso] --muestra-comun no hace nada en modo EIS: la muestra ya "
                  "es la de las firmas con EIS. El flag es para la corrida con AIIE.")
        else:
            extra, etiqueta = ["EIS", "naics3"], "--muestra-comun"
            OUTDIR = f"{OUTDIR}_muestra_comun"
    if flag("--muestra-vieja") or flag("--muestra-con-beta"):
        extra = list(dict.fromkeys(extra + Z_FUERA))
        etiqueta = (etiqueta + " --muestra-vieja").strip()
        OUTDIR = f"{OUTDIR}_muestra_vieja"

    CL2 = f"{CL1} + date_i"
    print(f"MODO: {modo.upper()}  |  exposicion = {AIIE}  |  muestra = firmas con "
          f"{MUESTRA_COL}  |  cluster = {CL1} (y {CL1} + fecha)  |  "
          f"salida = ./{OUTDIR}/")
    hp = huella(PANEL_CSV)
    print(f"PANEL: {hp}")

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "panel_usado.txt"), "a") as f:
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M}  "
                f"{' '.join(args) if args else '(base)'}  ->  {hp}\n")
    df = cargar(muestra_extra=extra, etiqueta=etiqueta)
    resolver_tope_fomc(df)

    cob = diagnostico(df)
    cob.to_csv(f"{OUTDIR}/COBERTURA.csv", index=False)
    sds = dict(zip(cob["shock"], cob["sd"]))

    corr = correlaciones(df)
    corr.to_csv(f"{OUTDIR}/CORRELACIONES.csv", index=False)

    corr_sh = correlaciones_shocks(df)
    corr_sh.to_csv(f"{OUTDIR}/CORRELACIONES_SHOCKS.csv", index=False)

    if flag("--solo-diagnostico"):
        print(f"\n-> {OUTDIR}/COBERTURA.csv  {OUTDIR}/CORRELACIONES.csv"
              f"  {OUTDIR}/CORRELACIONES_SHOCKS.csv")
        return

    tablas, resumen = {}, []

    print("\n" + "=" * 72 +
          "\nBASE  r ~ beta (AIIE x shock) + gamma' (Z x shock)   (se_1via vs se_2vias)\n"
          + "=" * 72)
    for x in TODOS:
        d = muestra(df, x)
        nota = f"  [muestra <= {FOMC_TOPE.date()}]" if x.startswith("FOMC") else ""
        filas, t = estimar_doble(pf, f_base(x), d, [[AIIE, x]], ["beta"],
                                 "base", x, sds.get(x))
        tablas[f"base_{x}"] = t
        resumen += filas
        if filas:
            f = filas[0]
            print(f"  {x:9s} beta={f['estimate']: .5g}  "
                  f"p1={f['p_1via']:.4f}  p2={f.get('p_2vias', float('nan')):.4f}"
                  f"  ef_1sd={f['efecto_1sd']: .5g}  N={f['N']:,}{nota}")

    res = pd.DataFrame(resumen)
    for c, n in [("p_1via", "signif_1via"), ("p_2vias", "signif_2vias")]:
        if c in res.columns:
            res[n] = pd.cut(res[c], [0, .01, .05, .10, 1],
                            labels=["***", "**", "*", "ns"], include_lowest=True)
    res.to_csv(f"{OUTDIR}/RESUMEN.csv", index=False)
    print("\n" + "=" * 72 + "\nRESUMEN\n" + "=" * 72)
    cols = [c for c in ["bloque", "shock", "parametro", "estimate", "efecto_1sd",
                        "p_1via", "signif_1via", "p_2vias", "signif_2vias", "N"]
            if c in res.columns]
    print(res[cols].to_string(index=False))

    extras = {"CORRELACIONES": corr, "CORRELACIONES_SHOCKS": corr_sh}

    if flag("--dispersion"):
        print("\n" + "=" * 72 +
              "\nTEST DE DISPERSION  (|residuo| sobre AIIE x |shock|)\n" + "=" * 72)
        print("  La interaccion con |shock| es un estimador de MEDIA. Este es el")
        print("  test literal: si el indice aumenta la DISPERSION, |residuo| sube.\n")
        filas = []
        for x in SH_ABS:
            d = muestra(df, x)
            need = [Y, AIIE, CL1, x, "permaticker", "sector_day"] + Z
            d = d.dropna(subset=need).reset_index(drop=True).copy()
            try:
                m1 = pf.feols(f"{Y} ~ {zx(x)} | {FE}", data=d,
                              vcov={"CRV1": CL1}, fixef_rm="none")
                r = np.asarray(m1.resid()).ravel()
                if len(r) != len(d):
                    print(f"  {x}: residuos {len(r):,} vs filas {len(d):,} -> "
                          f"intento alinear por indice del modelo")
                    idx = getattr(getattr(m1, "_data", None), "index", None)
                    if idx is not None and len(idx) == len(r):
                        d = d.loc[idx].copy()
                    else:
                        print(f"  {x}: no pude alinear -> salteado")
                        del m1; gc.collect(); continue
                d["_absres"] = np.abs(r).astype("float32")
                del m1; gc.collect()

                fl, t = estimar_doble(pf, f_base(x, dep="_absres"), d,
                                      [[AIIE, x]], ["beta_disp"],
                                      "dispersion", x, sds.get(x))
                tablas[f"disp_{x}"] = t
                filas += fl
                if fl:
                    f = fl[0]
                    print(f"  {x:9s} beta_disp={f['estimate']: .5g}  "
                          f"p1={f['p_1via']:.4f}  p2={f.get('p_2vias', float('nan')):.4f}")
            except Exception as e:
                print(f"  {x}: error -> {e}")
            gc.collect()
        if filas:
            dd = pd.DataFrame(filas)
            dd.to_csv(f"{OUTDIR}/DISPERSION.csv", index=False)
            extras["DISPERSION"] = dd

    def correr_jorda():
        pedido = valor("--jorda", "")
        shocks = [s.strip() for s in pedido.split(",") if s.strip()] or TODOS
        print("\n" + "=" * 72 + "\nJORDA\n" + "=" * 72)
        print("  h=0 = retorno del dia del shock (base: precio del dia anterior)")
        print("  se traza beta_h: la respuesta acumulada de la especificacion base\n")

        fechas = np.sort(df["date_i"].unique())
        col_de_fecha = pd.Series(np.arange(len(fechas)), index=fechas)
        T = len(fechas)

        todo = []
        for x in shocks:
            d = muestra(df, x).copy()
            firmas = np.sort(d["permaticker"].unique())
            fila_de_firma = pd.Series(np.arange(len(firmas)), index=firmas)
            ri = fila_de_firma.reindex(d["permaticker"]).to_numpy()
            ci = col_de_fecha.reindex(d["date_i"]).to_numpy()
            P = np.full((len(firmas), T), np.nan, dtype="float32")
            P[ri, ci] = d["closeadj"].to_numpy()
            den = np.full_like(P, np.nan)
            den[:, 1:] = P[:, :-1]
            filas = []
            for h in range(HMAX + 1):
                num = np.full_like(P, np.nan)
                if h == 0:
                    num[:] = P
                else:
                    num[:, :T - h] = P[:, h:]
                d["_fwd"] = (num / den - 1)[ri, ci].astype("float32")
                del num
                m = pf.feols(f_base(x, dep="_fwd"), data=d, vcov={"CRV1": CL1})
                t = m.tidy()
                n = buscar_coef(t, [AIIE, x])
                if n is not None:
                    b, se = t.loc[n, "Estimate"], t.loc[n, "Std. Error"]
                    filas.append({"shock": x, "h": h, "coef": "beta",
                                  "estimate": b, "se": se,
                                  "lo": b - 1.96*se, "hi": b + 1.96*se,
                                  "pvalue": t.loc[n, "Pr(>|t|)"]})
                    print(f"  {x:9s} h={h:2d}  beta={b: .5g}  "
                          f"p={t.loc[n,'Pr(>|t|)']:.4f}")
                del m, t
                d.drop(columns="_fwd", inplace=True)
                gc.collect()

            if not filas:
                continue
            irf = pd.DataFrame(filas)
            todo.append(irf)
            pd.concat(todo, ignore_index=True).to_csv(f"{OUTDIR}/JORDA.csv", index=False)

            try:
                import matplotlib; matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.axhline(0, color="black", lw=.8)
                ax.plot(irf["h"], irf["estimate"], marker="o", color="navy")
                ax.fill_between(irf["h"], irf["lo"], irf["hi"], alpha=.2, color="navy")
                ax.set_xlabel("horizonte h (dias)")
                ax.set_ylabel("respuesta acumulada")
                ax.set_title(f"Jorda -- {x}")
                fig.tight_layout()
                fig.savefig(f"{OUTDIR}/jorda_{x}.png", dpi=130)
                plt.close(fig)
            except Exception as e:
                print(f"    (no pude graficar: {e})")
            gc.collect()

        if todo:
            extras["JORDA"] = pd.concat(todo, ignore_index=True)

    if flag("--jorda"):
        correr_jorda()

    xlsx = f"{OUTDIR}/resultados.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
            res.to_excel(w, sheet_name="RESUMEN", index=False)
            cob.to_excel(w, sheet_name="COBERTURA", index=False)
            for k, v in extras.items():
                v.to_excel(w, sheet_name=k[:31], index=False)
            for nombre, t in tablas.items():
                t.to_excel(w, sheet_name=nombre[:31], index=False)
        print(f"\n-> {xlsx}")
    except Exception as e:
        print(f"\n(no pude escribir el Excel: {e}; quedan los CSV)")

    print(f"-> todo en ./{OUTDIR}/")
    print("Data from Sharadar. No redistribuir el crudo.")

if __name__ == "__main__":
    main()
