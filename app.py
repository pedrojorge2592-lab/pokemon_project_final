import os
import mlflow
import streamlit as st
import pandas as pd
from mlflow.tracking import MlflowClient

LOCAL_CSV = "data/pokemon.csv"
GITHUB_CSV = os.getenv(
    "POKEMON_CSV_URL",
    "https://raw.githubusercontent.com/pedrojorge2592-lab/pokemon_project_final/main/data/pokemon.csv",
)

csv_path = LOCAL_CSV if os.path.exists(LOCAL_CSV) else GITHUB_CSV
pokemon_df = pd.read_csv(csv_path)


# Optional: for single-file artifacts (.pkl)
try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="Pokémon Legendary Predictor")

# --- Pokéball CSS background ---
st.markdown(
    """
    <style>
    /* Apply Pokéball gradient to full app */
    .stApp {
      background: linear-gradient(
        to bottom,
        #ff1a1a 0%,    /* red top */
        #ff1a1a 48.5%,
        #000000 48.5%, /* black stripe */
        #000000 51.5%,
        #ffffff 51.5%, /* white bottom */
        #ffffff 100%
      ) !important;
    }

    /* Let the gradient show through */
    [data-testid="stAppViewContainer"] .main { background: transparent; }
    .block-container { background: transparent; }
    [data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Config from env ---

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
EXPERIMENT_NAME = os.getenv("MODEL_EXPERIMENT", "pokemon_legendary")
MODEL_URI = (os.getenv("MODEL_URI", "") or "").strip()

# Only set tracking URI if it's explicitly provided and not disabled
if TRACKING_URI and TRACKING_URI.lower() not in {"disabled", "none"}:
    mlflow.set_tracking_uri(TRACKING_URI)


# ---------- SMART MODEL LOADER (drop-in replacement) ----------
def _load_any_model(uri: str):
    """
    Try MLflow pyfunc first. If that fails (or uri is local),
    handle both single-file artifacts and MLflow model directories
    by loading the inner model.pkl with joblib when needed.
    """
    def _joblib_from_dir(path: str):
        if joblib is None:
            raise RuntimeError("joblib missing; cannot load file artifact")
        # typical layouts inside an MLflow model artifact
        for rel in ("model.pkl", "data/model.pkl"):
            p = os.path.join(path, rel)
            if os.path.exists(p):
                return joblib.load(p)
        # last resort: try pyfunc again on the directory
        return mlflow.pyfunc.load_model(path)

    # Local path or file://
    if uri.startswith("/") or uri.startswith("file://"):
        p = uri.replace("file://", "", 1)
        if os.path.isdir(p):
            return _joblib_from_dir(p)
        else:
            if joblib is None:
                raise RuntimeError("joblib missing; cannot load file artifact")
            return joblib.load(p)

    # MLflow URI (runs:/… or models:/…)
    try:
        return mlflow.pyfunc.load_model(uri)
    except Exception:
        local = mlflow.artifacts.download_artifacts(uri)
        if os.path.isdir(local):
            return _joblib_from_dir(local)
        else:
            if joblib is None:
                raise RuntimeError("joblib missing; cannot load file artifact")
            return joblib.load(local)


def _latest_run_ids(exp_name: str):
    """Return a list of newest-to-oldest finished run IDs for the experiment."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if not exp:
        return []
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=25,
    )
    return [r.info.run_id for r in runs]


def _auto_pick_model_uri():
    """
    Find newest usable artifact among recent runs.
    Prefer the 'model' directory; also try a raw rf_model.pkl if present.
    """
    run_ids = _latest_run_ids(EXPERIMENT_NAME)
    for rid in run_ids:
        candidates = (
            f"runs:/{rid}/model",            # standard logged model directory
            f"runs:/{rid}/model/model.pkl",  # direct file inside the model dir
            f"runs:/{rid}/rf_model.pkl",     # raw pickle if you logged one
        )
        for candidate in candidates:
            try:
                _ = _load_any_model(candidate)  # probe
                return candidate, rid
            except Exception:
                continue
    return None, None
# ---------- /SMART MODEL LOADER ----------

@st.cache_resource(show_spinner=True)
def get_model_and_meta(exp_name: str, model_uri: str | None):
    """Cache the loaded model + metadata across reruns."""
    chosen_uri = (model_uri or "").strip()
    chosen_run = None

    if not chosen_uri or chosen_uri.lower() in {"auto", "latest"}:
        chosen_uri, chosen_run = _auto_pick_model_uri()
        if not chosen_uri:
            raise RuntimeError(
                "Couldn't find a usable model. Train a run in MLflow that logs "
                "either 'model/' (mlflow.sklearn.log_model) or 'rf_model.pkl' as an artifact."
            )
    # If user pasted a specific URI, try to parse the run id (best effort)
    if chosen_uri and chosen_run is None and chosen_uri.startswith("runs:/"):
        try:
            chosen_run = chosen_uri.split("/")[2]
        except Exception:
            chosen_run = None

    model = _load_any_model(chosen_uri)
    return model, chosen_uri, chosen_run

# UI sidebar: show config + refresh
with st.sidebar:
    st.caption("MLflow")
    st.code(f"Tracking: {TRACKING_URI}")
    st.code(f"Experiment: {EXPERIMENT_NAME}")
    st.code(f"MODEL_URI: {MODEL_URI or 'auto (latest)'}")
    if st.button("🔄 Refresh model (pick newest)"):
        # Clear the cached model so next call re-picks
        get_model_and_meta.clear()
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

# Try to load a model
model = None
chosen_uri = None
chosen_run = None
try:
    model, chosen_uri, chosen_run = get_model_and_meta(EXPERIMENT_NAME, MODEL_URI)
    st.success(f"Loaded model from: {chosen_uri}")
    if chosen_run:
        st.caption(f"Run ID: {chosen_run}")
except Exception as e:
    st.error(f"Failed to load a model.\n\n{e}")

st.title("Is this Pokémon Legendary?")

col1, col2, col3 = st.columns(3)
HP = col1.number_input("HP", 1, 255, 60)
Attack = col1.number_input("Attack", 1, 255, 70)
Defense = col1.number_input("Defense", 1, 255, 65)
SpA = col2.number_input("Sp. Atk", 1, 255, 70)
SpD = col2.number_input("Sp. Def", 1, 255, 70)
Speed = col2.number_input("Speed", 1, 255, 70)
Total = col3.number_input("Total", 6, 1125, HP + Attack + Defense + SpA + SpD + Speed)
Type1 = col3.text_input("Type 1", "Water")
Type2 = col3.text_input("Type 2 (optional)", "") or None

if st.button("Predict") and model is not None:
    X = pd.DataFrame([{
        "HP": HP, "Attack": Attack, "Defense": Defense,
        "Sp. Atk": SpA, "Sp. Def": SpD, "Speed": Speed,
        "Total": Total, "Type 1": Type1, "Type 2": Type2
    }])

    # Prefer predict_proba, fallback to predict
    prob = None
    try:
        proba = model.predict_proba(X)
        prob = float(proba[0][1]) if getattr(proba, "shape", (0, 0))[1] >= 2 else float(proba[0])
    except Exception:
        y = model.predict(X)
        val = float(y[0]) if hasattr(y, "__len__") else float(y)
        prob = val if 0.0 <= val <= 1.0 else (1.0 if val >= 0.5 else 0.0)

    st.metric("Legendary probability", f"{prob:.2%}")
    st.write("Prediction:", "⭐ Legendary" if prob >= 0.5 else "Not Legendary")

# --- Professor Oak header with image ---
# --- Ask Professor Oak (header + Q&A) ---
import re

with st.container():
    col_img, col_txt = st.columns([1, 8])
    with col_img:
        try:
            st.image("oak.png", width=72)
        except Exception:
            pass
    with col_txt:
        st.subheader("💬 Ask Professor Oak")

# Load dataset used for Q&A (separate from your model inputs)
pokemon_df = pd.read_csv("data/pokemon.csv")

# --- Helpers ---------------------------------------------------------
STAT_ALIASES = {
    "Total":   ["total", "overall", "bst", "base total", "strength", "strongest"],
    "HP":      ["hp", "hit points", "health"],
    "Attack":  ["attack", "atk", "physical attack"],
    "Defense": ["defense", "def", "physical defense"],
    "Sp. Atk": ["sp. atk", "sp atk", "special attack", "spa", "spec atk"],
    "Sp. Def": ["sp. def", "sp def", "special defense", "spd", "spec def"],
    "Speed":   ["speed", "spe", "fastest", "quickest"],
}
ALIAS2CANON = {alias: canon for canon, aliases in STAT_ALIASES.items() for alias in aliases}

def _canonical_stat_from_text(text: str) -> str | None:
    t = text.lower()
    if any(w in t for w in ["fast", "fastest", "speed"]): return "Speed"
    if any(w in t for w in ["strong", "overall", "bst", "total"]): return "Total"
    for alias, canon in ALIAS2CANON.items():
        if alias in t:
            return canon
    if any(w in t for w in ["highest", "strongest", "best"]): return "Total"
    if any(w in t for w in ["lowest", "weakest", "worst"]): return "Total"
    return None

def _all_types(df: pd.DataFrame) -> set[str]:
    t1 = df["Type 1"].dropna().astype(str).str.strip()
    t2 = df["Type 2"].dropna().astype(str).str.strip()
    return set(pd.concat([t1, t2]).str.lower().unique())

def _detect_type(text: str, typeset: set[str]) -> str | None:
    # also supports 'fire-type' etc.
    words = re.findall(r"[a-zA-Z]+", text.lower())
    for w in words:
        if w in typeset:
            return w
    return None

def _filter_by_type(df: pd.DataFrame, type_name: str | None) -> pd.DataFrame:
    if not type_name:
        return df
    t = type_name.lower()
    return df[(df["Type 1"].str.lower() == t) | (df["Type 2"].fillna("").str.lower() == t)]

def _best_match_row(df: pd.DataFrame, name: str) -> pd.Series | None:
    n = name.strip().lower()
    if not n:
        return None
    exact = df[df["Name"].str.lower() == n]
    if len(exact): return exact.iloc[0]
    contains = df[df["Name"].str.lower().str.contains(re.escape(n))]
    if len(contains): return contains.iloc[0]
    return None

def format_top_rows(rows: pd.DataFrame, stat: str, n: int = 5) -> str:
    out = []
    for _, r in rows.head(n).iterrows():
        out.append(f"{len(out)+1}. **{r['Name']}** — {stat} = {int(r[stat])}")
    return "\n".join(out) if out else "_No matches._"

typeset = _all_types(pokemon_df)
has_generation = "Generation" in pokemon_df.columns

# --- Compare parsing & table ----------------------------------------
def _parse_compare(prompt: str) -> tuple[str, str] | None:
    """Extract two names from prompts like:
       'Gengar vs Alakazam', 'compare Pikachu with Raichu',
       'differences between Mew and Mewtwo'."""
    p = prompt.strip()
    m = re.search(r"compare\s+(.+?)\s+(?:vs|with|and)\s+(.+)", p, flags=re.I)
    if m: return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"(.+?)\s+vs\.?\s+(.+)", p, flags=re.I)
    if m: return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"differences?\s+(?:between|btw)\s+(.+?)\s+(?:and|&)\s+(.+)", p, flags=re.I)
    if m: return m.group(1).strip(), m.group(2).strip()
    return None

def _compare_table(ra: pd.Series, rb: pd.Series, has_generation_col: bool) -> str:
    """Return a markdown table comparing two rows, including deltas."""
    def _types(r):
        t2 = str(r.get("Type 2", "") or "").strip()
        return f"{r['Type 1']}" + (f"/{t2}" if t2 else "")

    def _legend(r):  # pretty legend flag
        return "⭐" if bool(r.get("Legendary", False)) else "—"

    def _gen(r):
        if has_generation_col and "Generation" in r:
            try:
                return int(r["Generation"])
            except Exception:
                pass
        return "—"

    stats = ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    # Header
    lines = [
        f"| **Stat** | **{ra['Name']}** | **{rb['Name']}** | **Δ (B−A)** |",
        "|:--|--:|--:|--:|",
        f"| Type | {_types(ra)} | {_types(rb)} | |",
        f"| Legendary | {_legend(ra)} | {_legend(rb)} | |",
        f"| Generation | {_gen(ra)} | {_gen(rb)} | |",
    ]

    for s in stats:
        av = int(ra[s]); bv = int(rb[s]); dv = bv - av
        arrow = "↑" if dv > 0 else ("↓" if dv < 0 else "→")
        lines.append(f"| {s} | {av} | {bv} | {dv:+d} {arrow} |")

    # Quick summary
    winner_total = ra["Name"] if int(ra["Total"]) > int(rb["Total"]) else (rb["Name"] if int(rb["Total"]) > int(ra["Total"]) else "Tie")
    fastest = ra["Name"] if int(ra["Speed"]) > int(rb["Speed"]) else (rb["Name"] if int(rb["Speed"]) > int(ra["Speed"]) else "Tie")

    summary = (
        "\n\n**Summary**\n"
        f"- Higher **Total**: **{winner_total}**\n"
        f"- Faster (**Speed**): **{fastest}**\n"
    )
    return "\n".join(lines) + summary

# --- Router ----------------------------------------------------------
def answer(prompt: str, df: pd.DataFrame) -> str:
    p = prompt.strip()
    pl = p.lower()

    # Full comparison table
    cmp = _parse_compare(p)
    if cmp:
        a, b = cmp
        ra, rb = _best_match_row(df, a), _best_match_row(df, b)
        if ra is None or rb is None:
            miss = a if ra is None else b
            return f"I couldn't find **{miss}**."
        title = f"**{ra['Name']}** _vs_ **{rb['Name']}**"
        table = _compare_table(ra, rb, has_generation)
        return f"{title}\n\n{table}"

    # Is <name> legendary?
    if "legendary" in pl and ("is " in pl or "?" in pl):
        name = re.sub(r"\bis\b|\?|legendary", "", p, flags=re.IGNORECASE).strip()
        r = _best_match_row(df, name)
        if r is None:
            return f"I couldn't find **{name}**."
        return f"**{r['Name']}** is {'⭐ Legendary' if bool(r['Legendary']) else 'not Legendary'}."

    # List legendaries (optionally top N)
    if ("list" in pl and "legend" in pl) or ("show" in pl and "legend" in pl):
        n = 10
        m = re.search(r"top\s+(\d+)", pl)
        if m: n = int(m.group(1))
        legends = df[df["Legendary"] == True].sort_values("Total", ascending=False)
        if legends.empty:
            return "There are no Legendary Pokémon in the dataset."
        return f"Top {n} legendaries by Total:\n\n" + format_top_rows(legends, "Total", n)

    # Counts
    if "how many" in pl or "count" in pl:
        if "legend" in pl:
            return f"There are **{int((df['Legendary'] == True).sum())}** Legendary Pokémon."
        if any(w in pl for w in ["dual", "two types", "dual-type"]):
            dual = int(df["Type 2"].fillna("").ne("").sum())
            return f"There are **{dual}** dual-type Pokémon."
        t = _detect_type(pl, typeset)
        if t:
            c = int(_filter_by_type(df, t).shape[0])
            return f"There are **{c}** **{t.title()}**-type Pokémon."
        if "pokemon" in pl or "pokémon" in pl:
            return f"The dataset has **{len(df)}** Pokémon."
        if "types" in pl:
            all_t = sorted(t.title() for t in typeset)
            return f"There are **{len(all_t)}** types: " + ", ".join(all_t)

    # Averages
    if "average" in pl or "mean" in pl:
        stat = _canonical_stat_from_text(pl) or "Total"
        t = _detect_type(pl, typeset)
        if t:
            sub = _filter_by_type(df, t)
            return f"Average **{stat}** for **{t.title()}**-type is **{sub[stat].mean():.1f}**."
        if "by type" in pl:
            means = (df.groupby("Type 1")[stat].mean().sort_values(ascending=False).head(10))
            lines = [f"Top types by average **{stat}**:"]
            for i, (typ, val) in enumerate(means.items(), 1):
                lines.append(f"{i}. **{typ}** — {val:.1f}")
            return "\n".join(lines)
        return f"Average **{stat}** across all Pokémon is **{df[stat].mean():.1f}**."

    # Top/Bottom N by stat (optional type & legendary filter)
    if any(w in pl for w in ["top", "best", "strongest", "fastest", "highest", "max", "bottom", "lowest", "weakest", "slowest", "min"]):
        stat = _canonical_stat_from_text(pl) or "Total"
        m = re.search(r"\b(\d+)\b", pl); n = int(m.group(1)) if m else 5
        t = _detect_type(pl, typeset)
        sub = _filter_by_type(df, t)
        if "non-legendary" in pl:
            sub = sub[sub["Legendary"] == False]
        elif "legendary" in pl and "non" not in pl:
            sub = sub[sub["Legendary"] == True]
        desc = not any(w in pl for w in ["bottom", "lowest", "weakest", "slowest", "min"])
        sub = sub.sort_values(stat, ascending=not desc)
        title = f"Top {n}" if desc else f"Bottom {n}"
        scope = f" among **{t.title()}**-type" if t else ""
        return f"{title} by **{stat}**{scope}:\n\n" + format_top_rows(sub, stat, n)

    # Single best/worst by stat (optional type)
    if any(w in pl for w in ["fastest", "slowest", "strongest", "weakest", "highest", "lowest", "max", "min"]):
        stat = _canonical_stat_from_text(pl) or "Total"
        t = _detect_type(pl, typeset)
        sub = _filter_by_type(df, t)
        desc = not any(w in pl for w in ["lowest", "weakest", "min", "slowest"])
        row = sub.sort_values(stat, ascending=not desc).iloc[0]
        adjective = "highest" if desc else "lowest"
        scope = f" among **{t.title()}**-type" if t else ""
        return f"The {adjective} **{stat}**{scope} is **{row['Name']}** with {stat} = **{int(row[stat])}**."

    # “Stats of <name>”
    if any(w in pl for w in ["stats of", "stat of", "tell me about", "show me", "details of"]):
        name = re.sub(r"stats? of|tell me about|show me|details of", "", pl).strip()
        r = _best_match_row(df, name)
        if r is None:
            return "I couldn't find that Pokémon."
        stats = ["Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]
        lines = [f"**{r['Name']}** — {'⭐ Legendary' if bool(r['Legendary']) else 'Not Legendary'}",
                 f"Type: **{r['Type 1']}**" + (f"/**{r['Type 2']}**" if pd.notna(r['Type 2']) and str(r['Type 2']).strip() else "")]
        for s in stats: lines.append(f"- {s}: {int(r[s])}")
        return "\n".join(lines)

    # Generation analytics (only if column exists)
    if has_generation:
        # "Which generation has the most legendaries?"
        if "generation" in pl and "legend" in pl and any(w in pl for w in ["most", "max"]):
            winners = (df[df["Legendary"] == True]
                       .groupby("Generation")
                       .size()
                       .sort_values(ascending=False))
            if winners.empty:
                return "No legendaries found in any generation."
            top_gen, top_cnt = int(winners.index[0]), int(winners.iloc[0])
            return f"Generation **{top_gen}** has the most legendaries (**{top_cnt}**)."

        # "average total by generation" / "mean speed by generation"
        if "generation" in pl and ("average" in pl or "mean" in pl):
            stat = _canonical_stat_from_text(pl) or "Total"
            means = df.groupby("Generation")[stat].mean().sort_values(ascending=False)
            lines = [f"Generations by average **{stat}**:"]
            for gen, v in means.items():
                lines.append(f"- Gen **{int(gen)}** — {v:.1f}")
            return "\n".join(lines)

        # "top 5 in generation 1"
        m = re.search(r"generation\s*(\d)\b", pl)
        if m and any(w in pl for w in ["top", "best", "strongest"]):
            gen = int(m.group(1))
            n = 5
            mN = re.search(r"\b(\d+)\b", pl)
            if mN: n = int(mN.group(1))
            sub = df[df["Generation"] == gen].sort_values("Total", ascending=False)
            return f"Top {n} in Gen **{gen}** by **Total**:\n\n" + format_top_rows(sub, "Total", n)

    # --- Flexible Name search ---
    if any(kw in pl for kw in [
        "find", "search", "contains", "containing",
        "starts with", "begins with",
        "ends with", "ending with",
        "name has", "name contains"
    ]):
        # Collect terms (prefer quoted, but fallback to single token after keywords)
        terms = [t1 or t2 for (t1, t2) in re.findall(r"'([^']+)'|\"([^\"]+)\"", p)]
        if not terms:
            m = re.search(r"(?:contains|containing|with|has)\s+([a-zA-Z0-9.\-']+)", pl) \
                or re.search(r"(?:starts with|begins with)\s+([a-zA-Z0-9.\-']+)", pl) \
                or re.search(r"(?:ends with|ending with)\s+([a-zA-Z0-9.\-']+)", pl) \
                or re.search(r"(?:find|search)\s+([a-zA-Z0-9.\-']+)", pl)
            if m: terms = [m.group(1)]

        # Match mode
        mode = "contains"
        if "starts with" in pl or "begins with" in pl: mode = "starts"
        elif "ends with" in pl or "ending with" in pl: mode = "ends"

        if not terms:
            return "Tell me what to search for (e.g., **find containing 'chu'** or **starts with dra**)."

        hits = df.copy()
        names = hits["Name"].astype(str).str.lower()

        # Apply all terms (AND semantics)
        for q in terms:
            ql = q.lower()
            if mode == "starts":
                mask = names.str.startswith(ql)
            elif mode == "ends":
                mask = names.str.endswith(ql)
            else:
                mask = names.str.contains(re.escape(ql))
            hits = hits[mask]
            names = hits["Name"].astype(str).str.lower()

        # Optional filters
        if "legendary" in pl and "non" not in pl:
            hits = hits[hits["Legendary"] == True]
        if "non-legendary" in pl:
            hits = hits[hits["Legendary"] == False]

        t = _detect_type(pl, typeset)
        if t:
            hits = _filter_by_type(hits, t)

        # Optional sort by stat
        stat = None
        for key in ["total","hp","attack","defense","sp. atk","sp. def","speed","overall","strongest","fastest","bst"]:
            if key in pl:
                stat = _canonical_stat_from_text(pl)
                break
        if stat:
            hits = hits.sort_values(stat, ascending=False)

        # Optional limit: "top N" / "first N"
        n = 20
        mN = re.search(r"\btop\s*(\d+)\b|\bfirst\s*(\d+)\b", pl)
        if mN:
            n = int(next(g for g in mN.groups() if g))

        if hits.empty:
            return "No matches."

        # Compact output
        out = []
        for _, r in hits.head(n).iterrows():
            t2 = r.get("Type 2", "")
            types = f"{r['Type 1']}" + (f"/{t2}" if pd.notna(t2) and str(t2).strip() else "")
            star = "⭐" if bool(r["Legendary"]) else "—"
            out.append(f"- **{r['Name']}** ({types}) — Total {int(r['Total'])}, Spe {int(r['Speed'])} {star}")
        return "\n".join(out)

    # --- Plain name fallback: just type a Pokémon name to get its stats ---
    q = re.sub(r"[\s\?\!\.]+$", "", pl)  # trim trailing punctuation
    q = re.sub(r"^(who is|what is|tell me about|info(?:rmation)? on|show me|stats?|stat)\s+", "", q, flags=re.I).strip()
    if len(q) >= 2:
        r = _best_match_row(df, q)
        if r is not None:
            stats = ["Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]
            types = f"{r['Type 1']}" + (f"/{r['Type 2']}" if pd.notna(r['Type 2']) and str(r['Type 2']).strip() else "")
            lines = [
                f"**{r['Name']}** — {'⭐ Legendary' if bool(r['Legendary']) else 'Not Legendary'}",
                f"Type: **{types}**",
            ]
            for s in stats:
                lines.append(f"- {s}: {int(r[s])}")
            return "\n".join(lines)

    # Default fallback
    return "I don’t know that one 🤔"

# --- Chat state + UI ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask me about the Pokémon dataset..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # ✅ Use the router
    try:
        response = answer(prompt, pokemon_df)
    except Exception as e:
        response = f"Oops — couldn’t read the dataset: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
