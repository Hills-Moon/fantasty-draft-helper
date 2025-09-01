# app.py — needs-aware, turn-aware draft helper with VORP, risk, tiers & LLM (no uploads)
import os, json, math, datetime as dt
from typing import Optional

import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv

from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # lets the OpenAI client find it
  
# ---------------------- Page & global styles ----------------------
st.set_page_config(page_title="Fantasy Draft Helper — simple & turn-aware",
                   page_icon="🟣", layout="wide")

PURPLE = "#7c3aed"
st.markdown(
    f'<h1 style="margin:0 0 .25rem 0; color:{PURPLE};">'
    'Fantasy Draft Helper — simple & turn-aware'
    '</h1>',
    unsafe_allow_html=True,
)
st.markdown(f"""
<style>
div.stButton > button {{
  border: 2px solid {PURPLE} !important;
  color: {PURPLE} !important;
  background: transparent !important;
  font-weight: 600;
}}
.badges {{ display:flex; flex-wrap:wrap; gap:.35rem; }}
.badge {{
  display:inline-block; padding:.15rem .5rem; border-radius:999px;
  background:#f5f3ff; color:{PURPLE}; border:1px solid {PURPLE}; font-weight:600;
  font-size:.85rem;
}}
.callout {{
  background:#f5f3ff; border-left:4px solid {PURPLE};
  padding:.75rem 1rem; margin:.75rem 0 1rem 0; color:#4c1d95;
}}
.callout b {{ color:#4c1d95; }}
h2, h3 {{ color:{PURPLE}; }}
</style>
""", unsafe_allow_html=True)

# ---------------------- Keys & scoring ----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
st.sidebar.write("ChatGPT:", "✅" if OPENAI_API_KEY else "❌ not connected")

# Full-PPR scoring (for last-season PPG proxy)
SCORING = dict(pass_yd=0.04, pass_td=4, int=-2,
               rush_yd=0.1, rush_td=6, rec_yd=0.1, rec=1, rec_td=6,
               fum_lost=-2)

# ---------------------- What this tool considers ----------------------
with st.expander("What this tool considers (tap to expand)", expanded=True):
    st.markdown(
        "- **ADP (Average Draft Position)** from Fantasy Football Calculator free API "
        "(https://fantasyfootballcalculator.com/api/v1/adp/*scoring*) — used as the market anchor.\n"
        "- **PPG (Points Per Game)**: always uses **last season full-PPR per-game** via `nfl_data_py`.\n"
        "- **VORP (Value Over Replacement Player)**: per position from your league’s starter counts.\n"
        "- **Risk**: probability a player is gone before your next pick (ADP spread model).\n"
        "- **Turn-aware**: accounts for how many picks happen before your next turn so you don’t get boxed out.\n"
        "- **Strategy levers**: late-round gates for QB/TE, small stacking bonus between your QB and WR/TE."
    )

# ---------------------- Data loaders ----------------------
@st.cache_data(show_spinner=True, ttl=60*60)
def load_adp(teams: int = 10, year: int = None, scoring: str = "ppr") -> pd.DataFrame:
    if year is None:
        year = dt.datetime.now().year
    url = f"https://fantasyfootballcalculator.com/api/v1/adp/{scoring}"
    params = {"teams": teams, "year": year, "position": "all"}
    r = requests.get(url, params=params, timeout=20); r.raise_for_status()
    players = r.json().get("players", [])
    if not players:
        return pd.DataFrame(columns=["name","pos","team","adp"])
    df = pd.DataFrame(players).rename(columns={"position":"pos"})
    df = df[["name","pos","team","adp"]].copy()
    df["pos"] = df["pos"].astype(str).str.upper().str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_history_latest(latest_season: int) -> pd.DataFrame:
    """Fallback 'projection' = last season PPR/PG."""
    try:
        from nfl_data_py import import_weekly_data
        wk = import_weekly_data([latest_season])

        def pick(*cands):
            for c in cands:
                if c in wk.columns: return c
            return None
        name_col = pick("player_name","player_display_name","name") or "player_name"
        pos_col  = pick("position","pos","position_group") or "position"
        cols = {
            "rec": pick("receptions","receiving_rec"),
            "rec_yds": pick("receiving_yards","receiving_yds"),
            "rec_tds": pick("receiving_tds","rec_td","receiving_touchdowns"),
            "rush_yds": pick("rushing_yards","rushing_yds"),
            "rush_tds": pick("rushing_tds","rush_td","rushing_touchdowns"),
            "pass_yds": pick("passing_yards","pass_yards","pass_yds"),
            "pass_tds": pick("passing_tds","pass_td","passing_touchdowns"),
            "ints": pick("interceptions","interceptions_thrown","passing_int"),
            "fum": pick("fumbles_lost","fum_lost"),
        }
        for norm, src in cols.items():
            wk[norm] = wk[src].fillna(0) if src else 0
        wk["ppr"] = (
            wk["rec"]*SCORING["rec"] + wk["rec_yds"]*SCORING["rec_yd"] + wk["rec_tds"]*SCORING["rec_td"] +
            wk["rush_yds"]*SCORING["rush_yd"] + wk["rush_tds"]*SCORING["rush_td"] +
            wk["pass_yds"]*SCORING["pass_yd"] + wk["pass_tds"]*SCORING["pass_td"] +
            wk["fum"]*SCORING["fum_lost"] + wk["ints"]*SCORING["int"]
        )
        g = wk.groupby([name_col, pos_col])["ppr"].agg(["sum","count"]).reset_index()
        g["proj_pg"] = (g["sum"] / g["count"].replace(0, pd.NA)).round(2)
        out = g.rename(columns={name_col:"name", pos_col:"pos"})[["name","pos","proj_pg"]]
        out["pos"] = out["pos"].astype(str).str.upper().str.strip()
        return out
    except Exception:
        return pd.DataFrame(columns=["name","pos","proj_pg"])

def _norm_name(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii")
         .str.replace(r"[^a-zA-Z ]", "", regex=True)
         .str.lower().str.strip()
    )

def join_board(adp_df: pd.DataFrame, proj_df: pd.DataFrame) -> pd.DataFrame:
    a = adp_df.copy(); p = proj_df.copy()
    a["_key"] = _norm_name(a["name"]); p["_key"] = _norm_name(p["name"])
    b = a.merge(p[["_key","pos","proj_pg"]], on=["_key","pos"], how="left").drop(columns="_key")
    b["value_score"] = (-b["adp"].fillna(999)) + b["proj_pg"].fillna(0) * 0.5
    return b

# ---------------------- Setup (collapsible) ----------------------
with st.expander("Setup (league, projections, strategy)", expanded=True):
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    TEAMS = c1.selectbox("League size", [8,10,12,14], index=1)
    SLOT  = c2.number_input("Your draft slot", 1, int(TEAMS), min(10,int(TEAMS)), 1)
    ADP_YEAR = c3.selectbox("ADP year (API)", [dt.datetime.now().year, dt.datetime.now().year-1], 0)
    SCORING_MODE = c4.selectbox("ADP scoring", ["ppr","half-ppr","standard"], 0)

    st.caption("Starter targets per position (FLEX counts as RB/WR/TE).")
    n1, n2, n3, n4, n5, n6, n7 = st.columns(7)
    NEEDS_TEMPLATE = {
        "QB":  n1.number_input("QB",  0, 3, 1, 1),
        "RB":  n2.number_input("RB",  0, 5, 2, 1),
        "WR":  n3.number_input("WR",  0, 5, 2, 1),
        "TE":  n4.number_input("TE",  0, 3, 1, 1),
        "FLEX":n5.number_input("FLEX",0, 3, 1, 1),
        "DST": n6.number_input("DST", 0, 2, 1, 1),
        "K":   n7.number_input("K",   0, 2, 1, 1),
    }

    st.caption("FLEX allocation (typical leagues).")
    fa1, fa2, fa3 = st.columns(3)
    FLEX_SHARE = {
        "RB":  fa1.slider("FLEX share → RB", 0.0, 1.0, 0.50, 0.05),
        "WR":  fa2.slider("FLEX share → WR", 0.0, 1.0, 0.50, 0.05),
        "TE":  fa3.slider("FLEX share → TE", 0.0, 1.0, 0.00, 0.05),
    }
    fs = sum(FLEX_SHARE.values()) or 1.0
    for k in FLEX_SHARE: FLEX_SHARE[k] = FLEX_SHARE[k] / fs

    st.caption("Position timing (earliest round you prefer to start selecting).")
    t1, t2, t3, t4 = st.columns(4)
    POS_MIN_ROUND = {
        "QB":  t1.number_input("Earliest QB",  1, 16, 7, 1),
        "TE":  t2.number_input("Earliest TE",  1, 16, 6, 1),
        "DST": t3.number_input("Earliest DST", 1, 16, 12,1),
        "K":   t4.number_input("Earliest K",   1, 16, 14,1),
        "RB":  1, "WR": 1,
    }
    STRONG_LATE_ROUND = st.checkbox("Make late-round QB/TE more strict", value=False)

# ---------------------- Build board (always last-season PPR/pg) ----------------------
adp  = load_adp(teams=int(TEAMS), year=int(ADP_YEAR), scoring=SCORING_MODE)
projections = load_history_latest(latest_season=dt.datetime.now().year - 1)
board = join_board(adp, projections)

# ---------------------- State & draft lists ----------------------
if "drafted" not in st.session_state: st.session_state.drafted = []
if "my_team" not in st.session_state: st.session_state.my_team = []

st.sidebar.header("Draft State")
st.sidebar.multiselect("Players drafted (anyone)", options=board["name"].tolist(), key="drafted")
st.sidebar.multiselect("Players on YOUR roster", options=board["name"].tolist(), key="my_team")

# ---------------------- Needs & badges ----------------------
def compute_needs(board: pd.DataFrame, roster: list, template: dict) -> dict:
    needs = template.copy()
    pos_lookup = board.set_index("name")["pos"].astype(str).str.upper().to_dict()
    for name in roster:
        pos = pos_lookup.get(name, "")
        if not pos: continue
        if pos in {"RB","WR","TE"}:
            if needs.get(pos,0) > 0: needs[pos] -= 1
            elif needs.get("FLEX",0) > 0: needs["FLEX"] -= 1
        elif pos in needs and needs[pos] > 0:
            needs[pos] -= 1
    for k in needs: needs[k] = max(0, int(needs[k]))
    return needs

needs = compute_needs(board, st.session_state.my_team, NEEDS_TEMPLATE)

def needs_badges_html(needs: dict) -> str:
    chips = ''.join(f'<span class="badge">{pos}: {int(cnt)}</span>'
                    for pos, cnt in needs.items() if int(cnt) > 0)
    return f'<div class="badges">{chips}</div>' if chips else "<div class='badges'><span class='badge'>All set ✅</span></div>"

with st.sidebar.container():
    st.markdown("### Remaining needs")
    st.markdown(needs_badges_html(needs), unsafe_allow_html=True)

# ---------------------- Snake turn math ----------------------
def snake_picks_for(slot: int, teams: int = 10, rounds: int = 16):
    out = []
    for r in range(1, rounds+1):
        overall = (r-1)*teams + slot if r % 2 == 1 else r*teams - slot + 1
        out.append((r, overall))
    return out

my_picks = snake_picks_for(int(SLOT), int(TEAMS), rounds=16)
c1, c2 = st.columns([1,3])
current_overall = c1.number_input("Current overall pick", 1, int(TEAMS)*16, int(SLOT), 1)

upcoming = [p for (_, p) in my_picks if p >= int(current_overall)]
if not upcoming:
    st.info("Your team is done drafting."); st.stop()
next_two = upcoming[:2] if len(upcoming) >= 2 else [upcoming[0], upcoming[0]]
after_list = [p for (_, p) in my_picks if p > next_two[-1]]
after_turn_next = after_list[0] if after_list else next_two[-1]
picks_in_between = max(0, after_turn_next - next_two[-1] - 1)
current_round = int(math.ceil(int(current_overall) / int(TEAMS)))
risk_cutoff = after_turn_next - 1

st.markdown(
    f'<div class="callout"><b>Your next picks:</b> <b>{next_two[0]}</b> and <b>{next_two[1]}</b><br>'
    f'<b>Picks before you again after this turn:</b> <b>{picks_in_between}</b></div>',
    unsafe_allow_html=True
)

# ---------------------- Remaining pool ----------------------
unavailable = set(st.session_state.drafted) | set(st.session_state.my_team)
remaining = board[~board["name"].isin(unavailable)].copy()

# ---------------------- VORP & replacement ----------------------
def replacement_index_for_pos(pos: str, teams: int, needs: dict, flex_share: dict) -> int:
    starters = teams * needs.get(pos, 0)
    flex_from_pos = teams * flex_share.get(pos, 0) * needs.get("FLEX", 0)
    return max(1, int(round(starters + flex_from_pos)))

def compute_vorp(df: pd.DataFrame, teams: int, needs_template: dict, flex_share: dict) -> pd.DataFrame:
    d = df.copy(); d["proj_pg"] = d["proj_pg"].fillna(0.0)
    repl = {}
    for pos in ["QB","RB","WR","TE","DST","K"]:
        n = replacement_index_for_pos(pos, teams, needs_template, flex_share)
        pool = d[d["pos"]==pos].sort_values("proj_pg", ascending=False)
        repl[pos] = float(pool["proj_pg"].iloc[n-1]) if len(pool) >= n else 0.0
    d["vorp"] = d.apply(lambda r: r["proj_pg"] - repl.get(str(r["pos"]), 0.0), axis=1)
    return d, repl

board_vorp, repl_map = compute_vorp(board, int(TEAMS), NEEDS_TEMPLATE, {"RB":0.5,"WR":0.5,"TE":0.0})
remaining = board_vorp[~board_vorp["name"].isin(unavailable)].copy()

# ---------------------- Risk model ----------------------
def norm_cdf(x):  # standard normal CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

ADP_SIGMA = 8.0
remaining["risk_prob"] = remaining["adp"].apply(
    lambda a: norm_cdf((risk_cutoff - float(a)) / ADP_SIGMA) if pd.notna(a) else 0.0
).round(3)

# ---------------------- Tiering ----------------------
def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy(); d["tier"] = ""
    for pos in ["RB","WR","TE","QB","DST","K"]:
        sub = d[d["pos"]==pos].sort_values("proj_pg", ascending=False).copy()
        if sub.empty: continue
        q1, q2, q3 = sub["proj_pg"].quantile([0.85, 0.7, 0.5])
        def tier_of(x):
            if x >= q1: return "Tier 1"
            if x >= q2: return "Tier 2"
            if x >= q3: return "Tier 3"
            return "Tier 4+"
        d.loc[sub.index, "tier"] = sub["proj_pg"].apply(tier_of)
    return d

remaining = assign_tiers(remaining)

# ---------------------- Needs-aware ranking (THIS pick) ----------------------
def needs_aware_rank(df: pd.DataFrame, needs: dict, pos_min_round: dict,
                     current_round: int, risk_cutoff: int, strong_late_qb_te: bool) -> pd.DataFrame:
    if df.empty: return df
    d = df.copy()
    boost = {
        "RB": 1.15 if needs.get("RB",0)>0 else 1.0,
        "WR": 1.15 if needs.get("WR",0)>0 else 1.0,
        "TE": 1.10 if needs.get("TE",0)>0 else 1.0,
        "QB": 1.05 if needs.get("QB",0)>0 else 1.0,
        "DST":1.05 if needs.get("DST",0)>0 else 1.0,
        "K":  1.02 if needs.get("K",0)>0 else 1.0,
    }
    d["need_boost"] = d["pos"].map(boost).fillna(1.0)
    d["allowed_now"] = d["pos"].map(lambda p: current_round >= pos_min_round.get(str(p),1))
    d["risk"] = d["adp"] <= risk_cutoff
    penalty = 0.2 if strong_late_qb_te else 0.25
    d["timing_mult"] = 1.0
    d.loc[(~d["allowed_now"]) & (~d["risk"]), "timing_mult"] = penalty
    d["risk_mult"] = d["risk"].map({True: 1.10, False: 1.0})

    # small stacking bonus: your QB with your WR/TE or vice versa
    my_pos = board_vorp.set_index("name")[["pos","team"]].to_dict("index")
    my_qb_teams = { my_pos[n]["team"] for n in st.session_state.my_team
                    if n in my_pos and my_pos[n]["pos"]=="QB" }
    def stack_bonus(row):
        t, p = row.get("team",""), row.get("pos","")
        if not t: return 0.0
        if p=="QB" and any(my_pos[n]["team"]==t and my_pos[n]["pos"] in {"WR","TE"}
                           for n in st.session_state.my_team if n in my_pos):
            return 0.03
        if p in {"WR","TE"} and t in my_qb_teams:
            return 0.03
        return 0.0
    d["stack_bonus"] = d.apply(stack_bonus, axis=1)

    def minmax(s):
        s = s.fillna(0.0); lo, hi = float(s.min()), float(s.max())
        if hi - lo < 1e-9: return pd.Series([0.5]*len(s), index=s.index)
        return (s - lo) / (hi - lo)

    d["adp_score"]  = 1 - minmax(d["adp"])
    d["proj_score"] = minmax(d["proj_pg"])
    d["vorp_score"] = minmax(d["vorp"])
    d["risk_score"] = d["risk_prob"].fillna(0.0)

    d["score"] = (
        0.45*d["adp_score"] + 0.30*d["proj_score"] +
        0.20*d["vorp_score"] + 0.05*d["risk_score"]
    )
    d["score"] = d["score"] * d["need_boost"] * d["risk_mult"] * d["timing_mult"] + d["stack_bonus"]
    return d.sort_values("score", ascending=False)

targets_now = needs_aware_rank(
    remaining, needs, POS_MIN_ROUND, current_round, risk_cutoff, STRONG_LATE_ROUND
)

# ---------------------- LLM suggestions (robust) ----------------------
def llm_suggest(available_df: pd.DataFrame,
                needs: dict,
                pick_overall: int,
                unavailable: set,
                roster: list,
                turn_info: Optional[dict] = None) -> str:
    if not OPENAI_API_KEY:
        return "_No OPENAI_API_KEY set; using algorithm only._"

    avail = available_df[~available_df["name"].isin(unavailable)].copy()
    if avail.empty:
        return "_No candidates available._"

    # Ensure required columns exist
    for col in ["pos", "adp", "ppg", "proj_pg", "team"]:
        if col not in avail.columns:
            avail[col] = pd.NA

    # Fallback score if 'score' is missing
    if "score" not in avail.columns:
        boost = {
            "RB": 1.15 if needs.get("RB", 0) > 0 else 1.0,
            "WR": 1.15 if needs.get("WR", 0) > 0 else 1.0,
            "TE": 1.10 if needs.get("TE", 0) > 0 else 1.0,
            "QB": 1.05 if needs.get("QB", 0) > 0 else 1.0,
            "DST": 1.05 if needs.get("DST", 0) > 0 else 1.0,
            "K":  1.02 if needs.get("K",  0) > 0 else 1.0,
        }
        avail["need_boost"] = avail["pos"].astype(str).map(boost).fillna(1.0)
        perf = avail["proj_pg"].fillna(avail["ppg"]).fillna(0)
        avail["score"] = (-avail["adp"].fillna(999)) * avail["need_boost"] + perf * 0.5

    # Sort candidates robustly
    sort_cols, asc = [], []
    if "score" in avail.columns:   sort_cols.append("score");   asc.append(False)
    if "adp" in avail.columns:     sort_cols.append("adp");     asc.append(True)
    if "proj_pg" in avail.columns: sort_cols.append("proj_pg"); asc.append(False)
    if sort_cols:
        avail = avail.sort_values(sort_cols, ascending=asc, na_position="last")

    candidates = (
        avail[["name", "team", "pos", "adp", "proj_pg"]]
        .head(15)
        .to_dict(orient="records")
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        payload = {
            "pick": int(pick_overall),
            "needs": needs,
            "candidates": candidates,
            "unavailable": sorted(unavailable),
            "roster": roster,
            "turn": (turn_info or {}),
            "hint": "full PPR; prefer WR unless RB/TE clearly better or needs require it; consider scarcity before next turn",
        }
        sys = (
            "You are a sharp fantasy football analyst for a 10-team full-PPR snake draft.\n"
            "Rules:\n"
            "1) Recommend ONLY players in 'candidates'.\n"
            "2) NEVER suggest anyone in 'unavailable'.\n"
            "3) Respect positional needs in 'needs' and scarcity before the next turn.\n"
            "Output EXACTLY three numbered items. For each: "
            "'Name (Team, Pos) — 2–3 sentences on fit, need, ADP risk, and PPR rationale.'"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.35,
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":str(payload)}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"_OpenAI call failed: {e}_"

# ---------------------- Top targets + quick adds ----------------------
st.subheader(f"Top targets for pick #{int(current_overall)}")
st.dataframe(
    targets_now[["name","team","pos","tier","adp","proj_pg","vorp","risk_prob","value_score"]]
        .rename(columns={"proj_pg":"proj/pg","risk_prob":"risk≤next"}),
    use_container_width=True
)

pick = st.selectbox("Quick add (from top targets)", targets_now["name"].head(20))
b1, b2, b3 = st.columns(3)
if b1.button("I drafted this"):
    if pick not in st.session_state.my_team: st.session_state.my_team.append(pick)
    if pick not in st.session_state.drafted: st.session_state.drafted.append(pick)
    st.rerun()
if b2.button("Someone else drafted this"):
    if pick not in st.session_state.drafted: st.session_state.drafted.append(pick)
    st.rerun()
if b3.button("Ask LLM for 3 picks"):
    st.write(
        llm_suggest(
            targets_now, needs, int(current_overall),
            unavailable=unavailable, roster=st.session_state.my_team,
            turn_info={"next_two": next_two, "picks_before_next_turn": picks_in_between},
        )
    )

# ---------------------- Small positional risk readout ----------------------
def positional_risk_note(df: pd.DataFrame) -> str:
    note = []
    for pos in ["RB","WR","TE","QB","DST","K"]:
        if needs.get(pos,0) <= 0: continue
        cnt = int(((df["pos"].astype(str)==pos) & (df["adp"] <= risk_cutoff)).sum())
        if cnt > 0: note.append(f"{pos}: {cnt} likely to go before your next turn")
    return " • ".join(note) if note else "No major positional risk before your next turn."

st.caption("Positional risk before next turn: " + positional_risk_note(remaining))
