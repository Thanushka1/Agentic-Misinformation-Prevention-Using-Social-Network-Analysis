"""Core logic: network, misinformation dataset, spread simulation, strategies, multi-agent system,
risk prediction, spread analysis and statistics."""
import os, math, heapq, random
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx

NETWORK_PATH = "data/twitter_combined.csv"
STORIES_PATH = "data/twitter_users_misinfo.csv"
STRATEGIES = ["No Protection", "Random", "Degree Centrality", "Betweenness Centrality",
              "MAS (Centrality Only)", "Multi-Agent System"]
MAS_NAMES = ("MAS (Centrality Only)", "Multi-Agent System")
MATCH_BOOST = 1.5          # a user is 1.5x more likely to be affected by a claim in the topic they are vulnerable to
FACT_CHECK_EFFECT = 0.8    # a fact-checked user blocks the claim with this probability (1.0 = perfect)


# ---------------------------------------------------------------- data
def load_twitter(path=NETWORK_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put the Twitter CSV in the data/ folder.")
    df = pd.read_csv(path, header=None, usecols=[0, 1], low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce").dropna().astype(int)    # also drops the header row
    G = nx.from_pandas_edgelist(df, 0, 1, create_using=nx.DiGraph()).reverse(copy=True)
    G.remove_edges_from(nx.selfloop_edges(G))      # edge u -> v : news flows from u to its follower v
    return G


def demo_graph(n=3000, seed=1):
    G = nx.DiGraph(nx.scale_free_graph(n, seed=seed))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def load_users(path=STORIES_PATH):
    """Per-user columns from the misinformation dataset (susceptibility, vulnerable topic, ...)."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "susceptibility" not in df.columns:
        return None
    return df.set_index("user_id")


def load_stories(users):
    """Misinformation claims and the real users who post them first (mixed category order)."""
    if users is None:
        return None
    df = users[users["is_misinfo_source"] == True].reset_index().sort_values("post_order")
    agg = dict(category=("category", "first"), tweet_text=("tweet_text", "first"),
               fact_check=("fact_check", "first"), prevention=("prevention", "first"),
               sources=("user_id", list))
    for c in ["emotional_intensity", "harm_level", "source_of_truth"]:      # optional claim-level columns
        if c in df.columns:
            agg[c] = (c, "first")
    out = df.groupby("claim_id").agg(**agg).reset_index()
    out["_r"] = out.groupby("category").cumcount()          # any 5 consecutive trials cover all 5 categories
    return out.sort_values(["_r", "category"]).drop(columns="_r").reset_index(drop=True)


class Network:
    """Graph + centrality rankings (the knowledge the agents use) + per-user susceptibility."""
    def __init__(self, G, betweenness_k=100, source_pool_percent=5, use_stories=True):
        self.G = G
        self.nodes = list(G.nodes())
        self.N = len(self.nodes)
        self.in_degree = dict(G.in_degree())
        self.out_degree = dict(G.out_degree())
        self.betweenness = nx.betweenness_centrality(G, k=min(betweenness_k, self.N), normalized=True, seed=42)
        self.rank_degree = sorted(self.nodes, key=lambda n: self.out_degree[n], reverse=True)
        self.rank_betweenness = sorted(self.nodes, key=lambda n: self.betweenness[n], reverse=True)
        self.source_pool = self.rank_degree[:max(1, int(self.N * source_pool_percent / 100))]
        self.users = load_users() if use_stories else None
        self.stories = load_stories(self.users)
        if self.users is not None:
            self.sus_factor = (0.4 + 1.2 * self.users["susceptibility"]).to_dict()   # average = 1.0
            self.vuln = self.users["vulnerable_category"].to_dict()
            self.src_type = self.users["source_type"].to_dict() if "source_type" in self.users.columns else {}
        else:
            self.sus_factor, self.vuln, self.src_type = {}, {}, {}

    def centrality_table(self):
        return pd.DataFrame({"User ID": self.nodes,
                             "Followers (Out-Degree)": [self.out_degree[n] for n in self.nodes],
                             "Following (In-Degree)": [self.in_degree[n] for n in self.nodes],
                             "Betweenness": [self.betweenness[n] for n in self.nodes]})

    def stats(self):
        return {"Total Users": self.N, "Total Connections": self.G.number_of_edges(),
                "Average Followers per User": round(self.G.number_of_edges() / self.N, 2),
                "Network Density": round(nx.density(self.G), 6)}

    def communities(self):
        comms = nx.community.louvain_communities(self.G.to_undirected(), seed=1)
        return comms, {n: i for i, c in enumerate(comms) for n in c}


def get_story(net, trial):
    if net.stories is None or len(net.stories) == 0:
        return None
    return net.stories.iloc[trial % len(net.stories)]


def story_category(net, trial):
    story = get_story(net, trial)
    return None if story is None else story["category"]


def story_emotion(net, trial):
    """Multiplier on the share probability: emotional claims spread faster (0.7x .. 1.3x)."""
    story = get_story(net, trial)
    if story is not None and "emotional_intensity" in story.index and pd.notna(story["emotional_intensity"]):
        return 0.7 + 0.6 * float(story["emotional_intensity"])
    return 1.0


def get_sources(net, trial, n_sources=5):
    """Starting spreaders: the users who posted the claim in the dataset."""
    story = get_story(net, trial)
    if story is not None:
        srcs = [s for s in story["sources"] if s in net.G]
        if srcs:
            return srcs[:n_sources]
    return random.Random(1000 + trial).sample(net.source_pool, min(n_sources, len(net.source_pool)))


# ---------------------------------------------------------------- spread simulation (Independent Cascade)
def simulate_spread(net, sources, protected, p, seed, hook=None, category=None,
                    boost=1.0, effect=1.0, parent=None):
    """A follower is affected with probability p x claim emotion x susceptibility
    (x MATCH_BOOST if the claim matches their topic). A fact-checked user blocks the claim with
    probability `effect`. If `parent` (a dict) is given, parent[v] = the user who infected v."""
    rng = random.Random(seed)
    sf, vu = net.sus_factor, net.vuln
    match = category is not None and bool(vu)
    infected, active = set(sources), list(sources)
    curve = [len(infected) / net.N]
    while active:
        if hook is not None:
            hook(active, infected, protected)
        newly = []
        for u in active:
            for v in net.G.successors(u):
                if v in infected:
                    continue
                if v in protected and (effect >= 1.0 or rng.random() < effect):
                    continue
                q = p * boost * sf.get(v, 1.0)
                if match and vu.get(v) == category:
                    q *= MATCH_BOOST
                if rng.random() < min(q, 1.0):
                    infected.add(v); newly.append(v)
                    if parent is not None:
                        parent[v] = u
        active = newly
        curve.append(len(infected) / net.N)
    return infected, curve


# ---------------------------------------------------------------- static strategies
def static_plan(net, strategy, budget, sources, rng):
    if strategy == "Random":
        cand = [n for n in net.nodes if n not in sources]
        return set(rng.sample(cand, min(budget, len(cand))))
    ranking = net.rank_degree if strategy == "Degree Centrality" else net.rank_betweenness
    chosen = []
    for n in ranking:
        if n not in sources:
            chosen.append(n)
            if len(chosen) == budget: break
    return set(chosen)


# ---------------------------------------------------------------- multi-agent system
class MonitoringAgent:
    """Agent 1: watches the spread and reports which users are about to be reached."""
    name = "Monitoring Agent"
    def __init__(self, net):
        self.net = net
    def observe(self, active, infected, protected):
        threatened = {v for u in active for v in self.net.G.successors(u)
                      if v not in infected and v not in protected}
        return {"from": self.name, "to": "Analyst Agent", "threatened": threatened}


class AnalystAgent:
    """Agent 2: ranks threatened users.
    aware=False -> by followers (reach), then betweenness (bridging)   [centrality only]
    aware=True  -> by followers x susceptibility x topic match, then betweenness  [uses the misinformation data]"""
    name = "Analyst Agent"
    def __init__(self, net, category=None, aware=True):
        self.net, self.cat, self.aware = net, category, aware
    def score(self, v):
        n = self.net
        s = float(n.out_degree[v])
        if self.aware:
            s *= n.sus_factor.get(v, 1.0)
            if self.cat is not None and n.vuln.get(v) == self.cat:
                s *= MATCH_BOOST
        return (s, n.betweenness[v])
    def analyse(self, msg, n):
        picks = heapq.nlargest(n, msg["threatened"], key=self.score)
        return {"from": self.name, "to": "Intervention Agent", "picks": picks}


class InterventionAgent:
    """Agent 3: owns the budget, decides the spend each round, fact-checks the chosen users."""
    name = "Intervention Agent"
    def __init__(self, budget, rounds, delay):
        self.left, self.delay, self.round = budget, delay, 0
        self.per_round = math.ceil(budget / rounds)
    def quota(self):
        self.round += 1
        if self.round <= self.delay or self.left <= 0:
            return 0
        return min(self.per_round, self.left)
    def act(self, msg, protected):
        protected.update(msg["picks"])
        self.left -= len(msg["picks"])


def agent_hook(net, budget, rounds=4, delay=1, log=None, category=None, aware=True):
    """log (optional) is a dict of lists: 'monitor', 'analyst', 'intervention'."""
    monitor, analyst, actor = MonitoringAgent(net), AnalystAgent(net, category, aware), InterventionAgent(budget, rounds, delay)
    def hook(active, infected, protected):
        before = actor.left
        n = actor.quota()                                       # Intervention: how much can I spend?
        m1 = monitor.observe(active, infected, protected) if (n > 0 or log is not None) else None
        if log is not None:
            log["monitor"].append({"Round": actor.round, "Active spreaders": len(active),
                                   "Infected so far": len(infected), "Users threatened": len(m1["threatened"])})
        if n == 0:
            if log is not None:
                log["intervention"].append({"Round": actor.round, "Budget before": before, "Quota": 0, "Spent": 0,
                                            "Budget left": actor.left, "Decision": "Wait (not noticed yet, or budget used)"})
            return
        m2 = analyst.analyse(m1, n)                             # Monitoring -> Analyst -> Intervention
        if log is not None:
            for v in m2["picks"][:5]:
                log["analyst"].append({"Round": actor.round, "User ID": v, "Followers": net.out_degree[v],
                                       "Susceptibility": round(net.sus_factor.get(v, 1.0), 2),
                                       "Topic match": net.vuln.get(v) == category if category else None,
                                       "Betweenness": round(net.betweenness[v], 6)})
        actor.act(m2, protected)
        if log is not None:
            log["intervention"].append({"Round": actor.round, "Budget before": before, "Quota": n,
                                        "Spent": len(m2["picks"]), "Budget left": actor.left,
                                        "Decision": "Fact-checked the Analyst's picks"})
    return hook


# ---------------------------------------------------------------- run / calibrate / summarise
def run(net, strategy, budget, sources, trial, p, rounds=4, delay=1, log=None,
        effect=None, rep=0, parent=None):
    """rep = repeat index: different random seeds for the same claim."""
    effect = FACT_CHECK_EFFECT if effect is None else effect
    cat = story_category(net, trial)
    protected, hook = set(), None
    if strategy in MAS_NAMES:
        hook = agent_hook(net, budget, rounds, delay, log, cat, aware=(strategy == "Multi-Agent System"))
    elif strategy != "No Protection":
        protected = static_plan(net, strategy, budget, set(sources), random.Random(2000 + trial + 100000 * rep))
    infected, curve = simulate_spread(net, sources, protected, p, 3000 + trial + 100000 * rep, hook, cat,
                                      story_emotion(net, trial), effect, parent)
    return infected, curve, protected


def calibrate(net, target=0.25, n_sources=5, runs=6, steps=8):
    """Find the share probability p so that, with no protection, about `target` of users are reached."""
    lo, hi = 0.0005, 0.2
    for _ in range(steps):
        mid = (lo + hi) / 2
        vals = [len(simulate_spread(net, get_sources(net, i, n_sources), set(), mid, 9000 + i, None,
                                    story_category(net, i), story_emotion(net, i))[0]) / net.N for i in range(runs)]
        lo, hi = (mid, hi) if np.mean(vals) < target else (lo, mid)
    return round((lo + hi) / 2, 4)


def outcome_table(net, infected, protected, sources, trial):
    """Per-user result of one simulation: who is affected and who is not."""
    category = story_category(net, trial)
    srcs, exposed = set(sources), set()
    for u in infected:
        exposed.update(net.G.successors(u))                    # every follower of an infected user saw the claim
    status = []
    for n in net.nodes:
        if n in srcs: status.append("Misinformation source")
        elif n in infected: status.append("Affected")
        elif n in protected: status.append("Fact-checked (protected)")
        elif n in exposed: status.append("Exposed but not affected")
        else: status.append("Not reached")
    df = pd.DataFrame({"User ID": net.nodes, "Followers": [net.out_degree[n] for n in net.nodes], "Status": status})
    if net.users is not None:
        df["Susceptibility level"] = df["User ID"].map(net.users["susceptibility_level"])
        df["Vulnerable topic"] = df["User ID"].map(net.users["vulnerable_category"])
        df["Claim topic matches"] = df["Vulnerable topic"] == category
    return df


def summarise(results_df):
    g = results_df.groupby(["Budget (%)", "Strategy"])["Infected Percentage"].agg(["mean", "std", "count"]).reset_index()
    g["ci"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"])
    s = g[["Budget (%)", "Strategy", "mean", "std", "ci"]].copy()
    s.columns = ["Budget (%)", "Strategy", "Average Infection (%)", "Std Dev", "95% CI (+/-)"]
    base = s.loc[s["Strategy"] == "No Protection", "Average Infection (%)"].iloc[0]
    s["Reduction (%)"] = (base - s["Average Infection (%)"]) / base * 100 if base > 0 else 0
    return s, base


# ---------------------------------------------------------------- statistics
def paired_test(results_df, a, b, budget):
    """Paired Wilcoxon signed-rank test: is strategy a different from strategy b (same claim, same seed)?"""
    d = results_df[results_df["Budget (%)"] == budget]
    col = "Rep" if "Rep" in d.columns else None
    idx = ["Trial", col] if col else ["Trial"]
    piv = d.pivot_table(index=idx, columns="Strategy", values="Infected Percentage")
    if a not in piv.columns or b not in piv.columns:
        return {"Comparison": f"{a} vs {b}", "Pairs": 0, "Mean difference (pp)": None, "p-value": None, "Significant (p<0.05)": "n/a"}
    diff = (piv[a] - piv[b]).dropna()
    pval = None
    try:
        from scipy.stats import wilcoxon
        if len(diff) >= 5 and (diff != 0).any():
            pval = float(wilcoxon(diff).pvalue)
    except Exception:
        pass
    return {"Comparison": f"{a} vs {b}", "Pairs": len(diff), "Mean difference (pp)": round(float(diff.mean()), 3),
            "p-value": None if pval is None else round(pval, 4),
            "Significant (p<0.05)": "n/a" if pval is None else ("Yes" if pval < 0.05 else "No")}


# ---------------------------------------------------------------- "who will be affected": prediction check
def risk_scores(net, category, p, boost=1.0):
    """Predicted chance that each user is affected BEFORE any simulation:
    exposure (accounts they follow) x share probability x susceptibility x topic match."""
    ids = net.nodes
    q = p * boost * np.array([net.sus_factor.get(n, 1.0) for n in ids])
    if category is not None and net.vuln:
        q = q * np.array([MATCH_BOOST if net.vuln.get(n) == category else 1.0 for n in ids])
    q = np.clip(q, 0, 1)
    indeg = np.array([net.in_degree[n] for n in ids])
    return pd.Series(1 - (1 - q) ** indeg, index=ids)


def precision_at_k(infected, sources, risk, k):
    """Of the k users we predicted as highest risk, how many were really affected?
    Returns (hit rate of the top-k, overall infection rate among non-source users)."""
    r = risk.drop(labels=[s for s in sources if s in risk.index])
    top = r.nlargest(min(k, len(r))).index
    hit = float(np.mean([n in infected for n in top]))
    base = len(set(infected) - set(sources)) / max(1, len(r))
    return hit, base


# ---------------------------------------------------------------- "where / how": spread analysis
def source_table(net, sources, comm_of=None):
    rows = []
    for s in sources:
        r = {"User ID": s, "Followers": net.out_degree[s], "Following": net.in_degree[s],
             "Source type": net.src_type.get(s, "n/a"), "Vulnerable topic": net.vuln.get(s, "n/a")}
        if comm_of is not None:
            r["Community"] = comm_of.get(s)
        rows.append(r)
    return pd.DataFrame(rows)


def super_spreaders(net, parent, top=10):
    cnt = Counter(parent.values()).most_common(top)
    return pd.DataFrame([{"User ID": u, "Users infected directly": c, "Followers": net.out_degree[u],
                          "Betweenness": round(net.betweenness[u], 6)} for u, c in cnt])


def new_per_round(net, curve):
    return (np.diff(np.array(curve)) * net.N).round().astype(int)


def community_infection(net, infected, comms, top=10):
    rows = []
    for i, c in enumerate(comms):
        k = len(set(c) & infected)
        if k:
            rows.append({"Community": i, "Size": len(c), "Infected": k, "Infected (%)": round(k / len(c) * 100, 1)})
    return pd.DataFrame(rows).sort_values("Infected", ascending=False).head(top).reset_index(drop=True)


def category_spread(net, p, n_sources=5):
    """No-protection spread of every claim: which TYPE of misinformation spreads most?"""
    if net.stories is None:
        return None
    rows = []
    for t in range(len(net.stories)):
        st_ = get_story(net, t)
        inf, curve, _ = run(net, "No Protection", 0, get_sources(net, t, n_sources), t, p)
        rows.append({"Claim": st_["claim_id"], "Category": st_["category"],
                     "Emotional intensity": st_.get("emotional_intensity", np.nan),
                     "Infected (%)": round(len(inf) / net.N * 100, 2), "Rounds": len(curve) - 1})
    return pd.DataFrame(rows)