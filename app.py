import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import sna_core as core


def show_df(df, **kw):
    """st.dataframe that hides the index when the Streamlit version supports it."""
    try:
        st.dataframe(df, hide_index=True, **kw)
    except TypeError:
        st.dataframe(df.reset_index(drop=True), **kw)


st.set_page_config(page_title="Misinformation Prevention using SNA", layout="wide")
st.title("Agentic Misinformation Prevention Using Social Network Analysis")
st.caption("Simulate fake news on a follower network and compare ways to spend a limited fact-checking budget.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    data_choice = "Twitter (SNAP dataset)"
    bet_k = st.slider("Betweenness sample size (k)", 20, 200, 100, 10)
    st.subheader("Experiment")
    budgets = st.multiselect("Fact-check budgets (% of users)", [1, 2, 5, 10, 15], default=[1, 5, 10])
    detail_budget = st.selectbox("Budget for detailed views (%)", [1, 2, 5, 10, 15], index=2)
    trials = st.slider("Trials per setting (one claim each)", 5, 25, 10)
    n_reps = st.slider("Random seeds per claim", 1, 3, 1, help="More seeds = more reliable averages and a stronger significance test.")
    n_sources = st.slider("Initial fake-news sources", 1, 10, 5)
    target = st.slider("No-protection spread to calibrate to", 0.05, 0.60, 0.25, 0.05)
    eff = st.slider("Fact-check effectiveness", 0.3, 1.0, core.FACT_CHECK_EFFECT, 0.05,
                    help="Chance that a fact-checked user really blocks the claim (1.0 = perfect).")
    st.subheader("Multi-agent system")
    a_rounds = st.slider("Rounds to spend the budget over", 1, 8, 4)
    a_delay = st.slider("Rounds before the agents notice", 0, 3, 1)

if detail_budget not in budgets:
    budgets = sorted(budgets + [detail_budget])

# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Loading network and computing centrality...")
def get_network(choice, k):
    is_twitter = choice.startswith("Twitter")
    G = core.load_twitter() if is_twitter else core.demo_graph()
    return core.Network(G, betweenness_k=k, use_stories=is_twitter)

try:
    net = get_network(data_choice, bet_k)
except Exception as e:
    st.error(f"Could not load the network: {e}. Check that data/twitter_combined.csv exists in the data folder.")
    st.stop()

@st.cache_data(show_spinner="Calibrating spread probability...")
def get_p(choice, k, target, n_sources):
    return core.calibrate(net, target, n_sources)

@st.cache_resource(show_spinner="Finding communities (slow on the full Twitter network)...")
def get_comms(choice, k):
    return net.communities()

P = get_p(data_choice, bet_k, target, n_sources)
strategies = core.STRATEGIES
tab_about, tab_net, tab_exp, tab_time, tab_vis, tab_who, tab_how, tab_sens = st.tabs(
    ["0. About", "1. Network & misinformation", "2. Strategy comparison", "3. Spread over time",
     "4. Network picture & agents", "5. Who is affected", "6. Where & how it spreads", "7. Sensitivity"])

with tab_about:
    st.subheader("Problem statement")
    st.write("Fake news spreads through follower networks, and fact-checkers can check only a limited number of users. "
             "Which users should be fact-checked first to reduce the spread the most?")
    st.subheader("Dataset collection")
    st.write("Network: SNAP Twitter follower network (Leskovec and McAuley), 81,306 users. "
             "Misinformation: 25 synthetic fake claims posted by real high-follower users, with a fact-check, prevention action "
             "and per-user susceptibility columns added to the Twitter users.")
    st.subheader("SNA tool and features used")
    st.write("Tool: NetworkX (Python). Features: directed graph, degree centrality, betweenness centrality, "
             "Louvain community detection, BFS neighbourhoods and graph layout.")
    st.subheader("AI approach")
    st.write("Agentic AI: the Multi-Agent System reacts to the spread round by round. Multi-agent AI: a Monitoring Agent, "
             "an Analyst Agent and an Intervention Agent, each with a separate job.")
# ------------------------------------------------------------------ tab 1
with tab_net:
    st.subheader("Network overview")
    show_df(pd.DataFrame(net.stats().items(), columns=["Metric", "Value"]))
    ct = net.centrality_table()
    c1, c2 = st.columns(2)
    c1.markdown("**Top 10 users by followers**")
    c1.write(ct.nlargest(10, "Followers (Out-Degree)").reset_index(drop=True))
    c2.markdown("**Top 10 users by betweenness**")
    c2.write(ct.nlargest(10, "Betweenness").reset_index(drop=True))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(ct["Followers (Out-Degree)"], bins=np.logspace(0, np.log10(max(2, ct["Followers (Out-Degree)"].max())), 40))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Followers"); ax.set_ylabel("Users")
    ax.set_title("A few users have a very large following")
    st.pyplot(fig)
    st.info(f"Calibrated spread probability P = {P} (about {int(target*100)}% of users reached with no protection).")

    st.subheader("Misinformation dataset")
    if net.stories is not None:
        n_spreaders = sum(len(s) for s in net.stories["sources"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Fake claims", len(net.stories))
        m2.metric("Users posting misinformation", n_spreaders)
        m3.metric("Share of network", f"{n_spreaders / net.N * 100:.2f}%")
        cols = [c for c in ["claim_id", "category", "tweet_text", "emotional_intensity", "harm_level",
                            "source_of_truth", "fact_check", "prevention"] if c in net.stories.columns]
        show_df(net.stories.sort_values("claim_id")[cols])
        st.markdown("**Who can be affected: user susceptibility in the dataset**")
        lv = net.users["susceptibility_level"].value_counts().reindex(["Low", "Medium", "High"])
        vc = net.users["vulnerable_category"].value_counts()
        s1, s2 = st.columns(2)
        s1.write(lv.rename("Users").to_frame())
        s2.write(vc.rename("Users vulnerable to this topic").to_frame())
        if "media_literacy" in net.users.columns:
            st.markdown("**What drives susceptibility**")
            drv = net.users.groupby("media_literacy")["susceptibility"].mean().reindex(["Low", "Medium", "High"]).round(3)
            d1, d2 = st.columns(2)
            d1.write(drv.rename("Average susceptibility").to_frame())
            if "age_group" in net.users.columns:
                d2.write(net.users.groupby("age_group")["susceptibility"].mean().round(3).rename("Average susceptibility").to_frame())
            st.caption("susceptibility = 0.40 x (1 - media literacy) + 0.30 x echo-chamber score + 0.15 x prior exposure + 0.15 x age factor.")
        st.caption("A follower is affected with probability P x claim emotion x susceptibility, and 1.5x higher when the claim is in "
                   "their vulnerable topic. The fake tweets and user attributes are synthetic; the users and followers are real SNAP data.")
    else:
                st.warning("Misinformation dataset not used (data/twitter_users_misinfo.csv is missing). Sources are picked at random.")

# ------------------------------------------------------------------ tab 2
with tab_exp:
    st.subheader("Compare strategies")
    st.write("Every strategy gets the same claim, sources, random seeds and budget. "
             "'MAS (Centrality Only)' is the ablation: the same agents without the susceptibility data.")
    if st.button("Run experiments", type="primary"):
        results, curves = [], {s: [] for s in strategies}
        bar, status, t0 = st.progress(0.0), st.empty(), time.time()
        for trial in range(trials):
            story = core.get_story(net, trial)
            claim = story["claim_id"] if story is not None else "random"
            cat = story["category"] if story is not None else "random"
            src = core.get_sources(net, trial, n_sources)
            for rep in range(n_reps):
                inf, curve, _ = core.run(net, "No Protection", 0, src, trial, P, effect=eff, rep=rep)
                for pct in budgets:
                    results.append({"Budget (%)": pct, "Trial": trial + 1, "Rep": rep + 1, "Claim": claim, "Category": cat,
                                    "Strategy": "No Protection", "Infected Percentage": len(inf) / net.N * 100,
                                    "Protected Users": 0})
                if rep == 0:
                    curves["No Protection"].append(curve)
                for pct in budgets:
                    budget = max(1, int(net.N * pct / 100))
                    for s in strategies[1:]:
                        inf, curve, prot = core.run(net, s, budget, src, trial, P, a_rounds, a_delay, effect=eff, rep=rep)
                        results.append({"Budget (%)": pct, "Trial": trial + 1, "Rep": rep + 1, "Claim": claim, "Category": cat,
                                        "Strategy": s, "Infected Percentage": len(inf) / net.N * 100,
                                        "Protected Users": len(prot)})
                        if pct == detail_budget and rep == 0:
                            curves[s].append(curve)
            bar.progress((trial + 1) / trials)
            status.caption(f"Trial {trial + 1}/{trials} ({time.time() - t0:.0f}s elapsed)")
        st.session_state["results"] = pd.DataFrame(results)
        st.session_state["curves"] = curves
        st.session_state["meta"] = dict(budgets=budgets, detail=detail_budget, P=P)

    if "results" in st.session_state:
        results_df = st.session_state["results"]
        meta = st.session_state["meta"]
        summary, baseline = core.summarise(results_df)
        st.success(f"No Protection baseline: {baseline:.2f}% of users infected (about {int(baseline / 100 * net.N):,} users).")
        pivot = summary.pivot(index="Strategy", columns="Budget (%)", values="Average Infection (%)").loc[strategies].round(2)
        pivot_red = summary.pivot(index="Strategy", columns="Budget (%)", values="Reduction (%)").loc[strategies].round(1)
        pivot_ci = summary.pivot(index="Strategy", columns="Budget (%)", values="95% CI (+/-)").loc[strategies].round(2)
        a, b, c = st.columns(3)
        a.markdown("**Average infected users (%)**"); a.dataframe(pivot)
        b.markdown("**Reduction vs No Protection (%)**"); b.dataframe(pivot_red)
        c.markdown("**95% confidence interval (+/- pp)**"); c.dataframe(pivot_ci)

        best = summary[summary["Strategy"] != "No Protection"]
        best = best.loc[best.groupby("Budget (%)")["Average Infection (%)"].idxmin()]
        for _, r in best.iterrows():
            saved = int((baseline - r["Average Infection (%)"]) / 100 * net.N)
            st.write(f"With **{int(r['Budget (%)'])}%** of users fact-checked, **{r['Strategy']}** is best: "
                     f"{r['Average Infection (%)']:.2f}% infected ({r['Reduction (%)']:.0f}% lower than no protection, "
                     f"about {saved:,} users saved).")

        st.markdown(f"**Is the difference real? Paired Wilcoxon test ({meta['detail']}% budget)**")
        pairs = [("Multi-Agent System", "Degree Centrality"), ("Multi-Agent System", "MAS (Centrality Only)"),
                 ("Multi-Agent System", "Betweenness Centrality"), ("Degree Centrality", "Betweenness Centrality")]
        show_df(pd.DataFrame([core.paired_test(results_df, x, y, meta["detail"]) for x, y in pairs]))
        st.caption("Negative mean difference = the first strategy infects fewer users. Needs at least 5 paired runs (use more trials or seeds).")

        nb = results_df[results_df["Strategy"] != "No Protection"]
        g = nb.groupby(["Budget (%)", "Strategy"])["Infected Percentage"]
        mean = g.mean().unstack().reindex(columns=strategies[1:])
        std = g.std().unstack().reindex(columns=strategies[1:])
        fig, ax = plt.subplots(figsize=(11, 5))
        mean.plot(kind="bar", yerr=std, ax=ax, capsize=3, rot=0)
        ax.axhline(baseline, color="gray", linestyle="--", label="No Protection")
        ax.set_xlabel("Fact-check budget (% of users)"); ax.set_ylabel("Average infected users (%)"); ax.legend()
        st.pyplot(fig)

        st.markdown(f"**Infected users (%) by claim category ({meta['detail']}% budget)**")
        d = results_df[results_df["Budget (%)"] == meta["detail"]]
        bycat = d.groupby(["Category", "Strategy"])["Infected Percentage"].mean().unstack().reindex(columns=strategies).round(2)
        st.dataframe(bycat)

        prot = results_df.groupby(["Strategy", "Budget (%)"])["Protected Users"].mean().unstack().round(0)
        st.markdown("**Fairness check: users actually fact-checked**")
        st.dataframe(prot.loc[strategies])
        for pct in meta["budgets"]:
            allowed = max(1, int(net.N * pct / 100))
            for s in strategies[1:]:
                if prot.loc[s, pct] < 0.95 * allowed:
                    st.caption(f"Note: at {pct}% budget, '{s}' used only {int(prot.loc[s, pct])} of {allowed} allowed users (the spread ended early).")
        st.download_button("Download results (CSV)", results_df.to_csv(index=False), "simulation_results.csv")
    else:
        st.info("Press **Run experiments**. With the full Twitter network, start with few trials.")

# ------------------------------------------------------------------ tab 3
with tab_time:
    if "curves" in st.session_state:
        curves = st.session_state["curves"]; d = st.session_state["meta"]["detail"]
        fig, ax = plt.subplots(figsize=(9, 5))
        for s in strategies:
            cs = curves[s]
            L = max(len(c) for c in cs)
            ax.plot(np.mean([c + [c[-1]] * (L - len(c)) for c in cs], axis=0) * 100, label=s)
        ax.set_xlabel("Spread round"); ax.set_ylabel("Infected users (%)")
        ax.set_title(f"Misinformation spread over time ({d}% fact-checked)"); ax.legend(); ax.grid(alpha=.3)
        st.pyplot(fig)
        if st.checkbox("Also count communities reached (slow on the full Twitter network)"):
            comms, comm_of = get_comms(data_choice, bet_k)
            with st.spinner("Running simulations..."):
                rows = []
                for s in strategies:
                    reach = []
                    for t in range(min(trials, 5)):
                        src = core.get_sources(net, t, n_sources)
                        b = 0 if s == "No Protection" else max(1, int(net.N * d / 100))
                        inf, _, _ = core.run(net, s, b, src, t, P, a_rounds, a_delay, effect=eff)
                        reach.append(len({comm_of[n] for n in inf}))
                    rows.append({"Strategy": s, "Avg communities reached": round(np.mean(reach), 1)})
            st.write(f"Communities found: {len(comms)}")
            show_df(pd.DataFrame(rows))
    else:
        st.info("Run the experiments in tab 2 first.")

# ------------------------------------------------------------------ claim picker (used by tabs 4, 5 and 6)
def pick_claim(key):
    if net.stories is None:
        return 0
    ids = sorted(net.stories["claim_id"])
    lookup = net.stories.set_index("claim_id")
    pick = st.selectbox("Choose a misinformation claim", ids, key=key,
                        format_func=lambda c: f"{c} ({lookup.loc[c, 'category']}): {lookup.loc[c, 'tweet_text']}")
    story = lookup.loc[pick]
    st.write(f"Fact-check: {story['fact_check']}")
    st.write(f"Prevention: {story['prevention']}")
    return int(net.stories.index[net.stories["claim_id"] == pick][0])

# ------------------------------------------------------------------ tab 4
with tab_vis:
    vis_trial = pick_claim("claim_vis")
    st.write("Red = infected, green = fact-checked, grey = not reached. Showing a 300-user neighbourhood of the first source.")
    if st.button("Draw network picture"):
        src0 = core.get_sources(net, vis_trial, n_sources)
        sample = list(nx.bfs_tree(net.G, src0[0], depth_limit=2).nodes())[:300]
        H = net.G.subgraph(sample)
        pos = nx.spring_layout(H, seed=3)
        budget_vis = max(1, int(net.N * detail_budget / 100))
        panels = ["No Protection", "Random", "Degree Centrality", "Multi-Agent System"]
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
        for ax, s in zip(axes, panels):
            inf, _, prot = core.run(net, s, 0 if s == "No Protection" else budget_vis, src0, vis_trial, P,
                                    a_rounds, a_delay, effect=eff)
            colors = ["red" if n in inf else "green" if n in prot else "lightgrey" for n in H]
            sizes = [35 if (n in inf or n in prot) else 8 for n in H]
            nx.draw_networkx(H, pos, ax=ax, node_color=colors, node_size=sizes, width=0.15, arrows=False, with_labels=False)
            ax.set_title(f"{s}\ninfected in full network: {len(inf) / net.N * 100:.1f}%"); ax.axis("off")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("**The three agents working separately**")
    if st.button("Show agents working separately"):
        log = {"monitor": [], "analyst": [], "intervention": []}
        src0 = core.get_sources(net, vis_trial, n_sources)
        core.run(net, "Multi-Agent System", max(1, int(net.N * detail_budget / 100)), src0, vis_trial, P,
                 a_rounds, a_delay, log=log, effect=eff)
        c1, c2, c3 = st.columns(3)
        c1.markdown("**1. Monitoring Agent**")
        c1.caption("Watches the spread and reports who is about to be reached.")
        c1.dataframe(pd.DataFrame(log["monitor"]).reset_index(drop=True))
        c2.markdown("**2. Analyst Agent**")
        c2.caption("Ranks threatened users by followers x susceptibility x topic match, then betweenness (top 5 per round shown).")
        c2.dataframe(pd.DataFrame(log["analyst"]).reset_index(drop=True))
        c3.markdown("**3. Intervention Agent**")
        c3.caption("Owns the budget, decides the spend each round, fact-checks the picks.")
        c3.dataframe(pd.DataFrame(log["intervention"]).reset_index(drop=True))

# ------------------------------------------------------------------ tab 5
with tab_who:
    st.write("For one claim and one strategy: which users are affected, which are exposed but resist, and which are never reached.")
    who_trial = pick_claim("claim_who")
    who_strategy = st.selectbox("Strategy", strategies, index=len(strategies) - 1)
    if st.button("Show who is affected"):
        src0 = core.get_sources(net, who_trial, n_sources)
        b = 0 if who_strategy == "No Protection" else max(1, int(net.N * detail_budget / 100))
        inf, _, prot = core.run(net, who_strategy, b, src0, who_trial, P, a_rounds, a_delay, effect=eff)
        out = core.outcome_table(net, inf, prot, src0, who_trial)
        counts = out["Status"].value_counts()
        order = ["Misinformation source", "Affected", "Exposed but not affected", "Fact-checked (protected)", "Not reached"]
        cols = st.columns(len(order))
        for col, name in zip(cols, order):
            col.metric(name, f"{int(counts.get(name, 0)):,}")
        exposed = out[out["Status"].isin(["Affected", "Exposed but not affected"])]
        if "Susceptibility level" in out.columns and len(exposed) > 0:
            st.markdown("**Of the users who saw the claim, how many were affected?**")
            r1, r2 = st.columns(2)
            by_level = exposed.groupby("Susceptibility level")["Status"].apply(lambda s: (s == "Affected").mean() * 100)
            r1.write(by_level.reindex(["Low", "Medium", "High"]).round(1).rename("Affected (%)").to_frame())
            by_match = exposed.groupby("Claim topic matches")["Status"].apply(lambda s: (s == "Affected").mean() * 100)
            by_match.index = by_match.index.map({True: "Claim matches their topic", False: "Different topic"})
            r2.write(by_match.round(1).rename("Affected (%)").to_frame())
        st.markdown("**Top 20 affected users by followers**")
        show_df(out[out["Status"] == "Affected"].nlargest(20, "Followers"))
        st.download_button("Download every user's status (CSV)", out.to_csv(index=False), "who_is_affected.csv")

# ------------------------------------------------------------------ tab 6
with tab_how:
    st.write("Answers four questions for one claim with no protection: **where** it starts, **how** it spreads, "
             "**what type** spreads most, and **who** will be affected (and whether we predicted it).")
    how_trial = pick_claim("claim_how")
    use_comm = st.checkbox("Include community analysis (slow on the full Twitter network)", key="how_comm")
    k_pred = st.number_input("Top-k users to predict as highest risk", 100, 20000, 1000, 100)
    if st.button("Analyse this claim"):
        src0 = core.get_sources(net, how_trial, n_sources)
        cat = core.story_category(net, how_trial)
        parent = {}
        inf, curve, _ = core.run(net, "No Protection", 0, src0, how_trial, P, parent=parent)
        comms = comm_of = None
        if use_comm:
            comms, comm_of = get_comms(data_choice, bet_k)

        st.markdown("### 1. Where does it start?")
        show_df(core.source_table(net, src0, comm_of))
        st.caption("These are the real users who post the claim first. Source type comes from the misinformation dataset.")
        if comms is not None:
            st.write(f"The claim ends up in these communities (top 10 of {len(comms)}):")
            show_df(core.community_infection(net, inf, comms))

        st.markdown("### 2. How does it spread?")
        h1, h2, h3 = st.columns(3)
        h1.metric("Users affected", f"{len(inf):,} ({len(inf) / net.N * 100:.1f}%)")
        h2.metric("Spread rounds (cascade depth)", len(curve) - 1)
        h3.metric("Claim topic", cat if cat else "n/a")
        npr = core.new_per_round(net, curve)
        st.bar_chart(pd.DataFrame({"New users affected": npr}, index=range(1, len(npr) + 1)))
        st.markdown("**Super-spreaders: users who directly infected the most followers**")
        show_df(core.super_spreaders(net, parent))

        st.markdown("### 3. Who will be affected? Prediction check")
        risk = core.risk_scores(net, cat, P, core.story_emotion(net, how_trial))
        hit, base = core.precision_at_k(inf, src0, risk, int(k_pred))
        p1, p2, p3 = st.columns(3)
        p1.metric(f"Top-{int(k_pred):,} predicted users really affected", f"{hit * 100:.1f}%")
        p2.metric("Average user affected", f"{base * 100:.1f}%")
        p3.metric("Lift over average", f"{hit / base:.1f}x" if base > 0 else "n/a")
        st.caption("Before the simulation, each user gets a risk score = exposure (accounts they follow) x susceptibility x topic match. "
                   "Lift above 1x shows the score really identifies who will be affected.")

    st.markdown("---")
    st.markdown("### 4. What type of misinformation spreads most?")
    if st.button("Run all claims with no protection"):
        cs = core.category_spread(net, P, n_sources)
        if cs is None:
            st.warning("Needs the misinformation dataset (Twitter network).")
        else:
            show_df(cs)
            by = cs.groupby("Category")["Infected (%)"].mean().round(2).sort_values(ascending=False)
            st.bar_chart(by)
            st.caption("Average spread of the 5 claims in each category, with no protection.")

# ------------------------------------------------------------------ tab 7
with tab_sens:
    st.write("How robust is the multi-agent advantage? Change one setting and compare strategies at the detail budget.")
    param = st.selectbox("Setting to vary", ["Rounds before the agents notice", "Rounds to spend the budget over",
                                             "Fact-check effectiveness", "Spread probability (x calibrated P)"])
    grid = {"Rounds before the agents notice": [0, 1, 2, 3], "Rounds to spend the budget over": [1, 2, 4, 8],
            "Fact-check effectiveness": [0.5, 0.65, 0.8, 1.0], "Spread probability (x calibrated P)": [0.5, 0.75, 1.0, 1.25, 1.5]}[param]
    n_tr = st.slider("Claims to average over", 3, 10, 5, key="sens_trials")
    if st.button("Run sensitivity"):
        budget = max(1, int(net.N * detail_budget / 100))
        rows, bar = [], st.progress(0.0)
        total, done = len(grid) * 3, 0
        for val in grid:
            kw = dict(rounds=a_rounds, delay=a_delay, effect=eff); p_use = P
            if param.startswith("Rounds before"): kw["delay"] = val
            elif param.startswith("Rounds to spend"): kw["rounds"] = val
            elif param.startswith("Fact-check"): kw["effect"] = val
            else: p_use = P * val
            for s in ["Degree Centrality", "MAS (Centrality Only)", "Multi-Agent System"]:
                vals = []
                for t in range(n_tr):
                    src = core.get_sources(net, t, n_sources)
                    inf, _, _ = core.run(net, s, budget, src, t, p_use, **kw)
                    vals.append(len(inf) / net.N * 100)
                rows.append({param: val, "Strategy": s, "Infected (%)": np.mean(vals)})
                done += 1; bar.progress(done / total)
        sens = pd.DataFrame(rows).pivot(index=param, columns="Strategy", values="Infected (%)").round(2)
        st.line_chart(sens)
        st.dataframe(sens)
        st.caption("If the multi-agent lines stay below Degree Centrality across the range, the result does not depend on one lucky setting.")