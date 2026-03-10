"""
Energy-Use Profiles Explorer
Streamlit dashboard for the customer segmentation project.
Loads pre-computed artifacts .
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page config 
st.set_page_config(
    page_title="Energy-Use Profiles Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Load artifacts 
ARTIFACTS = "artifacts"


@st.cache_data
def load_data():
    llm = json.load(open(f"{ARTIFACTS}/llm_results.json"))
    profiles = pd.read_csv(f"{ARTIFACTS}/cluster_profiles.csv", index_col="cluster")
    profiles_scaled = pd.read_csv(f"{ARTIFACTS}/cluster_profiles_scaled.csv", index_col="cluster")
    hourly = pd.read_csv(f"{ARTIFACTS}/hourly_by_cluster.csv")
    meta = pd.read_csv(f"{ARTIFACTS}/metadata_with_clusters.csv")
    assignments = pd.read_csv(f"{ARTIFACTS}/cluster_assignments.csv")
    proposals = json.load(open(f"{ARTIFACTS}/agent_proposals.json"))
    return llm, profiles, profiles_scaled, hourly, meta, assignments, proposals


llm, profiles, profiles_scaled, hourly, meta, assignments, proposals = load_data()

n_clusters = len(llm)
cluster_ids = sorted(llm.keys(), key=int)
COLORS = px.colors.qualitative.Set2[:n_clusters]
color_map = {int(cid): COLORS[i] for i, cid in enumerate(cluster_ids)}

#  Sidebar navigation 
st.sidebar.title("⚡ Energy-Use Profiles")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Explore a Profile", "Compare Profiles", "Agent Transparency"],
    index=0,
)

st.sidebar.markdown("---")


# ═══════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("⚡ Energy-Use Profiles Overview")
    st.markdown(
        "**500 households** clustered into **7 energy-use profiles** based on "
        "1 year of hourly smart-meter data (electricity + gas). "
        "Profiles are named and interpreted by a multi-agent LLM pipeline."
    )

    # Summary table 
    summary_rows = []
    for cid in cluster_ids:
        d = llm[cid]
        summary_rows.append({
            "Profile": int(cid),
            "Name": d["name"],
            "Households": d["size"],
            "Share (%)": round(d["size"] / 500 * 100, 1),
            "Electricity (kWh/day)": d["avg_electricity_kwh_day"],
            "Gas (kWh/day)": d["avg_gas_kwh_day"],
        })
    summary_df = pd.DataFrame(summary_rows)

    st.dataframe(
        summary_df.style.format({"Share (%)": "{:.1f}", "Electricity (kWh/day)": "{:.2f}", "Gas (kWh/day)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # Cluster distribution
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            summary_df, x="Profile", y="Households", color="Name",
            color_discrete_sequence=COLORS,
            title="Households per Profile",
            text="Households",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Profile", yaxis_title="Households")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            summary_df, values="Households", names="Name",
            color_discrete_sequence=COLORS,
            title="Population Share",
        )
        fig.update_traces(textinfo="percent+label", textposition="inside")
        st.plotly_chart(fig, use_container_width=True)

    # Daily consumption comparison
    st.subheader("Average Daily Consumption by Profile")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            summary_df, x="Profile", y="Electricity (kWh/day)",
            color="Profile", color_discrete_map=color_map,
            title="Electricity",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            summary_df, x="Profile", y="Gas (kWh/day)",
            color="Profile", color_discrete_map=color_map,
            title="Gas",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Average daily curves (all profiles)
    st.subheader("Average Daily Consumption Curves")
    fuel = st.radio("Fuel", ["Electricity", "Gas"], horizontal=True, key="overview_fuel")
    col_name = "electricity_kWh" if fuel == "Electricity" else "gas_kWh"

    fig = px.line(
        hourly, x="hour", y=col_name, color="cluster",
        color_discrete_map=color_map,
        labels={"hour": "Hour of Day", col_name: f"{fuel} (kWh)", "cluster": "Profile"},
        title=f"Average Hourly {fuel} Consumption by Profile",
    )
    fig.update_layout(xaxis=dict(dtick=2))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: Explore a Profile
# ═══════════════════════════════════════════════════════════════
elif page == "Explore a Profile":
    st.title("🔍 Explore an Energy-Use Profile")

    # Profile selector
    options = {int(cid): f"Profile {cid} — {llm[cid]['name']} ({llm[cid]['size']} HH)" for cid in cluster_ids}
    selected = st.selectbox("Select a profile", list(options.keys()), format_func=lambda x: options[x])
    cid_str = str(selected)
    d = llm[cid_str]

    #  Header metrics
    st.markdown(f"## Profile {selected} — *{d['name']}*")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Households", d["size"])
    c2.metric("Population Share", f"{d['size'] / 500 * 100:.1f}%")
    c3.metric("Avg Electricity", f"{d['avg_electricity_kwh_day']:.2f} kWh/day")
    c4.metric("Avg Gas", f"{d['avg_gas_kwh_day']:.2f} kWh/day")

    #  LLM Interpretation
    st.markdown("### Profile Interpretation")
    st.markdown(d["interpretation"])

    st.markdown("---")

    #  Daily consumption curve
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Daily Consumption Curve")
        cluster_hourly = hourly[hourly["cluster"] == selected]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cluster_hourly["hour"], y=cluster_hourly["electricity_kWh"],
            mode="lines+markers", name="Electricity", line=dict(color="#1f77b4", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=cluster_hourly["hour"], y=cluster_hourly["gas_kWh"],
            mode="lines+markers", name="Gas", line=dict(color="#ff7f0e", width=3),
        ))
        # Population average
        pop_hourly = hourly.groupby("hour")[["electricity_kWh", "gas_kWh"]].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=pop_hourly["hour"], y=pop_hourly["electricity_kWh"],
            mode="lines", name="Electricity (pop. avg)", line=dict(color="#1f77b4", width=1, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=pop_hourly["hour"], y=pop_hourly["gas_kWh"],
            mode="lines", name="Gas (pop. avg)", line=dict(color="#ff7f0e", width=1, dash="dash"),
        ))
        fig.update_layout(
            xaxis_title="Hour of Day", yaxis_title="kWh",
            xaxis=dict(dtick=2), height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    #  Radar chart
    with col2:
        st.markdown("### Feature Fingerprint")
        radar_features = [
            "avg_daily_electricity_usage", "avg_daily_gas_usage",
            "evening_peak_share_electricity", "prop_daily_electricity_overnight",
            "ratio_winter_summer_electricity_usage", "weekend_weekday_ratio_electricity",
            "peak_to_avg_electricity", "std_daily_electricity_usage",
        ]
        available = [f for f in radar_features if f in profiles_scaled.columns]

        vals = profiles_scaled.loc[selected, available].values.tolist()
        pop_vals = [0.0] * len(available)  # standardized mean = 0
        labels = [f.replace("_", " ").title() for f in available]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=labels + [labels[0]],
            fill="toself", name=f"Profile {selected}",
            line=dict(color=color_map[selected], width=2),
        ))
        fig.add_trace(go.Scatterpolar(
            r=pop_vals + [pop_vals[0]], theta=labels + [labels[0]],
            fill="none", name="Population avg",
            line=dict(color="gray", width=1, dash="dash"),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-2.5, 3])),
            height=400, showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    #  Metadata breakdown
    st.markdown("### Household Characteristics")
    cluster_meta = meta[meta["cluster"] == selected]

    cat_cols = ["property_type", "heating_type", "income_band", "occupancy_pattern",
                "insulation_quality", "tenure_type", "cooking_fuel"]
    available_cats = [c for c in cat_cols if c in cluster_meta.columns]

    cols = st.columns(min(len(available_cats), 3))
    for i, col_name in enumerate(available_cats):
        with cols[i % 3]:
            dist = cluster_meta[col_name].value_counts(normalize=True).head(5).reset_index()
            dist.columns = [col_name, "share"]
            dist["share"] = (dist["share"] * 100).round(1)
            fig = px.bar(
                dist, x="share", y=col_name, orientation="h",
                title=col_name.replace("_", " ").title(),
                labels={"share": "%"},
                color_discrete_sequence=[color_map[selected]],
            )
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    #  Binary features
    binary_cols = ["solar_pv", "ev_ownership", "children_present"]
    available_bin = [c for c in binary_cols if c in cluster_meta.columns]
    if available_bin:
        st.markdown("### Technology & Household Indicators")
        cols = st.columns(len(available_bin))
        for i, col_name in enumerate(available_bin):
            with cols[i]:
                pct = (cluster_meta[col_name].str.lower() == "yes").mean() * 100
                pop_pct = (meta[col_name].str.lower() == "yes").mean() * 100
                delta = pct - pop_pct
                st.metric(
                    col_name.replace("_", " ").title(),
                    f"{pct:.0f}%",
                    delta=f"{delta:+.0f} percentage points vs population",
                    delta_color="normal",
                )


# ═══════════════════════════════════════════════════════════════
# PAGE: Compare Profiles
# ═══════════════════════════════════════════════════════════════
elif page == "Compare Profiles":
    st.title("📊 Compare Energy-Use Profiles")

    options = {int(cid): f"Profile {cid} — {llm[cid]['name']}" for cid in cluster_ids}
    selected_profiles = st.multiselect(
        "Select profiles to compare",
        list(options.keys()),
        default=[0, 1, 5],
        format_func=lambda x: options[x],
    )

    if len(selected_profiles) < 2:
        st.warning("Please select at least 2 profiles to compare.")
    else:
        # Daily curves comparison 
        st.subheader("Daily Consumption Curves")
        fuel = st.radio("Fuel", ["Electricity", "Gas"], horizontal=True, key="compare_fuel")
        col_name = "electricity_kWh" if fuel == "Electricity" else "gas_kWh"

        filtered = hourly[hourly["cluster"].isin(selected_profiles)]
        fig = px.line(
            filtered, x="hour", y=col_name, color="cluster",
            color_discrete_map=color_map,
            labels={"hour": "Hour of Day", col_name: f"{fuel} (kWh)", "cluster": "Profile"},
        )
        fig.update_layout(xaxis=dict(dtick=2))
        st.plotly_chart(fig, use_container_width=True)

        #  Radar overlay
        st.subheader("Feature Fingerprint Comparison")
        radar_features = [
            "avg_daily_electricity_usage", "avg_daily_gas_usage",
            "evening_peak_share_electricity", "prop_daily_electricity_overnight",
            "ratio_winter_summer_electricity_usage", "weekend_weekday_ratio_electricity",
            "peak_to_avg_electricity", "std_daily_electricity_usage",
        ]
        available = [f for f in radar_features if f in profiles_scaled.columns]
        labels = [f.replace("_", " ").title() for f in available]

        fig = go.Figure()
        for cid in selected_profiles:
            vals = profiles_scaled.loc[cid, available].values.tolist()
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=labels + [labels[0]],
                fill="toself", name=f"Profile {cid} — {llm[str(cid)]['name']}",
                line=dict(color=color_map[cid], width=2),
                opacity=0.6,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-2.5, 3])),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        #  Side-by-side metadata comparison
        st.subheader("Metadata Comparison")
        compare_col = st.selectbox(
            "Select characteristic",
            ["property_type", "heating_type", "income_band", "occupancy_pattern",
             "insulation_quality", "tenure_type", "cooking_fuel"],
        )

        compare_data = []
        for cid in selected_profiles:
            cluster_meta = meta[meta["cluster"] == cid]
            dist = cluster_meta[compare_col].value_counts(normalize=True).reset_index()
            dist.columns = [compare_col, "share"]
            dist["share"] = (dist["share"] * 100).round(1)
            dist["Profile"] = f"{cid} — {llm[str(cid)]['name']}"
            compare_data.append(dist)

        compare_df = pd.concat(compare_data)
        fig = px.bar(
            compare_df, x="share", y=compare_col, color="Profile",
            orientation="h", barmode="group",
            color_discrete_sequence=COLORS,
            labels={"share": "%"},
            title=compare_col.replace("_", " ").title(),
        )
        fig.update_layout(yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: Agent Transparency
# ═══════════════════════════════════════════════════════════════
elif page == "Agent Transparency":
    st.title("🤖 Multi-Agent Naming Transparency")
    st.markdown(
        "Three role-specialized agents independently proposed segment names. "
        "A Judge agent then synthesized the final consensus name for each profile."
    )

    # Build comparison table
    rows = []
    for cid in cluster_ids:
        row = {"Profile": int(cid)}
        for agent_name, agent_props in proposals.items():
            row[agent_name] = agent_props.get(cid, "—")
        entry = llm[cid]
        row["Final Name"] = entry["name"]
        row["Judge Rationale"] = entry.get("rationale", "")
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Individual profile deep dive
    st.markdown("---")
    st.subheader("Naming Rationale per Profile")
    for cid in cluster_ids:
        d = llm[cid]
        with st.expander(f"Profile {cid} — **{d['name']}** ({d['size']} HH)"):
            st.markdown(f"**Judge rationale:** {d.get('rationale', 'N/A')}")
            st.markdown("**Agent proposals:**")
            for agent_name, agent_props in proposals.items():
                st.markdown(f"- *{agent_name}*: {agent_props.get(cid, '—')}")
