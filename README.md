# Customer Segmentation from Smart-Meter Energy Data

A hybrid customer segmentation framework that combines **unsupervised clustering** of smart-meter time-series data with a **multi-agent LLM architecture** for automated segment naming and interpretation 
## Objective

Energy utilities need to understand how different households consume energy to design targeted interventions (tariffs, demand-response, retrofit programmes). Traditional clustering produces statistically valid groups but often lacks interpretability.

This project addresses that gap by:

1. Extracting **behavioral features** from hourly smart-meter data (magnitude, time-of-day, seasonality, variability)
2. Applying **K-Means clustering** to identify homogeneous consumption groups
3. Using a **role-specialized multi-agent LLM pipeline** to generate stable segment names and rich, data-driven profile interpretations

**Key design principle:** Profiles are built from **energy consumption only**. Household metadata (property type, income, heating system, etc.) is brought in *after* clustering to enrich interpretation, never to influence the grouping itself.

## Results

K-Means with **K = 7** (silhouette score = 0.473) segments 500 households into three macro-groups:

| Macro-group | Clusters | Households | Defining characteristic |
|-------------|----------|------------|------------------------|
| Gas-heated | 0, 3, 4, 6 | 309 (61.8%) | Rely on gas for heating (winter/summer gas ratio ≈ 12×) |
| All-electric | 1, 2 | 159 (31.8%) | Zero gas; electric heating drives high seasonal electricity (7×) |
| Low-energy no-gas | 5 | 32 (6.4%) | Minimal consumption on both fuels |

Within these groups, clusters are further differentiated by consumption magnitude, time-of-day shape, and day-to-day variability.

## Multi-Agent LLM Architecture

Five agents collaborate across two stages:

**Stage 1 — Segment Naming**
| Agent | Role |
|-------|------|
| Energy Domain Expert | Utility-sector naming conventions |
| Behavioral Analyst | Lifestyle and behavioral patterns |
| Data Storyteller | Non-technical stakeholder communication |
| **Judge** | Synthesizes the best consensus name per cluster |

**Stage 2 — Profile Interpretation**
| Agent | Role |
|-------|------|
| Interpreter | Produces structured narratives (headline finding, energy pattern, key characteristics, policy implications) |

The interpreter prompt follows the [Nesta energy-use profiles](https://www.nesta.org.uk/report/understanding-gb-energy-consumption-patterns/) report format , each profile includes a bold headline finding, data-driven narrative, distinguishing bullet points, and actionable strategy recommendations.

## Notebook Structure

| Section | Description |
|---------|-------------|
| **1. EDA** | Hourly/daily/monthly consumption distributions, time-of-day patterns, metadata overview |
| **2. Feature Engineering** | 25+ features: magnitude, time-of-day breakdown, behavioral ratios, seasonality, variability, gas flag |
| **3. Preprocessing** | Skewness correction (Box-Cox, log1p, sqrt), standardization (StandardScaler), PCA (5 components, 96.95% variance) |
| **4. Clustering** | Silhouette analysis (K=2–10), K-Means with K=7, cluster profiling, radar charts, metadata cross-tabulation |
| **5. Agent-Based Interpretation** | OpenAI API (gpt-4o-mini), multi-agent naming consensus, Nesta-style profile interpretations |

## Data

Synthetic dataset simulating a UK smart-meter rollout:

- **`hope_city_hourly_consumption_v2_2025.csv.gz`** — 500 households × 8,760 hours (4.38M rows), with `electricity_kWh` and `gas_kWh` columns
- **`hope_city_households_metadata.csv`** — property type, construction age, floor area, insulation quality, tenure, occupants, income band, occupancy pattern, heating type, cooling system, cooking fuel, solar PV, EV ownership
- **`hope_city_dataset_generator.py`** — script to regenerate the synthetic data

## Setup

### Requirements

```
Python 3.10+
```

### Install dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly openai
```

### OpenAI API key (for Section 5 only)

Section 5 requires an [OpenAI API key](https://platform.openai.com/api-keys). Set it as an environment variable or enter it when prompted:

```bash
export OPENAI_API_KEY="sk-..."
```

The full pipeline (11 API calls with `gpt-4o-mini`) costs under **$0.01**.

### Run the notebook

Open `clustering.ipynb` in Jupyter or VS Code and run all cells sequentially. Sections 1–4 run without any API key. Section 5 requires the OpenAI key. Section 6 exports all artifacts for the dashboard.

### Run the Streamlit dashboard

The dashboard loads pre-computed artifacts — **no API key needed** at runtime.

```bash
# Install dashboard dependencies
pip install -r requirements.txt

# First, run the notebook through Section 6 to generate artifacts/
# Then launch the dashboard:
streamlit run app.py
```

The app has four pages:
- **Overview** — summary table, cluster distribution, daily consumption curves
- **Explore a Profile** — deep-dive into a single profile with LLM interpretation, daily curves, radar chart, metadata breakdown
- **Compare Profiles** — side-by-side comparison of selected profiles
- **Agent Transparency** — see how the 3 naming agents and the judge arrived at each segment name

## Project Structure

```
Customer Segmentation/
├── clustering.ipynb                  # Main notebook (116 cells)
├── app.py                            # Streamlit dashboard
├── requirements.txt                  # Dashboard dependencies
├── README.md
├── artifacts/                        # Generated by notebook Section 6
│   ├── cluster_assignments.csv
│   ├── cluster_profiles.csv
│   ├── cluster_profiles_scaled.csv
│   ├── hourly_by_cluster.csv
│   ├── metadata_with_clusters.csv
│   ├── llm_results.json
│   └── agent_proposals.json
└── dataset/
    ├── hope_city_hourly_consumption_v2_2025.csv.gz
    ├── hope_city_households_metadata.csv
    └── hope_city_dataset_generator.py
```

## Deployment

The Streamlit app can be deployed for free on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push the repo (including `artifacts/`) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app"
3. Point to `app.py` in your repo
4. Deploy — no API keys or secrets needed

## Acknowledgements

- Synthetic data generated to simulate patterns observed in UK smart-meter datasets


