# CityWatch: product brief

## Problem

Municipal road and asset maintenance is mostly **reactive**. Defects are found by
patrols or complaint calls, triaged by hand, and repaired in arrival order. Two
things are missing:

1. **Fast, structured intake**: residents see problems first, but complaint lines
   capture vague locations and no photo, so staff re-inspect before acting.
2. **Forward-looking prioritisation**: crews have no view of *where failures are
   likely next*, so preventive work (crack sealing before freeze-thaw season,
   sign replacement) is hard to target.

## Users

| User | Job to be done |
|---|---|
| Resident | Report a problem in under 30 seconds and see that it was handled |
| City operations staff | Triage incoming reports, dispatch crews, close the loop |
| Asset / capital planner | Decide which neighbourhoods and assets get preventive budget |

## Solution scope (v2)

| Capability | How |
|---|---|
| Photo + GPS reporting | Expo app; upload goes through the API (no keys on device) |
| Automatic triage | MobileNetV2 classifier (10 issue classes) + severity score |
| Context enrichment | Weather at report time (Open-Meteo), nearest asset (PostGIS KNN) |
| De-duplication | Reports on an asset with an open incident join that incident |
| Status workflow | open → in_progress → resolved / rejected, staff-only |
| Risk forecasting | Random Forest per horizon (30/90/365 days) per asset, from ~125 point-in-time features |
| Region view | 100 H3 hexagons over Waterloo, ranked by forecast risk |
| Engagement | Leaderboard of resident reporters |

**Out of scope for v2:** work-order integration (e.g. Cityworks), crew routing,
real (non-simulated) failure history, staff web console.

## Success metrics

| Metric | Definition | Where it is measured |
|---|---|---|
| Triage accuracy | CV test accuracy / macro-F1 on held-out images | `ml/artifacts/cv_metrics.json` |
| Forecast lift | Recall of next-30-day failures when inspecting the top 10% of assets, vs. ranking by past incidents | `ml/artifacts/risk_metrics.json` |
| Time to resolve | Median days from report to `resolved` (last 90 days) | `GET /api/stats` |
| Intake volume | Reports in the last 7 days | `GET /api/stats` |
| Duplicate rate | Share of reports that attach to an existing open incident | `reports.incident_id` |

## Key decisions

| Decision | Why |
|---|---|
| Single FastAPI service instead of Express + Python sidecar | One language for ML and API; removes a network hop per report; the CV model and feature code are imported directly |
| Postgres + PostGIS as the only datastore | Spatial joins (nearest asset, radius queries) run in the database with GIST indexes; works locally and on Supabase unchanged |
| H3 hexagons for regions | Equal-area cells make region risk comparable; stable ids for caching and URLs |
| Time-based, purged evaluation split | Random splits leak future information into training and overstate accuracy |
| Simulated failure history, clearly flagged | No public per-asset failure data exists for Waterloo; simulation is driven by real assets and real weather, and every row carries `is_simulated = true` |
| Heuristic severity fallback | The workflow must not depend on a paid LLM key; Gemini is an optional upgrade |

## Roadmap

1. **Now (v2)**: everything above, CI, reproducible data + training pipeline.
2. **Next**: staff web console for triage; push notifications on status change;
   Auth0 JWT verification on the API (replacing the shared staff token).
3. **Later**: ingest real City of Waterloo service-request data to replace the
   simulated history; recalibrate the risk models on it; add crew scheduling
   recommendations using forecast risk × repair cost.

## Risks

| Risk | Mitigation |
|---|---|
| Model trained on simulated failures may not transfer | Pipeline is data-agnostic; retrain on real service requests when available. Metrics are reported as simulation results |
| Misclassified photos create bad incidents | Only asset-failure classes open incidents; staff can reject; confidence is stored |
| Public Overpass / Open-Meteo rate limits | Mirror fallback for Overpass; weather responses cached per hour and location |
