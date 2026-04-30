import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import json
import os

st.set_page_config(
    page_title="Zerve Analytics — Hackathon",
    layout="wide"
)

# ── Load data from files ──────────────────────────────────────────────────────
@st.cache_data
def load_funnel():
    df = pd.read_parquet("funnel.parquet")
    if 'at_risk' not in df.columns:
        df['at_risk'] = (df['stage_number'] == 9).astype(int)
    if 'main_stage_name' not in df.columns:
        df['main_stage_name'] = df['stage_name']
    return df

# ── Constants ─────────────────────────────────────────────────────────────────
EVER_REACHED = {
    'New':                  17541,
    'Exploring':            15471,
    'Activated':             8147,
    'Wrote Code':            5576,
    'Integrated':            4806,
    'Consistently Engaged':  3637,
    'Upgrade Intent':        3139,
    'Upgraded':               854,
    'At Risk':               2403,
}
CONV_RATES = {
    'New':                   4.9,
    'Exploring':             5.5,
    'Activated':            10.5,
    'Wrote Code':           15.3,
    'Integrated':           17.8,
    'Consistently Engaged': 23.5,
    'Upgrade Intent':       27.2,
    'Upgraded':            100.0,
    'At Risk':              16.6,
}
STAGE_ACTIONS = {
    'New':                  'Nudge toward first sign-in or page view',
    'Exploring':            'Trigger first AI interaction or content creation',
    'Activated':            'Guide toward running their first block of code',
    'Wrote Code':           'Encourage file upload or dataset connection',
    'Integrated':           'Highlight benefits of returning on multiple days',
    'Consistently Engaged': 'Trigger upgrade nudge — high conversion stage',
    'Upgrade Intent':       'Reduce checkout friction — offer trial extension',
    'Upgraded':             'Focus on retention and feature expansion',
    'At Risk':              'Re-engage within 7 days — personalized outreach',
}
STAGE_ORDER = [
    'New','Exploring','Activated','Wrote Code',
    'Integrated','Consistently Engaged','Upgrade Intent','Upgraded'
]
STAGE_COLORS = {
    'New':                  '#95a5a6',
    'Exploring':            '#3498db',
    'Activated':            '#2ecc71',
    'Wrote Code':           '#f1c40f',
    'Integrated':           '#e67e22',
    'Consistently Engaged': '#e74c3c',
    'Upgrade Intent':       '#9b59b6',
    'Upgraded':             '#1abc9c',
    'At Risk':              '#c0392b',
}
STAGE_NUM_NAMES = {
    0:'New', 1:'Exploring', 2:'Activated', 3:'Wrote Code',
    4:'Integrated', 5:'Consistently Engaged',
    6:'Upgrade Intent', 7:'Upgraded'
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.header-bar {
    background: linear-gradient(90deg, #1a1a2e 0%, #0f3460 100%);
    padding: 20px 30px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.header-title {
    font-size: 28px;
    font-weight: 900;
    color: white;
    margin: 0;
}
.header-sub {
    font-size: 13px;
    color: rgba(255,255,255,0.65);
    margin-top: 4px;
}
.metric-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 12px;
    color: white;
    display: inline-block;
    margin: 2px;
}
.metric-pill span {
    color: #2ecc71;
    font-weight: 800;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-bar">
    <div class="header-title"> Zerve Analytics</div>
    <div class="header-sub">ODSC Hackathon · April 2026 · Data Science Team</div>
    <div style="margin-top:10px;">
        <span class="metric-pill">AUC-PR <span>0.2122</span></span>
        <span class="metric-pill">Lift <span>4.0x</span></span>
        <span class="metric-pill">Recall@0.30 <span>71%</span></span>
        <span class="metric-pill">Recall@0.10 <span>97.7%</span></span>
        <span class="metric-pill">Users <span>17,541</span></span>
        <span class="metric-pill">Upgraded <span>854</span></span>
        <span class="metric-pill">At Risk <span>2,005</span></span>
        <span class="metric-pill">Funnel stages <span>8 + At Risk</span></span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Page navigation buttons ───────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'funnel'

col_btn1, col_btn2, col_spacer = st.columns([1, 1, 4])
with col_btn1:
    if st.button(
        " Funnel Overview",
        type="primary" if st.session_state.page == 'funnel' else "secondary",
        use_container_width=True
    ):
        st.session_state.page = 'funnel'
        st.rerun()

with col_btn2:
    if st.button(
        " Upgrade Prediction Model",
        type="primary" if st.session_state.page == 'model' else "secondary",
        use_container_width=True
    ):
        st.session_state.page = 'model'
        st.rerun()

st.markdown("---")
page = st.session_state.page

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FUNNEL OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == 'funnel':
    st.title(" Zerve User Funnel")
    st.markdown(
        "Every user is in **exactly one stage** at any point in time. "
        "Stages are grounded in observed behavior — not assumptions. "
        "Data shows **55% use AI before creating content** and "
        "**45% create content first** — therefore both are merged into "
        "a single **Activated** stage."
    )
    st.markdown("---")

    funnel_df = load_funnel()
    total     = len(funnel_df)
    upgraded  = int(funnel_df['is_upgrader'].sum())
    at_risk   = int((funnel_df['stage_name']=='At Risk').sum())
    intent    = int((funnel_df['stage_name']=='Upgrade Intent').sum())
    exploring = int((funnel_df['stage_name']=='Exploring').sum())
    activated = int((funnel_df['stage_name']=='Activated').sum())

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Users",     f"{total:,}")
    c2.metric("Upgraded",        f"{upgraded:,}",
              f"{upgraded/total*100:.1f}% conversion")
    c3.metric("At Risk",         f"{at_risk:,}",
              f"{at_risk/total*100:.1f}% of users")
    c4.metric("Upgrade Intent",  f"{intent:,}",
              f"{intent/total*100:.1f}% of users")
    c5.metric("Activation Rate", f"{activated/(exploring+activated)*100:.1f}%",
              "Exploring → Activated")

    st.markdown("---")

    # ── Funnel shape ──────────────────────────────────────────────────────────
    st.subheader("Funnel Shape — Ever Reached per Stage")
    st.caption(
        "Width = users who ever reached this stage. "
        "Conv% = eventual upgrade rate."
    )

    fig_funnel = go.Figure(go.Funnel(
        y            = STAGE_ORDER,
        x            = [EVER_REACHED[s] for s in STAGE_ORDER],
        textinfo     = "value+percent initial",
        texttemplate = "%{value:,}<br>%{percentInitial:.1%}",
        marker       = dict(
            color = [STAGE_COLORS[s] for s in STAGE_ORDER],
            line  = dict(width=2, color='white')
        ),
        connector = dict(line=dict(color='#bdc3c7', width=2))
    ))
    fig_funnel.update_layout(
        height=500,
        margin=dict(l=20,r=20,t=20,b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    # ── Funnel table ──────────────────────────────────────────────────────────
    st.subheader("Funnel Summary Table")
    summary = (
        funnel_df.groupby(['stage_number','stage_name'])
        .agg(current_users=('person_id','count'),
             upgraders=('is_upgrader','sum'))
        .reset_index().sort_values('stage_number')
    )
    summary['pct_total']    = (summary['current_users']/total*100).round(1)
    summary['ever_reached'] = summary['stage_name'].map(EVER_REACHED)
    summary['conv_pct']     = summary['stage_name'].map(CONV_RATES)
    summary['action']       = summary['stage_name'].map(STAGE_ACTIONS)

    display = summary[['stage_number','stage_name','current_users',
                        'pct_total','ever_reached','conv_pct','action']].copy()
    display.columns = ['Stage #','Stage','Current Users',
                        '% Total','Ever Reached','Conv %','Recommended Action']
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("---")

    
    # ── Key insights ──────────────────────────────────────────────────────────
    st.subheader("💡 Key Business Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
**1. Activation Cliff**
{exploring:,} exploring → only {activated:,} activated **(49% drop)**

First AI or content creation doubles upgrade likelihood: **5.5% → 10.5%**

→ *Stronger onboarding nudge to first AI use*
        """)
        st.info(f"""
**3. Conversion Grows Monotonically**
**4.9% (New) → 27.2% (Upgrade Intent)**

Every stage advancement increases upgrade probability.
Integrated users convert at **17.8% vs 5.5%** for explorers.

→ *Push users through each milestone systematically*
        """)
    with col2:
        st.info(f"""
**2. Upgrade Intent Gap**
{intent:,} showed intent → only {upgraded:,} converted **(27.2%)**

**717 high-intent users abandoned** — checkout friction is the gap.

→ *Reduce friction, offer trial extension at upgrade click*
        """)
        st.info(f"""
**4. Credit Wall Signal**
548 users hit credit limit but **never clicked upgrade**

Warm leads stopped at pricing — most actionable re-engagement target.

→ *Personalized email within 7 days of credits_exceeded*
        """)

    st.markdown("---")
    with st.expander("📋 Stage Transition Rules — deterministic, engineer-implementable"):
        st.markdown("""
| Transition | Rule |
|------------|------|
| 0→1 New → Exploring | `$pageview` OR `submit_onboarding_form` OR `skip_onboarding_form` OR `sign_in` |
| 1→2 Exploring → Activated | `canvas_create` OR `block_create` OR `$ai_generation` OR `agent_new_chat` — whichever fires first |
| 2→3 Activated → Wrote Code | `run_block` OR `run_all_blocks` |
| 3→4 Wrote Code → Integrated | `files_upload` OR `quickstart_add_dataset` OR `notebook_deployment_created` |
| 4→5 Integrated → Consistently Engaged | `active_calendar_days >= 3` AND `$ai_generation` or `agent_message` ever fired |
| 5→6 Consistently Engaged → Upgrade Intent | `clicked_upgrade` OR `claim_free_offer_clicked` OR `referral_modal_open` |
| 6→7 Upgrade Intent → Upgraded | `upgrade_subscription` fires |
| 2-6→9 Any engaged → At Risk | 14d inactivity AND days_active≥3, OR credits_exceeded AND no upgrade click, OR sessions drop 50%+ |

**Design note:** Content creation and AI use merged into Activated because
data shows 55% use AI first, 45% create content first — not a strict sequence.
        """)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL (fully static)
# ═════════════════════════════════════════════════════════════════════════════
elif page == 'model':
    st.title("Upgrade Prediction Model")
    st.markdown(
        "XGBoost model trained to predict which Zerve users will upgrade "
        "to a paid plan — based purely on behavioral signals with no leakage."
    )
    st.markdown("---")

    # ── Why AUC-PR ────────────────────────────────────────────────────────────
    st.subheader(" Why AUC-PR? The Right Metric for This Problem")
    st.markdown("""
The dataset has a **5.2% upgrade rate** — 854 upgraders out of 17,541 users.
This severe class imbalance makes ROC-AUC misleading:

- **ROC-AUC** inflates because it gives equal weight to true negatives —
  correctly classifying the easy 95% non-upgraders boosts the score artificially
- **AUC-PR** only evaluates performance on the positive class — upgraders
- A random classifier gets AUC-PR = **0.053** (the base rate)
- Our model gets **0.2122 — a 4.0x lift over random**

We optimize **Recall** over Precision because the cost of a false positive
is near zero (just an email nudge), while missing a real upgrader is costly.
    """)

    why1, why2, why3 = st.columns(3)
    why1.metric("Baseline AUC-PR", "0.053",  "Random classifier")
    why2.metric("Our AUC-PR",      "0.2122", "+0.159 improvement")
    why3.metric("Lift",            "4.0x",   "Better than random")

    st.markdown("---")

    # ── Model architecture ────────────────────────────────────────────────────
    st.subheader("🏗️ Model Architecture & Design Decisions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Algorithm:** XGBoost Config B + Enhanced Features
- `n_estimators`: 1000 (best iteration: 993)
- `learning_rate`: 0.01
- `max_depth`: 4
- `min_child_weight`: 10
- `gamma`: 1
- `scale_pos_weight`: 10
- `subsample`: 0.8
- `colsample_bytree`: 0.7
- `reg_alpha`: 0.1 · `reg_lambda`: 1.0
        """)
        st.markdown("""
**Why these choices?**
- `scale_pos_weight=10` handles 19:1 imbalance natively
  — better than SMOTE which caused probability compression
- `max_depth=4` prevents overfitting on 9K training samples
- `min_child_weight=10` requires more samples per leaf
- `gamma=1` minimum loss reduction — avoids noisy splits
- No calibration — raw probabilities were more stable
        """)
    with col2:
        st.markdown("""
**Split strategy:** Stratified temporal 60/20/20
- Users sorted chronologically by last event timestamp
- Every 5th → val, every 5th+1 → test
- ~5% upgraders in all three splits
- Val used ONLY for threshold tuning
- Test set truly unseen throughout all experiments
        """)
        st.markdown("""
**Leakage prevention — excluded from all features:**
- `upgrade_subscription` — the target variable itself
- `clicked_upgrade` — fires at the upgrade decision point
- `claim_free_offer_clicked` — fires at upgrade decision
- `offer_declined` — fires at upgrade decision
- `credits_awarded`, `credits_received` — post-upgrade events

Applied per-user temporal cutoff: only events before
upgrade timestamp used for each upgrading user.
        """)

    st.markdown("---")

    # ── Performance metrics ───────────────────────────────────────────────────
    st.subheader(" Model Performance — Truly Unseen Test Set")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC-PR",      "0.2122", "4.0x lift over 0.053")
    m2.metric("Recall@0.10", "97.7%",  "Catches 97.7% of upgraders")
    m3.metric("Recall@0.30", "71.0%",  "At email campaign threshold")
    m4.metric("F2 Score",    "0.321",  "Recall-weighted metric")
    m5.metric("Best iter",   "993",    "of 1000 trees")

    st.markdown("---")

    # ── Threshold chart ───────────────────────────────────────────────────────
    st.subheader(" Recall & F2 vs Threshold")
    st.caption("Tuned on val set only — test set never seen during threshold selection")

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    recalls    = [0.977, 0.847, 0.710, 0.449, 0.290, 0.222]
    f2s        = [0.251, 0.285, 0.332, 0.310, 0.275, 0.242]
    flagged    = [2719, 1907, 1176, 569, 223, 102]

    fig_thresh = go.Figure()
    fig_thresh.add_trace(go.Scatter(
        x=thresholds, y=recalls,
        name='Recall', mode='lines+markers',
        line=dict(color='#3498db', width=3),
        marker=dict(size=10)
    ))
    fig_thresh.add_trace(go.Scatter(
        x=thresholds, y=f2s,
        name='F2 Score', mode='lines+markers',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=10)
    ))
    fig_thresh.add_vline(
        x=0.30, line_dash='dash', line_color='#f39c12', line_width=2,
        annotation_text='Email t=0.30 → 71% recall',
        annotation_position='top right'
    )
    fig_thresh.add_vline(
        x=0.10, line_dash='dash', line_color='#2ecc71', line_width=2,
        annotation_text='Broad t=0.10 → 97.7% recall',
        annotation_position='top left'
    )
    fig_thresh.update_layout(
        height=380,
        xaxis_title='Threshold',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.15], gridcolor='#ecf0f1'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=20,r=20,t=50,b=20)
    )
    st.plotly_chart(fig_thresh, use_container_width=True)

    # ── Operating points table ────────────────────────────────────────────────
    st.subheader("Operating Points Table")
    threshold_data = pd.DataFrame({
        'Threshold':    thresholds,
        'Recall':       [f"{r:.1%}" for r in recalls],
        'F2':           [f"{f:.3f}" for f in f2s],
        'Users flagged':[f"{u:,}"   for u in flagged],
        'Upgraders caught (of 176)': [int(r*176) for r in recalls],
        'Recommended use': [
            'Broad email campaign — max recall',
            'In-app banner',
            'Targeted email — best balance',
            'Priority outreach',
            'High-touch sales',
            'Personal outreach — high precision'
        ]
    })
    st.dataframe(threshold_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Model comparison ──────────────────────────────────────────────────────
    st.subheader("📈 Model Comparison — Config B vs Config B + Enhanced Features")
    st.caption("Both evaluated on truly unseen test set — no leakage")

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("""
**Config B — Base model**
- scale_pos_weight: 10
- Features: 50 base behavioral signals
- AUC-PR: **0.2037** (3.8x lift)
- Recall@0.30: **68.2%**
- Recall@0.10: **97.7%**
- F2: **0.325**
- Best iteration: 999
        """)
    with comp_col2:
        st.markdown("""
**Config B + Enhanced Features — Final model ✅**
- scale_pos_weight: 10 (same)
- Features: 56 signals (6 new added)
- AUC-PR: **0.2122** (4.0x lift) ↑
- Recall@0.30: **71.0%** ↑
- Recall@0.10: **97.7%** (same)
- F2: **0.321**
- Best iteration: 993
        """)

    comparison = pd.DataFrame({
        'Model': [
            'Random baseline',
            'GBM stratified temporal',
            'XGBoost Config A (spw=5)',
            'XGBoost Config B (spw=10)',
            'XGBoost Optuna 50 trials',
            'Config B + Enhanced Features ✅'
        ],
        'AUC-PR':      [0.053, 0.197, 0.200, 0.204, 0.187, 0.212],
        'Recall@0.30': ['-','0.136','0.386','0.682','0.352','0.710'],
        'Notes': [
            'Baseline — random classifier',
            'Earlier GBM attempt',
            'Too conservative — low scale_pos_weight',
            'Best manual config',
            'Optuna overfit to val set — worse on test',
            'Final — best on all metrics'
        ]
    })

    colors_comp = ['#95a5a6','#bdc3c7','#f39c12',
                   '#e67e22','#e74c3c','#2ecc71']
    fig_comp = go.Figure(go.Bar(
        x            = comparison['Model'],
        y            = comparison['AUC-PR'],
        marker_color = colors_comp,
        text         = [f"{v:.3f}" for v in comparison['AUC-PR']],
        textposition = 'outside'
    ))
    fig_comp.add_hline(
        y=0.053, line_dash='dash', line_color='gray', line_width=1.5,
        annotation_text='Random baseline (0.053)'
    )
    fig_comp.update_layout(
        height       = 420,
        yaxis_title  = 'AUC-PR (higher = better)',
        yaxis        = dict(range=[0, 0.26], gridcolor='#ecf0f1'),
        paper_bgcolor= 'rgba(0,0,0,0)',
        plot_bgcolor = 'rgba(0,0,0,0)',
        showlegend   = False,
        margin       = dict(l=20,r=20,t=20,b=130),
        xaxis        = dict(tickangle=-15)
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Feature importance (hardcoded) ────────────────────────────────────────
    st.subheader("🔑 Feature Importance — Top 20 Signals")
    st.caption("XGBoost gain-based importance from final model")

    feat_data = pd.DataFrame({
        'Feature': [
            'hit_exceeded','distinct_event_types','n_credits_exceeded',
            'ai_output_tokens','tool_calls_per_ai_call','unique_events',
            'n_canvas_created','activity_x_ai','credits_x_ai',
            'completed_onboarding','events_last_7d','avg_session_depth',
            'n_tool_calls','countdown_depth','agent_event_ratio',
            'n_ai_calls','device_type','n_run_all',
            'n_blocks_created','ai_input_tokens'
        ],
        'Importance': [
            0.051121, 0.035587, 0.033730,
            0.030469, 0.030461, 0.030095,
            0.028412, 0.026962, 0.023954,
            0.022711, 0.022103, 0.021831,
            0.021753, 0.021496, 0.020390,
            0.020304, 0.019805, 0.018565,
            0.018326, 0.018263
        ]
    })

    fig_imp = px.bar(
        feat_data, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='RdYlGn'
    )
    fig_imp.update_layout(
        height=580,
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False,
        margin=dict(l=20,r=20,t=20,b=20),
        xaxis=dict(gridcolor='#ecf0f1')
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Feature explanations ──────────────────────────────────────────────────
    st.subheader("💡 Top Feature Explanations")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
**🥇 hit_exceeded (rank 1 · 0.051)**
User exhausted ALL credits. Strongest single signal —
hitting a hard wall without upgrading is the clearest
indicator of willingness to pay.
        """)
        st.info("""
**🥉 n_credits_exceeded (rank 3 · 0.034)**
How many times the user hit the credit limit.
Repeated exhaustion = persistent high demand.
These users keep coming back despite the restriction.
        """)
        st.info("""
**5th tool_calls_per_ai_call (0.030)**
Tools invoked per AI call — power users who invoke
multiple tools per AI request extract deep value.
Strong signal of platform investment.
        """)
    with col2:
        st.info("""
**🥈 distinct_event_types (rank 2 · 0.036)**
Unique event types ever fired — breadth of exploration.
Users who try more features are more invested.
New enhanced feature — jumped straight to rank 2.
        """)
        st.info("""
**4th ai_output_tokens (0.030)**
Total AI output tokens consumed.
More tokens = more substantive AI interactions.
Users getting real value from AI features.
        """)
        st.info("""
**credits_x_ai interaction (0.024)**
Credit usage × AI calls combined.
Captures the most upgrade-ready profile:
heavy AI users who also hit credit limits.
        """)

    st.markdown("---")

    # ── Why this model wins ───────────────────────────────────────────────────
    st.subheader("🏆 Why This Model?")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.success("""
** No SMOTE**

SMOTE caused probability compression — scores collapsed
near zero making threshold decisions unreliable.

`scale_pos_weight=10` handles imbalance cleanly inside
XGBoost without distorting probability distribution.
        """)
    with w2:
        st.success("""
**No Calibration**

Isotonic calibration on val set destroyed probability spread.
Raw XGBoost probabilities with proper hyperparameters
were more stable, better distributed, and more actionable.
        """)
    with w3:
        st.success("""
** Strict Temporal Split**

Users sorted by last event before splitting — prevents
temporal leakage. Val and test reflect real deployment
conditions. Test set never seen during any design decision.
        """)

    st.markdown("---")

    # ── Enhanced features ─────────────────────────────────────────────────────
    st.subheader("🔬 Enhanced Features — What Improved the Model")
    st.markdown(
        "Config B + Enhanced Features adds **6 new engineered features** "
        "improving AUC-PR from 0.2037 → 0.2122 (+4.2% relative improvement):"
    )
    enh_data = pd.DataFrame({
        'Feature':    ['distinct_event_types','agent_event_ratio',
                       'peak_session_length','hours_to_first_ai',
                       'returned_day2','ever_used_ai'],
        'Rank':       [2, 15, 25, 45, 52, 56],
        'Importance': [0.0356, 0.0204, 0.0168, 0.0132, 0.0093, 0.0000],
        'Description':['Breadth of platform use — unique event types fired',
                       'Fraction of events that are agent interactions',
                       'Max events in a single session — power user signal',
                       'Hours from first event to first AI use',
                       'Did user return on day 2 after signup',
                       'Did user ever use AI at all']
    })
    st.dataframe(enh_data, use_container_width=True, hide_index=True)
    st.caption(
        "distinct_event_types (rank 2) and agent_event_ratio (rank 15) "
        "were the main contributors. Model improved from AUC-PR 0.2037 → 0.2122 "
        "— a +0.0085 gain (+4.2% relative improvement)."
    )
