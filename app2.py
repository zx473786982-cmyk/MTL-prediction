"""
MTL 污泥电渗透脱水预测系统 —— Streamlit 网页预测系统
功能：用户输入污泥特性 + 操作参数 → 集成预测 FMC(%) 和 SEC(kWh/kg·RW)
"""

import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import streamlit as st

# ============================================================================
# 1. 模型定义（必须和训练代码完全一致）
# ============================================================================

class IFPNetwork(nn.Module):
    """IF-P 软参数共享多任务神经网络"""

    def __init__(self, input_dim, encoder_dim, fusion_dim, dropout):
        super().__init__()

        self.encoder_fmc = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.encoder_sec = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim * 2, fusion_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.head_fmc = nn.Sequential(
            nn.Linear(encoder_dim + fusion_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )
        self.head_sec = nn.Sequential(
            nn.Linear(encoder_dim + fusion_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        f_fmc   = self.encoder_fmc(x)
        f_sec   = self.encoder_sec(x)
        shared  = self.fusion(torch.cat([f_fmc, f_sec], dim=-1))
        out_fmc = self.head_fmc(torch.cat([f_fmc, shared], dim=-1)).squeeze(-1)
        out_sec = self.head_sec(torch.cat([f_sec, shared], dim=-1)).squeeze(-1)
        return out_fmc, out_sec


# ============================================================================
# 2. 配置
# ============================================================================

SEEDS = [20, 52, 111, 340, 888]
ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

CONTINUOUS_FEATURES = [
    'pH', 'IMC(%)', 'Zeta(mV)', 'EC(ms/cm)', 'VS/TS(%)',
    'PT(min)', 'AV(V)', 'SS(g)', 'MP(kPa)'
]

ST_OPTIONS  = ['AS', 'AGS', 'DS']
EFT_OPTIONS = ['EOD', 'PEOD', 'UEOD', 'IEOD', 'ACEO']

ST_LABELS = {
    'AS':  '活性污泥 (AS)',
    'AGS': '好氧颗粒污泥 (AGS)',
    'DS':  '厌氧消化污泥 (DS)',
    'OTHER': '其他污泥类型',
}
EFT_LABELS = {
    'EOD':  '电渗透脱水 (EOD)',
    'PEOD': '外加压力电渗透 (PEOD)',
    'UEOD': '恒压电渗透 (UEOD)',
    'IEOD': '恒流电渗透 (IEOD)',
    'ACEO': '交变电场电渗透 (ACEO)',
}

ST_OPTIONS_WITH_OTHER = ST_OPTIONS + ['OTHER']


# ============================================================================
# 3. 加载模型与预处理器（缓存，只加载一次）
# ============================================================================

@st.cache_resource
def load_all_models():
    """加载 5 个 seed 的模型和预处理器"""
    models_info = []

    for seed in SEEDS:
        ohe_st     = joblib.load(os.path.join(ARTIFACT_DIR, f"ohe_st_seed_{seed}.pkl"))
        ohe_eft    = joblib.load(os.path.join(ARTIFACT_DIR, f"ohe_eft_seed_{seed}.pkl"))
        scaler_X   = joblib.load(os.path.join(ARTIFACT_DIR, f"scaler_X_seed_{seed}.pkl"))
        scaler_fmc = joblib.load(os.path.join(ARTIFACT_DIR, f"scaler_fmc_seed_{seed}.pkl"))
        scaler_sec = joblib.load(os.path.join(ARTIFACT_DIR, f"scaler_sec_seed_{seed}.pkl"))

        n_cont    = len(CONTINUOUS_FEATURES)
        n_st_cat  = len(ohe_st.categories_[0])
        n_eft_cat = len(ohe_eft.categories_[0])
        input_dim = n_cont + n_st_cat + n_eft_cat

        state_dict = torch.load(
            os.path.join(ARTIFACT_DIR, f"IFP_model_seed_{seed}.pth"),
            map_location="cpu",
            weights_only=True,
        )

        encoder_dim = state_dict["encoder_fmc.0.weight"].shape[0]
        fusion_dim  = state_dict["fusion.0.weight"].shape[0]
        dropout = 0.0

        model = IFPNetwork(input_dim, encoder_dim, fusion_dim, dropout)
        model.load_state_dict(state_dict)
        model.eval()

        models_info.append({
            "seed":       seed,
            "model":      model,
            "ohe_st":     ohe_st,
            "ohe_eft":    ohe_eft,
            "scaler_X":   scaler_X,
            "scaler_fmc": scaler_fmc,
            "scaler_sec": scaler_sec,
        })

    return models_info


# ============================================================================
# 4. 预测函数（返回每个 seed 的独立预测，用于计算范围）
# ============================================================================

def predict_ensemble_with_range(models_info, features_cont, st_value, eft_value):
    """
    5 个 seed 模型分别预测，返回均值和各 seed 的预测数组
    """
    fmc_preds = []
    sec_preds = []

    for info in models_info:
        cont_arr = np.array(features_cont, dtype=np.float32).reshape(1, -1)
        st_ohe   = info["ohe_st"].transform([[st_value]]).astype(np.float32)
        eft_ohe  = info["ohe_eft"].transform([[eft_value]]).astype(np.float32)

        X_raw = np.hstack([cont_arr, st_ohe, eft_ohe])
        X     = info["scaler_X"].transform(X_raw).astype(np.float32)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            pf_s, ps_s = info["model"](X_tensor)

        pf_s = pf_s.numpy().flatten()
        ps_s = ps_s.numpy().flatten()

        fmc = info["scaler_fmc"].inverse_transform(pf_s.reshape(-1, 1)).flatten()
        fmc = np.clip(fmc, 0, 100)

        sec_log = info["scaler_sec"].inverse_transform(ps_s.reshape(-1, 1)).flatten()
        sec = np.expm1(sec_log)
        sec = np.clip(sec, 0, None)

        fmc_preds.append(fmc[0])
        sec_preds.append(sec[0])

    fmc_arr = np.array(fmc_preds)
    sec_arr = np.array(sec_preds)

    return float(np.mean(fmc_arr)), float(np.mean(sec_arr)), fmc_arr, sec_arr


# ============================================================================
# 5. 页面样式
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
}
.stApp {
    background: #f7f9fc;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── 全局文字 ── */
.main .stMarkdown p { color: #334155; }
.main h1, .main h2, .main h3 { color: #0f172a; }

/* ── 侧边栏 ── */
section[data-testid="stSidebar"] {
    background: #eef3f8;
    border-right: 1px solid #d6dee8;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: #334155 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #475569 !important;
    font-size: 0.85rem;
    font-weight: 500;
}

/* ── 输入框 / 选择框 ── */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    border-radius: 8px !important;
}

/* ── 横幅 ── */
.hero-banner {
    position: relative;
    background: linear-gradient(135deg, #ffffff 0%, #f3f7fb 100%);
    border: 1px solid #d6dee8;
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(15, 23, 42, 0.05);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -45%; right: -18%;
    width: 460px; height: 460px;
    background: radial-gradient(circle, rgba(30, 64, 175, 0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
    position: relative;
    z-index: 1;
}
.hero-title span {
    background: linear-gradient(135deg, #1d4ed8, #0f766e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.92rem;
    color: #475569;
    margin: 0;
    line-height: 1.6;
    position: relative;
    z-index: 1;
}
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1rem;
    position: relative;
    z-index: 1;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #16a34a;
    box-shadow: 0 0 6px rgba(22,163,74,0.35);
    animation: pdot 2s ease-in-out infinite;
}
@keyframes pdot { 0%,100%{opacity:1;} 50%{opacity:0.5;} }
.status-text {
    font-size: 0.8rem;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── 侧边栏分组 ── */
.sidebar-section {
    background: rgba(29, 78, 216, 0.06);
    border-left: 3px solid #1d4ed8;
    padding: 0.5rem 0.8rem;
    margin: 1.2rem 0 0.8rem 0;
    border-radius: 0 6px 6px 0;
}
.sidebar-section h3 {
    font-size: 0.82rem;
    font-weight: 600;
    color: #1e3a8a !important;
    margin: 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── 警告 ── */
.warning-banner {
    background: #fff8e6;
    border: 1px solid #f3d37a;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0;
}
.warning-banner p {
    color: #9a6700 !important;
    font-size: 0.8rem;
    margin: 0;
    line-height: 1.5;
}

/* ── 结果卡片 ── */
.result-card {
    border-radius: 12px;
    padding: 2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
}
.result-card:hover { transform: translateY(-2px); }

.result-card-fmc {
    background: linear-gradient(145deg, #ffffff, #f7fbff);
    border: 1px solid #bfdbfe;
}
.result-card-sec {
    background: linear-gradient(145deg, #ffffff, #f4fbfa);
    border: 1px solid #b7e4dc;
}

.result-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0 0 0.2rem 0;
}
.result-card-fmc .result-label { color: #1d4ed8; }
.result-card-sec .result-label { color: #0f766e; }

.result-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0.4rem 0;
    line-height: 1;
}
.result-card-fmc .result-value { color: #1e3a8a; }
.result-card-sec .result-value { color: #115e59; }

.result-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    margin: 0.5rem 0 0 0;
    padding-top: 0.7rem;
    border-top: 1px solid #e2e8f0;
}
.result-card-fmc .result-range { color: #3b82f6; }
.result-card-sec .result-range { color: #0f766e; }

.result-unit {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.5rem;
}

/* ── 指标格子 ── */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1.2rem;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #dbe3ec;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03);
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 0.3rem 0;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.2rem;
    font-weight: 600;
    color: #0f172a;
    margin: 0;
}

/* ── 参数表 ── */
.param-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.5rem;
    background: #ffffff;
}
.param-table th {
    text-align: left;
    padding: 0.4rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #475569;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-bottom: 1px solid #dbe3ec;
    background: #f8fafc;
}
.param-table td {
    padding: 0.4rem 0.7rem;
    font-size: 0.85rem;
    color: #334155;
    border-bottom: 1px solid #edf2f7;
}

/* ── 架构说明 ── */
.arch-box {
    background: #ffffff;
    border: 1px solid #dbe3ec;
    border-radius: 10px;
    padding: 1.2rem;
}
.arch-box h4 {
    color: #1e3a8a;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0 0 0.8rem 0;
}
.arch-item {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}
.arch-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    color: #1d4ed8;
    background: rgba(29, 78, 216, 0.08);
    padding: 0.12rem 0.45rem;
    border-radius: 3px;
    white-space: nowrap;
}
.arch-desc {
    font-size: 0.8rem;
    color: #475569;
    line-height: 1.4;
}

/* ── 提示 ── */
.hint-block {
    background: #f8fbff;
    border: 1px solid #d6e6f5;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.hint-block p {
    color: #475569 !important;
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.6;
}

/* ── 页脚 ── */
.footer-bar {
    text-align: center;
    padding: 2rem 0 1rem 0;
    margin-top: 2rem;
    border-top: 1px solid #dbe3ec;
}
.footer-bar p {
    font-size: 0.72rem;
    color: #64748b;
    margin: 0;
}

/* ── 按钮 ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.25) !important;
}

/* ── expander ── */
details {
    background: #ffffff;
    border: 1px solid #dbe3ec;
    border-radius: 10px;
    padding: 0.25rem 0.5rem;
}
</style>
"""


# ============================================================================
# 6. 页面主函数
# ============================================================================

def main():
    st.set_page_config(
        page_title="MTL 污泥电渗透脱水预测系统",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # 加载模型
    try:
        models_info = load_all_models()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        model_error = str(e)

    # ================================================================
    # 侧边栏
    # ================================================================
    with st.sidebar:
        st.markdown(
            '<div style="padding:0.5rem 0 0.2rem 0;">'
            '<span style="font-size:1.1rem;font-weight:700;color:#e8edf3;">'
            'MTL 参数配置</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="font-size:0.78rem;color:#5a7a94;margin:0 0 0.5rem 0;">'
            '设置以下参数后点击主页面「开始预测」</p>',
            unsafe_allow_html=True
        )

        # ── 分类变量 ──
        st.markdown('<div class="sidebar-section"><h3>分类变量</h3></div>', unsafe_allow_html=True)

        st_display = [ST_LABELS[k] for k in ST_OPTIONS_WITH_OTHER]
        st_selected = st.selectbox("污泥类型 (ST)", st_display)
        st_value = ST_OPTIONS_WITH_OTHER[st_display.index(st_selected)]

        is_other_sludge = (st_value == 'OTHER')
        if is_other_sludge:
            st.markdown(
                '<div class="warning-banner"><p>'
                '「其他污泥类型」不在模型训练范围内，预测结果仅供参考。'
                '</p></div>',
                unsafe_allow_html=True
            )

        eft_display = [EFT_LABELS[k] for k in EFT_OPTIONS]
        eft_selected = st.selectbox("电场类型 (EFT)", eft_display)
        eft_value = EFT_OPTIONS[eft_display.index(eft_selected)]

        # ── 污泥特性 ──
        st.markdown('<div class="sidebar-section"><h3>污泥特性</h3></div>', unsafe_allow_html=True)

        ph   = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        imc  = st.number_input("初始含水率 IMC (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
        zeta = st.number_input("Zeta 电位 (mV)", min_value=-100.0, max_value=100.0, value=-20.0, step=0.1)
        ec   = st.number_input("电导率 EC (ms/cm)", min_value=0.0, max_value=50.0, value=2.0, step=0.01)
        vsts = st.number_input("有机质含量 VS/TS (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

        # ── 操作参数 ──
        st.markdown('<div class="sidebar-section"><h3>操作参数</h3></div>', unsafe_allow_html=True)

        pt = st.number_input("处理时间 PT (min)", min_value=0.0, max_value=1000.0, value=60.0, step=1.0)
        av = st.number_input("外加电压 AV (V)", min_value=0.0, max_value=200.0, value=30.0, step=0.1)
        ss = st.number_input("污泥质量 SS (g)", min_value=0.0, max_value=5000.0, value=100.0, step=1.0)
        mp = st.number_input("外加压力 MP (kPa)", min_value=0.0, max_value=10000.0, value=200.0, step=1.0)

    # ================================================================
    # 主区域
    # ================================================================

    # 头部
    status_html = (
        '<div class="status-bar">'
        '<div class="status-dot"></div>'
        f'<span class="status-text">{len(models_info)} models loaded · ensemble ready</span>'
        '</div>'
    ) if model_loaded else (
        '<div class="status-bar">'
        '<div class="status-dot" style="background:#ef4444;box-shadow:0 0 6px rgba(239,68,68,0.5);"></div>'
        '<span class="status-text">model loading failed</span>'
        '</div>'
    )

    st.markdown(f"""
    <div class="hero-banner">
        <h1 class="hero-title"><span>MTL</span> 污泥电渗透脱水预测系统</h1>
        <p class="hero-sub">
            基于多任务学习模型（Multi-Task Learning），集成软参数共享网络与 NSGA-II
            多目标超参数优化，同时预测最终含水率 FMC 与单位能耗 SEC。
        </p>
        {status_html}
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"模型加载失败：{model_error}")
        st.stop()

    # 预测按钮
    predict_clicked = st.button("开始预测", use_container_width=True, type="primary")

    if predict_clicked:
        features_cont = [ph, imc, zeta, ec, vsts, pt, av, ss, mp]
        st_for_model = st_value if st_value != 'OTHER' else 'UNKNOWN'

        with st.spinner("模型推理中 ..."):
            fmc_mean, sec_mean, fmc_all, sec_all = predict_ensemble_with_range(
                models_info, features_cont, st_for_model, eft_value
            )

        fmc_std = float(np.std(fmc_all, ddof=1)) if len(fmc_all) > 1 else 0.0
        sec_std = float(np.std(sec_all, ddof=1)) if len(sec_all) > 1 else 0.0
        fmc_lo, fmc_hi = max(0, fmc_mean - 2*fmc_std), min(100, fmc_mean + 2*fmc_std)
        sec_lo, sec_hi = max(0, sec_mean - 2*sec_std), sec_mean + 2*sec_std

        if is_other_sludge:
            st.warning("当前污泥类型「其他」不在模型训练范围内，以下结果仅供参考。")

        # 结果卡片
        col_fmc, col_sec = st.columns(2)
        with col_fmc:
            st.markdown(f"""
            <div class="result-card result-card-fmc">
                <p class="result-label">最终含水率 FMC</p>
                <p class="result-value">{fmc_mean:.2f}%</p>
                <p class="result-range">预测范围 {fmc_lo:.2f}% — {fmc_hi:.2f}%</p>
                <p class="result-unit">Final Moisture Content</p>
            </div>
            """, unsafe_allow_html=True)
        with col_sec:
            st.markdown(f"""
            <div class="result-card result-card-sec">
                <p class="result-label">单位能耗 SEC</p>
                <p class="result-value">{sec_mean:.4f}</p>
                <p class="result-range">预测范围 {sec_lo:.4f} — {sec_hi:.4f}</p>
                <p class="result-unit">kWh/kg·RW · Specific Energy Consumption</p>
            </div>
            """, unsafe_allow_html=True)

        # 模型指标
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <p class="metric-label">FMC 模型 R²</p>
                <p class="metric-value">0.8607</p>
            </div>
            <div class="metric-card">
                <p class="metric-label">SEC 模型 R²</p>
                <p class="metric-value">0.8032</p>
            </div>
            <div class="metric-card">
                <p class="metric-label">集成模型数</p>
                <p class="metric-value">{len(SEEDS)}</p>
            </div>
            <div class="metric-card">
                <p class="metric-label">集成标准差 FMC / SEC</p>
                <p class="metric-value">{fmc_std:.3f} / {sec_std:.4f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 参数汇总
        with st.expander("查看本次输入参数"):
            st.markdown(f"""
            <table class="param-table">
                <tr><th>参数</th><th>数值</th><th>参数</th><th>数值</th></tr>
                <tr><td>污泥类型</td><td>{ST_LABELS[st_value]}</td>
                    <td>电场类型</td><td>{EFT_LABELS[eft_value]}</td></tr>
                <tr><td>pH</td><td>{ph}</td>
                    <td>IMC</td><td>{imc}%</td></tr>
                <tr><td>Zeta</td><td>{zeta} mV</td>
                    <td>EC</td><td>{ec} ms/cm</td></tr>
                <tr><td>VS/TS</td><td>{vsts}%</td>
                    <td>PT</td><td>{pt} min</td></tr>
                <tr><td>AV</td><td>{av} V</td>
                    <td>SS</td><td>{ss} g</td></tr>
                <tr><td>MP</td><td>{mp} kPa</td>
                    <td></td><td></td></tr>
            </table>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="hint-block">'
            '<p>在左侧面板中设置污泥特性与操作参数，然后点击上方「开始预测」按钮获取结果。</p>'
            '</div>',
            unsafe_allow_html=True
        )

    # 模型说明
    with st.expander("模型架构与技术说明"):
        st.markdown("""
        <div class="arch-box">
            <h4>MTL Multi-Task Learning Architecture</h4>
            <div class="arch-item">
                <span class="arch-tag">NETWORK</span>
                <span class="arch-desc">软参数共享 — 双编码器分别提取 FMC / SEC 特征，Fusion 层实现跨任务信息交互</span>
            </div>
            <div class="arch-item">
                <span class="arch-tag">NSGA-II</span>
                <span class="arch-desc">多目标进化算法同时优化两个任务的验证损失，搜索 Pareto 最优超参数</span>
            </div>
            <div class="arch-item">
                <span class="arch-tag">ENSEMBLE</span>
                <span class="arch-desc">5 种子集成取均值，提升预测稳定性并量化不确定性范围</span>
            </div>
            <div class="arch-item">
                <span class="arch-tag">LEAK-FREE</span>
                <span class="arch-desc">严格无泄露 — 缩放器仅在训练子集上拟合，杜绝测试信息渗入</span>
            </div>
            <div class="arch-item">
                <span class="arch-tag">PARETO</span>
                <span class="arch-desc">自适应权重 — 损失驱动的动态任务平衡机制</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 页脚
    st.markdown(
        '<div class="footer-bar">'
        '<p>MTL Multi-Task Learning · Soft Parameter Sharing + NSGA-II + Ensemble</p>'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
