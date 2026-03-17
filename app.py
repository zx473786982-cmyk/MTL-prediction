"""
IF-P 多任务学习模型 —— Streamlit 网页预测系统
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
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "ensemble_artifacts")

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
}
EFT_LABELS = {
    'EOD':  '电渗透脱水 (EOD)',
    'PEOD': '外加压力电渗透 (PEOD)',
    'UEOD': '恒压电渗透 (UEOD)',
    'IEOD': '恒流电渗透 (IEOD)',
    'ACEO': '交变电场电渗透 (ACEO)',
}


# ============================================================================
# 3. 加载模型与预处理器（缓存，只加载一次）
# ============================================================================

@st.cache_resource
def load_all_models():
    """加载 5 个 seed 的模型和预处理器"""
    models_info = []

    for seed in SEEDS:
        # 加载预处理器
        ohe_st     = joblib.load(os.path.join(ARTIFACT_DIR, f"ohe_st_seed_{seed}.pkl"))
        ohe_eft    = joblib.load(os.path.join(ARTIFACT_DIR, f"ohe_eft_seed_{seed}.pkl"))
        scaler_X   = joblib.load(os.path.join(ARTIFACT_DIR, f"scaler_X_seed_{seed}.pkl"))
        scaler_fmc = joblib.load(os.path.join(ARTIFACT_DIR, f"scaler_fmc_seed_{seed}.pkl"))
        scaler_sec = joblib.load(os.path.join(ARTIFACT_DIR, f"scaler_sec_seed_{seed}.pkl"))

        # 计算输入维度
        n_cont    = len(CONTINUOUS_FEATURES)
        n_st_cat  = len(ohe_st.categories_[0])
        n_eft_cat = len(ohe_eft.categories_[0])
        input_dim = n_cont + n_st_cat + n_eft_cat

        # 从第一个 seed 的预处理器获取网络参数
        # （所有 seed 用的超参数相同，区别只是随机种子）
        state_dict = torch.load(
            os.path.join(ARTIFACT_DIR, f"IFP_model_seed_{seed}.pth"),
            map_location="cpu",
            weights_only=True,
        )

        # 从权重反推 encoder_dim 和 fusion_dim
        encoder_dim = state_dict["encoder_fmc.0.weight"].shape[0]
        fusion_dim  = state_dict["fusion.0.weight"].shape[0]
        # dropout 在 eval 模式下不影响，给个占位值即可
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
# 4. 预测函数
# ============================================================================

def predict_ensemble(models_info, features_cont, st_value, eft_value):
    """
    5 个 seed 模型分别预测，取平均值
    返回: (fmc_pred, sec_pred)
    """
    fmc_preds = []
    sec_preds = []

    for info in models_info:
        # --- 特征处理 ---
        cont_arr = np.array(features_cont, dtype=np.float32).reshape(1, -1)
        st_ohe   = info["ohe_st"].transform([[st_value]]).astype(np.float32)
        eft_ohe  = info["ohe_eft"].transform([[eft_value]]).astype(np.float32)

        X_raw = np.hstack([cont_arr, st_ohe, eft_ohe])
        X     = info["scaler_X"].transform(X_raw).astype(np.float32)

        # --- 模型推理 ---
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            pf_s, ps_s = info["model"](X_tensor)

        pf_s = pf_s.numpy().flatten()
        ps_s = ps_s.numpy().flatten()

        # --- 反标准化 ---
        fmc = info["scaler_fmc"].inverse_transform(pf_s.reshape(-1, 1)).flatten()
        fmc = np.clip(fmc, 0, 100)

        sec_log = info["scaler_sec"].inverse_transform(ps_s.reshape(-1, 1)).flatten()
        sec = np.expm1(sec_log)
        sec = np.clip(sec, 0, None)

        fmc_preds.append(fmc[0])
        sec_preds.append(sec[0])

    return float(np.mean(fmc_preds)), float(np.mean(sec_preds))


# ============================================================================
# 5. Streamlit 页面
# ============================================================================

def main():
    st.set_page_config(
        page_title="IF-P 污泥电渗透脱水预测系统",
        page_icon="⚡",
        layout="wide",
    )

    # --- 自定义样式 ---
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-card h2 {
        font-size: 2.5rem;
        margin: 0.5rem 0;
        color: white;
    }
    .result-card p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }
    .result-card-sec {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-card-sec h2 {
        font-size: 2.5rem;
        margin: 0.5rem 0;
        color: white;
    }
    .result-card-sec p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }
    .info-box {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- 标题 ---
    st.markdown('<h1 class="main-title">⚡ IF-P 污泥电渗透脱水预测系统</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">基于多任务学习（软参数共享 + NSGA-II + 多种子集成）的智能预测平台</p>',
        unsafe_allow_html=True
    )

    # --- 加载模型 ---
    try:
        models_info = load_all_models()
    except Exception as e:
        st.error(f"模型加载失败，请检查 ensemble_artifacts 文件夹是否完整。\n\n错误信息：{e}")
        st.stop()

    st.success(f"✅ 已加载 {len(models_info)} 个集成模型，准备就绪！")

    # --- 输入区域 ---
    st.markdown("---")
    st.subheader("📋 请输入参数")

    col_left, col_mid, col_right = st.columns(3)

    # ---- 左列：分类变量 + 基本特性 ----
    with col_left:
        st.markdown("**🔬 污泥类型与电场类型**")

        st_display = [ST_LABELS[k] for k in ST_OPTIONS]
        st_selected = st.selectbox("污泥类型 (ST)", st_display)
        st_value = ST_OPTIONS[st_display.index(st_selected)]

        eft_display = [EFT_LABELS[k] for k in EFT_OPTIONS]
        eft_selected = st.selectbox("电场类型 (EFT)", eft_display)
        eft_value = EFT_OPTIONS[eft_display.index(eft_selected)]

        st.markdown("**📊 污泥特性**")
        ph  = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        imc = st.number_input("初始含水率 IMC (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)

    # ---- 中列：更多特性 ----
    with col_mid:
        st.markdown("**📊 污泥特性（续）**")
        zeta = st.number_input("Zeta 电位 (mV)", min_value=-100.0, max_value=100.0, value=-20.0, step=0.1)
        ec   = st.number_input("电导率 EC (ms/cm)", min_value=0.0, max_value=50.0, value=2.0, step=0.01)
        vsts = st.number_input("有机质含量 VS/TS (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

    # ---- 右列：操作参数 ----
    with col_right:
        st.markdown("**⚙️ 操作参数**")
        pt = st.number_input("处理时间 PT (min)", min_value=0.0, max_value=1000.0, value=60.0, step=1.0)
        av = st.number_input("外加电压 AV (V)", min_value=0.0, max_value=200.0, value=30.0, step=0.1)
        ss = st.number_input("污泥质量 SS (g)", min_value=0.0, max_value=5000.0, value=100.0, step=1.0)
        mp = st.number_input("外加压力 MP (kPa)", min_value=0.0, max_value=10000.0, value=200.0, step=1.0)

    # ---- 预测按钮 ----
    st.markdown("---")

    if st.button("🚀 开始预测", use_container_width=True, type="primary"):

        # 按 CONTINUOUS_FEATURES 的顺序组装
        # ['pH', 'IMC(%)', 'Zeta(mV)', 'EC(ms/cm)', 'VS/TS(%)',
        #  'PT(min)', 'AV(V)', 'SS(g)', 'MP(kPa)']
        features_cont = [ph, imc, zeta, ec, vsts, pt, av, ss, mp]

        with st.spinner("模型推理中..."):
            fmc_pred, sec_pred = predict_ensemble(models_info, features_cont, st_value, eft_value)

        # ---- 显示结果 ----
        st.markdown("---")
        st.subheader("📊 预测结果")

        res_left, res_right = st.columns(2)

        with res_left:
            st.markdown(f"""
            <div class="result-card">
                <p>最终含水率 FMC</p>
                <h2>{fmc_pred:.2f}%</h2>
                <p>Final Moisture Content</p>
            </div>
            """, unsafe_allow_html=True)

        with res_right:
            st.markdown(f"""
            <div class="result-card-sec">
                <p>单位能耗 SEC</p>
                <h2>{sec_pred:.4f}</h2>
                <p>kWh/kg·RW (Specific Energy Consumption)</p>
            </div>
            """, unsafe_allow_html=True)

        # ---- 输入参数汇总 ----
        with st.expander("📋 查看本次输入参数"):
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                st.write(f"**污泥类型:** {ST_LABELS[st_value]}")
                st.write(f"**电场类型:** {EFT_LABELS[eft_value]}")
                st.write(f"**pH:** {ph}")
                st.write(f"**IMC:** {imc}%")
                st.write(f"**Zeta:** {zeta} mV")
            with param_col2:
                st.write(f"**EC:** {ec} ms/cm")
                st.write(f"**VS/TS:** {vsts}%")
                st.write(f"**PT:** {pt} min")
                st.write(f"**AV:** {av} V")
                st.write(f"**SS:** {ss} g")
                st.write(f"**MP:** {mp} kPa")

    # --- 页脚 ---
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#999; font-size:0.85rem;">'
        'IF-P Multi-Task Learning Model · Soft Parameter Sharing + NSGA-II + Ensemble'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
