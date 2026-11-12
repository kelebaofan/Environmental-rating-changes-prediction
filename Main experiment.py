import os
import re
import gc
import time
import hashlib
import platform
import warnings
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, precision_recall_fscore_support,
    average_precision_score,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── 可选依赖：CatBoost ──────────────────────────────────────────────────
HAS_CB = False
try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    pass

# ── 可选依赖：imblearn（SMOTE + 平衡集成） ──────────────────────────────
HAS_IMBLEARN = False
HAS_BALANCED_RF = False
HAS_EASY_ENSEMBLE = False
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    pass
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_BALANCED_RF = True
except ImportError:
    pass
try:
    from imblearn.ensemble import EasyEnsembleClassifier
    HAS_EASY_ENSEMBLE = True
except ImportError:
    pass

# ── 可选依赖：Optuna ────────────────────────────────────────────────────
HAS_OPTUNA = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    pass

# ── 可选依赖：LightGBM ──────────────────────────────────────────────────
HAS_LGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    pass

# ── 可选依赖：XGBoost ───────────────────────────────────────────────────
HAS_XGB = False
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    pass

# ── 可选依赖：matplotlib（混淆矩阵热力图） ──────────────────────────────
HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    pass

print(f"[ENV] CB={HAS_CB} IMBLEARN={HAS_IMBLEARN} OPTUNA={HAS_OPTUNA}"
      f" LGB={HAS_LGB} XGB={HAS_XGB} MPL={HAS_MPL} OS={platform.system()}\n")

# ════════════════════════════════════════════════════════════════════════
# 全局配置
# ════════════════════════════════════════════════════════════════════════
DATA_PATH = r"data source"
SHEET_NAME = "Sheet name"
OUTPUT_PATH = ""

# ── Bootstrap 统计检验参数 ──────────────────────────────────────────────
N_BOOT = 2000          # 主模型逐窗口 CI 重采样次数
BOOT_ALPHA = 0.05      # 95% 置信区间
BOOT_SEED = 42         # 随机种子（保证可复现）
BOOT_MIN_OK = 10       # 有效重采样数下限（低于此值判为不可用）

# ── ★ 全模型扩展检验参数 ────────────────────────────────────────────────
N_BOOT_MODEL = N_BOOT  # 逐模型逐年/池化配对检验与池化SD的重采样次数
PAIRED_SIDE = "two"    # "two"=双侧 H1: Full≠No-Gas（默认，推荐）
                       # "one"=单侧 H1: Full>No-Gas
PAIRED_CLUSTER = True  # True=按企业 id 聚类重抽样（池化口径推荐）

# ── ★ 混淆矩阵输出配置（仅主模型） ──────────────────────────────────────
CONFUSION_PLOT_DIR = "confusion_figures"
CONFUSION_DPI = 300
# Excel 表内类名（中文）；PNG 图内自动改用英文标签以规避字体缺字形
CONFUSION_CLASS_NAMES: Dict[int, str] = {-1: "下降(↓)", 0: "稳定(平)", 1: "上升(↑)"}

YEAR_COL = "year"
ID_COL = "id"
INDUSTRY_COL = "industry"
TARGET_COL = "Wind E"
RANDOM_STATE = 42

if platform.system() == "Windows":
    N_JOBS = 1
    N_JOBS_CATBOOST = 8
else:
    N_JOBS = 16
    N_JOBS_CATBOOST = 16

CAP_LOWER_PCT = 0.01
CAP_UPPER_PCT = 0.99
FEAT_IMP_THRESHOLD = 0.001
FEAT_MIN_KEEP = 50
FEAT_TOPK = 75
CORR_THRESHOLD = 0.85
PCA_VARIANCE = 0.95
OPTUNA_N_TRIALS_MAX = 40
OPTUNA_N_TRIALS_MIN = 15
OPTUNA_TRIALS_PER_N = 200
OPTUNA_VAL_RATIO = 0.5
CW_MODE = "sqrt"
OPTUNA_L2_MAX = 15.0

POLL_ANOMALY_GASES = ["CH4", "NO2", "CO", "SO2"]
RAW_POLLUTION_COLS_DROP = {"mean_NO2", "mean_SO2", "mean_CH4", "mean_CO"}
POLL_TREND_COLS = [
    "mean_SO2", "mean_NO2", "mean_CH4", "mean_CO",
    "SO2_chang", "NO2_chang", "CH4_chang", "CO_chang",
]
CARBON_NEUTRAL_YEAR = 2021

CORE_NUM_FEATURES = [
    "owner", "independent_rate", "diet", "recognize", "prize",
    "punish_num", "punish_money", "patent",
    "CH4_CV", "NO2_CV", "CO_CV", "SO2_CV",
    "mean_NO2", "mean_SO2", "mean_CH4", "mean_CO",
    "NO2_chang", "SO2_chang", "CH4_chang", "CO_chang",
    "Top1", "Lnage", "Lev", "ROA", "Size", "Growth",
]
STRICT_WHITELIST = [
    "SO2_exceed_diff", "CH4_zscore",
    "Lev_x_SO2_exceed", "Size_x_CO_zscore",
]
IND_STAT_CURRENT_COLS = {
    "ind_change_rate", "ind_up_rate", "ind_down_rate",
}
BASE_DROP_ROOTS: Set[str] = {"mean_CH4", "mean_NO2", "mean_CO", "mean_SO2"}

# ════════════════════════════════════════════════════════════════════════
# 消融实验配置（Full vs No-Gas 两档对比）
# ════════════════════════════════════════════════════════════════════════
EXPERIMENT_CONFIGS: Dict[str, Dict] = {
    "Full": {
        "description": "完整模型（含全部卫星气体污染物特征及派生特征）",
        "drop_gas_features": False,
    },
    "No-Gas": {
        "description": "消融模型（移除全部卫星气体污染物特征及派生特征）",
        "drop_gas_features": True,
    },
}

# ── 气体相关的原始列名前缀（严格前缀匹配，避免宽泛子串误伤） ────────────
GAS_RAW_PREFIXES: List[str] = [
    "mean_CH4", "mean_NO2", "mean_CO", "mean_SO2",   # 原始浓度
    "CH4_CV", "NO2_CV", "CO_CV", "SO2_CV",           # 变异系数
    "CH4_chang", "NO2_chang", "CO_chang", "SO2_chang",  # 变化率
    "CH4_zscore", "NO2_zscore", "CO_zscore", "SO2_zscore",  # z 分数
    "CH4_exceed", "NO2_exceed", "CO_exceed", "SO2_exceed",  # 超标
    "poll_abs_sum", "poll_cv_mean", "poll_mean_sum",  # 复合指标
    "gas_zscore_range", "gas_zscore_std",             # 跨气体统计
    "gas_anomaly_composite", "gas_anomaly_max",       # 异常综合
    "gas_exceed_composite", "gas_exceed_count",       # 超标综合
    "gas_composite_x_",                               # 气体×财务交互
    "NO2_SO2_ratio", "CH4_CO_ratio",                  # 气体比值
]
# ── 气体相关的交互项 / 派生命名模式 ─────────────────────────────────────
GAS_INTERACTION_PATTERNS: List[str] = [
    "_x_CH4", "_x_NO2", "_x_CO", "_x_SO2",   # 交互项后缀
    "CH4_", "NO2_", "CO_", "SO2_",           # 气体名开头
    "_CH4_", "_NO2_", "_CO_", "_SO2_",       # 包含气体名
    "Lev_x_SO2", "Size_x_CO", "Size_x_NO2", "Size_x_CH4",  # 特定交互
]
# CO 的词素级正则：大写 CO 且前后不能紧邻英文字母
_RE_CO_ISOLATED = re.compile(r"(?<![A-Za-z])CO(?![A-Za-z])")
_RE_CO_TOKEN = re.compile(r"(?:^|_)CO(?:_|$)")
_GAS_UNAMBIGUOUS = ["CH4", "NO2", "SO2"]

CONFIG_COL = "config（消融配置）"

# ── 模型分流集合 ────────────────────────────────────────────────────────
LINEAR_MODELS: set = {"LR", "LinearSVM"}
NO_SMOTE_MODELS: set = {"BalancedRF", "EasyEnsemble"}
TREE_MODELS: set = {
    "RF", "XGBoost", "LightGBM", "HistGradBoost",
}
# ★ CatBoost 基线已删除，仅保留主模型 10_CatBoost_Main

# ── 模型列表（★ 基线 9 → 8，CatBoost 仅保留主模型） ────────────────────
BASELINE_MODEL_NAMES: List[str] = [
    "LR", "RF", "BalancedRF", "EasyEnsemble",
    "XGBoost", "LightGBM", "LinearSVM", "HistGradBoost",
]
MAIN_MODEL_NAME = "10_CatBoost_Main"
ALL_MODEL_NAMES = BASELINE_MODEL_NAMES + [MAIN_MODEL_NAME]
# ★ 回填混淆矩阵目标模型（仅主模型）
CONFUSION_MODEL_NAMES: List[str] = [MAIN_MODEL_NAME]

# ── 标签体系 ────────────────────────────────────────────────────────────
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL = {0: -1, 1: 0, 2: 1}
LABEL_LIST = [-1, 0, 1]

# ── 指标字段名常量 ──────────────────────────────────────────────────────
_MK_F1 = "Macro-F1（宏平均F1分数）"
_MK_AUC_ROC = "Macro-AUC-ROC（宏平均受试者工作特征曲线下面积）"
_MK_AUC_PR = "AUC-PR（宏平均精确率-召回率曲线下面积）"
_MK_REC_DN = "Recall-下降(↓)（下降类别召回率）"
_MK_REC_ST = "Recall-稳定(平)（稳定类别召回率）"
_MK_REC_UP = "Recall-上升(↑)（上升类别召回率）"
_MK_REC_MAC = "Macro-Recall（宏平均召回率）"
_MK_ACC = "Accuracy（准确率）"
_MK_PRE_MAC = "Macro-Precision（宏平均精确率）"
_MK_F1_DN = "F1-下降(↓)（下降类别F1分数）"
_MK_F1_ST = "F1-稳定(平)（稳定类别F1分数）"
_MK_F1_UP = "F1-上升(↑)（上升类别F1分数）"
_MK_PRE_DN = "Precision-下降(↓)（下降类别精确率）"
_MK_PRE_ST = "Precision-稳定(平)（稳定类别精确率）"
_MK_PRE_UP = "Precision-上升(↑)（上升类别精确率）"
_MK_SUP_DN = "Support-下降(↓)（下降类别样本数）"
_MK_SUP_ST = "Support-稳定(平)（稳定类别样本数）"
_MK_SUP_UP = "Support-上升(↑)（上升类别样本数）"

# ★★ 主评价指标口径（★本版核心修改）：
# Macro-Recall / AUC-ROC / AUC-PR / Recall-下降 / Macro-Precision / Accuracy
SUMMARY_METRIC_COLS = [
    _MK_REC_MAC, _MK_AUC_ROC, _MK_AUC_PR, _MK_REC_DN, _MK_PRE_MAC, _MK_ACC,
]
# 逐窗口明细 Sheet 仍输出全部指标（含 Macro-F1 等，作对照）
ALL_METRIC_COLS = SUMMARY_METRIC_COLS + [
    _MK_F1, _MK_REC_ST, _MK_REC_UP,
    _MK_F1_DN, _MK_F1_ST, _MK_F1_UP,
    _MK_PRE_DN, _MK_PRE_ST, _MK_PRE_UP,
    _MK_SUP_DN, _MK_SUP_ST, _MK_SUP_UP,
]

FIXED_WINDOWS: List[Dict] = [
    {"window_id": "W1_train2018-2020_test2022",
     "train_years": [2018, 2019, 2020], "val_year": 2021, "test_year": 2022},
    {"window_id": "W2_train2019-2021_test2023",
     "train_years": [2019, 2020, 2021], "val_year": 2022, "test_year": 2023},
    {"window_id": "W3_train2020-2022_test2024",
     "train_years": [2020, 2021, 2022], "val_year": 2023, "test_year": 2024},
]

# ════════════════════════════════════════════════════════════════════════
# 标签映射
# ════════════════════════════════════════════════════════════════════════
def map_labels(y):
    return np.array([LABEL_MAP[int(v)] for v in y])


def inv_labels(y):
    return np.array([INV_LABEL[int(v)] for v in y])

# ════════════════════════════════════════════════════════════════════════
# 污染特征列匹配与删除
# ════════════════════════════════════════════════════════════════════════
def drop_pollution_cols_from_avail(avail, drop_roots):
    if not drop_roots:
        return []
    to_drop: Set[str] = set()
    for col in avail:
        for root in drop_roots:
            if col.startswith(root):
                to_drop.add(col); break
            if f"_{root}_" in col or col.endswith(f"_{root}"):
                to_drop.add(col); break
            if f"_x_{root}" in col:
                to_drop.add(col); break
            gas_name = root.split("_")[0]
            if (gas_name in POLL_ANOMALY_GASES and f"_x_{gas_name}" in col
                    and any(r.startswith(gas_name) for r in drop_roots)):
                to_drop.add(col); break
    return [c for c in avail if c in to_drop]


def get_avail_after_base_drop(avail, base_drop_roots):
    if not base_drop_roots:
        return avail
    dropped = drop_pollution_cols_from_avail(avail, base_drop_roots)
    kept = [c for c in avail if c not in set(dropped)]
    print(f"  [BASE_DROP] 删除 {len(dropped)} 列 → 剩余 {len(kept)} 列")
    return kept

# ════════════════════════════════════════════════════════════════════════
# 消融实验：气体特征精确识别与移除
# ════════════════════════════════════════════════════════════════════════
def identify_gas_features(feat_names: List[str]) -> Set[str]:
    """精确识别所有气体污染物相关特征（含派生特征）。"""
    gas_features: Set[str] = set()
    for col in feat_names:
        if not isinstance(col, str):
            continue
        if any(col.startswith(prefix) for prefix in GAS_RAW_PREFIXES):
            gas_features.add(col); continue
        if any(pattern in col for pattern in GAS_INTERACTION_PATTERNS):
            gas_features.add(col); continue
        if any(gas in col for gas in _GAS_UNAMBIGUOUS):
            gas_features.add(col); continue
        if _RE_CO_ISOLATED.search(col) or _RE_CO_TOKEN.search(col):
            gas_features.add(col)
    return gas_features


def remove_gas_features(feat_names: List[str], verbose: bool = True) -> List[str]:
    """从特征列表中移除所有气体相关特征（保持原有顺序）。"""
    gas_feats = identify_gas_features(feat_names)
    kept = [f for f in feat_names if f not in gas_feats]
    if verbose:
        print(f"  [消融] 识别到 {len(gas_feats)} 个气体特征，"
              f"移除后剩余 {len(kept)} 个特征（原始 {len(feat_names)} 个）")
        sample = sorted(gas_feats)[:20]
        print(f"  [消融] 移除的气体特征示例（前20个）: {sample}")
        leftover = [f for f in kept if any(g in f for g in _GAS_UNAMBIGUOUS)]
        if leftover:
            print(f"  [消融] ⚠ 残留可疑气体列: {leftover[:10]}")
    return kept


def stable_feature_hash(feat_names: List[str]) -> str:
    """特征列集合的稳定指纹（md5 前12位，跨进程可复现）。"""
    payload = "|".join(sorted(feat_names)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]

# ════════════════════════════════════════════════════════════════════════
# 标签完整性检查
# ════════════════════════════════════════════════════════════════════════
def _check_label_integrity(dtr, dte, window_id):
    def _label3(df_):
        return np.where(df_[TARGET_COL] > 0, 1,
                        np.where(df_[TARGET_COL] < 0, -1, 0)).astype(int)
    ytr = _label3(dtr); yte = _label3(dte)
    utr = set(np.unique(ytr).tolist()); ute = set(np.unique(yte).tolist())
    need = {-1, 0, 1}
    prefix = f"  [标签校验][{window_id}]"
    missing_train = need - utr
    if missing_train:
        reason = f"训练集缺少类别 {missing_train}，实际只有 {utr}，跳过"
        print(f"{prefix} ❌ {reason}"); return False, reason
    if len(ute) < 2:
        reason = f"测试集只有 1 个类别 {ute}，AUC 无法计算，跳过"
        print(f"{prefix} ❌ {reason}"); return False, reason
    from collections import Counter
    cnt = Counter(ytr.tolist())
    min_cls, min_n = min(cnt.items(), key=lambda x: x[1])
    label_name_map = {-1: "Down(↓)", 0: "Stable(平)", 1: "Up(↑)"}
    if min_n < 5:
        reason = (f"训练集类别 {label_name_map[min_cls]} 仅有 {min_n} 条"
                  f"（<5），SMOTE 无法运行，跳过")
        print(f"{prefix} ❌ {reason}"); return False, reason
    dist_tr = " ".join(f"{label_name_map[k]}:{cnt[k]}" for k in [-1, 0, 1])
    nte = Counter(yte.tolist())
    dist_te = " ".join(f"{label_name_map[k]}:{nte.get(k, 0)}"
                       for k in [-1, 0, 1])
    print(f"{prefix} ✅ 训练集={len(ytr)}行 [{dist_tr}]"
          f" 测试集={len(yte)}行 [{dist_te}]")
    return True, "ok"

# ════════════════════════════════════════════════════════════════════════
# 行业统计安全计算
# ════════════════════════════════════════════════════════════════════════
def compute_industry_stats_from_train(dtr, dv, dte):
    if INDUSTRY_COL not in dtr.columns:
        return dtr, dv, dte

    def _up(s):
        ch = s[s["y_bin"] == 1]
        return float((ch[TARGET_COL] > 0).mean()) if len(ch) > 0 else 0.0

    def _dn(s):
        ch = s[s["y_bin"] == 1]
        return float((ch[TARGET_COL] < 0).mean()) if len(ch) > 0 else 0.0

    ist = (dtr.groupby([INDUSTRY_COL, YEAR_COL])["y_bin"]
           .agg(["mean", "count"]).reset_index())
    ist.columns = [INDUSTRY_COL, YEAR_COL,
                   "ind_change_rate_train", "ind_n_train"]
    ur = (dtr.groupby([INDUSTRY_COL, YEAR_COL]).apply(_up)
          .reset_index(name="ind_up_rate_train"))
    dr = (dtr.groupby([INDUSTRY_COL, YEAR_COL]).apply(_dn)
          .reset_index(name="ind_down_rate_train"))
    ist = ist.merge(ur, on=[INDUSTRY_COL, YEAR_COL], how="left")
    ist = ist.merge(dr, on=[INDUSTRY_COL, YEAR_COL], how="left")
    ist.sort_values([INDUSTRY_COL, YEAR_COL], inplace=True)
    for col in ["ind_change_rate_train", "ind_up_rate_train",
                "ind_down_rate_train"]:
        g = ist.groupby(INDUSTRY_COL)[col]
        ist[f"{col}_lag1"] = g.shift(1)
        ist[f"{col}_ma3_lag1"] = (g.rolling(3, min_periods=2).mean()
                                  .reset_index(level=0, drop=True).shift(1))
    max_train_year = int(dtr[YEAR_COL].max())
    ist_latest = ist[ist[YEAR_COL] == max_train_year].copy()
    rename_map = {c: c.replace("_train", "_safe") for c in ist.columns}
    ist = ist.rename(columns=rename_map)
    ist_latest = ist_latest.rename(columns=rename_map)

    def _is_leaky(col_name):
        for stem in ["ind_change_rate", "ind_up_rate", "ind_down_rate"]:
            if stem in col_name and "_safe" not in col_name:
                return True
        return False

    dtr_c = dtr.drop(columns=[c for c in dtr.columns if _is_leaky(c)],
                     errors="ignore").copy()
    dv_c = dv.drop(columns=[c for c in dv.columns if _is_leaky(c)],
                   errors="ignore").copy()
    dte_c = dte.drop(columns=[c for c in dte.columns if _is_leaky(c)],
                     errors="ignore").copy()
    train_join = ([INDUSTRY_COL, YEAR_COL]
                  + [c for c in ist.columns
                     if c not in [INDUSTRY_COL, YEAR_COL, "ind_n_safe"]])
    dtr_c = dtr_c.merge(ist[train_join],
                        on=[INDUSTRY_COL, YEAR_COL], how="left")
    safe_lag_cols = ([INDUSTRY_COL]
                     + [c for c in ist_latest.columns
                        if "_safe_lag1" in c or "_safe_ma3_lag1" in c])
    ist_j = ist_latest[[c for c in safe_lag_cols
                        if c in ist_latest.columns]].copy()
    dv_c = dv_c.merge(ist_j, on=INDUSTRY_COL, how="left")
    dte_c = dte_c.merge(ist_j, on=INDUSTRY_COL, how="left")
    for df_ in [dtr_c, dv_c, dte_c]:
        sc = [c for c in df_.columns if "_safe" in c and df_[c].dtype in
              [np.float64, np.float32, np.int64, np.int32]]
        df_[sc] = df_[sc].fillna(0)
    added = [c for c in dtr_c.columns if "_safe" in c]
    print(f"  [FIX-2] 安全行业特征={len(added)}列 "
          f"max_train_year={max_train_year}")
    return dtr_c, dv_c, dte_c

# ════════════════════════════════════════════════════════════════════════
# 动态类权重
# ════════════════════════════════════════════════════════════════════════
def compute_class_weights_3c(y_raw, mode=CW_MODE):
    from collections import Counter
    counts = Counter(int(v) for v in y_raw)
    n_major = max(counts.values())
    if mode == "log":
        cw = {lbl: float(np.log(1.0 + n_major / max(counts.get(lbl, 1), 1)))
              for lbl in [-1, 0, 1]}
    else:
        cw = {lbl: float(np.sqrt(n_major / max(counts.get(lbl, 1), 1)))
              for lbl in [-1, 0, 1]}
    scale = max(cw.get(0, 1.0), 1e-9)
    return {lbl: round(v / scale, 4) for lbl, v in cw.items()}


def build_sample_weights(y_raw, class_weight_dict):
    sw = np.array([class_weight_dict.get(int(v), 1.0) for v in y_raw],
                  dtype=np.float32)
    msw = sw.mean()
    if msw > 1e-9:
        sw = sw / msw
    return sw

# ════════════════════════════════════════════════════════════════════════
# 特征工程
# ════════════════════════════════════════════════════════════════════════
def feature_engineering(df, core_features=None):
    if core_features is None:
        core_features = CORE_NUM_FEATURES
    df = df.copy()
    df.sort_values([ID_COL, YEAR_COL], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["y_bin"] = (df[TARGET_COL] != 0).astype(int)
    df["y_dir"] = np.where(df[TARGET_COL] > 0, 1, 0)
    grp = df.groupby(ID_COL)
    for f in core_features:
        if f not in df.columns:
            continue
        df[f"{f}_lag1"] = grp[f].shift(1)
    df["y_lag1"] = grp[TARGET_COL].shift(1)
    df["y_bin_lag1"] = grp["y_bin"].shift(1)
    for f in core_features:
        if f not in df.columns:
            continue
        r = grp[f].rolling(3, min_periods=2)
        df[f"{f}_ma3"] = r.mean().reset_index(level=0, drop=True)
        df[f"{f}_std3"] = r.std().reset_index(level=0, drop=True)
        df[f"{f}_trend2"] = grp[f].diff(1)
    for f in core_features:
        if f not in df.columns:
            continue
        yg = df.groupby(YEAR_COL)[f]
        df[f"{f}_yr_rank"] = yg.rank(pct=True)
        ym_ = yg.transform("mean"); ys_ = yg.transform("std").replace(0, 1)
        df[f"{f}_yr_zscore"] = (df[f] - ym_) / ys_
        if INDUSTRY_COL in df.columns:
            yi = df.groupby([YEAR_COL, INDUSTRY_COL])[f]
            df[f"{f}_yi_rank"] = yi.rank(pct=True)
            yim = yi.transform("mean"); yis = yi.transform("std").replace(0, 1)
            df[f"{f}_yi_zscore"] = (df[f] - yim) / yis
    for sx in ["_yr_rank", "_yr_zscore", "_yi_rank", "_yi_zscore"]:
        for f in core_features:
            c = f"{f}{sx}"
            if c in df.columns:
                df[f"{c}_lag1"] = df.groupby(ID_COL)[c].shift(1)
    pc = [c for c in ["NO2_chang", "SO2_chang", "CH4_chang", "CO_chang"]
          if c in df.columns and c in core_features]
    pcv = [c for c in ["CH4_CV", "NO2_CV", "CO_CV", "SO2_CV"]
           if c in df.columns and c in core_features]
    pm = [c for c in ["mean_NO2", "mean_SO2", "mean_CH4", "mean_CO"]
          if c in df.columns and c in core_features]
    if pc:
        df["poll_abs_sum"] = df[pc].abs().sum(axis=1)
    if pcv:
        df["poll_cv_mean"] = df[pcv].mean(axis=1)
    if pm:
        df["poll_mean_sum"] = df[pm].sum(axis=1)
    for comp in ["poll_abs_sum", "poll_cv_mean", "poll_mean_sum"]:
        if comp in df.columns and INDUSTRY_COL in df.columns:
            df[f"{comp}_yi_rank"] = df.groupby(
                [YEAR_COL, INDUSTRY_COL])[comp].rank(pct=True)
            df[f"{comp}_yi_rank_lag1"] = df.groupby(
                ID_COL)[f"{comp}_yi_rank"].shift(1)
    if "punish_num" in df.columns and "punish_money" in df.columns:
        df["punish_interact"] = df["punish_num"] * df["punish_money"]
    if "punish_num" in df.columns and "recognize" in df.columns:
        df["punish_recognize"] = df["punish_num"] * df["recognize"]
    if "prize" in df.columns and "patent" in df.columns:
        df["prize_patent"] = df["prize"] * df["patent"]
    if "Size_yr_zscore" in df.columns:
        for poll in POLL_TREND_COLS:
            tc = f"{poll}_trend2"
            if tc in df.columns:
                df[f"{poll}_trend2_x_size_zscore"] = (
                    df[tc] * df["Size_yr_zscore"])
    for cv_col in ["CH4_CV", "NO2_CV", "CO_CV", "SO2_CV"]:
        if cv_col in df.columns:
            df[f"{cv_col}_x_size_zscore"] = df[cv_col] * df["Size_yr_zscore"]
    gas_grp = df.groupby(ID_COL)
    for gas in POLL_ANOMALY_GASES:
        zs_col = f"{gas}_zscore"; ex_col = f"{gas}_exceed"
        if zs_col in df.columns:
            df[f"{gas}_zscore_lag1"] = gas_grp[zs_col].shift(1)
            df[f"{gas}_zscore_sq"] = df[zs_col] ** 2
        if ex_col in df.columns:
            df[f"{gas}_exceed_diff"] = gas_grp[ex_col].diff(1)
            df[f"{gas}_exceed_sum2"] = (
                gas_grp[ex_col].rolling(2, min_periods=1)
                .sum().reset_index(level=0, drop=True))
        if zs_col in df.columns and "Size_yr_zscore" in df.columns:
            df[f"{gas}_zscore_x_size"] = df[zs_col] * df["Size_yr_zscore"]
        if ex_col in df.columns and "Lev" in df.columns:
            df[f"{gas}_exceed_x_lev"] = df[ex_col] * df["Lev"]
    if "SO2_exceed_x_lev" in df.columns and "Lev_x_SO2_exceed" not in df.columns:
        df["Lev_x_SO2_exceed"] = df["SO2_exceed_x_lev"]
    if "CO_zscore_x_size" in df.columns and "Size_x_CO_zscore" not in df.columns:
        df["Size_x_CO_zscore"] = df["CO_zscore_x_size"]
    df["After_Carbon_Neutral_Goal"] = (
        df[YEAR_COL] >= CARBON_NEUTRAL_YEAR).astype(np.int8)
    for gas in POLL_ANOMALY_GASES:
        zs_col = f"{gas}_zscore"
        if zs_col in df.columns:
            df[f"{gas}_zscore_x_carbon"] = (
                df[zs_col] * df["After_Carbon_Neutral_Goal"])
    if "poll_abs_sum" in df.columns:
        df["poll_abs_sum_x_carbon"] = (
            df["poll_abs_sum"] * df["After_Carbon_Neutral_Goal"])
    if INDUSTRY_COL in df.columns:
        def _up2(s):
            ch = s[s["y_bin"] == 1]
            return float((ch[TARGET_COL] > 0).mean()) if len(ch) > 0 else 0.0

        def _dn2(s):
            ch = s[s["y_bin"] == 1]
            return float((ch[TARGET_COL] < 0).mean()) if len(ch) > 0 else 0.0

        ist = (df.groupby([INDUSTRY_COL, YEAR_COL])["y_bin"]
               .agg(["mean", "sum", "count"]).reset_index())
        ist.columns = [INDUSTRY_COL, YEAR_COL,
                       "ind_change_rate", "ind_change_count", "ind_total"]
        ur = (df.groupby([INDUSTRY_COL, YEAR_COL]).apply(_up2)
              .reset_index(name="ind_up_rate"))
        dr = (df.groupby([INDUSTRY_COL, YEAR_COL]).apply(_dn2)
              .reset_index(name="ind_down_rate"))
        ist = ist.merge(ur, on=[INDUSTRY_COL, YEAR_COL], how="left")
        ist = ist.merge(dr, on=[INDUSTRY_COL, YEAR_COL], how="left")
        ist.sort_values([INDUSTRY_COL, YEAR_COL], inplace=True)
        for col in ["ind_change_rate", "ind_up_rate", "ind_down_rate"]:
            g = ist.groupby(INDUSTRY_COL)[col]
            ist[f"{col}_lag1"] = g.shift(1)
            ist[f"{col}_ma3_lag1"] = (
                g.rolling(3, min_periods=2).mean()
                .reset_index(level=0, drop=True).shift(1))
        df = df.merge(
            ist.drop(columns=["ind_change_count", "ind_total"],
                     errors="ignore"),
            on=[INDUSTRY_COL, YEAR_COL], how="left")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].fillna("")
    return df


def onehot_industry(df):
    if INDUSTRY_COL not in df.columns:
        return df
    return pd.concat(
        [df, pd.get_dummies(df[INDUSTRY_COL], prefix="ind", dtype=np.int8)],
        axis=1)


def get_feature_columns_fixed(df):
    exclude = {YEAR_COL, ID_COL, INDUSTRY_COL, TARGET_COL, "y_bin", "y_dir"}
    exclude.update(RAW_POLLUTION_COLS_DROP)
    for stem in IND_STAT_CURRENT_COLS:
        for suffix in ["", "_lag1", "_ma3_lag1"]:
            exclude.add(f"{stem}{suffix}")
    return [c for c in df.columns
            if c not in exclude and df[c].dtype in
            [np.float64, np.float32, np.int64, np.int32, np.int8]]

# ════════════════════════════════════════════════════════════════════════
# 分位数截断
# ════════════════════════════════════════════════════════════════════════
def cap_fit(X):
    return {j: (float(np.percentile(X[:, j], CAP_LOWER_PCT * 100)),
                float(np.percentile(X[:, j], CAP_UPPER_PCT * 100)))
            for j in range(X.shape[1])}


def cap_transform(X, bounds):
    X = X.copy()
    for j, (lo, hi) in bounds.items():
        X[:, j] = np.clip(X[:, j], lo, hi)
    return X

# ════════════════════════════════════════════════════════════════════════
# 特征选择 / 共线性过滤
# ════════════════════════════════════════════════════════════════════════
def select_features_rf(Xtr, ytr, names):
    if len(np.unique(ytr)) < 2:
        return np.ones(len(names), dtype=bool), list(names)
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, class_weight="balanced",
        n_jobs=N_JOBS, random_state=RANDOM_STATE)
    rf.fit(Xtr, ytr)
    imp = rf.feature_importances_
    mask = imp >= FEAT_IMP_THRESHOLD
    if mask.sum() < FEAT_MIN_KEEP:
        ti = np.argsort(imp)[::-1][:FEAT_MIN_KEEP]
        mask = np.zeros(len(imp), dtype=bool); mask[ti] = True
    return mask, [f for f, v in zip(names, mask) if v]


def select_features_catboost(Xtr, ytr, names, whitelist=None):
    """CatBoost 重要性 Top-K 特征选择 + 白名单强制保留。"""
    if whitelist is None:
        whitelist = STRICT_WHITELIST
    strict_present = [f for f in whitelist if f in names]
    if HAS_CB:
        try:
            selector = CatBoostClassifier(
                iterations=500, depth=6, l2_leaf_reg=10,
                random_strength=2, bootstrap_type="Bernoulli",
                subsample=0.8, verbose=0,
                random_seed=RANDOM_STATE, thread_count=N_JOBS_CATBOOST)
            selector.fit(Xtr, ytr)
            importances = np.array(selector.get_feature_importance())
            actual_k = min(FEAT_TOPK, len(names))
            topk_idx = set(np.argsort(importances)[::-1][:actual_k])
            mask = np.zeros(len(names), dtype=bool)
            mask[list(topk_idx)] = True
            for f in strict_present:
                if f in names and not mask[names.index(f)]:
                    mask[names.index(f)] = True
            return mask, [f for f, s in zip(names, mask) if s]
        except Exception as e:
            print(f"  [WARN] CatBoost特征选择失败（{e}），降级→RF")
    mask, sel = select_features_rf(Xtr, ytr, names)
    for f in strict_present:
        if f in names and not mask[names.index(f)]:
            mask[names.index(f)] = True
    return mask, [f for f, s in zip(names, mask) if s]


def filter_high_correlation(Xtr, feat_names, threshold=CORR_THRESHOLD):
    n_feat = Xtr.shape[1]
    if n_feat <= 1:
        return Xtr, feat_names, np.ones(n_feat, dtype=bool)
    var_order = np.argsort(-Xtr.var(axis=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_mat = np.corrcoef(Xtr.T)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)
    keep_set = []; drop_mask = np.zeros(n_feat, dtype=bool)
    for idx in var_order:
        if drop_mask[idx]:
            continue
        if any(abs(corr_mat[idx, k]) > threshold for k in keep_set):
            drop_mask[idx] = True
        else:
            keep_set.append(idx)
    keep_mask = ~drop_mask
    kept_names = [f for f, k in zip(feat_names, keep_mask) if k]
    Xtr_filtered = Xtr[:, keep_mask]
    print(f"  [共线性过滤] |r|>{threshold:.2f} "
          f"删除={int(drop_mask.sum())} 保留={len(kept_names)}")
    return Xtr_filtered, kept_names, keep_mask


def try_smote_3class(X, y):
    if not HAS_IMBLEARN:
        return X, y, "skip: no imblearn"
    mc = pd.Series(y).value_counts().min()
    k = min(5, mc - 1)
    if k < 1:
        return X, y, f"skip: minority={mc}"
    try:
        sm = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
        Xr, yr = sm.fit_resample(X, y)
        if len(np.unique(yr)) < len(np.unique(y)):
            return X, y, "skip: collapsed"
        return Xr, yr, f"ok: {len(y)}→{len(Xr)}"
    except Exception as e:
        return X, y, f"fail: {str(e)[:60]}"


def split_val_for_optuna(Xv, yv, dv, ratio=OPTUNA_VAL_RATIO, rs=RANDOM_STATE):
    rng = np.random.RandomState(rs)
    companies = np.array(sorted(dv[ID_COL].unique()))
    rng.shuffle(companies)
    n_tune = max(1, int(len(companies) * ratio))
    tune_ids = set(companies[:n_tune])
    tune_mask = dv[ID_COL].isin(tune_ids).values
    eval_mask = ~tune_mask
    if tune_mask.sum() == 0 or eval_mask.sum() == 0:
        return Xv, yv, Xv, yv
    print(f"  [验证集拆分] tune={tune_mask.sum()}行 "
          f"eval={eval_mask.sum()}行")
    return (Xv[tune_mask], yv[tune_mask], Xv[eval_mask], yv[eval_mask])

# ════════════════════════════════════════════════════════════════════════
# 线性模型预处理：StandardScaler + PCA
# ════════════════════════════════════════════════════════════════════════
def apply_pca_for_linear(
        Xtr: np.ndarray, Xv: np.ndarray, Xte: np.ndarray,
        var: float = PCA_VARIANCE,
        random_state=RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_feat = Xtr.shape[1]
    if n_feat < 2:
        sc = StandardScaler()
        return (sc.fit_transform(Xtr).astype(np.float32),
                sc.transform(Xv).astype(np.float32),
                sc.transform(Xte).astype(np.float32))
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xv_sc = scaler.transform(Xv)
    Xte_sc = scaler.transform(Xte)
    pca = PCA(n_components=var, random_state=random_state)
    Xtr_pca = pca.fit_transform(Xtr_sc)
    Xv_pca = pca.transform(Xv_sc)
    Xte_pca = pca.transform(Xte_sc)
    print(f"  [线性PCA] {n_feat}维 → {pca.n_components_}主成分"
          f"（解释方差 ≥ {var:.0%}）")
    return (Xtr_pca.astype(np.float32), Xv_pca.astype(np.float32),
            Xte_pca.astype(np.float32))

# ════════════════════════════════════════════════════════════════════════
# 指标计算
# ════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, proba3=None):
    m = {}; labels = LABEL_LIST
    m[_MK_ACC] = round(accuracy_score(y_true, y_pred), 5)
    m[_MK_F1] = round(f1_score(y_true, y_pred, average="macro",
                               zero_division=0, labels=labels), 5)
    m[_MK_PRE_MAC] = round(precision_score(y_true, y_pred, average="macro",
                                           zero_division=0, labels=labels), 5)
    m[_MK_REC_MAC] = round(recall_score(y_true, y_pred, average="macro",
                                        zero_division=0, labels=labels), 5)
    pc_, rc_, fc_, sc_ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    for i, (pk, rk, fk, sk) in enumerate([
            (_MK_PRE_DN, _MK_REC_DN, _MK_F1_DN, _MK_SUP_DN),
            (_MK_PRE_ST, _MK_REC_ST, _MK_F1_ST, _MK_SUP_ST),
            (_MK_PRE_UP, _MK_REC_UP, _MK_F1_UP, _MK_SUP_UP)]):
        m[pk] = round(pc_[i], 5); m[rk] = round(rc_[i], 5)
        m[fk] = round(fc_[i], 5); m[sk] = int(sc_[i])
    if proba3 is not None and proba3.shape[1] == 3:
        try:
            yb = label_binarize(y_true, classes=labels)
            m[_MK_AUC_ROC] = round(
                roc_auc_score(yb, proba3, multi_class="ovr", average="macro"),
                5)
        except Exception:
            m[_MK_AUC_ROC] = np.nan
        try:
            yb = label_binarize(y_true, classes=labels)
            ap_list = [average_precision_score(yb[:, ci], proba3[:, ci])
                       for ci in range(3) if yb[:, ci].sum() > 0]
            m[_MK_AUC_PR] = round(float(np.mean(ap_list)), 5) if ap_list \
                else np.nan
        except Exception:
            m[_MK_AUC_PR] = np.nan
    else:
        m[_MK_AUC_ROC] = np.nan
        m[_MK_AUC_PR] = np.nan
    return m


def _safe_normalize_proba(p3):
    p3 = np.where(np.isfinite(p3), p3, 0.0).astype(np.float32)
    rs = p3.sum(axis=1, keepdims=True)
    rs = np.where(rs > 1e-9, rs, 1.0)
    return p3 / rs


def _align_proba_to_label_list(mdl, X: np.ndarray) -> np.ndarray:
    """将模型 predict_proba 输出对齐到 LABEL_LIST 的列顺序。"""
    proba_raw = mdl.predict_proba(X)
    model_classes = list(mdl.classes_)
    n = len(X)
    aligned = np.zeros((n, 3), dtype=np.float32)
    for ti, lbl in enumerate(LABEL_LIST):
        if lbl in model_classes:
            si = model_classes.index(lbl)
            if si < proba_raw.shape[1]:
                aligned[:, ti] = proba_raw[:, si]
    return aligned

# ════════════════════════════════════════════════════════════════════════
# Bootstrap 统计检验（置信区间 + 配对显著性）
# ════════════════════════════════════════════════════════════════════════
def _boot_metric_vector(y_t, y_p, p3) -> Dict[str, float]:
    """单个 bootstrap 重采样上的指标向量（AUC 基于 proba3 重算）。"""
    labels = LABEL_LIST
    out: Dict[str, float] = {}
    try:
        out[_MK_ACC] = float(accuracy_score(y_t, y_p))
    except Exception:
        out[_MK_ACC] = np.nan
    try:
        out[_MK_F1] = float(f1_score(y_t, y_p, average="macro",
                                     zero_division=0, labels=labels))
    except Exception:
        out[_MK_F1] = np.nan
    try:
        pc_, rc_, _, _ = precision_recall_fscore_support(
            y_t, y_p, labels=labels, zero_division=0)
        dn = labels.index(-1)
        # 说明：重采样中缺失 -1 类时 sklearn 返回 0.0（zero_division=0）。
        # 配对检验中两模型使用完全相同的 idx，处理方式一致，差值仍可比。
        out[_MK_REC_DN] = float(rc_[dn])
        out[_MK_PRE_DN] = float(pc_[dn])
        out[_MK_REC_MAC] = float(np.mean(rc_))
        out[_MK_PRE_MAC] = float(np.mean(pc_))
    except Exception:
        for k in (_MK_REC_DN, _MK_PRE_DN, _MK_REC_MAC, _MK_PRE_MAC):
            out[k] = np.nan
    if p3 is not None and getattr(p3, "shape", (0, 0))[1] == 3:
        try:
            yb = label_binarize(y_t, classes=labels)
            if yb.shape[1] == 3 and all(yb[:, i].sum() > 0 for i in range(3)):
                out[_MK_AUC_ROC] = float(roc_auc_score(
                    yb, p3, multi_class="ovr", average="macro"))
            else:
                out[_MK_AUC_ROC] = np.nan
        except Exception:
            out[_MK_AUC_ROC] = np.nan
        try:
            yb = label_binarize(y_t, classes=labels)
            ap_list = [average_precision_score(yb[:, ci], p3[:, ci])
                       for ci in range(3) if yb[:, ci].sum() > 0]
            out[_MK_AUC_PR] = float(np.mean(ap_list)) if ap_list else np.nan
        except Exception:
            out[_MK_AUC_PR] = np.nan
    else:
        out[_MK_AUC_ROC] = np.nan
        out[_MK_AUC_PR] = np.nan
    return out


def _summarize_boot(vals, alpha=BOOT_ALPHA, min_ok=BOOT_MIN_OK) -> Dict:
    """将 bootstrap 重复值汇总为 mean / CI / SD。"""
    arr = np.asarray(vals, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) > min_ok:
        return {
            "mean": float(np.mean(valid)),
            "ci_low": float(np.percentile(valid, 100 * alpha / 2)),
            "ci_high": float(np.percentile(valid, 100 * (1 - alpha / 2))),
            "sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            "n_valid": int(len(valid)),
        }
    return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "sd": np.nan, "n_valid": int(len(valid))}


def bootstrap_metrics(y_true, y_pred, proba3, n_boot=N_BOOT,
                      alpha=BOOT_ALPHA, seed=BOOT_SEED) -> Dict:
    """分层 bootstrap 计算各指标的 95% 置信区间（AUC 基于 proba3 重算）。"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if proba3 is not None:
        proba3 = np.asarray(proba3, dtype=np.float32)
    rng = np.random.RandomState(seed)
    metric_keys = [_MK_F1, _MK_AUC_ROC, _MK_AUC_PR, _MK_REC_DN,
                   _MK_PRE_DN, _MK_ACC, _MK_REC_MAC, _MK_PRE_MAC]
    boot_results = {k: [] for k in metric_keys}
    class_indices = [np.where(y_true == c)[0] for c in np.unique(y_true)]
    class_indices = [ix for ix in class_indices if len(ix) > 0]
    if not class_indices:
        return {k: _summarize_boot([], alpha) for k in metric_keys}
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(ix, size=len(ix), replace=True)
            for ix in class_indices])
        p3_b = proba3[idx] if proba3 is not None else None
        m = _boot_metric_vector(y_true[idx], y_pred[idx], p3_b)
        for k in metric_keys:
            boot_results[k].append(m.get(k, np.nan))
    return {k: _summarize_boot(v, alpha) for k, v in boot_results.items()}


def paired_bootstrap_test(y_true, pred_a, pred_b, proba_a, proba_b,
                          n_boot=N_BOOT_MODEL, seed=BOOT_SEED, side="two",
                          cluster_ids=None) -> Dict:
    """
    ★ 配对 bootstrap 检验：模型A（Full） vs 模型B（No-Gas）。
    ★本版：检验指标集与主评价指标口径一致（六项）。
    配对设计：同一重采样 idx 同时用于两个模型，
    差值的方差只反映模型差异，不反映样本抽样差异。
    side="two"（默认）: 双侧 H1: A≠B —— p = 2·min(p_left, p_right)；
    side="one" : 单侧 H1: A>B。
    cluster_ids: 提供时按企业 id 聚类重抽样。
    """
    y_true = np.asarray(y_true).ravel()
    pred_a = np.asarray(pred_a).ravel()
    pred_b = np.asarray(pred_b).ravel()
    if proba_a is not None:
        proba_a = np.asarray(proba_a, dtype=np.float32)
    if proba_b is not None:
        proba_b = np.asarray(proba_b, dtype=np.float32)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if cluster_ids is not None:
        ids = np.asarray(cluster_ids).ravel()
        uniq = np.array(sorted(pd.unique(ids)))
        rid = {v: i for i, v in enumerate(uniq)}
        rows_by = defaultdict(list)
        for i, v in enumerate(ids):
            rows_by[rid[v]].append(i)
        row_lists = [rows_by[i] for i in range(len(uniq))]

        def _draw():
            chosen = rng.choice(len(uniq), size=len(uniq), replace=True)
            return np.concatenate([row_lists[c] for c in chosen])
    else:
        def _draw():
            return rng.randint(0, n, size=n)
    # ★ 检验指标集 = 主评价指标口径（六项）
    metric_keys = [_MK_REC_MAC, _MK_AUC_ROC, _MK_AUC_PR,
                   _MK_REC_DN, _MK_PRE_MAC, _MK_ACC]
    diffs = {k: [] for k in metric_keys}
    for _ in range(n_boot):
        idx = _draw()
        y_t = y_true[idx]
        m_a = _boot_metric_vector(
            y_t, pred_a[idx], proba_a[idx] if proba_a is not None else None)
        m_b = _boot_metric_vector(
            y_t, pred_b[idx], proba_b[idx] if proba_b is not None else None)
        for k in metric_keys:
            va = m_a.get(k, np.nan); vb = m_b.get(k, np.nan)
            diffs[k].append(va - vb if not (np.isnan(va) or np.isnan(vb))
                            else np.nan)
    results: Dict[str, Dict] = {}
    for k in metric_keys:
        arr = np.asarray(diffs[k], dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) > BOOT_MIN_OK:
            nv = len(valid)
            ci_low = float(np.percentile(valid, 2.5))
            ci_high = float(np.percentile(valid, 97.5))
            # +1 连续性校正避免 p 恰为 0
            p_greater = float((np.sum(valid <= 0) + 1) / (nv + 1))  # H1: A>B
            p_less = float((np.sum(valid >= 0) + 1) / (nv + 1))    # H1: A<B
            if side == "one":
                p_value = p_greater
                significant = bool(ci_low > 0)
            else:
                p_value = float(min(1.0, 2.0 * min(p_greater, p_less)))
                significant = bool((ci_low > 0) or (ci_high < 0))
            results[k] = {
                "diff_mean": float(np.mean(valid)),
                "diff_ci_low": ci_low,
                "diff_ci_high": ci_high,
                "diff_sd": float(np.std(valid, ddof=1)),
                "p_value": p_value,
                "significant_005": significant,
                "side": side,
                "n_valid_boot": int(nv),
            }
        else:
            results[k] = {
                "diff_mean": np.nan, "diff_ci_low": np.nan,
                "diff_ci_high": np.nan, "diff_sd": np.nan,
                "p_value": np.nan, "significant_005": False,
                "side": side, "n_valid_boot": int(len(valid)),
            }
    return results

# ════════════════════════════════════════════════════════════════════════
# ★ 池化（多窗口合并）预测拼接与池化 bootstrap
# ════════════════════════════════════════════════════════════════════════
def pool_config_preds(all_predictions, config_name, model_name,
                      window_order=None) -> Optional[Dict]:
    """
    将某配置下某模型的各窗口测试预测纵向拼接（池化）。
    返回 {"y_true","y_pred","proba3","ids","windows"}；无可用窗口返回 None。
    """
    if window_order is None:
        window_order = [w["window_id"] for w in FIXED_WINDOWS]
    yts, yps, p3s, idl, used = [], [], [], [], []
    for wid in window_order:
        p = all_predictions.get((config_name, wid), {}).get(model_name)
        if p is None:
            continue
        yts.append(np.asarray(p[0]))
        yps.append(np.asarray(p[1]))
        p3s.append(np.asarray(p[2], dtype=np.float32))
        idl.append(np.asarray(p[3]))
        used.append(wid)
    if not yts:
        return None
    return {
        "y_true": np.concatenate(yts),
        "y_pred": np.concatenate(yps),
        "proba3": np.vstack(p3s),
        "ids": np.concatenate(idl),
        "windows": used,
    }


def pooled_bootstrap_metrics(pc: Dict, n_boot=N_BOOT_MODEL, alpha=BOOT_ALPHA,
                             seed=BOOT_SEED, cluster=True) -> Tuple[Dict, Dict]:
    """
    ★ 池化（多窗口合并）点估计 + bootstrap SD / 95% CI。
    cluster=True 且企业 id 存在跨窗口重复时，按企业聚类重抽样；
    否则按类别分层 IID 重抽样。
    """
    y_true = pc["y_true"]; y_pred = pc["y_pred"]; proba3 = pc["proba3"]
    point = compute_metrics(y_true, y_pred, proba3)
    rng = np.random.RandomState(seed)
    metric_keys = [_MK_F1, _MK_AUC_ROC, _MK_AUC_PR, _MK_REC_DN,
                   _MK_PRE_DN, _MK_ACC, _MK_REC_MAC, _MK_PRE_MAC]
    ids = np.asarray(pc["ids"]).ravel()
    use_cluster = bool(cluster) and len(pd.unique(ids)) < len(ids)
    if use_cluster:
        uniq = np.array(sorted(pd.unique(ids)))
        rid = {v: i for i, v in enumerate(uniq)}
        rows_by = defaultdict(list)
        for i, v in enumerate(ids):
            rows_by[rid[v]].append(i)
        row_lists = [rows_by[i] for i in range(len(uniq))]

        def _draw():
            chosen = rng.choice(len(uniq), size=len(uniq), replace=True)
            return np.concatenate([row_lists[c] for c in chosen])
    else:
        cls_ix = [np.where(y_true == c)[0] for c in np.unique(y_true)]
        cls_ix = [ix for ix in cls_ix if len(ix) > 0]

        def _draw():
            return np.concatenate([
                rng.choice(ix, size=len(ix), replace=True)
                for ix in cls_ix])
    boot = {k: [] for k in metric_keys}
    for _ in range(n_boot):
        idx = _draw()
        m = _boot_metric_vector(y_true[idx], y_pred[idx], proba3[idx])
        for k in metric_keys:
            boot[k].append(m.get(k, np.nan))
    return point, {k: _summarize_boot(v, alpha) for k, v in boot.items()}

# ════════════════════════════════════════════════════════════════════════
# ★ 混淆矩阵（主模型池化）：计算 + 绘图
# ════════════════════════════════════════════════════════════════════════
def compute_confusion_matrices(y_true, y_pred):
    """
    按 LABEL_LIST 顺序（-1,0,1 → 行=真实，列=预测）计算 3×3 混淆矩阵：
    返回 (计数矩阵 cm, 行归一化 cm_row[召回率视角], 列归一化 cm_col[精确率视角])。
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(np.asarray(y_true).ravel(),
                          np.asarray(y_pred).ravel(),
                          labels=LABEL_LIST).astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    col_sums = cm.sum(axis=0, keepdims=True)
    cm_row = np.divide(cm, row_sums, out=np.zeros_like(cm),
                       where=row_sums > 0)
    cm_col = np.divide(cm, col_sums, out=np.zeros_like(cm),
                       where=col_sums > 0)
    return cm, cm_row, cm_col


def plot_confusion_matrix_pair(cm_counts, cm_row_norm, title, save_path,
                               dpi=CONFUSION_DPI) -> bool:
    """池化混淆矩阵双子图 PNG：左=计数（普通），右=行归一化（每类召回率）。"""
    if not HAS_MPL:
        return False
    labels_en = ["Down (-1)", "Stable (0)", "Up (+1)"]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4),
                             constrained_layout=True)
    panels = [
        (axes[0], cm_counts, "Counts", lambda v: f"{int(v)}"),
        (axes[1], cm_row_norm, "Row-normalized (recall)",
         lambda v: f"{v * 100:.1f}%"),
    ]
    for ax, mat, sub, fmt in panels:
        is_norm = sub.startswith("Row")
        vmax = 1.0 if is_norm else max(float(mat.max()), 1e-9)
        im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=vmax)
        thresh = vmax * 0.6
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, fmt(float(mat[i, j])),
                        ha="center", va="center", fontsize=12,
                        color="white" if mat[i, j] > thresh else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(labels_en, fontsize=10)
        ax.set_yticks(range(3)); ax.set_yticklabels(labels_en, fontsize=10)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label", fontsize=11)
        ax.set_title(sub, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=13)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True

# ════════════════════════════════════════════════════════════════════════
# 基线模型工厂（★ CatBoost 基线已删除）
# ════════════════════════════════════════════════════════════════════════
def _auto_params(n: int) -> Dict:
    if n < 10_000:
        return dict(ne=300, depth=6, lr=0.05, ml=20)
    elif n < 20_000:
        return dict(ne=500, depth=8, lr=0.05, ml=12)
    else:
        return dict(ne=600, depth=9, lr=0.04, ml=10)


def build_baseline_model(
        model_name: str,
        class_weight_dict: dict = None,
        random_state: int = RANDOM_STATE,
):
    """返回 (model_instance, needs_mapped_labels)。★ 无 CatBoost 基线。"""
    cw_sklearn = class_weight_dict if class_weight_dict else "balanced"
    if model_name == "LR":
        import sklearn
        lr_kwargs = dict(solver="saga", max_iter=2000,
                         class_weight=cw_sklearn, C=0.1,
                         random_state=random_state, n_jobs=N_JOBS)
        sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if sk_ver < (1, 3):
            lr_kwargs["multi_class"] = "multinomial"
        return LogisticRegression(**lr_kwargs), False
    elif model_name == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight=cw_sklearn, n_jobs=N_JOBS,
            random_state=random_state), False
    elif model_name == "BalancedRF":
        if not HAS_BALANCED_RF:
            return None, False
        return BalancedRandomForestClassifier(
            n_estimators=300, replacement=True, n_jobs=N_JOBS,
            random_state=random_state), False
    elif model_name == "EasyEnsemble":
        if not HAS_EASY_ENSEMBLE:
            return None, False
        return EasyEnsembleClassifier(
            n_estimators=10, n_jobs=N_JOBS,
            random_state=random_state), False
    elif model_name == "XGBoost":
        if not HAS_XGB:
            return None, True
        a = _auto_params(500)
        return xgb.XGBClassifier(
            n_estimators=a["ne"], max_depth=a["depth"],
            learning_rate=a["lr"], subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", tree_method="hist",
            random_state=random_state, n_jobs=N_JOBS, verbosity=0,
            use_label_encoder=False), True
    elif model_name == "LightGBM":
        if not HAS_LGB:
            return None, True
        return None, True
    elif model_name == "LinearSVM":
        base = LinearSVC(multi_class="ovr", class_weight=cw_sklearn,
                         max_iter=2000, random_state=random_state, C=0.1)
        return CalibratedClassifierCV(
            estimator=base, method="sigmoid", cv=3), False
    elif model_name == "HistGradBoost":
        try:
            mdl = HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=0.1,
                class_weight="balanced", random_state=random_state)
        except TypeError:
            mdl = HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=0.1,
                random_state=random_state)
        return mdl, False
    else:
        raise ValueError(f"未知基线模型: {model_name}")


def train_and_predict_baseline(
        model_name: str,
        Xtr: np.ndarray, ytr: np.ndarray,
        Xv: np.ndarray, yv: np.ndarray,
        Xte: np.ndarray, yte: np.ndarray,
        class_weight_dict: dict = None,
        random_state: int = RANDOM_STATE,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """训练基线模型并返回 (metrics, y_true, y_pred, proba3)。"""
    if class_weight_dict is None:
        class_weight_dict = {-1: 1.0, 0: 1.0, 1: 1.0}
    mdl, needs_mapped = build_baseline_model(model_name, class_weight_dict,
                                             random_state)
    if mdl is None and model_name != "LightGBM":
        return {}, yte, np.zeros_like(yte), np.zeros((len(yte), 3))
    sample_sw = build_sample_weights(ytr, class_weight_dict)
    ytr_fit = map_labels(ytr) if needs_mapped else ytr
    try:
        if model_name == "LightGBM":
            if not HAS_LGB:
                return {}, yte, np.zeros_like(yte), np.zeros((len(yte), 3))
            a = _auto_params(len(ytr))
            params = {
                "objective": "multiclass", "num_class": 3,
                "metric": "multi_logloss", "seed": random_state,
                "learning_rate": a["lr"], "num_leaves": 63,
                "feature_fraction": 0.8, "bagging_fraction": 0.8,
                "bagging_freq": 5, "min_child_samples": a["ml"],
                "verbosity": -1, "min_gain_to_split": 0.01,
            }
            lgb_model = lgb.train(
                params,
                lgb.Dataset(Xtr, label=map_labels(ytr),
                            weight=sample_sw.astype(np.float32)),
                num_boost_round=a["ne"])
            proba_raw = lgb_model.predict(Xte)
            y_pred = inv_labels(np.argmax(proba_raw, axis=1))
        elif model_name == "XGBoost":
            a = _auto_params(len(ytr))
            mdl.n_estimators = a["ne"]
            mdl.max_depth = a["depth"]
            mdl.learning_rate = a["lr"]
            mdl.fit(Xtr, ytr_fit, sample_weight=sample_sw.astype(np.float32))
            proba_raw = mdl.predict_proba(Xte)
            y_pred = inv_labels(mdl.predict(Xte))
        elif model_name == "HistGradBoost":
            try:
                mdl.fit(Xtr, ytr_fit,
                        sample_weight=sample_sw.astype(np.float32))
            except TypeError:
                mdl.fit(Xtr, ytr_fit)
            proba_raw = _align_proba_to_label_list(mdl, Xte)
            y_pred = mdl.predict(Xte)
        else:
            mdl.fit(Xtr, ytr_fit)
            proba_raw = _align_proba_to_label_list(mdl, Xte)
            y_pred = mdl.predict(Xte)
        proba_raw = _safe_normalize_proba(
            np.array(proba_raw, dtype=np.float32))
        metrics = compute_metrics(yte, y_pred, proba_raw)
        return metrics, yte, y_pred, proba_raw
    except Exception as e:
        print(f"  [{model_name}] 训练/预测失败: {e}")
        return {}, yte, np.zeros_like(yte), np.zeros((len(yte), 3))

# ════════════════════════════════════════════════════════════════════════
# Optuna 目标函数（主模型）—— 调参目标保持 Macro-F1（其他不变）
# ════════════════════════════════════════════════════════════════════════
def _catboost_optuna_objective(trial, Xtr, ytr, Xv_tune, yv_tune, sample_sw):
    if not HAS_CB:
        return 0.0
    bootstrap_type = trial.suggest_categorical("bootstrap_type",
                                               ["Bernoulli", "Bayesian"])
    params = {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, OPTUNA_L2_MAX, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "bootstrap_type": bootstrap_type,
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "verbose": 0, "random_seed": RANDOM_STATE,
        "thread_count": N_JOBS_CATBOOST,
    }
    if bootstrap_type == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    else:
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 2.0)
    try:
        mdl = CatBoostClassifier(**params)
        mdl.fit(Xtr, ytr, sample_weight=sample_sw)
        pred = inv_labels(mdl.predict(Xv_tune).flatten().astype(int))
        return f1_score(yv_tune, pred, average="macro", zero_division=0)
    except Exception:
        return 0.0

# ════════════════════════════════════════════════════════════════════════
# 主模型 Pipeline（CatBoost + Optuna）
# ════════════════════════════════════════════════════════════════════════
def run_catboost_main_window(Xtr_tree, ytr, Xtr_smote, ytr_smote,
                             Xv_tree, yv, Xte_tree, yte, dv):
    """主模型（CatBoost + Optuna）单窗口训练。
    返回: (metrics, debug_info, y_true_test, y_pred_test, proba_test)"""
    if not HAS_CB:
        return ({}, {"skip_reason": "no_catboost"},
                np.asarray(yte), np.zeros_like(np.asarray(yte)),
                np.zeros((len(yte), 3), dtype=np.float32))
    debug_info = {}
    Xv_tune, yv_tune, Xv_eval, yv_eval = split_val_for_optuna(Xv_tree, yv, dv)
    cw_smote = compute_class_weights_3c(ytr_smote)
    sample_sw = build_sample_weights(ytr_smote, cw_smote).astype(np.float32)
    ytr_mapped = map_labels(ytr_smote)
    n_trials = int(np.clip(len(Xtr_smote) // OPTUNA_TRIALS_PER_N,
                           OPTUNA_N_TRIALS_MIN, OPTUNA_N_TRIALS_MAX))
    print(f"  [Optuna] n_trials={n_trials}（训练集{len(Xtr_smote)}行）")
    debug_info["n_trials_actual"] = n_trials
    if HAS_OPTUNA:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                               n_warmup_steps=0))
        study.optimize(
            lambda trial: _catboost_optuna_objective(
                trial, Xtr_smote, ytr_mapped, Xv_tune, yv_tune, sample_sw),
            n_trials=n_trials, show_progress_bar=False, n_jobs=1)
        best_params = study.best_params
        debug_info["optuna_best_val_f1"] = round(study.best_value, 5)
        debug_info["optuna_best_params"] = best_params
        print(f"  [Optuna] 最优val F1={study.best_value:.4f}")
    else:
        best_params = {
            "iterations": 500, "depth": 6, "learning_rate": 0.05,
            "l2_leaf_reg": 5.0, "subsample": 0.8, "colsample_bylevel": 0.8,
            "random_strength": 1.0, "min_data_in_leaf": 10,
            "bootstrap_type": "Bernoulli",
        }
        debug_info["optuna_best_val_f1"] = np.nan
        debug_info["optuna_best_params"] = best_params
    bootstrap_type = best_params.get("bootstrap_type", "Bernoulli")
    final_params = {
        "iterations": best_params.get("iterations", 500),
        "depth": best_params.get("depth", 6),
        "learning_rate": best_params.get("learning_rate", 0.05),
        "l2_leaf_reg": best_params.get("l2_leaf_reg", 5.0),
        "colsample_bylevel": best_params.get("colsample_bylevel", 0.8),
        "random_strength": best_params.get("random_strength", 1.0),
        "min_data_in_leaf": best_params.get("min_data_in_leaf", 10),
        "bootstrap_type": bootstrap_type,
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "random_seed": RANDOM_STATE, "thread_count": N_JOBS_CATBOOST,
    }
    if bootstrap_type == "Bernoulli":
        final_params["subsample"] = best_params.get("subsample", 0.8)
    else:
        final_params["bagging_temperature"] = best_params.get(
            "bagging_temperature", 0.5)
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(Xtr_smote, ytr_mapped, sample_weight=sample_sw)
    print(f"  [最终模型] iters={final_params['iterations']}"
          f" depth={final_params['depth']}"
          f" lr={final_params['learning_rate']:.4f}")
    proba_te = _safe_normalize_proba(
        final_model.predict_proba(Xte_tree).astype(np.float32))
    pred_te = inv_labels(final_model.predict(Xte_tree).flatten().astype(int))
    metrics = compute_metrics(yte, pred_te, proba_te)
    pred_ve = inv_labels(final_model.predict(Xv_eval).flatten().astype(int))
    val_mf1 = f1_score(yv_eval, pred_ve, average="macro", zero_division=0)
    debug_info["val_eval_macro_f1"] = round(val_mf1, 5)
    print(f"  [{MAIN_MODEL_NAME}] val F1={val_mf1:.4f}"
          f" test Macro-Recall="
          f"{metrics.get(_MK_REC_MAC, float('nan')):.4f}")
    return metrics, debug_info, np.asarray(yte), pred_te, proba_te

# ════════════════════════════════════════════════════════════════════════
# 记录构建辅助
# ════════════════════════════════════════════════════════════════════════
def _make_record(model_name, window_id, train_years, val_year, test_year,
                 train_n, test_n, n_after_base, metrics, elapsed,
                 preprocessing_route, extra=None):
    """构建统一的逐窗口指标记录（基线与主模型共用字段命名）。"""
    rec = {
        "window_id（窗口标识）": window_id,
        "model（模型名称）": model_name,
        "train_years（训练年份）": str(train_years),
        "val_year（验证年份）": val_year,
        "test_year（测试年份）": test_year,
        "train_n": train_n,
        "test_n": test_n,
        "n_feats_after_base_drop": n_after_base,
        "preprocessing_route（预处理管道）": preprocessing_route,
        "elapsed_sec": elapsed,
    }
    if extra:
        rec.update(extra)
    for mk in ALL_METRIC_COLS:
        rec[mk] = metrics.get(mk, np.nan)
    return rec

# ════════════════════════════════════════════════════════════════════════
# 单窗口训练 Pipeline（先跑全部基线模型，再跑主模型）
# ════════════════════════════════════════════════════════════════════════
def run_fixed_window(
        dtr_raw, dv_raw, dte_raw, feat_names, window_id,
        train_years, val_year, test_year,
        base_drop_roots=None, drop_gas: bool = False,
        config_name: str = "Full",
) -> Tuple[List[Dict], Dict[str, Tuple]]:
    """
    返回:
      records: List[Dict] — 本窗口所有模型的指标记录
      preds_by_model: {model_name: (y_true, y_pred, proba3, test_ids)}
    """
    if base_drop_roots is None:
        base_drop_roots = BASE_DROP_ROOTS
    ok, reason = _check_label_integrity(dtr_raw, dte_raw, window_id)
    if not ok:
        return [], {}
    records: List[Dict] = []
    preds_by_model: Dict[str, Tuple] = {}
    # ── 行业统计（防泄漏） ───────────────────────────────────────────
    dtr, dv, dte = compute_industry_stats_from_train(dtr_raw, dv_raw, dte_raw)
    avail_all = [f for f in feat_names
                 if f in dtr.columns and f in dv.columns
                 and f in dte.columns]
    avail = get_avail_after_base_drop(avail_all, base_drop_roots)
    # ── 消融：移除气体特征 ────────────────────────────────────────────
    n_gas_removed = 0
    if drop_gas:
        n_before = len(avail)
        avail = remove_gas_features(avail)
        n_gas_removed = n_before - len(avail)
        effective_whitelist: List[str] = []
        print(f"  [消融] STRICT_WHITELIST 已清空"
              f"（原 {len(STRICT_WHITELIST)} 项全为气体特征）")
    else:
        effective_whitelist = list(STRICT_WHITELIST)
    if not avail:
        print(f"  [SKIP] 剔除后无可用特征列")
        return [], {}
    # ── ★ 测试集企业 id（聚类重抽样用）★ ────────────────────────────
    test_ids = (dte[ID_COL].values if ID_COL in dte.columns
                else np.arange(len(dte)))
    test_ids = np.asarray(test_ids)
    # ── 原始特征矩阵构建 ─────────────────────────────────────────────
    def _bX(df_):
        arr = df_[avail].values.astype(np.float32)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    Xtr_r = _bX(dtr); Xv_r = _bX(dv); Xte_r = _bX(dte)
    bounds = cap_fit(Xtr_r)
    Xtr_c = cap_transform(Xtr_r, bounds)
    Xv_c = cap_transform(Xv_r, bounds)
    Xte_c = cap_transform(Xte_r, bounds)

    def _l3(df_):
        return np.where(df_[TARGET_COL] > 0, 1,
                        np.where(df_[TARGET_COL] < 0, -1, 0)).astype(int)

    ytr3 = _l3(dtr); yv3 = _l3(dv); yte3 = _l3(dte)
    if len(np.unique(ytr3)) < 2:
        print("  [SKIP] 训练集类别数不足")
        return [], {}
    cw_dict = compute_class_weights_3c(ytr3)
    print(f"  [动态类权重] {cw_dict}")
    # ── 特征选择 → 共线性过滤 ────────────────────────────────────────
    ytrb = dtr["y_bin"].values
    feat_mask, sel_feats = select_features_catboost(
        Xtr_c, ytrb, avail, whitelist=effective_whitelist)
    Xtr_s = Xtr_c[:, feat_mask]; Xv_s = Xv_c[:, feat_mask]
    Xte_s = Xte_c[:, feat_mask]
    print(f"  [特征选择] {feat_mask.sum()} / {len(avail)}"
          f"（白名单强制保留 "
          f"{len([f for f in effective_whitelist if f in avail])} 项）")
    Xtr_f, kept_feats, corr_mask = filter_high_correlation(Xtr_s, sel_feats)
    Xv_f = Xv_s[:, corr_mask]; Xte_f = Xte_s[:, corr_mask]
    # ── 防缓存 / 防污染校验 ──────────────────────────────────────────
    feat_hash = stable_feature_hash(kept_feats)
    gas_in_final = identify_gas_features(kept_feats)
    n_gas_in_final = len(gas_in_final)
    print(f"  [验证] 配置={'No-Gas' if drop_gas else 'Full'}"
          f" 窗口={window_id} 特征矩阵shape={Xtr_f.shape}"
          f" 特征列哈希={feat_hash}")
    print(f"  [验证] 最终特征集残留气体列数={n_gas_in_final}"
          f" 基底后移除气体列数={n_gas_removed}")
    if drop_gas and n_gas_in_final > 0:
        print(f"  [验证] ❌ 警告：No-Gas 档仍残留气体特征 "
              f"{sorted(gas_in_final)[:10]}")
    common_extra = {
        "drop_gas": bool(drop_gas),
        "n_gas_features_removed": int(n_gas_removed),
        "n_feats_final": int(Xtr_f.shape[1]),
        "feature_set_hash": feat_hash,
        "n_gas_left_in_final": int(n_gas_in_final),
        "whitelist_used": ",".join(effective_whitelist) or "(空)",
    }
    # ── 三条预处理管道 ────────────────────────────────────────────────
    Xtr_tree_sm, ytr_tree_sm, smote_tree = try_smote_3class(Xtr_f, ytr3)
    cw_tree = compute_class_weights_3c(ytr_tree_sm)
    print(f"  [树模型SMOTE] {smote_tree}")
    Xtr_lin, Xv_lin, Xte_lin = apply_pca_for_linear(Xtr_f, Xv_f, Xte_f)
    Xtr_lin_sm, ytr_lin_sm, smote_lin = try_smote_3class(Xtr_lin, ytr3)
    cw_lin = compute_class_weights_3c(ytr_lin_sm)
    print(f"  [线性模型SMOTE] {smote_lin}")
    cw_nosmote = compute_class_weights_3c(ytr3)

    # ══════════════════════════════════════════════════════════════
    # 基线模型循环（★ 8 个基线，无 CatBoost）
    # ══════════════════════════════════════════════════════════════
    for model_name in BASELINE_MODEL_NAMES:
        t0 = time.time()
        print(f"\n  → [{config_name}][{window_id}] 基线: {model_name}")
        if model_name in LINEAR_MODELS:
            Xtr_u, ytr_u = Xtr_lin_sm, ytr_lin_sm
            Xv_u, Xte_u = Xv_lin, Xte_lin
            cw_u = cw_lin
            route_tag = "Linear+PCA+SMOTE"
        elif model_name in NO_SMOTE_MODELS:
            Xtr_u, ytr_u = Xtr_f, ytr3
            Xv_u, Xte_u = Xv_f, Xte_f
            cw_u = cw_nosmote
            route_tag = "NoSMOTE（内置平衡）"
        else:
            Xtr_u, ytr_u = Xtr_tree_sm, ytr_tree_sm
            Xv_u, Xte_u = Xv_f, Xte_f
            cw_u = cw_tree
            route_tag = "Tree+SMOTE"
        metrics, y_t, y_p, p3 = train_and_predict_baseline(
            model_name=model_name, Xtr=Xtr_u, ytr=ytr_u,
            Xv=Xv_u, yv=yv3, Xte=Xte_u, yte=yte3,
            class_weight_dict=cw_u, random_state=RANDOM_STATE)
        elapsed = round(time.time() - t0, 1)
        rec = _make_record(
            model_name=model_name, window_id=window_id,
            train_years=train_years, val_year=val_year,
            test_year=test_year, train_n=len(dtr), test_n=len(dte),
            n_after_base=len(avail), metrics=metrics, elapsed=elapsed,
            preprocessing_route=route_tag, extra=dict(common_extra))
        records.append(rec)
        if metrics:
            preds_by_model[model_name] = (
                np.asarray(y_t), np.asarray(y_p),
                np.asarray(p3, dtype=np.float32), test_ids)
            print(f"  [{model_name}] Macro-Recall="
                  f"{metrics.get(_MK_REC_MAC, np.nan):.4f}"
                  f" AUC={metrics.get(_MK_AUC_ROC, np.nan):.4f} ({elapsed}s)")
        else:
            print(f"  [{model_name}] 无有效结果（依赖缺失或训练失败）")

    # ══════════════════════════════════════════════════════════════
    # 主模型：CatBoost + Optuna
    # ══════════════════════════════════════════════════════════════
    print(f"\n  → [{config_name}][{window_id}] 主模型: {MAIN_MODEL_NAME}")
    t_main = time.time()
    (metrics_main, dbg, y_true_main, pred_te_main, proba_te_main
     ) = run_catboost_main_window(
        Xtr_tree=Xtr_f, ytr=ytr3,
        Xtr_smote=Xtr_tree_sm, ytr_smote=ytr_tree_sm,
        Xv_tree=Xv_f, yv=yv3, Xte_tree=Xte_f, yte=yte3, dv=dv_raw)
    elapsed_main = round(time.time() - t_main, 1)
    if not metrics_main:
        print(f"  [SKIP] 主模型训练失败")
        records = [{CONFIG_COL: config_name, **r} for r in records]
        return records, preds_by_model
    # ★ 留存主模型测试集预测 ★
    preds_by_model[MAIN_MODEL_NAME] = (
        np.asarray(y_true_main), np.asarray(pred_te_main),
        np.asarray(proba_te_main, dtype=np.float32), test_ids)
    print(f"\n  ★ [{config_name}][{window_id}] 主模型 "
          f"Macro-Recall={metrics_main.get(_MK_REC_MAC, np.nan):.4f}"
          f" AUC-ROC={metrics_main.get(_MK_AUC_ROC, np.nan):.4f}"
          f" ({elapsed_main}s)")
    main_extra = dict(common_extra)
    main_extra.update({
        "n_feats_after_selection": int(feat_mask.sum()),
        "n_feats_after_corr_filter": Xtr_f.shape[1],
        "smote_info": smote_tree,
        "optuna_best_val_f1": dbg.get("optuna_best_val_f1", np.nan),
        "val_eval_macro_f1": dbg.get("val_eval_macro_f1", np.nan),
        "n_optuna_trials": dbg.get("n_trials_actual", np.nan),
        "best_iterations": dbg.get("optuna_best_params", {}).get(
            "iterations", np.nan),
        "best_depth": dbg.get("optuna_best_params", {}).get("depth", np.nan),
        "best_lr": dbg.get("optuna_best_params", {}).get(
            "learning_rate", np.nan),
        "best_l2_leaf_reg": dbg.get("optuna_best_params", {}).get(
            "l2_leaf_reg", np.nan),
    })
    rec_main = _make_record(
        model_name=MAIN_MODEL_NAME, window_id=window_id,
        train_years=train_years, val_year=val_year, test_year=test_year,
        train_n=len(dtr), test_n=len(dte), n_after_base=len(avail),
        metrics=metrics_main, elapsed=elapsed_main,
        preprocessing_route="Tree+SMOTE+Optuna", extra=main_extra)
    records.append(rec_main)
    records = [{CONFIG_COL: config_name, **r} for r in records]
    gc.collect()
    return records, preds_by_model

# ════════════════════════════════════════════════════════════════════════
# 跨窗口均值汇总（按模型分组）
# ════════════════════════════════════════════════════════════════════════
def build_summary(df_detail):
    """跨窗口汇总：按（消融配置 × 模型）分组，输出 mean ± SD。
    ★ 汇总指标 = 主评价指标口径（六项）。"""
    if df_detail.empty:
        return pd.DataFrame()
    exist_sum = [c for c in SUMMARY_METRIC_COLS if c in df_detail.columns]
    model_col = "model（模型名称）"
    group_cols = ([CONFIG_COL] if CONFIG_COL in df_detail.columns else []) \
        + [model_col]
    rows = []
    for keys, g in df_detail.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["有效窗口数"] = len(g)
        for c in exist_sum:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            mu = float(vals.mean()) if len(vals) else np.nan
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{c} mean±SD"] = (f"{mu:.4f} ± {sd:.4f}"
                                   if pd.notna(mu) else "NaN")
            row[f"{c}_mean"] = round(mu, 5) if pd.notna(mu) else np.nan
            row[f"{c}_sd"] = round(sd, 5) if pd.notna(sd) else np.nan
        rows.append(row)
    df_summary = pd.DataFrame(rows)
    m_order = {m: i for i, m in enumerate(ALL_MODEL_NAMES)}
    c_order = {c: i for i, c in enumerate(EXPERIMENT_CONFIGS.keys())}
    if CONFIG_COL in df_summary.columns:
        df_summary["_ordc"] = df_summary[CONFIG_COL].map(
            lambda x: c_order.get(x, 999))
    else:
        df_summary["_ordc"] = 0
    df_summary["_ordm"] = df_summary[model_col].map(
        lambda x: m_order.get(x, 999))
    df_summary.sort_values(["_ordc", "_ordm"], inplace=True)
    df_summary.drop(columns=["_ordc", "_ordm"], inplace=True)
    df_summary.reset_index(drop=True, inplace=True)
    return df_summary

# ── Bootstrap 结果表格构建辅助 ──────────────────────────────────────────
# ★ CI 指标集 = 主评价指标口径（六项）：
# Macro-Recall / AUC-ROC / AUC-PR / Recall-下降 / Macro-Precision / Accuracy
CI_METRIC_KEYS: List[str] = [
    _MK_REC_MAC, _MK_AUC_ROC, _MK_AUC_PR, _MK_REC_DN, _MK_PRE_MAC, _MK_ACC,
]
METRIC_SHORT: Dict[str, str] = {
    _MK_F1: "MacroF1",
    _MK_AUC_ROC: "AUC-ROC",
    _MK_AUC_PR: "AUC-PR",
    _MK_REC_DN: "Recall-下降",
    _MK_PRE_DN: "Precision-下降",
    _MK_ACC: "Accuracy",
    _MK_REC_MAC: "Macro-Recall",
    _MK_PRE_MAC: "Macro-Precision",
}
_WIN_COL = "window_id（窗口标识）"
_MODEL_COL = "model（模型名称）"


def _r5(v):
    """安全保留5位小数（NaN / None 原样返回）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return round(f, 5) if np.isfinite(f) else np.nan


def build_ablation_ci_table(df_detail, ci_results):
    """Sheet：Full vs No-Gas 主模型逐窗口指标 + Bootstrap 95% CI（含差值行）。
    ★ 指标 = 主评价指标口径（六项）。"""
    if df_detail.empty:
        return pd.DataFrame()
    df_main = df_detail[df_detail[_MODEL_COL] == MAIN_MODEL_NAME]
    if df_main.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    point_by_key: Dict[Tuple[str, str], Dict[str, float]] = {}
    for _, r in df_main.iterrows():
        cfg = r.get(CONFIG_COL, "Full")
        wid = r[_WIN_COL]
        ci = ci_results.get((cfg, wid), {})
        row = {
            CONFIG_COL: cfg, _WIN_COL: wid,
            "test_year（测试年份）": r.get("test_year（测试年份）"),
            "test_n": r.get("test_n"),
            "n_feats_final": r.get("n_feats_final"),
            "n_gas_features_removed": r.get("n_gas_features_removed"),
            "n_gas_left_in_final": r.get("n_gas_left_in_final"),
            "feature_set_hash": r.get("feature_set_hash"),
        }
        pts = {}
        for mk in CI_METRIC_KEYS:
            s = METRIC_SHORT.get(mk, mk)
            point = r.get(mk, np.nan)
            pts[mk] = point
            c = ci.get(mk, {})
            row[f"{s}（点估计）"] = _r5(point)
            row[f"{s} Boot均值"] = _r5(c.get("mean", np.nan))
            row[f"{s} CI下限(2.5%)"] = _r5(c.get("ci_low", np.nan))
            row[f"{s} CI上限(97.5%)"] = _r5(c.get("ci_high", np.nan))
            row[f"{s} BootSD"] = _r5(c.get("sd", np.nan))
            row[f"{s} 有效重采样数"] = c.get("n_valid", np.nan)
        point_by_key[(cfg, wid)] = pts
        rows.append(row)
    window_ids: List[str] = []
    for (_cfg, wid) in point_by_key.keys():
        if wid not in window_ids:
            window_ids.append(wid)
    for wid in window_ids:
        pf = point_by_key.get(("Full", wid))
        pn = point_by_key.get(("No-Gas", wid))
        if pf is None or pn is None:
            continue
        drow = {CONFIG_COL: "Δ(Full − No-Gas)", _WIN_COL: wid}
        for mk in CI_METRIC_KEYS:
            s = METRIC_SHORT.get(mk, mk)
            try:
                d = float(pf.get(mk, np.nan)) - float(pn.get(mk, np.nan))
            except (TypeError, ValueError):
                d = np.nan
            drow[f"{s}（点估计）"] = _r5(d)
        rows.append(drow)
    return pd.DataFrame(rows)


def build_paired_table(pack: List[Tuple[str, str, Dict]],
                       side: str = PAIRED_SIDE,
                       clustered: bool = PAIRED_CLUSTER) -> pd.DataFrame:
    """★ 将配对检验结果包构建为长表（指标 = 主评价指标口径）。"""
    rows: List[Dict] = []
    for model_name, scope_label, res in pack:
        for mk in CI_METRIC_KEYS:
            d = res.get(mk)
            if d is None:
                continue
            p = d.get("p_value", np.nan)
            sig = bool(d.get("significant_005", False))
            dm = d.get("diff_mean", np.nan)
            if sig and pd.notna(dm) and dm > 0:
                concl = "Full 显著更高"
            elif sig and pd.notna(dm) and dm < 0:
                concl = "No-Gas 显著更高"
            else:
                concl = "无显著差异"
            rows.append({
                "model（模型名称）": model_name,
                "范围": scope_label,
                "指标": METRIC_SHORT.get(mk, mk),
                "Δmean(Full−No-Gas)": _r5(dm),
                "ΔCI下限(2.5%)": _r5(d.get("diff_ci_low", np.nan)),
                "ΔCI上限(97.5%)": _r5(d.get("diff_ci_high", np.nan)),
                "ΔSD": _r5(d.get("diff_sd", np.nan)),
                "p值": _r5(p),
                "95%CI不含0": "是" if sig else "否",
                "有效重采样数": d.get("n_valid_boot", np.nan),
                "检验方向": ("双侧" if side == "two"
                             else "单侧(Full>No-Gas)"),
                "重抽样": "聚类(企业id)" if clustered else "IID",
                "结论": concl,
            })
    return pd.DataFrame(rows)


def build_pooled_sd_table(pooled_stats: Dict[Tuple[str, str], Tuple]) -> pd.DataFrame:
    """★ 池化 SD 汇总表（“点估计±SD”列可直接粘贴进论文主表）。
    ★ 指标 = 主评价指标口径（六项）。"""
    rows: List[Dict] = []
    for (model_name, cfg), (point, summ) in pooled_stats.items():
        for mk in CI_METRIC_KEYS:
            s = METRIC_SHORT.get(mk, mk)
            c = summ.get(mk, {})
            pt = point.get(mk, np.nan)
            sd = c.get("sd", np.nan)
            rows.append({
                "model（模型名称）": model_name,
                "config（消融配置）": cfg,
                "指标": s,
                "池化点估计": _r5(pt),
                "Boot均值": _r5(c.get("mean", np.nan)),
                "CI下限(2.5%)": _r5(c.get("ci_low", np.nan)),
                "CI上限(97.5%)": _r5(c.get("ci_high", np.nan)),
                "BootSD": _r5(sd),
                "点估计±SD": (f"{pt:.4f} ± {sd:.4f}"
                             if pd.notna(pt) and pd.notna(sd) else "NaN"),
                "有效重采样数": c.get("n_valid", np.nan),
            })
    return pd.DataFrame(rows)


def build_confusion_sheet_rows(confusion_results) -> pd.DataFrame:
    """
    ★ 将池化混淆矩阵构建为 Excel 长表（仅主模型）。
    confusion_results: {(config, model): {"y_true","y_pred","n_windows","windows"}}
    """
    rows: List[Dict] = []
    pred_cols = [f"预测={CONFUSION_CLASS_NAMES[l]}" for l in LABEL_LIST]
    for (cfg, mname), info in confusion_results.items():
        cm, cm_row, cm_col = compute_confusion_matrices(
            info["y_true"], info["y_pred"])
        n_total = int(cm.sum())
        for mtype, mat in (("计数", cm),
                           ("行归一化(召回率视角)", cm_row),
                           ("列归一化(精确率视角)", cm_col)):
            is_count = (mtype == "计数")
            for i, tl in enumerate(LABEL_LIST):
                row_total = int(cm[i].sum())
                rec = float(cm[i, i] / row_total) if row_total > 0 else np.nan
                rows.append({
                    CONFIG_COL: cfg,
                    "model（模型名称）": mname,
                    "范围": f"Pooled({info['n_windows']}窗)",
                    "窗口": "+".join(info["windows"]),
                    "矩阵类型": mtype,
                    "真实类": CONFUSION_CLASS_NAMES[tl],
                    pred_cols[0]: (int(mat[i, 0]) if is_count
                                   else round(float(mat[i, 0]), 5)),
                    pred_cols[1]: (int(mat[i, 1]) if is_count
                                   else round(float(mat[i, 1]), 5)),
                    pred_cols[2]: (int(mat[i, 2]) if is_count
                                   else round(float(mat[i, 2]), 5)),
                    "行合计(真实类样本数)": row_total,
                    "召回率(对角/行合计)": (round(rec, 5)
                                          if np.isfinite(rec) else np.nan),
                    "池化总样本": n_total,
                })
    return pd.DataFrame(rows)

# ════════════════════════════════════════════════════════════════════════
# 实验主函数
# ════════════════════════════════════════════════════════════════════════
def run_main_experiment(df, feat_names, fixed_windows=None, configs=None):
    """
    消融实验主循环：逐配置（Full / No-Gas）× 逐窗口 独立训练。
    返回:
      df_detail: 逐窗口明细（含 config 列）
      all_predictions: {(config_name, window_id):
                        {model_name: (y_true, y_pred, proba3, test_ids)}}
    """
    if fixed_windows is None:
        fixed_windows = FIXED_WINDOWS
    if configs is None:
        configs = EXPERIMENT_CONFIGS
    all_records: List[Dict] = []
    all_predictions: Dict[Tuple[str, str], Dict[str, Tuple]] = {}
    t_total = time.time()
    for config_name, config in configs.items():
        drop_gas = bool(config.get("drop_gas_features", False))
        print(f"\n{'█'*72}")
        print(f" 实验配置: {config_name} — {config['description']}")
        print(f" drop_gas_features = {drop_gas}"
              f" STRICT_WHITELIST = "
              f"{'[]（已清空）' if drop_gas else STRICT_WHITELIST}")
        print(f"{'█'*72}")
        for win in fixed_windows:
            window_id = win["window_id"]
            train_years = win["train_years"]
            val_year = win["val_year"]
            test_year = win["test_year"]
            dtr_raw = df[df[YEAR_COL].isin(train_years)].copy()
            dv_raw = df[df[YEAR_COL] == val_year].copy()
            dte_raw = df[df[YEAR_COL] == test_year].copy()
            if len(dtr_raw) == 0 or len(dv_raw) == 0 or len(dte_raw) == 0:
                print(f"  [SKIP] {config_name} / {window_id} 数据不足")
                continue
            print(f"\n  {'─'*60}")
            print(f"  [{config_name}] 窗口: {window_id}")
            print(f"  训练={train_years} 验证={val_year} 测试={test_year}")
            print(f"  规模 训练={len(dtr_raw)} 验证={len(dv_raw)}"
                  f" 测试={len(dte_raw)}")
            print(f"  {'─'*60}")
            records, preds_by_model = run_fixed_window(
                dtr_raw=dtr_raw, dv_raw=dv_raw, dte_raw=dte_raw,
                feat_names=feat_names, window_id=window_id,
                train_years=train_years, val_year=val_year,
                test_year=test_year,
                base_drop_roots=BASE_DROP_ROOTS,
                drop_gas=drop_gas, config_name=config_name)
            all_records.extend(records)
            if preds_by_model:
                all_predictions[(config_name, window_id)] = preds_by_model
            del dtr_raw, dv_raw, dte_raw
            gc.collect()
    n_model_preds = sum(len(v) for v in all_predictions.values())
    print(f"\n 训练完成 总记录={len(all_records)} "
          f"留存预测（配置×窗口×模型）={n_model_preds} 组 "
          f"耗时={round(time.time()-t_total,1)}s")
    if not all_records:
        return pd.DataFrame(), all_predictions
    df_detail = pd.DataFrame(all_records)
    return df_detail, all_predictions

# ════════════════════════════════════════════════════════════════════════
# 结果保存
# ════════════════════════════════════════════════════════════════════════
def save_results(df_detail, df_summary, df_ablation_ci,
                 df_paired_yearly, df_paired_pooled, df_pooled_sd,
                 df_confusion=None, output_path=OUTPUT_PATH):
    """保存七个 Sheet + CSV。"""
    os.makedirs(os.path.dirname(output_path)
                if os.path.dirname(output_path) else ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if not df_detail.empty:
            df_detail.to_excel(writer, sheet_name="逐窗口明细", index=False)
        if df_ablation_ci is not None and not df_ablation_ci.empty:
            df_ablation_ci.to_excel(writer, sheet_name="消融对比(带CI)",
                                    index=False)
        if df_paired_yearly is not None and not df_paired_yearly.empty:
            df_paired_yearly.to_excel(writer, sheet_name="逐年配对检验(全模型)",
                                      index=False)
        if df_paired_pooled is not None and not df_paired_pooled.empty:
            df_paired_pooled.to_excel(writer, sheet_name="池化配对检验(全模型)",
                                      index=False)
        if df_pooled_sd is not None and not df_pooled_sd.empty:
            df_pooled_sd.to_excel(writer, sheet_name="池化SD汇总(全模型)",
                                  index=False)
        if df_confusion is not None and not df_confusion.empty:
            df_confusion.to_excel(writer, sheet_name="混淆矩阵(池化)",
                                  index=False)
        if not df_summary.empty:
            df_summary.to_excel(writer, sheet_name="跨窗口汇总", index=False)
        side_desc = ("双侧 H1: Full≠No-Gas" if PAIRED_SIDE == "two"
                     else "单侧 H1: Full>No-Gas")
        cfg = [
            {"配置项": "版本", "内容": "主模型（CatBoost+Optuna） + 8基线模型"
                            " + 消融实验 + Bootstrap检验"
                            " + 全模型逐年/池化配对检验 + 池化SD"
                            " + 主模型池化混淆矩阵"
                            "（★主评价指标口径版）"},
            {"配置项": "★主评价指标口径", "内容": "Macro-Recall / AUC-ROC / "
                            "AUC-PR / Recall-下降(↓) / Macro-Precision / "
                            "Accuracy —— 汇总/CI/配对检验/池化SD 表统一采用；"
                            "逐窗口明细 Sheet 仍保留全部指标（含 Macro-F1）"},
            {"配置项": "模型说明", "内容": "★ 基线 CatBoost 已删除；"
                            "CatBoost 仅以主模型（Optuna 调参）形式保留；"
                            "select_features_catboost 为特征选择工具，不受影响"},
            {"配置项": "Optuna调参目标", "内容": "Macro-F1（TotalF1:average=Macro）"
                            "—— 调参目标未随报告口径改动，保持与其他流程一致"},
            {"配置项": "消融配置", "内容": " | ".join(
                f"{k}: {v['description']}"
                for k, v in EXPERIMENT_CONFIGS.items())},
            {"配置项": "No-Gas处理", "内容": "移除全部气体特征 + "
                            "清空 STRICT_WHITELIST；"
                            "与 Full 完全独立训练，无缓存共享"},
            {"配置项": "气体特征识别", "内容": "严格前缀匹配 + 交互项模式"
                            " + CH4/NO2/SO2 子串 + CO 词素级正则"},
            {"配置项": "Bootstrap次数(逐窗口CI)", "内容": f"{N_BOOT}"
                            f"（分层重采样，seed={BOOT_SEED}）"},
            {"配置项": "Bootstrap次数(全模型检验)", "内容": f"{N_BOOT_MODEL}"},
            {"配置项": "配对检验范围", "内容": f"全模型（{len(ALL_MODEL_NAMES)}个）×"
                            " 逐年（逐窗口） + 池化（三窗口合并）"},
            {"配置项": "配对检验方向", "内容": side_desc},
            {"配置项": "配对检验重抽样", "内容": "聚类（按企业id）"
                            if PAIRED_CLUSTER else "IID"},
            {"配置项": "配对检验设计", "内容": "同一重采样idx同时用于两配置，"
                            "差值方差只反映模型差异；p值带+1连续性校正；"
                            "检验指标集=主评价指标口径（六项）"},
            {"配置项": "池化SD", "内容": "三窗口测试样本合并后直接计算点估计，"
                            "聚类/分层 bootstrap 给出 SD 与 95% CI；"
                            "「池化SD汇总(全模型)」的 点估计±SD 列可直接入论文主表"},
            {"配置项": "混淆矩阵", "内容": f"仅主模型（{MAIN_MODEL_NAME}）；"
                            "池化口径（三窗口测试样本合并）；普通计数 + 行归一化"
                            "（召回率视角）+ 列归一化（精确率视角）；"
                            "PNG 输出至 " + CONFUSION_PLOT_DIR},
            {"配置项": "AUC计算", "内容": "★ 每次重采样均基于 proba3[idx] 重算，"
                            "避免 NaN"},
            {"配置项": "基线模型", "内容": ", ".join(BASELINE_MODEL_NAMES)},
            {"配置项": "主模型", "内容": MAIN_MODEL_NAME},
            {"配置项": "预处理管道", "内容": "线性=PCA+SMOTE；树=SMOTE；"
                            "平衡集成=NoSMOTE"},
            {"配置项": "PCA解释方差", "内容": f"{PCA_VARIANCE:.0%}"},
            {"配置项": "共线性阈值", "内容": f"|r|>{CORR_THRESHOLD:.2f}"},
            {"配置项": "特征选择TopK", "内容": f"{FEAT_TOPK}"},
            {"配置项": "Optuna试验数", "内容": f"{OPTUNA_N_TRIALS_MIN}–"
                            f"{OPTUNA_N_TRIALS_MAX}"},
            {"配置项": "类权重模式", "内容": CW_MODE},
            {"配置项": "滚动窗口数", "内容": str(len(FIXED_WINDOWS))},
            {"配置项": "标签体系", "内容": "{-1:下降, 0:稳定, 1:上升}"
                            "（Recall-下降 即 -1 类召回率）"},
            {"配置项": "STRICT_WHITELIST", "内容": ", ".join(STRICT_WHITELIST)
                            + "（全为气体特征）"},
            {"配置项": "生成时间", "内容": f"{datetime.now():%Y-%m-%d %H:%M:%S}"},
        ]
        pd.DataFrame(cfg).to_excel(writer, sheet_name="配置说明", index=False)
    print(f"  [保存] Excel → {output_path}")
    if not df_detail.empty:
        cp = output_path.replace(".xlsx", "_detail.csv")
        df_detail.to_csv(cp, index=False, encoding="utf-8-sig")
        print(f"  [保存] CSV → {cp}")
    for df_x, tag in [(df_paired_yearly, "yearly_paired"),
                      (df_paired_pooled, "pooled_paired"),
                      (df_pooled_sd, "pooled_sd"),
                      (df_confusion, "confusion_pooled")]:
        if df_x is not None and not df_x.empty:
            cp2 = output_path.replace(".xlsx", f"_{tag}.csv")
            df_x.to_csv(cp2, index=False, encoding="utf-8-sig")
            print(f"  [保存] CSV → {cp2}")

# ════════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    side_desc = ("双侧 H1: Full≠No-Gas" if PAIRED_SIDE == "two"
                 else "单侧 H1: Full>No-Gas")
    print(f"{'='*72}")
    print(f" 主模型（CatBoost + Optuna） + 8 基线模型 + 消融实验"
          f" + 全模型逐年/池化配对检验 + 池化SD + 主模型池化混淆矩阵")
    print(f" ★ 主评价指标口径: Macro-Recall / AUC-ROC / AUC-PR /"
          f" Recall-下降 / Macro-Precision / Accuracy")
    print(f" 基线: {', '.join(BASELINE_MODEL_NAMES)}")
    print(f" 主模型: {MAIN_MODEL_NAME}（★ 基线 CatBoost 已删除）")
    print(f" 消融配置: {' vs '.join(EXPERIMENT_CONFIGS.keys())}")
    print(f" 配对检验: 全模型×(逐年+池化)，{side_desc}，"
          f"聚类={'开' if PAIRED_CLUSTER else '关'}，n_boot={N_BOOT_MODEL}")
    print(f" Bootstrap: {N_BOOT} 次重采样，{100*(1-BOOT_ALPHA):.0f}% CI")
    print(f" {datetime.now():%Y-%m-%d %H:%M:%S} OS={platform.system()}")
    print(f"{'='*72}\n")

    if not os.path.exists(DATA_PATH):
        print(f" ERROR: 数据文件未找到: {DATA_PATH}")
        return

    print(" [1/7] 数据加载...")
    df_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    for c in CORE_NUM_FEATURES:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
    for gas in POLL_ANOMALY_GASES:
        for sfx in ["_zscore", "_exceed"]:
            col = f"{gas}{sfx}"
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    print(f"  原始数据: {df_raw.shape}"
          f" 年份: {sorted(df_raw[YEAR_COL].unique())}")

    print(f"\n [2/7] 特征工程...")
    df = feature_engineering(df_raw)
    df = onehot_industry(df)
    feats = get_feature_columns_fixed(df)
    print(f"  建模特征总数: {len(feats)}")
    # ── 气体特征预检 ─────────────────────────────────────────────────
    gas_all = identify_gas_features(feats)
    feats_ng = [f for f in feats if f not in gas_all]
    print(f"  [预检] Full 档特征数 = {len(feats)}")
    print(f"  [预检] No-Gas 档特征数 = {len(feats_ng)}"
          f"（移除 {len(gas_all)} 个气体特征）")
    print(f"  [预检] 移除示例（前15）: {sorted(gas_all)[:15]}")
    kept_susp = [f for f in feats_ng if any(g in f for g in _GAS_UNAMBIGUOUS)]
    print(f"  [预检] 保留列中可疑气体残留: {kept_susp if kept_susp else '无'}")

    print(f"\n [3/7] 消融实验开始"
          f"（Full / No-Gas × {len(FIXED_WINDOWS)} 窗口）...\n")
    df_detail, all_predictions = run_main_experiment(
        df=df, feat_names=feats, fixed_windows=FIXED_WINDOWS,
        configs=EXPERIMENT_CONFIGS)
    if df_detail.empty:
        print(" [WARNING] 实验未产生任何有效结果")
        return

    # ══════════════════════════════════════════════════════════════
    # [4/7] 主模型逐窗口 Bootstrap CI
    # ══════════════════════════════════════════════════════════════
    print(f"\n [4/7] [Bootstrap CI] 主模型逐窗口"
          f" {100*(1-BOOT_ALPHA):.0f}% 置信区间（n_boot={N_BOOT}）...")
    ci_results: Dict[Tuple[str, str], Dict] = {}
    for (config_name, window_id), preds_by_model in all_predictions.items():
        p_main = preds_by_model.get(MAIN_MODEL_NAME)
        if p_main is None:
            continue
        y_true, y_pred, proba3 = p_main[0], p_main[1], p_main[2]
        tb = time.time()
        ci = bootstrap_metrics(y_true, y_pred, proba3, n_boot=N_BOOT,
                               alpha=BOOT_ALPHA, seed=BOOT_SEED)
        ci_results[(config_name, window_id)] = ci
        c_rec = ci.get(_MK_REC_MAC, {}); c_auc = ci.get(_MK_AUC_ROC, {})
        print(f"  [{config_name}][{window_id}] n={len(y_true)}"
              f" MacroRecall={c_rec.get('mean', np.nan):.4f}"
              f" [{c_rec.get('ci_low', np.nan):.4f},"
              f" {c_rec.get('ci_high', np.nan):.4f}]"
              f" AUC={c_auc.get('mean', np.nan):.4f}"
              f" [{c_auc.get('ci_low', np.nan):.4f},"
              f" {c_auc.get('ci_high', np.nan):.4f}]"
              f" ({round(time.time()-tb,1)}s)")

    # ══════════════════════════════════════════════════════════════
    # [5/7] ★ 全模型逐年配对检验 + 池化配对检验 + 池化 SD ★
    # ══════════════════════════════════════════════════════════════
    print(f"\n [5/7] [全模型检验] 逐年 + 池化配对检验与池化 SD"
          f"（{side_desc}，聚类={'开' if PAIRED_CLUSTER else '关'}，"
          f"n_boot={N_BOOT_MODEL}；共 {len(ALL_MODEL_NAMES)} 模型；"
          f"指标=主评价指标口径六项）...")
    # ── 5a. 逐年（逐窗口）配对检验：全模型 ──────────────────────────
    yearly_pack: List[Tuple[str, str, Dict]] = []
    for model_name in ALL_MODEL_NAMES:
        for win in FIXED_WINDOWS:
            wid = win["window_id"]
            pf = all_predictions.get(("Full", wid), {}).get(model_name)
            pn = all_predictions.get(("No-Gas", wid), {}).get(model_name)
            if pf is None or pn is None:
                continue
            y_true_f, pred_f, proba_f, ids_f = pf
            y_true_n, pred_n, proba_n, ids_n = pn
            if not np.array_equal(np.asarray(y_true_f),
                                  np.asarray(y_true_n)):
                print(f"  [SKIP] {model_name}/{wid} 两配置测试标签不一致")
                continue
            res = paired_bootstrap_test(
                y_true_f, pred_f, pred_n, proba_f, proba_n,
                n_boot=N_BOOT_MODEL, seed=BOOT_SEED, side=PAIRED_SIDE,
                cluster_ids=(np.asarray(ids_f) if PAIRED_CLUSTER else None))
            yearly_pack.append((model_name, wid, res))
            if model_name == MAIN_MODEL_NAME:
                for mk in CI_METRIC_KEYS:
                    d = res.get(mk, {})
                    star = "★" if d.get("significant_005") else " "
                    print(f"  {star}[主模型][{wid}]"
                          f" {METRIC_SHORT.get(mk, mk):<16}"
                          f" Δ={d.get('diff_mean', np.nan):+.4f}"
                          f" p={d.get('p_value', np.nan):.4f}")
        print(f"  ✔ 逐年配对检验完成: {model_name}")
    df_paired_yearly = build_paired_table(yearly_pack, side=PAIRED_SIDE,
                                          clustered=PAIRED_CLUSTER)

    # ── 5b. 池化配对检验 + 池化 SD：全模型 ──────────────────────────
    pooled_pack: List[Tuple[str, str, Dict]] = []
    pooled_stats: Dict[Tuple[str, str], Tuple] = {}
    for model_name in ALL_MODEL_NAMES:
        pc_f = pool_config_preds(all_predictions, "Full", model_name)
        pc_n = pool_config_preds(all_predictions, "No-Gas", model_name)
        if pc_f is not None and pc_n is not None:
            common_w = [w for w in pc_f["windows"] if w in pc_n["windows"]]
            if common_w:
                pc_f2 = pool_config_preds(all_predictions, "Full",
                                          model_name, window_order=common_w)
                pc_n2 = pool_config_preds(all_predictions, "No-Gas",
                                          model_name, window_order=common_w)
                if (pc_f2 is not None and pc_n2 is not None and
                        np.array_equal(pc_f2["y_true"], pc_n2["y_true"])):
                    res = paired_bootstrap_test(
                        pc_f2["y_true"], pc_f2["y_pred"], pc_n2["y_pred"],
                        pc_f2["proba3"], pc_n2["proba3"],
                        n_boot=N_BOOT_MODEL, seed=BOOT_SEED,
                        side=PAIRED_SIDE,
                        cluster_ids=(pc_f2["ids"]
                                     if PAIRED_CLUSTER else None))
                    pooled_pack.append(
                        (model_name, f"Pooled({len(common_w)}窗)", res))
        for cfg_name, pc in (("Full", pc_f), ("No-Gas", pc_n)):
            if pc is None:
                continue
            point, summ = pooled_bootstrap_metrics(
                pc, n_boot=N_BOOT_MODEL, alpha=BOOT_ALPHA, seed=BOOT_SEED,
                cluster=PAIRED_CLUSTER)
            pooled_stats[(model_name, cfg_name)] = (point, summ)
        print(f"  ✔ 池化检验/SD完成: {model_name}")
    df_paired_pooled = build_paired_table(pooled_pack, side=PAIRED_SIDE,
                                          clustered=PAIRED_CLUSTER)
    df_pooled_sd = build_pooled_sd_table(pooled_stats)

    print(f"\n 【池化配对检验摘要】（{side_desc}，"
          f"聚类={'开' if PAIRED_CLUSTER else '关'}）")
    print(f"  {'模型':<20} {'显著指标/6':>10} {'最小p':>10}"
          f" {'MacroRecall方向':>14}")
    for model_name, _scope, res in pooled_pack:
        sig_n = sum(1 for mk in CI_METRIC_KEYS
                    if res.get(mk, {}).get("significant_005"))
        ps = [res.get(mk, {}).get("p_value", np.nan)
              for mk in CI_METRIC_KEYS]
        ps = [x for x in ps if pd.notna(x)]
        pmin = min(ps) if ps else np.nan
        d_rec = res.get(_MK_REC_MAC, {}).get("diff_mean", np.nan)
        direc = ("Full更高" if (pd.notna(d_rec) and d_rec > 0)
                 else ("No-Gas更高" if pd.notna(d_rec) else "—"))
        print(f"  {model_name:<20} {sig_n:>10} "
              f"{(f'{pmin:.4f}' if pd.notna(pmin) else 'NaN'):>10} {direc:>14}")

    for cfg_name in ["Full", "No-Gas"]:
        ps_ = pooled_stats.get((MAIN_MODEL_NAME, cfg_name))
        if ps_ is None:
            continue
        point_, summ_ = ps_
        print(f"\n 【主模型池化点估计 ± BootSD】配置={cfg_name}"
              f"（论文主表口径）")
        for mk in CI_METRIC_KEYS:
            s = METRIC_SHORT.get(mk, mk)
            pt = point_.get(mk, np.nan)
            sd = summ_.get(mk, {}).get("sd", np.nan)
            print(f"  {s:<16} {pt:.4f} ± {sd:.4f}")

    # ══════════════════════════════════════════════════════════════
    # [6/7] ★ 主模型（CatBoost）全样本池化混淆矩阵 ★
    # ══════════════════════════════════════════════════════════════
    print(f"\n [6/7] [混淆矩阵] 主模型（{MAIN_MODEL_NAME}）"
          f"全样本池化混淆矩阵...")
    confusion_results: Dict[Tuple[str, str], Dict] = {}
    for mname in CONFUSION_MODEL_NAMES:
        for cfg_name in EXPERIMENT_CONFIGS.keys():
            pc = pool_config_preds(all_predictions, cfg_name, mname)
            if pc is None:
                print(f"  [SKIP] {cfg_name}/{mname} 无池化预测")
                continue
            confusion_results[(cfg_name, mname)] = {
                "y_true": pc["y_true"], "y_pred": pc["y_pred"],
                "n_windows": len(pc["windows"]), "windows": pc["windows"],
            }
            cm, cm_row, _ = compute_confusion_matrices(
                pc["y_true"], pc["y_pred"])
            rec_list = [round(float(cm[i, i] / cm[i].sum()), 3)
                        if cm[i].sum() > 0 else np.nan for i in range(3)]
            print(f"  [{cfg_name}][{mname}] n={int(cm.sum())}"
                  f"（{len(pc['windows'])}窗）"
                  f" 总体准确率={np.trace(cm) / cm.sum():.4f}"
                  f" 各类召回(↓/平/↑)={rec_list}")
    df_confusion = build_confusion_sheet_rows(confusion_results)
    if HAS_MPL:
        os.makedirs(CONFUSION_PLOT_DIR, exist_ok=True)
        for (cfg_name, mname), info in confusion_results.items():
            cm, cm_row, _ = compute_confusion_matrices(
                info["y_true"], info["y_pred"])
            fp = os.path.join(
                CONFUSION_PLOT_DIR,
                f"confusion_{cfg_name}_{mname.replace(' ', '_')}_pooled.png")
            if plot_confusion_matrix_pair(
                    cm, cm_row,
                    title=f"Confusion Matrix — {mname} ({cfg_name}, pooled)",
                    save_path=fp):
                print(f"  [混淆矩阵] ✔ PNG → {fp}")
            else:
                print(f"  [混淆矩阵] matplotlib 不可用，仅输出数值表")

    # ══════════════════════════════════════════════════════════════
    # [7/7] 汇总与保存
    # ══════════════════════════════════════════════════════════════
    df_summary = build_summary(df_detail)
    df_ablation_ci = build_ablation_ci_table(df_detail, ci_results)
    SEP = "─" * 110
    print(f"\n{'='*110}\n 【结果汇总：跨窗口均值（按配置 × 模型，"
          f"主评价指标口径）】\n{'='*110}")
    print(f"  {'配置':<8} {'模型':<20} {'Macro-Recall':>12} {'AUC-ROC':>10}"
          f" {'AUC-PR':>10} {'Rec-↓':>8} {'Macro-Prec':>11} {'Acc':>8}")
    print(f" {SEP}")
    if not df_summary.empty:
        for _, row in df_summary.iterrows():
            def _g(k, w=8):
                v = row.get(f"{k}_mean", np.nan)
                return f"{v:{w}.4f}" if pd.notna(v) else f"{'NaN':>{w}}"
            print(f"  {str(row.get(CONFIG_COL, ''))[:8]:<8}"
                  f" {str(row.get(_MODEL_COL, ''))[:20]:<20}"
                  f" {_g(_MK_REC_MAC, 12)} {_g(_MK_AUC_ROC, 10)}"
                  f" {_g(_MK_AUC_PR, 10)} {_g(_MK_REC_DN, 8)}"
                  f" {_g(_MK_PRE_MAC, 11)} {_g(_MK_ACC, 8)}")
    print(f"{'='*110}")
    save_results(df_detail, df_summary, df_ablation_ci,
                 df_paired_yearly, df_paired_pooled, df_pooled_sd,
                 df_confusion, OUTPUT_PATH)

    print(f"\n 总耗时: {(time.time()-t0)/60:.1f} 分钟")
    print(f"{'='*72}")
    print(f" 结果文件 → {OUTPUT_PATH}")
    print(f" Sheet: 逐年配对检验(全模型) / 池化配对检验(全模型) /"
          f" 池化SD汇总(全模型) / 混淆矩阵(池化)")
    print(f" 混淆矩阵PNG → {CONFUSION_PLOT_DIR}/")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
