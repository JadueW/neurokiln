"""Portable pipeline templates and paper reproduction metadata.

Templates are ordinary graph specs, so importing one never requires a special
runtime. The reproduction catalog is intentionally conservative: it documents
what the implementation covers and what it does not, instead of claiming an
exact numerical reproduction.
"""

from __future__ import annotations

from typing import Any

FORMAT = "neurokiln.pipeline-template"
FORMAT_VERSION = 1


def _node(
    node_id: str,
    node_type: str,
    params: dict[str, Any],
    label: str,
    x: int,
    y: int,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "params": params,
        "label": label,
        "x": x,
        "y": y,
    }


_CSP_LDA_SPEC = {
    "nodes": [
        _node("bp", "preprocess.bandpass", {"low": 8.0, "high": 30.0, "order": 4}, "8–30 Hz 带通", 40, 120),
        _node("csp", "feature.csp", {"n_components": 6, "reg": 1e-6}, "CSP 空域滤波", 340, 120),
        _node("lda", "decoder.lda", {"shrinkage": "auto"}, "收缩 LDA", 640, 120),
    ],
    "edges": [
        {"source": "bp", "target": "csp"},
        {"source": "csp", "target": "lda"},
    ],
}

_FBCSP_LDA_SPEC = {
    "nodes": [
        _node("fbcsp", "feature.fbcsp", {"n_components": 2, "reg": 1e-6}, "滤波器组 CSP", 340, 120),
        _node("lda", "decoder.lda", {"shrinkage": "auto"}, "收缩 LDA", 640, 120),
    ],
    "edges": [{"source": "fbcsp", "target": "lda"}],
}

_RIEMANN_MDM_SPEC = {
    "nodes": [
        _node("bp", "preprocess.bandpass", {"low": 8.0, "high": 30.0, "order": 4}, "8–30 Hz 带通", 40, 120),
        _node("mdm", "decoder.riemann_mdm", {"metric": "riemann", "covariance": "oas"}, "Riemann MDM", 340, 120),
    ],
    "edges": [{"source": "bp", "target": "mdm"}],
}

_TS_LR_SPEC = {
    "nodes": [
        _node("bp", "preprocess.bandpass", {"low": 8.0, "high": 30.0, "order": 4}, "8–30 Hz 带通", 40, 120),
        _node(
            "ts",
            "decoder.tangent_space_lr",
            {"metric": "riemann", "covariance": "oas", "C": 1.0},
            "切空间 + LR",
            340,
            120,
        ),
    ],
    "edges": [{"source": "bp", "target": "ts"}],
}

_XDAWN_LDA_SPEC = {
    "nodes": [
        _node("bp", "preprocess.bandpass", {"low": 1.0, "high": 15.0, "order": 4}, "1–15 Hz 带通", 40, 120),
        _node(
            "xdawn",
            "decoder.xdawn_lda",
            {"n_filters": 4, "shrinkage": "auto"},
            "xDAWN + LDA",
            340,
            120,
        ),
    ],
    "edges": [{"source": "bp", "target": "xdawn"}],
}

_SSVEP_CCA_SPEC = {
    "nodes": [
        _node(
            "cca",
            "decoder.ssvep_cca",
            {"n_harmonics": 5, "n_bands": 0},
            "SSVEP CCA",
            340,
            120,
        )
    ],
    "edges": [],
}

_EEGNET_SPEC = {
    "nodes": [
        _node(
            "net",
            "decoder.eegnet",
            {"max_epochs": 60, "batch_size": 32, "learning_rate": 0.001, "dropout": 0.5},
            "EEGNet",
            340,
            120,
        )
    ],
    "edges": [],
}

_CSP_SPACE = [
    {"node_id": "bp", "param": "low", "kind": "float", "low": 8.0, "high": 12.0, "step": 0.5},
    {"node_id": "bp", "param": "high", "kind": "float", "low": 20.0, "high": 32.0, "step": 0.5},
    {"node_id": "csp", "param": "n_components", "kind": "int", "low": 2, "high": 12, "step": 1},
    {"node_id": "csp", "param": "reg", "kind": "select", "options": ["1e-6", "1e-4", "1e-3"]},
]

_RIEMANN_SPACE = [
    {"node_id": "bp", "param": "low", "kind": "float", "low": 8.0, "high": 12.0, "step": 0.5},
    {"node_id": "bp", "param": "high", "kind": "float", "low": 20.0, "high": 32.0, "step": 0.5},
    {"node_id": "mdm", "param": "metric", "kind": "select", "options": ["riemann", "logeuclid"]},
    {"node_id": "mdm", "param": "covariance", "kind": "select", "options": ["oas", "lwf", "scm"]},
]


def _template(
    template_id: str,
    name: str,
    family: str,
    paradigm: str,
    description: str,
    spec: dict[str, Any],
    search_space: list[dict[str, Any]] | None,
    paper: dict[str, Any],
    reproduction: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": template_id,
        "name": name,
        "family": family,
        "paradigm": paradigm,
        "description": description,
        "tags": [paradigm, *reproduction.get("tags", [])],
        "spec": spec,
        "search_space": search_space or [],
        "paper": paper,
        "reproduction": reproduction,
        "source": "builtin",
    }


PORTABLE_TEMPLATES: tuple[dict[str, Any], ...] = (
    _template(
        "csp_lda",
        "CSP + LDA",
        "classical",
        "MI",
        "运动想象经典基线：带通、CSP 空域滤波、收缩 LDA。",
        _CSP_LDA_SPEC,
        _CSP_SPACE,
        {
            "title": "Optimizing spatial filters for robust EEG single-trial analysis",
            "authors": "Ramoser, H., Müller-Gerking, J., Pfurtscheller, G.",
            "year": 2000,
            "venue": "IEEE Transactions on Rehabilitation Engineering",
            "doi": "10.1109/86.847816",
        },
        {
            "tags": ["经典", "运动想象", "CSP"],
            "protocol": "within_subject 5 折，训练折内拟合 CSP",
            "datasets": ["BCI Competition IV 2a", "BCI Competition IV 2b"],
            "caveats": "未复刻论文的原始预处理与试次级伪迹剔除；数值不保证逐位一致。",
        },
    ),
    _template(
        "fbcsp_lda",
        "FBCSP + LDA",
        "classical",
        "MI",
        "滤波器组 CSP（FBCSP）按子带提取 CSP 特征后拼接分类。",
        _FBCSP_LDA_SPEC,
        [
            {"node_id": "fbcsp", "param": "n_components", "kind": "int", "low": 2, "high": 6, "step": 1},
            {"node_id": "fbcsp", "param": "reg", "kind": "select", "options": ["1e-6", "1e-4", "1e-3"]},
                ],
        {
            "title": "Filter Bank Common Spatial Pattern (FBCSP) in brain-computer interface",
            "authors": "Ang, K. K., Chin, Z. Y., Zhang, H., Guan, C.",
            "year": 2008,
            "venue": "IJCNN 2008",
            "doi": "10.1109/IJCNN.2008.4634130",
        },
        {
            "tags": ["经典", "运动想象", "滤波器组"],
            "protocol": "within_subject 5 折；滤波器组在训练折内拟合",
            "datasets": ["BCI Competition IV 2a"],
            "caveats": "论文包含互信息特征选择；当前模板保留全部子带特征，可选 Top-K 节点。",
        },
    ),
    _template(
        "riemann_mdm",
        "Riemann MDM",
        "riemann",
        "MI",
        "协方差矩阵的黎曼几何基线：OAS 协方差 + 最小距离均值。",
        _RIEMANN_MDM_SPEC,
        _RIEMANN_SPACE,
        {
            "title": "Multiclass brain-computer interface classification by Riemannian geometry",
            "authors": "Barachant, A., Bonnet, S., Congedo, M., Jutten, C.",
            "year": 2012,
            "venue": "IEEE Transactions on Biomedical Engineering",
            "doi": "10.1109/TBME.2011.2145362",
        },
        {
            "tags": ["黎曼几何", "运动想象"],
            "protocol": "within_subject 5 折；协方差仅在训练折估计",
            "datasets": ["BCI Competition IV 2a", "PhysioNet EEG Motor Imagery"],
            "caveats": "未实现论文中的多类概率校准扩展。",
        },
    ),
    _template(
        "ts_lr",
        "Tangent Space + LR",
        "riemann",
        "MI",
        "切空间投影后用逻辑回归分类；与 Riemann MDM 形成几何对照。",
        _TS_LR_SPEC,
        [
            {"node_id": "bp", "param": "low", "kind": "float", "low": 8.0, "high": 12.0, "step": 0.5},
            {"node_id": "bp", "param": "high", "kind": "float", "low": 20.0, "high": 32.0, "step": 0.5},
            {"node_id": "ts", "param": "metric", "kind": "select", "options": ["riemann", "logeuclid"]},
            {"node_id": "ts", "param": "C", "kind": "float", "low": 0.01, "high": 10.0, "step": 0.05},
        ],
        {
            "title": "Multiclass brain-computer interface classification by Riemannian geometry",
            "authors": "Barachant, A., Bonnet, S., Congedo, M., Jutten, C.",
            "year": 2012,
            "venue": "IEEE Transactions on Biomedical Engineering",
            "doi": "10.1109/TBME.2011.2145362",
        },
        {
            "tags": ["黎曼几何", "切空间"],
            "protocol": "within_subject 5 折；切空间参考点仅用训练折",
            "datasets": ["BCI Competition IV 2a"],
            "caveats": "切空间参考点与分类器均在训练折内拟合。",
        },
    ),
    _template(
        "xdawn_lda",
        "xDAWN + LDA",
        "erp",
        "P300",
        "P300 经典空间滤波：xDAWN 估计诱发响应子空间，再用 LDA。",
        _XDAWN_LDA_SPEC,
        [
            {"node_id": "xdawn", "param": "n_filters", "kind": "int", "low": 2, "high": 8, "step": 1},
        ],
        {
            "title": "xDAWN Algorithm to Enhance Evoked Potentials: Application to Brain-Computer Interface",
            "authors": "Rivet, B., Souloumiac, A., Attina, V., Gibert, G.",
            "year": 2009,
            "venue": "IEEE Transactions on Biomedical Engineering",
            "doi": "10.1109/TBME.2009.2012869",
        },
        {
            "tags": ["ERP", "P300", "空间滤波"],
            "protocol": "within_subject 分层 k 折；xDAWN 仅用训练折",
            "datasets": ["BNCI2014_009", "合成 P300"],
            "caveats": "未复刻论文的刺激时序与个体化窗口选择。",
        },
    ),
    _template(
        "ssvep_cca",
        "SSVEP CCA",
        "ssvep",
        "SSVEP",
        "典型相关分析检测刺激频率；需要数据集提供 stimulus_frequencies。",
        _SSVEP_CCA_SPEC,
        [],
        {
            "title": "Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs",
            "authors": "Lin, Z., Zhang, C., Wu, W., Gao, X.",
            "year": 2006,
            "venue": "IEEE Transactions on Biomedical Engineering",
            "doi": "10.1109/TBME.2006.886577",
        },
        {
            "tags": ["SSVEP", "CCA"],
            "protocol": "within_subject 分层 k 折；按刺激频率评估",
            "datasets": ["Nakanishi 2015", "合成 SSVEP"],
            "caveats": "依赖数据集的刺激频率元数据；缺少元数据时无法建立参考信号。",
        },
    ),
    _template(
        "eegnet",
        "EEGNet",
        "deep",
        "MI",
        "紧凑卷积网络基线；CPU 训练较慢，建议小预算 HPO。",
        _EEGNET_SPEC,
        [
            {"node_id": "net", "param": "learning_rate", "kind": "float", "low": 0.0001, "high": 0.01, "step": 0.0001},
            {"node_id": "net", "param": "dropout", "kind": "float", "low": 0.1, "high": 0.7, "step": 0.05},
        ],
        {
            "title": "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces",
            "authors": "Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., Lance, B. J.",
            "year": 2018,
            "venue": "Journal of Neural Engineering",
            "doi": "10.1088/1741-2552/aace8c",
        },
        {
            "tags": ["深度学习", "CNN"],
            "protocol": "within_subject 分层 k 折；训练轮数需固定",
            "datasets": ["BCI Competition IV 2a", "PhysioNet EEG Motor Imagery"],
            "caveats": "未复刻论文的完整训练策略与数据增强；不同随机种子会有波动。",
        },
    ),
)

_BY_ID = {item["id"]: item for item in PORTABLE_TEMPLATES}


def template_catalog() -> list[dict[str, Any]]:
    return [dict(item) for item in PORTABLE_TEMPLATES]


def get_template(template_id: str) -> dict[str, Any]:
    item = _BY_ID.get(str(template_id))
    if item is None:
        raise KeyError(f"unknown template: {template_id}")
    return dict(item)


def reproduction_catalog() -> list[dict[str, Any]]:
    entries = []
    for item in PORTABLE_TEMPLATES:
        reproduction = item["reproduction"]
        entries.append(
            {
                "id": item["id"],
                "template_id": item["id"],
                "name": item["name"],
                "paradigm": item["paradigm"],
                "family": item["family"],
                "description": item["description"],
                "paper": item["paper"],
                "protocol": reproduction["protocol"],
                "datasets": reproduction["datasets"],
                "caveats": reproduction["caveats"],
                "tags": item["tags"],
            }
        )
    return entries


__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "PORTABLE_TEMPLATES",
    "get_template",
    "reproduction_catalog",
    "template_catalog",
]
