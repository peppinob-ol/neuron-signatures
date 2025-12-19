"""
Streamlit EDA app for neuron activation runs (POC 1 + POC 2 outputs).

Run:
  streamlit run neuron_signatures/eda_streamlit.py

Then select:
  - run_dir (must contain manifest.json + activations.pt)
  - analysis_dir (optional; if missing, the app can still plot a chosen neuron)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st
import torch


try:
    # Normal usage when repo root is on sys.path (e.g., `python -m ...`)
    from neuron_signatures.eda_heatmap import (
        HeatmapConfig,
        compute_max_abs,
        render_stacked_prompts_heatmap_html,
        render_token_heatmap_html,
    )
except ModuleNotFoundError:
    # Streamlit `run neuron_signatures/eda_streamlit.py` often adds only the
    # script directory to sys.path; fall back to local import in that case.
    from eda_heatmap import (  # type: ignore
        HeatmapConfig,
        compute_max_abs,
        render_stacked_prompts_heatmap_html,
        render_token_heatmap_html,
    )


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_activations(run_dir: Path) -> Dict[str, torch.Tensor]:
    return torch.load(run_dir / "activations.pt", map_location="cpu")


def _load_top_neurons(analysis_dir: Path) -> Optional[List[Dict[str, Any]]]:
    path = analysis_dir / "top_neurons.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return None


def _list_influence_rank_files(analysis_dir: Path) -> List[Path]:
    """
    List influence ranking files in analysis_dir.

    We prefer CSV (`influence_*.csv`) when present, but will also list JSON
    (`influence_*.json`) while excluding `*_meta.json` and `*_write_scores*.json`.
    Sorted by modification time (newest first).
    """
    if not analysis_dir.exists():
        return []
    files = list(analysis_dir.glob("influence_*.csv")) + list(analysis_dir.glob("influence_*.json"))
    out: List[Path] = []
    for p in files:
        if p.name.endswith("_meta.json"):
            continue
        if "_write_scores" in p.name:
            continue
        out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


@st.cache_data(show_spinner=False)
def _load_write_scores_cached(path_str: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Load write scores JSON and index by (layer, neuron).
    
    Returns dict: {(layer, neuron): {"top_pos": [...], "top_neg": [...], "explicit": [...]}}
    """
    path = Path(path_str)
    if not path.exists():
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    result: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for n in data.get("neurons", []):
        key = (int(n.get("layer", 0)), int(n.get("neuron", 0)))
        ws = n.get("write_scores")
        if ws:
            result[key] = ws
    return result


def _find_write_scores_file(analysis_dir: Path) -> Optional[Path]:
    """Find the most recent write_scores JSON file in analysis_dir."""
    if not analysis_dir.exists():
        return None
    files = list(analysis_dir.glob("*_write_scores*.json"))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _render_write_scores_compact(ws: Dict[str, Any], n_show: int = 5) -> str:
    """Render write scores as compact HTML spans (negative left, positive right)."""
    top_pos = ws.get("top_pos", [])[:n_show]
    top_neg = ws.get("top_neg", [])[:n_show]
    
    neg_parts = []
    for t in top_neg:
        score = t.get("score", 0)
        tok = t.get("token", "?").replace("<", "&lt;").replace(">", "&gt;")
        neg_parts.append(f'<span style="background:#f8d7da;padding:1px 4px;border-radius:3px;margin:1px;font-size:12px;">{tok} <small>{score:.2f}</small></span>')
    
    pos_parts = []
    for t in top_pos:
        score = t.get("score", 0)
        tok = t.get("token", "?").replace("<", "&lt;").replace(">", "&gt;")
        pos_parts.append(f'<span style="background:#d4edda;padding:1px 4px;border-radius:3px;margin:1px;font-size:12px;">{tok} <small>+{score:.2f}</small></span>')
    
    html = '<div style="line-height:1.8;">'
    html += '<b style="color:#dc3545;font-size:11px;">-</b> ' + " ".join(neg_parts)
    html += ' &nbsp; <b style="color:#28a745;font-size:11px;">+</b> ' + " ".join(pos_parts)
    html += '</div>'
    return html


def _load_influence_ranking(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return None


@st.cache_data(show_spinner=False)
def _load_influence_csv_cached(path_str: str) -> Dict[str, Any]:
    """
    Load an influence CSV produced by POC2 into compact tensors for filtering.

    Returns:
        dict with keys: layer_i, neuron_i, activation, influence, abs_influence, meta
    """
    import csv

    layers: List[int] = []
    neurons: List[int] = []
    activations: List[float] = []
    influences: List[float] = []
    abs_influences: List[float] = []
    meta: Dict[str, Any] = {}

    with open(path_str, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not meta:
                # Small prompt-level metadata from the first row (repeated in file).
                meta = {
                    "prompt_id": str(row.get("prompt_id", "")),
                    "target_token_id": int(row.get("target_token_id", 0) or 0),
                    "target_token_ascii": str(row.get("target_token_ascii", "")),
                    "target_pos": int(row.get("target_pos", 0) or 0),
                    "ctx_idx": int(row.get("ctx_idx", 0) or 0),
                    "metric": str(row.get("metric", "")),
                    "target_mode": str(row.get("target_mode", "")),
                }

            layers.append(int(row["layer"]))
            neurons.append(int(row["neuron"]))
            activations.append(float(row["activation"]))
            influences.append(float(row["influence"]))
            abs_influences.append(float(row["abs_influence"]))

    return {
        "layer_i": torch.tensor(layers, dtype=torch.int16),
        "neuron_i": torch.tensor(neurons, dtype=torch.int32),
        "activation": torch.tensor(activations, dtype=torch.float32),
        "influence": torch.tensor(influences, dtype=torch.float32),
        "abs_influence": torch.tensor(abs_influences, dtype=torch.float32),
        "meta": meta,
    }


def _get_prompt_entry(manifest: Dict[str, Any], prompt_id: str) -> Dict[str, Any]:
    for p in manifest.get("prompts", []):
        if p.get("probe_id") == prompt_id:
            return p
    raise KeyError(f"prompt_id not found in manifest: {prompt_id}")


def _plot_activation_for_prompt(
    prompt_id: str,
    tokens_ascii: List[str],
    token_ids: List[int],
    values: List[float],
) -> go.Figure:
    x = list(range(len(values)))
    hover = [
        f"ctx_idx={i}<br>token_id={token_ids[i]}<br>token={tokens_ascii[i]}<br>act={values[i]:+.6f}"
        for i in x
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=values,
            mode="lines+markers",
            name=prompt_id,
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=f"Activation vs ctx_idx ({prompt_id})",
        xaxis_title="ctx_idx",
        yaxis_title="activation (hook_post)",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _plot_overlay(
    series: List[Tuple[str, List[int], List[str], List[float]]],
) -> go.Figure:
    fig = go.Figure()
    for prompt_id, token_ids, tokens_ascii, values in series:
        x = list(range(len(values)))
        hover = [
            f"prompt={prompt_id}<br>ctx_idx={i}<br>token_id={token_ids[i]}<br>token={tokens_ascii[i]}<br>act={values[i]:+.6f}"
            for i in x
        ]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines",
                name=prompt_id,
                hovertext=hover,
                hoverinfo="text",
            )
        )
    fig.update_layout(
        title="Overlay across prompts (same layer/neuron)",
        xaxis_title="ctx_idx",
        yaxis_title="activation (hook_post)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Neuron Signatures EDA", layout="wide")
    st.title("Neuron Signatures - EDA (POC 1/2)")

    st.sidebar.header("Inputs")
    default_run = "runs/poc1_seedprompt_gpu6"
    run_dir_str = st.sidebar.text_input("run_dir", value=default_run)
    run_dir = Path(run_dir_str)

    if not (run_dir / "manifest.json").exists() or not (run_dir / "activations.pt").exists():
        st.error("run_dir must contain manifest.json and activations.pt")
        st.stop()

    analysis_dir = run_dir / "analysis"
    analysis_dir_str = st.sidebar.text_input("analysis_dir", value=str(analysis_dir))
    analysis_dir = Path(analysis_dir_str)

    top_k_limit = st.sidebar.number_input("top_k_limit", min_value=10, max_value=2000, value=200, step=10)

    with st.spinner("Loading manifest and activations..."):
        manifest = _load_manifest(run_dir)
        activations = _load_activations(run_dir)

    prompts = manifest.get("prompts", [])
    prompt_ids = [p.get("probe_id") for p in prompts]

    # Session state for keyboard-driven navigation
    n_layers = int(manifest["n_layers"])
    d_mlp = int(manifest["d_mlp"])
    if "layer" not in st.session_state:
        st.session_state.layer = 0
    if "neuron" not in st.session_state:
        st.session_state.neuron = 0
    if "candidate_idx" not in st.session_state:
        st.session_state.candidate_idx = 0
    if "_candidate_pairs" not in st.session_state:
        st.session_state._candidate_pairs = []
    if "_candidate_labels" not in st.session_state:
        st.session_state._candidate_labels = []

    # Handle pending navigation from chart click (must happen BEFORE widgets render)
    if "_pending_nav" in st.session_state:
        nav = st.session_state._pending_nav
        st.session_state.layer = int(nav["layer"])
        st.session_state.neuron = int(nav["neuron"])
        if "candidate_idx" in nav:
            st.session_state.candidate_idx = int(nav["candidate_idx"])
            st.session_state.candidate_choice_idx = int(nav["candidate_idx"])
        del st.session_state._pending_nav

    # Store bounds for shortcut callbacks (kept in session_state for simplicity)
    st.session_state._max_layer = n_layers - 1
    st.session_state._max_neuron = d_mlp - 1

    # Candidate list navigation helper
    def _apply_candidate_idx() -> None:
        pairs = list(st.session_state.get("_candidate_pairs", []))
        if not pairs:
            return
        i = int(st.session_state.get("candidate_idx", 0))
        i = max(0, min(i, len(pairs) - 1))
        st.session_state.candidate_idx = i
        layer_i, neuron_i = pairs[i]
        st.session_state.layer = int(layer_i)
        st.session_state.neuron = int(neuron_i)

    # Callback functions for keyboard navigation buttons
    def kb_prev() -> None:
        if st.session_state.get("_candidate_pairs"):
            st.session_state.candidate_idx = max(int(st.session_state.get("candidate_idx", 0)) - 1, 0)
            _apply_candidate_idx()
        else:
            st.session_state.layer = max(int(st.session_state.layer) - 1, 0)

    def kb_next() -> None:
        if st.session_state.get("_candidate_pairs"):
            pairs = list(st.session_state.get("_candidate_pairs", []))
            st.session_state.candidate_idx = min(int(st.session_state.get("candidate_idx", 0)) + 1, len(pairs) - 1)
            _apply_candidate_idx()
        else:
            st.session_state.layer = min(int(st.session_state.layer) + 1, int(st.session_state._max_layer))

    def kb_prev10() -> None:
        if st.session_state.get("_candidate_pairs"):
            st.session_state.candidate_idx = max(int(st.session_state.get("candidate_idx", 0)) - 10, 0)
            _apply_candidate_idx()
        else:
            st.session_state.neuron = max(int(st.session_state.neuron) - 1, 0)

    def kb_next10() -> None:
        if st.session_state.get("_candidate_pairs"):
            pairs = list(st.session_state.get("_candidate_pairs", []))
            st.session_state.candidate_idx = min(int(st.session_state.get("candidate_idx", 0)) + 10, len(pairs) - 1)
            _apply_candidate_idx()
        else:
            st.session_state.neuron = min(int(st.session_state.neuron) + 1, int(st.session_state._max_neuron))

    # Navigation buttons (keyboard shortcuts not supported in Streamlit sandbox)
    st.sidebar.header("Quick navigation")
    nav_cols = st.sidebar.columns(4)
    nav_cols[0].button("Prev", key="kb_prev_btn", on_click=kb_prev, help="Previous (-1)")
    nav_cols[1].button("Next", key="kb_next_btn", on_click=kb_next, help="Next (+1)")
    nav_cols[2].button("-10", key="kb_prev10_btn", on_click=kb_prev10, help="Back 10")
    nav_cols[3].button("+10", key="kb_next10_btn", on_click=kb_next10, help="Forward 10")

    st.sidebar.header("Neuron selection")
    top_neurons = _load_top_neurons(analysis_dir) if analysis_dir.exists() else None
    influence_files = _list_influence_rank_files(analysis_dir)

    selection_modes = ["manual"]
    if top_neurons:
        selection_modes.append("peaks_top_neurons")
    if influence_files:
        selection_modes.append("influence_json")

    # Default to influence if present, otherwise peaks, otherwise manual.
    default_mode = "manual"
    if "influence_json" in selection_modes:
        default_mode = "influence_json"
    elif "peaks_top_neurons" in selection_modes:
        default_mode = "peaks_top_neurons"

    if "selection_mode" not in st.session_state:
        st.session_state.selection_mode = default_mode

    st.sidebar.selectbox(
        "selection_mode",
        options=selection_modes,
        index=selection_modes.index(st.session_state.selection_mode),
        key="selection_mode",
        help="Choose how to select/browse neurons: manual, peaks ranking, or influence ranking JSON.",
    )

    candidates: List[Dict[str, Any]] = []
    candidate_pairs: List[Tuple[int, int]] = []
    candidate_labels: List[str] = []
    # Store influence info keyed by (layer, neuron) for display in heatmap section
    influence_info: Dict[Tuple[int, int], Dict[str, Any]] = {}

    if st.session_state.selection_mode == "peaks_top_neurons":
        limited = (top_neurons or [])[: int(top_k_limit)]
        candidates = limited
        for n in limited:
            layer_i = int(n.get("layer", 0))
            neuron_i = int(n.get("neuron", 0))
            candidate_pairs.append((layer_i, neuron_i))
            candidate_labels.append(
                f"L{layer_i} N{neuron_i} | mean_peak_abs={float(n.get('mean_peak_abs', 0.0)):.6f} | mode_tok={str(n.get('mode_peak_token_ascii',''))}"
            )
        st.sidebar.caption("Source: peaks ranking from analysis/top_neurons.json")

    elif st.session_state.selection_mode == "influence_json":
        # Pick influence file (prefer CSV if available)
        def fmt_file(p: Path) -> str:
            return p.name

        if "influence_file_idx" not in st.session_state:
            st.session_state.influence_file_idx = 0
        st.sidebar.selectbox(
            "influence_file",
            options=list(range(len(influence_files))),
            format_func=lambda i: fmt_file(influence_files[i]),
            key="influence_file_idx",
        )
        file_idx = int(st.session_state.influence_file_idx)
        file_idx = max(0, min(file_idx, len(influence_files) - 1))
        influence_path = influence_files[file_idx]
        if influence_path.suffix.lower() == ".csv":
            data = _load_influence_csv_cached(str(influence_path))

            metric = st.sidebar.selectbox(
                "influence_filter_metric",
                options=["abs_influence", "influence", "activation"],
                index=0,
                help="Filter the influence-ranked neurons by a scalar column.",
            )
            values = data[metric]
            v_min = float(values.min().item())
            v_max = float(values.max().item())
            lo, hi = st.sidebar.slider(
                "influence_filter_range",
                min_value=v_min,
                max_value=v_max,
                value=(v_min, v_max),
                format="%.6g",
                help="Keep neurons whose selected metric is within this range.",
            )

            mask = (values >= float(lo)) & (values <= float(hi))
            idx_all = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            n_total = int(values.numel())
            n_keep = int(idx_all.numel())
            k_show = min(int(top_k_limit), n_keep)
            idx = idx_all[:k_show]

            st.sidebar.caption(
                f"Source: {influence_path.name} | kept {n_keep}/{n_total} (showing {k_show})"
            )

            layer_t = data["layer_i"][idx].tolist()
            neuron_t = data["neuron_i"][idx].tolist()
            act_t = data["activation"][idx].tolist()
            inf_t = data["influence"][idx].tolist()
            abs_inf_t = data["abs_influence"][idx].tolist()
            metric_t = data[metric][idx].tolist()

            for layer_i, neuron_i, act_v, inf_v, abs_v, m_v in zip(
                layer_t, neuron_t, act_t, inf_t, abs_inf_t, metric_t
            ):
                key = (int(layer_i), int(neuron_i))
                candidate_pairs.append(key)
                candidate_labels.append(
                    f"L{int(layer_i)} N{int(neuron_i)} | {metric}={float(m_v):.6g} | abs_inf={float(abs_v):.6g} | act={float(act_v):.6g}"
                )
                influence_info[key] = {
                    "activation": float(act_v),
                    "influence": float(inf_v),
                    "abs_influence": float(abs_v),
                    "meta": data.get("meta", {}),
                }

        else:
            st.sidebar.warning("Influence JSON can be very large; prefer the CSV for filtering.")
            infl = _load_influence_ranking(influence_path) or []
            limited = infl[: int(top_k_limit)]
            candidates = limited
            for n in limited:
                layer_i = int(n.get("layer", 0))
                neuron_i = int(n.get("neuron", 0))
                abs_inf = float(n.get("abs_influence", 0.0))
                act_v = float(n.get("activation", 0.0))
                inf_v = float(n.get("influence", 0.0))
                tok = str(n.get("target_token_ascii", ""))
                key = (layer_i, neuron_i)
                candidate_pairs.append(key)
                candidate_labels.append(f"L{layer_i} N{neuron_i} | abs_influence={abs_inf:.6g} | target={tok}")
                influence_info[key] = {
                    "activation": act_v,
                    "influence": inf_v,
                    "abs_influence": abs_inf,
                    "meta": {
                        "target_token_ascii": tok,
                        "target_token_id": int(n.get("target_token_id", 0)),
                        "target_pos": int(n.get("target_pos", 0)),
                        "metric": str(n.get("metric", "")),
                    },
                }
            st.sidebar.caption(f"Source: {influence_path.name} (showing first {len(limited)})")

    # Persist candidate pairs for keyboard callbacks
    st.session_state._candidate_pairs = candidate_pairs
    st.session_state._candidate_labels = candidate_labels
    st.session_state._influence_info = influence_info

    # Load write scores if available
    write_scores_file = _find_write_scores_file(analysis_dir)
    write_scores_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if write_scores_file:
        write_scores_data = _load_write_scores_cached(str(write_scores_file))
    st.session_state._write_scores = write_scores_data


    if candidate_pairs:
        # Keep indices in range if filters changed.
        if int(st.session_state.get("candidate_idx", 0)) >= len(candidate_pairs):
            st.session_state.candidate_idx = 0
        if int(st.session_state.get("candidate_choice_idx", 0)) >= len(candidate_pairs):
            st.session_state.candidate_choice_idx = 0

        # Candidate picker
        def apply_candidate_choice() -> None:
            st.session_state.candidate_idx = int(st.session_state.get("candidate_choice_idx", 0))
            _apply_candidate_idx()

        # Initialize choice index from current candidate_idx
        if "candidate_choice_idx" not in st.session_state:
            st.session_state.candidate_choice_idx = int(st.session_state.candidate_idx)

        st.sidebar.selectbox(
            "Pick from filtered list",
            options=list(range(len(candidate_pairs))),
            format_func=lambda i: candidate_labels[i],
            key="candidate_choice_idx",
            on_change=apply_candidate_choice,
        )
        st.sidebar.caption("Tip: use Prev/Next buttons to step through, or -10/+10 for jumps.")
        # Ensure layer/neuron are consistent if candidate_idx changed elsewhere (keyboard)
        _apply_candidate_idx()
    else:
        st.sidebar.info("No ranking file found (or manual mode). Use manual layer/neuron inputs.")

    # Manual controls always visible (keeps state in sync with arrow keys)
    st.sidebar.number_input("layer", min_value=0, max_value=n_layers - 1, step=1, key="layer")
    st.sidebar.number_input("neuron", min_value=0, max_value=d_mlp - 1, step=1, key="neuron")

    layer = int(st.session_state.layer)
    neuron = int(st.session_state.neuron)

    st.subheader("Selected neuron")
    st.write(f"Layer: {layer}  Neuron: {neuron}")

    # Build per-prompt series for plots
    series: List[Tuple[str, List[int], List[str], List[float]]] = []
    for pid in prompt_ids:
        entry = _get_prompt_entry(manifest, pid)
        token_ids = [int(x) for x in entry.get("token_ids", [])]
        tokens_ascii = [str(x) for x in entry.get("tokens_ascii", [])]
        x = activations[pid]  # [L, seq, d_mlp]
        v = x[layer, :, neuron].float().tolist()
        series.append((pid, token_ids, tokens_ascii, v))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Overlay across prompts")
        st.plotly_chart(_plot_overlay(series), width="stretch")

    with col2:
        st.markdown("### Single prompt view")
        pid_sel = st.selectbox("prompt_id", prompt_ids, index=0)
        entry = _get_prompt_entry(manifest, pid_sel)
        token_ids = [int(x) for x in entry.get("token_ids", [])]
        tokens_ascii = [str(x) for x in entry.get("tokens_ascii", [])]
        v = activations[pid_sel][layer, :, neuron].float().tolist()
        st.plotly_chart(_plot_activation_for_prompt(pid_sel, tokens_ascii, token_ids, v), width="stretch")

    st.markdown("### Token table (selected prompt)")
    rows = []
    for i, (tid, tok, val) in enumerate(zip(token_ids, tokens_ascii, v)):
        rows.append({"ctx_idx": i, "token_id": tid, "token": tok, "activation": float(val)})
    st.dataframe(rows, width="stretch", height=320)

    st.markdown("### Token heatmap (Neuronpedia-style)")
    st.caption("Green = positive activations, orange = negative activations. Opacity uses log scaling.")

    hm_col1, hm_col2, hm_col3 = st.columns([1, 1, 1])
    with hm_col1:
        tokens_per_row = st.number_input("tokens_per_row", min_value=10, max_value=200, value=50, step=5)
        show_values = st.checkbox("show_values_on_tokens", value=True)
        exclude_bos = st.checkbox("exclude_bos_from_max", value=True)
    with hm_col2:
        normalize_mode = st.selectbox("normalization", ["per_prompt", "global_across_prompts"], index=1)
        value_decimals = st.number_input("value_decimals", min_value=0, max_value=6, value=2, step=1)
        minimum_threshold = st.number_input(
            "minimum_threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.00005,
            step=0.00005,
            help="Values with abs(value) <= threshold are not highlighted.",
        )
        minimum_opacity = st.number_input(
            "minimum_opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Minimum background opacity for highlighted tokens.",
        )
    with hm_col3:
        compact_view = st.checkbox("compact_view", value=True, help="Hide per-prompt titles and meta lines")
        view_write_scores = st.checkbox("view_write_scores", value=True, help="Show write scores for neuron")

    cfg = HeatmapConfig(
        tokens_per_row=int(tokens_per_row),
        show_values=bool(show_values),
        exclude_bos=bool(exclude_bos),
        value_decimals=int(value_decimals),
        minimum_threshold=float(minimum_threshold),
        minimum_opacity=float(minimum_opacity),
    )

    # Heatmap for selected prompt
    if normalize_mode == "per_prompt":
        max_abs = compute_max_abs(tokens_ascii, v, cfg)
    else:
        global_max = 0.0
        for _, _, toks, vals in series:
            m = compute_max_abs(toks, vals, cfg)
            if m > global_max:
                global_max = m
        max_abs = global_max

    heatmap_html = render_token_heatmap_html(
        tokens_ascii=tokens_ascii,
        token_ids=token_ids,
        values=v,
        cfg=cfg,
        max_abs_override=max_abs,
        title=f"Prompt {pid_sel} | Layer {layer} | Neuron {neuron}",
    )
    st.markdown(heatmap_html, unsafe_allow_html=True)

    st.markdown("### Stacked prompts heatmap (same neuron)")

    # Show influence info and write scores in compact form
    inf_info = st.session_state.get("_influence_info", {}).get((layer, neuron))
    ws_info = st.session_state.get("_write_scores", {}).get((layer, neuron))
    
    # Write scores expander (above the stacked heatmap) - only if checkbox enabled
    if view_write_scores and ws_info:
        with st.expander("Write scores details", expanded=False):
            top_pos = ws_info.get("top_pos", [])
            top_neg = ws_info.get("top_neg", [])
            explicit = ws_info.get("explicit", [])
            
            col_n, col_p = st.columns(2)
            with col_n:
                st.markdown("**Suppresses (-)**")
                for t in top_neg[:10]:
                    st.markdown(f"- `{t['token']}` ({t['score']:+.3f})")
            with col_p:
                st.markdown("**Promotes (+)**")
                for t in top_pos[:10]:
                    st.markdown(f"- `{t['token']}` ({t['score']:+.3f})")
            
            if explicit:
                st.markdown("**Explicit tokens:**")
                exp_parts = [f"`{e['token']}`({e['score']:+.2f})" for e in explicit]
                st.markdown(" | ".join(exp_parts))
    
    # Build subtitle_html (write scores compact line) and meta_extra (influence info)
    subtitle_html = ""
    meta_extra = ""
    
    if view_write_scores and ws_info:
        subtitle_html = _render_write_scores_compact(ws_info, n_show=5)
    
    if inf_info:
        meta = inf_info.get("meta", {})
        target_tok = meta.get("target_token_ascii", "")
        target_pos = meta.get("target_pos", "")
        meta_extra = f"| act={inf_info['activation']:.3g} | inf={inf_info['influence']:+.3g}"
        if target_tok:
            meta_extra += f" | seed_target={target_tok}@{target_pos}"

    prompt_series = [(pid, toks, tids, vals) for (pid, tids, toks, vals) in series]
    stacked_html = render_stacked_prompts_heatmap_html(
        prompt_series=prompt_series,
        cfg=cfg,
        global_max_abs=(None if normalize_mode == "per_prompt" else max_abs),
        title=f"Layer {layer} | Neuron {neuron}",
        subtitle_html=subtitle_html,
        meta_extra=meta_extra,
        compact=compact_view,
    )
    st.markdown(stacked_html, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Influence Distribution Charts (only when influence data is loaded)
    # -------------------------------------------------------------------------
    influence_info_all = st.session_state.get("_influence_info", {})
    if influence_info_all and st.session_state.get("selection_mode") == "influence_json":
        st.markdown("---")
        st.markdown("### Influence Distribution (all neurons in ranking)")

        # Build lists for charting
        chart_layers: List[int] = []
        chart_neurons: List[int] = []
        chart_acts: List[float] = []
        chart_infs: List[float] = []
        chart_abs_infs: List[float] = []
        for (l, n), info in influence_info_all.items():
            chart_layers.append(l)
            chart_neurons.append(n)
            chart_acts.append(info["activation"])
            chart_infs.append(info["influence"])
            chart_abs_infs.append(info["abs_influence"])

        if chart_layers:
            # Compute ranks by abs_influence (descending)
            sorted_idx = sorted(range(len(chart_abs_infs)), key=lambda i: -chart_abs_infs[i])
            ranks = [0] * len(chart_abs_infs)
            for rank, i in enumerate(sorted_idx, start=1):
                ranks[i] = rank

            # Selected neuron mask
            is_selected = [(chart_layers[i] == layer and chart_neurons[i] == neuron) for i in range(len(chart_layers))]

            # Chart controls
            chart_cols = st.columns([1, 1, 1])
            with chart_cols[0]:
                chart_type = st.selectbox(
                    "Chart type",
                    ["layer_scatter", "act_vs_inf", "rank_bars"],
                    index=0,
                    key="infl_chart_type",
                    help="layer_scatter: layer vs influence; act_vs_inf: activation vs influence; rank_bars: ranked bar chart",
                )
            with chart_cols[1]:
                y_metric = st.selectbox(
                    "Y metric",
                    ["abs_influence", "influence", "activation"],
                    index=0,
                    key="infl_y_metric",
                )
            with chart_cols[2]:
                # Handle small datasets: min_value must be <= max_value
                n_items = len(chart_layers)
                min_top_n = min(10, n_items)
                # Clear cached value if it's out of the valid range
                cached_key = "infl_show_top_n"
                if cached_key in st.session_state:
                    cached_val = st.session_state[cached_key]
                    if cached_val < min_top_n or cached_val > n_items:
                        del st.session_state[cached_key]
                show_top_n = st.number_input(
                    "Show top N",
                    min_value=min_top_n,
                    max_value=n_items,
                    value=min(200, n_items),
                    step=10,
                    key=cached_key,
                )

            # Filter to top N by abs_influence
            top_n = int(show_top_n)
            top_indices = sorted_idx[:top_n]

            # Prepare filtered data
            f_layers = [chart_layers[i] for i in top_indices]
            f_neurons = [chart_neurons[i] for i in top_indices]
            f_acts = [chart_acts[i] for i in top_indices]
            f_infs = [chart_infs[i] for i in top_indices]
            f_abs_infs = [chart_abs_infs[i] for i in top_indices]
            f_ranks = [ranks[i] for i in top_indices]
            f_selected = [is_selected[i] for i in top_indices]

            # Get y values based on metric
            if y_metric == "abs_influence":
                f_y = f_abs_infs
            elif y_metric == "influence":
                f_y = f_infs
            else:
                f_y = f_acts

            # Build the chart
            fig = go.Figure()

            if chart_type == "layer_scatter":
                # Scatter: x=layer (jittered), y=metric, color by layer
                import random
                random.seed(42)  # reproducible jitter
                jitter = [random.uniform(-0.3, 0.3) for _ in f_layers]
                x_jittered = [l + j for l, j in zip(f_layers, jitter)]

                # Non-selected points
                other_x = [x_jittered[i] for i in range(len(f_selected)) if not f_selected[i]]
                other_y = [f_y[i] for i in range(len(f_selected)) if not f_selected[i]]
                other_layers = [f_layers[i] for i in range(len(f_selected)) if not f_selected[i]]
                other_neurons = [f_neurons[i] for i in range(len(f_selected)) if not f_selected[i]]
                other_ranks = [f_ranks[i] for i in range(len(f_selected)) if not f_selected[i]]
                other_acts = [f_acts[i] for i in range(len(f_selected)) if not f_selected[i]]
                other_abs = [f_abs_infs[i] for i in range(len(f_selected)) if not f_selected[i]]

                if other_x:
                    fig.add_trace(go.Scatter(
                        x=other_x,
                        y=other_y,
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=other_layers,
                            colorscale="Viridis",
                            opacity=0.7,
                            colorbar=dict(title="Layer", x=1.02),
                        ),
                        text=[f"L{other_layers[i]} N{other_neurons[i]}<br>rank={other_ranks[i]}<br>act={other_acts[i]:.4g}<br>abs_inf={other_abs[i]:.4g}"
                              for i in range(len(other_x))],
                        hoverinfo="text",
                        name="neurons",
                        customdata=list(zip(other_layers, other_neurons)),
                    ))

                # Selected point (highlighted)
                sel_x = [x_jittered[i] for i in range(len(f_selected)) if f_selected[i]]
                sel_y = [f_y[i] for i in range(len(f_selected)) if f_selected[i]]
                sel_layers = [f_layers[i] for i in range(len(f_selected)) if f_selected[i]]
                sel_neurons = [f_neurons[i] for i in range(len(f_selected)) if f_selected[i]]
                sel_ranks = [f_ranks[i] for i in range(len(f_selected)) if f_selected[i]]
                if sel_x:
                    fig.add_trace(go.Scatter(
                        x=sel_x,
                        y=sel_y,
                        mode="markers",
                        marker=dict(size=16, color="red", symbol="star", line=dict(width=2, color="black")),
                        text=[f"SELECTED: L{sel_layers[i]} N{sel_neurons[i]}<br>rank={sel_ranks[i]}" for i in range(len(sel_x))],
                        hoverinfo="text",
                        name="selected",
                    ))

                fig.update_layout(
                    title=f"Layer vs {y_metric} (top {top_n})",
                    xaxis_title="Layer",
                    yaxis_title=y_metric,
                    height=450,
                )

            elif chart_type == "act_vs_inf":
                # Scatter: x=activation, y=influence, color by layer
                other_idx = [i for i in range(len(f_selected)) if not f_selected[i]]
                sel_idx = [i for i in range(len(f_selected)) if f_selected[i]]

                if other_idx:
                    fig.add_trace(go.Scatter(
                        x=[f_acts[i] for i in other_idx],
                        y=[f_infs[i] for i in other_idx],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=[f_layers[i] for i in other_idx],
                            colorscale="Viridis",
                            opacity=0.7,
                            colorbar=dict(title="Layer", x=1.02),
                        ),
                        text=[f"L{f_layers[i]} N{f_neurons[i]}<br>rank={f_ranks[i]}<br>act={f_acts[i]:.4g}<br>inf={f_infs[i]:.4g}"
                              for i in other_idx],
                        hoverinfo="text",
                        name="neurons",
                        customdata=[(f_layers[i], f_neurons[i]) for i in other_idx],
                    ))

                if sel_idx:
                    fig.add_trace(go.Scatter(
                        x=[f_acts[i] for i in sel_idx],
                        y=[f_infs[i] for i in sel_idx],
                        mode="markers",
                        marker=dict(size=16, color="red", symbol="star", line=dict(width=2, color="black")),
                        text=[f"SELECTED: L{f_layers[i]} N{f_neurons[i]}<br>rank={f_ranks[i]}" for i in sel_idx],
                        hoverinfo="text",
                        name="selected",
                    ))

                fig.update_layout(
                    title=f"Activation vs Influence (top {top_n})",
                    xaxis_title="Activation",
                    yaxis_title="Influence",
                    height=450,
                )

            elif chart_type == "rank_bars":
                # Bar chart: x=rank, y=metric, color by layer
                colors = ["red" if f_selected[i] else f"hsl({f_layers[i] * 360 / n_layers}, 70%, 50%)" for i in range(len(f_ranks))]
                fig.add_trace(go.Bar(
                    x=f_ranks,
                    y=f_y,
                    marker_color=colors,
                    text=[f"L{f_layers[i]} N{f_neurons[i]}" for i in range(len(f_ranks))],
                    hovertemplate="Rank %{x}<br>%{text}<br>" + y_metric + "=%{y:.4g}<extra></extra>",
                    customdata=list(zip(f_layers, f_neurons)),
                ))
                fig.update_layout(
                    title=f"Influence Ranking (top {top_n}) - color by layer",
                    xaxis_title="Rank",
                    yaxis_title=y_metric,
                    height=450,
                )

            # Enable selection
            fig.update_layout(clickmode="event+select", dragmode="lasso")
            fig.update_traces(
                selector=dict(type="scatter"),
                unselected=dict(marker=dict(opacity=0.3)),
            )

            # Render chart with selection support
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                key="infl_dist_chart",
                on_select="rerun",
                selection_mode=["points", "box", "lasso"],
            )

            # Handle selection to navigate (use pending nav to avoid widget state conflict)
            if event and hasattr(event, "selection") and event.selection:
                points = getattr(event.selection, "points", None) or []
                if points:
                    pt = points[0]
                    customdata = getattr(pt, "customdata", None)
                    if customdata is None and isinstance(pt, dict):
                        customdata = pt.get("customdata")
                    if customdata and len(customdata) >= 2:
                        clicked_layer, clicked_neuron = int(customdata[0]), int(customdata[1])
                        if (clicked_layer, clicked_neuron) != (layer, neuron):
                            # Store pending navigation and rerun
                            nav = {"layer": clicked_layer, "neuron": clicked_neuron}
                            pairs = st.session_state.get("_candidate_pairs", [])
                            for i, (l, n) in enumerate(pairs):
                                if l == clicked_layer and n == clicked_neuron:
                                    nav["candidate_idx"] = i
                                    break
                            st.session_state._pending_nav = nav
                            st.rerun()

            st.caption(
                "Tip: Use lasso/box selection (Plotly toolbar) to select a neuron and navigate to it. "
                "Selected neuron is shown as a red star. Color indicates layer."
            )

            # Fallback: quick jump selectbox
            with st.expander("Quick jump (fallback)", expanded=False):
                jump_labels = [
                    f"#{f_ranks[i]}: L{f_layers[i]} N{f_neurons[i]} | {y_metric}={f_y[i]:.4g}"
                    for i in range(len(f_ranks))
                ]
                jump_idx = st.selectbox(
                    "Select neuron",
                    options=list(range(len(jump_labels))),
                    format_func=lambda i: jump_labels[i],
                    key="chart_jump_sel",
                )
                if st.button("Go", key="chart_jump_btn"):
                    target_layer = f_layers[int(jump_idx)]
                    target_neuron = f_neurons[int(jump_idx)]
                    nav = {"layer": target_layer, "neuron": target_neuron}
                    pairs = st.session_state.get("_candidate_pairs", [])
                    for i, (l, n) in enumerate(pairs):
                        if l == target_layer and n == target_neuron:
                            nav["candidate_idx"] = i
                            break
                    st.session_state._pending_nav = nav
                    st.rerun()

    st.markdown("### Notes")
    st.write(
        "This app visualizes raw hook_post activations for a single (layer, neuron) across prompts. "
        "To get ranked candidates, run POC 2 on the same run_dir (peaks or influence mode) and refresh the app."
    )


if __name__ == "__main__":
    main()



