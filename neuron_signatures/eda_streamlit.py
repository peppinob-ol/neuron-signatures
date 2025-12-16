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
    # Optional: keyboard shortcuts for arrow navigation
    from streamlit_shortcuts import add_shortcuts  # type: ignore
except ModuleNotFoundError:
    add_shortcuts = None

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
    (`influence_*.json`) while excluding `*_meta.json`.
    Sorted by modification time (newest first).
    """
    if not analysis_dir.exists():
        return []
    files = list(analysis_dir.glob("influence_*.csv")) + list(analysis_dir.glob("influence_*.json"))
    out: List[Path] = []
    for p in files:
        if p.name.endswith("_meta.json"):
            continue
        out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


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

    # Store bounds for shortcut callbacks (kept in session_state for simplicity)
    st.session_state._max_layer = n_layers - 1
    st.session_state._max_neuron = d_mlp - 1

    # Keyboard navigation toggle
    st.sidebar.header("Keyboard navigation")
    kb_enabled = st.sidebar.checkbox("enable_arrow_keys", value=True)
    st.sidebar.caption("Up/Down: layer  Left/Right: neuron (or filtered list, if enabled below)")
    if kb_enabled and add_shortcuts is None:
        st.sidebar.warning("Install streamlit-shortcuts to enable arrow keys: uv pip install -r requirements.txt")

    # Candidate list navigation (set later after we load rankings)
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

    # Callback functions for arrow navigation (used by hidden/visible buttons).
    def kb_layer_up() -> None:
        if st.session_state.get("arrow_keys_candidates", False) and st.session_state.get("_candidate_pairs"):
            st.session_state.candidate_idx = max(int(st.session_state.candidate_idx) - 1, 0)
            _apply_candidate_idx()
            return
        st.session_state.layer = max(int(st.session_state.layer) - 1, 0)

    def kb_layer_down() -> None:
        if st.session_state.get("arrow_keys_candidates", False) and st.session_state.get("_candidate_pairs"):
            pairs = list(st.session_state.get("_candidate_pairs", []))
            st.session_state.candidate_idx = min(int(st.session_state.candidate_idx) + 1, len(pairs) - 1)
            _apply_candidate_idx()
            return
        st.session_state.layer = min(int(st.session_state.layer) + 1, int(st.session_state._max_layer))

    def kb_neuron_left() -> None:
        if st.session_state.get("arrow_keys_candidates", False) and st.session_state.get("_candidate_pairs"):
            st.session_state.candidate_idx = max(int(st.session_state.candidate_idx) - 10, 0)
            _apply_candidate_idx()
            return
        st.session_state.neuron = max(int(st.session_state.neuron) - 1, 0)

    def kb_neuron_right() -> None:
        if st.session_state.get("arrow_keys_candidates", False) and st.session_state.get("_candidate_pairs"):
            pairs = list(st.session_state.get("_candidate_pairs", []))
            st.session_state.candidate_idx = min(int(st.session_state.candidate_idx) + 10, len(pairs) - 1)
            _apply_candidate_idx()
            return
        st.session_state.neuron = min(int(st.session_state.neuron) + 1, int(st.session_state._max_neuron))

    # Register shortcuts (only if installed and enabled).
    # streamlit-shortcuts binds keyboard keys to Streamlit elements *by key*,
    # so we create buttons with matching keys and wire them to these callbacks.
    if kb_enabled and add_shortcuts is not None:
        # Buttons can be clicked too; arrow keys will click them automatically.
        nav = st.sidebar.container()
        nav_cols = nav.columns(4)
        nav_cols[0].button("Up", key="kb_layer_up", on_click=kb_layer_up, help="ArrowUp: layer -1")
        nav_cols[1].button("Down", key="kb_layer_down", on_click=kb_layer_down, help="ArrowDown: layer +1")
        nav_cols[2].button("Left", key="kb_neuron_left", on_click=kb_neuron_left, help="ArrowLeft: neuron -1")
        nav_cols[3].button("Right", key="kb_neuron_right", on_click=kb_neuron_right, help="ArrowRight: neuron +1")

        # Bind arrow keys to those element keys. `e.key` becomes lowercase in the
        # library implementation, so use 'arrowup'/'arrowdown'/etc.
        add_shortcuts(
            kb_layer_up="arrowup",
            kb_layer_down="arrowdown",
            kb_neuron_left="arrowleft",
            kb_neuron_right="arrowright",
        )

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
                candidate_pairs.append((int(layer_i), int(neuron_i)))
                candidate_labels.append(
                    f"L{int(layer_i)} N{int(neuron_i)} | {metric}={float(m_v):.6g} | abs_inf={float(abs_v):.6g} | act={float(act_v):.6g}"
                )

        else:
            st.sidebar.warning("Influence JSON can be very large; prefer the CSV for filtering.")
            infl = _load_influence_ranking(influence_path) or []
            limited = infl[: int(top_k_limit)]
            candidates = limited
            for n in limited:
                layer_i = int(n.get("layer", 0))
                neuron_i = int(n.get("neuron", 0))
                abs_inf = float(n.get("abs_influence", 0.0))
                tok = str(n.get("target_token_ascii", ""))
                candidate_pairs.append((layer_i, neuron_i))
                candidate_labels.append(f"L{layer_i} N{neuron_i} | abs_influence={abs_inf:.6g} | target={tok}")
            st.sidebar.caption(f"Source: {influence_path.name} (showing first {len(limited)})")

    # Persist candidate pairs for keyboard callbacks
    st.session_state._candidate_pairs = candidate_pairs
    st.session_state._candidate_labels = candidate_labels

    # Toggle: arrow keys browse candidates (if available)
    st.sidebar.checkbox(
        "arrow_keys_candidates",
        value=(bool(candidate_pairs)),
        disabled=(not bool(candidate_pairs)),
        help="If enabled, arrow keys browse within the filtered candidate list (Up/Down: +/-1, Left/Right: +/-10).",
    )

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
        st.sidebar.caption("Tip: enable arrow_keys_candidates to step through this list with the keyboard.")
        # Ensure layer/neuron are consistent if candidate_idx changed elsewhere (keyboard)
        _apply_candidate_idx()
    else:
        st.sidebar.info("No ranking file found (or manual mode). Use manual indices or arrow keys.")

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

    hm_col1, hm_col2 = st.columns([1, 1])
    with hm_col1:
        tokens_per_row = st.number_input("tokens_per_row", min_value=10, max_value=200, value=50, step=5)
        show_values = st.checkbox("show_values_on_tokens", value=False)
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
    prompt_series = [(pid, toks, tids, vals) for (pid, tids, toks, vals) in series]
    stacked_html = render_stacked_prompts_heatmap_html(
        prompt_series=prompt_series,
        cfg=cfg,
        global_max_abs=(None if normalize_mode == "per_prompt" else max_abs),
        title=f"Layer {layer} | Neuron {neuron}",
    )
    st.markdown(stacked_html, unsafe_allow_html=True)

    st.markdown("### Notes")
    st.write(
        "This app visualizes raw hook_post activations for a single (layer, neuron) across prompts. "
        "To get ranked candidates, run POC 2 on the same run_dir (peaks or influence mode) and refresh the app."
    )


if __name__ == "__main__":
    main()



