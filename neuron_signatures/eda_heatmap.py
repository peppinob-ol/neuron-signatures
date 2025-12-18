"""
Neuronpedia-style token heatmap rendering for neuron activations.

Adapted for POC1 run outputs:
- tokens come from manifest.json as tokens_ascii + token_ids
- values are a single neuron's activations over token positions

Implementation notes:
- Logarithmic opacity scaling like Neuronpedia.
- Green for positive activations, orange for negative activations.
- tokens_ascii are expected to be ASCII-safe already (from the pipeline).
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# Colors from Neuronpedia (RGB 0-255)
EMERALD_RGB = (52, 211, 153)
ORANGE_RGB = (251, 146, 60)

# Neuronpedia-style constants
MINIMUM_OPACITY = 0.05
MINIMUM_THRESHOLD = 0.00005


@dataclass(frozen=True)
class HeatmapConfig:
    tokens_per_row: int = 50
    show_values: bool = True
    exclude_bos: bool = True
    value_decimals: int = 2
    minimum_threshold: float = MINIMUM_THRESHOLD
    minimum_opacity: float = MINIMUM_OPACITY
    color_scale: str = "log"  # "log", "linear", "sqrt", "rank"


# Valid color scale modes
COLOR_SCALE_MODES = ["log", "linear", "sqrt", "power_0.3", "power_2"]


def calculate_opacity(value_abs: float, max_abs: float, cfg: HeatmapConfig) -> float:
    """Calculate opacity using configured scaling.
    
    Scaling modes:
    - log: Neuronpedia-style logarithmic scaling (default)
    - linear: Simple linear ratio
    - sqrt: Square root - exaggerates low values, compresses highs
    - power_0.3: Strong exaggeration of differences (gamma 0.3)
    - power_2: Compresses low values, exaggerates highs (gamma 2)
    """
    if max_abs <= 0.0 or value_abs <= cfg.minimum_threshold:
        return 0.0

    ratio = value_abs / max_abs
    if ratio <= 0.0:
        return 0.0
    if ratio > 1.0:
        ratio = 1.0

    scale = 1.0 - cfg.minimum_opacity
    
    mode = cfg.color_scale
    if mode == "linear":
        # Simple linear scaling
        scaled_ratio = ratio
    elif mode == "sqrt":
        # Square root: exaggerates small differences
        scaled_ratio = math.sqrt(ratio)
    elif mode == "power_0.3":
        # Strong exaggeration (gamma = 0.3)
        scaled_ratio = math.pow(ratio, 0.3)
    elif mode == "power_2":
        # Compress small values, exaggerate large (gamma = 2)
        scaled_ratio = ratio * ratio
    else:
        # Default: Neuronpedia-style log scaling
        scaled_ratio = math.log10(1.0 + 9.0 * ratio) / math.log10(10.0)
    
    opacity = cfg.minimum_opacity + scaled_ratio * scale
    return max(0.0, min(1.0, opacity))


def rgba_for_value(value: float, max_abs: float, cfg: HeatmapConfig) -> Tuple[int, int, int, float]:
    """Return RGBA where r,g,b are 0-255 and a in [0,1]."""
    opacity = calculate_opacity(abs(value), max_abs, cfg)
    r, g, b = EMERALD_RGB if value >= 0.0 else ORANGE_RGB
    return r, g, b, opacity


def text_color_for_opacity(opacity: float) -> str:
    """Choose black or white text depending on background intensity."""
    return "#000000" if opacity < 0.5 else "#FFFFFF"


def compute_max_abs(tokens_ascii: Sequence[str], values: Sequence[float], cfg: HeatmapConfig) -> float:
    """Compute max(abs(values)), optionally excluding BOS."""
    if not values:
        return 0.0

    if not cfg.exclude_bos:
        return max(abs(float(v)) for v in values)

    max_abs = 0.0
    for tok, v in zip(tokens_ascii, values):
        if tok == "<bos>":
            continue
        a = abs(float(v))
        if a > max_abs:
            max_abs = a
    return max_abs


def _chunk_indices(n: int, chunk: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        j = min(n, i + chunk)
        spans.append((i, j))
        i = j
    return spans


def render_token_heatmap_html(
    tokens_ascii: Sequence[str],
    token_ids: Sequence[int],
    values: Sequence[float],
    cfg: HeatmapConfig,
    max_abs_override: Optional[float] = None,
    title: str = "",
) -> str:
    """Render a token heatmap as an HTML block (Streamlit: unsafe_allow_html=True)."""
    if len(tokens_ascii) != len(values) or len(token_ids) != len(values):
        raise ValueError("tokens_ascii, token_ids, and values must have the same length")

    max_abs = float(max_abs_override) if max_abs_override is not None else compute_max_abs(tokens_ascii, values, cfg)
    rows = _chunk_indices(len(values), max(1, int(cfg.tokens_per_row)))

    title_html = ""
    if title:
        title_html = f"<div class='ns-heatmap-title'>{html.escape(title)}</div>"

    info_line = (
        f"<div class='ns-heatmap-meta'>max_abs={max_abs:.6f} | tokens={len(values)} | per_row={int(cfg.tokens_per_row)}</div>"
    )

    css = (
        "<style>"
        ".ns-heatmap-title { font-weight: 600; margin: 6px 0 6px 0; }"
        ".ns-heatmap-meta { color: #666; font-size: 12px; margin: 0 0 6px 0; }"
        # Neuronpedia-like chips: fixed width, no squishing. Row scrolls horizontally if needed.
        ".ns-heatmap-row { display: flex; flex-wrap: nowrap; gap: 4px; margin-bottom: 8px; overflow-x: auto; padding-bottom: 2px; }"
        ".ns-heatmap-cell { flex: 0 0 auto; width: 86px; }"
        ".ns-heatmap-token { border: 1px solid #ddd; border-radius: 4px; padding: 6px 6px; text-align: center; }"
        ".ns-heatmap-token-text { font-weight: 700; font-size: 13px; line-height: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; "
        "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }"
        ".ns-heatmap-token-val { font-size: 11px; line-height: 11px; opacity: 0.85; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; "
        "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }"
        "</style>"
    )

    blocks: List[str] = [css, title_html, info_line]

    def display_token(tok: str) -> str:
        # Mimic Neuronpedia-like space marker using ASCII '.' prefix.
        # Example: " Texas" -> ".Texas"
        if tok.startswith(" "):
            stripped = tok.lstrip(" ")
            if stripped:
                return "." + stripped
            return "."
        return tok

    for start, end in rows:
        row_cells: List[str] = []
        for i in range(start, end):
            tok = str(tokens_ascii[i])
            tid = int(token_ids[i])
            val = float(values[i])

            r, g, b, a = rgba_for_value(val, max_abs, cfg)
            fg = text_color_for_opacity(a)

            tok_disp = html.escape(display_token(tok))
            tooltip = f"ctx_idx={i} | token_id={tid} | token={tok} | act={val:+.6f}"
            tooltip_attr = html.escape(tooltip, quote=True)

            value_line = ""
            if cfg.show_values and abs(val) > cfg.minimum_threshold:
                fmt = f"{{0:.{int(cfg.value_decimals)}f}}"
                value_line = f"<div class='ns-heatmap-token-val'>{html.escape(fmt.format(val))}</div>"

            cell = (
                f"<div class='ns-heatmap-cell' title=\"{tooltip_attr}\">"
                f"<div class='ns-heatmap-token' style='background-color: rgba({r},{g},{b},{a:.4f}); color: {fg};'>"
                f"<div class='ns-heatmap-token-text'>{tok_disp}</div>"
                f"{value_line}"
                "</div>"
                "</div>"
            )
            row_cells.append(cell)

        blocks.append("<div class='ns-heatmap-row'>" + "".join(row_cells) + "</div>")

    return "".join(blocks)


def render_stacked_prompts_heatmap_html(
    prompt_series: Sequence[Tuple[str, Sequence[str], Sequence[int], Sequence[float]]],
    cfg: HeatmapConfig,
    global_max_abs: Optional[float] = None,
    title: str = "",
) -> str:
    """Render stacked prompts heatmap for the same neuron across prompts."""
    if global_max_abs is None:
        gm = 0.0
        for _, toks, _, vals in prompt_series:
            m = compute_max_abs(toks, vals, cfg)
            if m > gm:
                gm = m
        global_max_abs = gm

    css = (
        "<style>"
        ".ns-stacked-title { font-weight: 700; margin: 8px 0 4px 0; }"
        ".ns-stacked-meta { color: #666; font-size: 12px; margin: 0 0 8px 0; }"
        ".ns-stacked-prompt-id { margin: 10px 0 4px 0; font-weight: 600; }"
        "</style>"
    )

    parts: List[str] = [css]
    if title:
        parts.append(f"<div class='ns-stacked-title'>{html.escape(title)}</div>")
    parts.append(
        f"<div class='ns-stacked-meta'>global_max_abs={float(global_max_abs):.6f} | prompts={len(prompt_series)}</div>"
    )

    for prompt_id, toks, tids, vals in prompt_series:
        parts.append(f"<div class='ns-stacked-prompt-id'>prompt: {html.escape(prompt_id)}</div>")
        parts.append(
            render_token_heatmap_html(
                tokens_ascii=toks,
                token_ids=tids,
                values=vals,
                cfg=cfg,
                max_abs_override=float(global_max_abs),
                title="",
            )
        )

    return "".join(parts)


