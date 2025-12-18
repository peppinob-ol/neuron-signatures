"""
POC 1 browser: interactive TUI to inspect saved activations (manifest.json + activations.pt).

Run:
    python -m neuron_signatures.poc1_browse_activations --run_dir runs/poc1_test_gpu6

Keybindings (ASCII-only):
    - Left/Right: layer -/+ 1
    - Up/Down: ctx_idx (token position) -/+ 1
    - n / b: neuron +1 / -1
    - PageDown / PageUp: neuron +100 / -100
    - ] / [: next / previous prompt
    - a: toggle all-tokens view vs windowed view
    - g: go to neuron index
    - l: go to layer index
    - c: go to ctx_idx
    - ?: help
    - q: quit
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    return s[: max_len - 3] + "..."


def _format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def _clamp(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s.strip())
    except Exception:
        return None


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "manifest.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_activations(run_dir: Path) -> Dict[str, torch.Tensor]:
    path = run_dir / "activations.pt"
    return torch.load(path, map_location="cpu")


@dataclass
class BrowserState:
    run_dir: Path
    manifest: Dict[str, Any]
    activations: Dict[str, torch.Tensor]
    prompts: List[Dict[str, Any]]
    prompt_idx: int = 0
    layer: int = 0
    neuron: int = 0
    ctx_idx: int = 0
    show_all_tokens: bool = False
    window: int = 15

    @property
    def n_layers(self) -> int:
        return int(self.manifest.get("n_layers", 0))

    @property
    def d_mlp(self) -> int:
        return int(self.manifest.get("d_mlp", 0))


def _current_prompt(state: BrowserState) -> Dict[str, Any]:
    return state.prompts[state.prompt_idx]


def _current_tensor(state: BrowserState) -> torch.Tensor:
    pid = _current_prompt(state)["probe_id"]
    return state.activations[pid]


def _sync_state_bounds(state: BrowserState) -> None:
    state.prompt_idx = _clamp(state.prompt_idx, 0, len(state.prompts) - 1)
    state.layer = _clamp(state.layer, 0, state.n_layers - 1)
    state.neuron = _clamp(state.neuron, 0, state.d_mlp - 1)

    seq_len = int(_current_tensor(state).shape[1])
    state.ctx_idx = _clamp(state.ctx_idx, 0, seq_len - 1)


def _stats_for_vector(v: torch.Tensor) -> Dict[str, Any]:
    v_f = v.float()
    max_val, max_idx = torch.max(v_f, dim=0)
    min_val, min_idx = torch.min(v_f, dim=0)
    mean_val = torch.mean(v_f)
    std_val = torch.std(v_f, unbiased=False)
    max_abs = torch.max(torch.abs(v_f)).item()
    return {
        "max_val": float(max_val.item()),
        "max_idx": int(max_idx.item()),
        "min_val": float(min_val.item()),
        "min_idx": int(min_idx.item()),
        "mean": float(mean_val.item()),
        "std": float(std_val.item()),
        "max_abs": float(max_abs),
    }


def _render(state: BrowserState) -> str:
    _sync_state_bounds(state)

    prompt = _current_prompt(state)
    pid = prompt["probe_id"]
    text = prompt.get("text", "")
    token_ids: List[int] = prompt.get("token_ids", [])
    tokens_ascii: List[str] = prompt.get("tokens_ascii", [])

    x = _current_tensor(state)  # [n_layers, seq_len, d_mlp]
    seq_len = int(x.shape[1])

    v = x[state.layer, :, state.neuron]  # [seq_len]
    v_f = v.float()
    s = _stats_for_vector(v_f)

    ctx_idx = state.ctx_idx
    cursor_val = float(v_f[ctx_idx].item())
    cursor_tid = token_ids[ctx_idx] if ctx_idx < len(token_ids) else -1
    cursor_tok = tokens_ascii[ctx_idx] if ctx_idx < len(tokens_ascii) else ""

    max_idx = s["max_idx"]
    min_idx = s["min_idx"]

    # Token window
    if state.show_all_tokens:
        start = 0
        end = seq_len
    else:
        start = max(0, ctx_idx - state.window)
        end = min(seq_len, ctx_idx + state.window + 1)

    # Header
    lines: List[str] = []
    lines.append(f"Run dir: {state.run_dir.as_posix()}")
    lines.append(
        f"Model: {state.manifest.get('model_name','')} | hook: {state.manifest.get('hook_name','')}"
    )
    lines.append(
        f"Prompt {state.prompt_idx + 1}/{len(state.prompts)}: {pid} | seq_len={seq_len}"
    )
    lines.append(f"Text: {text}")
    lines.append(
        f"Layer {state.layer}/{state.n_layers - 1} | Neuron {state.neuron}/{state.d_mlp - 1}"
    )
    lines.append(
        "Cursor: "
        f"ctx_idx={ctx_idx} | token_id={cursor_tid} | token={_truncate(repr(cursor_tok), 40)} | "
        f"act={cursor_val:+.6f}"
    )
    max_tok = tokens_ascii[max_idx] if max_idx < len(tokens_ascii) else ""
    min_tok = tokens_ascii[min_idx] if min_idx < len(tokens_ascii) else ""
    lines.append(
        "Stats: "
        f"max={s['max_val']:+.6f}@{max_idx}({_truncate(repr(max_tok), 18)}) | "
        f"min={s['min_val']:+.6f}@{min_idx}({_truncate(repr(min_tok), 18)}) | "
        f"mean={s['mean']:+.6f} | std={s['std']:+.6f}"
    )
    lines.append("")
    lines.append(
        "Keys: [ ] prompt | Left/Right layer | Up/Down ctx | n/b neuron +/-1 | "
        "PageUp/PageDown neuron +/-100 | a all/window | g goto_neuron | l goto_layer | c goto_ctx | ? help | q quit"
    )
    lines.append("")

    # Table header
    lines.append("  ctx_idx  token_id  token                 activation      bar")
    lines.append("  ------  --------  --------------------  ------------  --------------------")

    max_abs = float(s["max_abs"])
    bar_max = 20

    for i in range(start, end):
        tok = tokens_ascii[i] if i < len(tokens_ascii) else ""
        tid = token_ids[i] if i < len(token_ids) else -1
        val = float(v_f[i].item())

        cursor_mark = ">" if i == ctx_idx else " "
        max_mark = "*" if i == max_idx else " "

        if max_abs > 0:
            bar_len = int(round(abs(val) / max_abs * bar_max))
        else:
            bar_len = 0
        bar = "#" * bar_len

        tok_disp = _truncate(repr(tok), 20)
        lines.append(
            f"{cursor_mark}{max_mark} {i:6d}  {tid:8d}  {tok_disp:<20}  {val:+12.6f}  {bar:<20}"
        )

    if not state.show_all_tokens and (seq_len > (end - start)):
        lines.append("")
        lines.append(
            f"[Windowed view] showing ctx_idx {start}..{end-1} of {seq_len}. Press 'a' to toggle all tokens."
        )

    return "\n".join(lines)


def _run_tui(state: BrowserState) -> None:
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import HSplit
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Frame, TextArea
        from prompt_toolkit.shortcuts.dialogs import input_dialog, message_dialog
    except Exception as e:
        print("Missing dependency: prompt_toolkit. Install with: uv pip install -r requirements.txt")
        raise

    kb = KeyBindings()
    text_area = TextArea(text=_render(state), read_only=True)

    def refresh() -> None:
        text_area.text = _render(state)

    @kb.add("q")
    def _(event) -> None:
        event.app.exit()

    @kb.add("left")
    def _(event) -> None:
        state.layer = _clamp(state.layer - 1, 0, state.n_layers - 1)
        refresh()

    @kb.add("right")
    def _(event) -> None:
        state.layer = _clamp(state.layer + 1, 0, state.n_layers - 1)
        refresh()

    @kb.add("up")
    def _(event) -> None:
        seq_len = int(_current_tensor(state).shape[1])
        state.ctx_idx = _clamp(state.ctx_idx - 1, 0, seq_len - 1)
        refresh()

    @kb.add("down")
    def _(event) -> None:
        seq_len = int(_current_tensor(state).shape[1])
        state.ctx_idx = _clamp(state.ctx_idx + 1, 0, seq_len - 1)
        refresh()

    @kb.add("n")
    def _(event) -> None:
        state.neuron = _clamp(state.neuron + 1, 0, state.d_mlp - 1)
        refresh()

    @kb.add("b")
    def _(event) -> None:
        state.neuron = _clamp(state.neuron - 1, 0, state.d_mlp - 1)
        refresh()

    @kb.add("pageup")
    def _(event) -> None:
        state.neuron = _clamp(state.neuron - 100, 0, state.d_mlp - 1)
        refresh()

    @kb.add("pagedown")
    def _(event) -> None:
        state.neuron = _clamp(state.neuron + 100, 0, state.d_mlp - 1)
        refresh()

    @kb.add("[")
    def _(event) -> None:
        state.prompt_idx = _clamp(state.prompt_idx - 1, 0, len(state.prompts) - 1)
        state.ctx_idx = 0
        refresh()

    @kb.add("]")
    def _(event) -> None:
        state.prompt_idx = _clamp(state.prompt_idx + 1, 0, len(state.prompts) - 1)
        state.ctx_idx = 0
        refresh()

    @kb.add("a")
    def _(event) -> None:
        state.show_all_tokens = not state.show_all_tokens
        refresh()

    @kb.add("g")
    def _(event) -> None:
        result = input_dialog(
            title="Go to neuron",
            text=f"Enter neuron index [0..{state.d_mlp - 1}]:",
        ).run()
        if result is None:
            return
        idx = _safe_int(result)
        if idx is None:
            message_dialog(title="Invalid input", text="Neuron index must be an integer.").run()
            return
        state.neuron = _clamp(idx, 0, state.d_mlp - 1)
        refresh()

    @kb.add("l")
    def _(event) -> None:
        result = input_dialog(
            title="Go to layer",
            text=f"Enter layer index [0..{state.n_layers - 1}]:",
        ).run()
        if result is None:
            return
        idx = _safe_int(result)
        if idx is None:
            message_dialog(title="Invalid input", text="Layer index must be an integer.").run()
            return
        state.layer = _clamp(idx, 0, state.n_layers - 1)
        refresh()

    @kb.add("c")
    def _(event) -> None:
        seq_len = int(_current_tensor(state).shape[1])
        result = input_dialog(
            title="Go to ctx_idx",
            text=f"Enter ctx_idx [0..{seq_len - 1}]:",
        ).run()
        if result is None:
            return
        idx = _safe_int(result)
        if idx is None:
            message_dialog(title="Invalid input", text="ctx_idx must be an integer.").run()
            return
        state.ctx_idx = _clamp(idx, 0, seq_len - 1)
        refresh()

    @kb.add("?")
    def _(event) -> None:
        help_text = (
            "POC 1 activation browser\\n\\n"
            "Navigation:\\n"
            "- [ / ]: previous / next prompt\\n"
            "- Left / Right: layer -/+ 1\\n"
            "- Up / Down: ctx_idx (token position) -/+ 1\\n"
            "- n / b: neuron +1 / -1\\n"
            "- PageDown / PageUp: neuron +100 / -100\\n"
            "- a: toggle all tokens vs windowed tokens\\n\\n"
            "Jump:\\n"
            "- g: go to neuron index\\n"
            "- l: go to layer index\\n"
            "- c: go to ctx_idx\\n\\n"
            "Quit:\\n"
            "- q\\n"
        )
        message_dialog(title="Help", text=help_text).run()

    root = HSplit([Frame(text_area, title="Activation Browser")])
    style = Style.from_dict({})
    app = Application(layout=Layout(root), key_bindings=kb, full_screen=True, style=style)
    app.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="POC 1: Browse saved activations in a TUI")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory containing manifest.json and activations.pt",
    )
    parser.add_argument(
        "--prompt_idx",
        type=int,
        default=0,
        help="Initial prompt index (default: 0)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Initial layer index (default: 0)",
    )
    parser.add_argument(
        "--neuron",
        type=int,
        default=0,
        help="Initial neuron index (default: 0)",
    )
    parser.add_argument(
        "--ctx_idx",
        type=int,
        default=0,
        help="Initial token position ctx_idx (default: 0)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=15,
        help="Window radius for token display (default: 15). Ignored in all-tokens mode.",
    )
    parser.add_argument(
        "--all_tokens",
        action="store_true",
        help="Start in all-tokens mode (instead of windowed mode).",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    manifest = _load_manifest(run_dir)
    activations = _load_activations(run_dir)

    prompts = manifest.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("manifest.json has no prompts list")

    # Quick integrity checks (lightweight)
    missing = []
    for p in prompts:
        pid = p.get("probe_id")
        if pid not in activations:
            missing.append(pid)
    if missing:
        raise ValueError(f"activations.pt missing prompt ids: {missing[:10]}")

    state = BrowserState(
        run_dir=run_dir,
        manifest=manifest,
        activations=activations,
        prompts=prompts,
        prompt_idx=args.prompt_idx,
        layer=args.layer,
        neuron=args.neuron,
        ctx_idx=args.ctx_idx,
        show_all_tokens=bool(args.all_tokens),
        window=int(args.window),
    )

    # Display one-line startup info (ASCII-only)
    total_bytes = 0
    for t in activations.values():
        total_bytes += int(t.numel() * t.element_size())
    print(f"Loaded {len(prompts)} prompt(s). Total activation bytes: {_format_mib(total_bytes)}")

    _run_tui(state)


if __name__ == "__main__":
    main()







