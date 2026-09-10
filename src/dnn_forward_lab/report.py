"""Standalone, escaped research reports. All chart values come from exported returns."""

import base64
import html
import io
from importlib.resources import files
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

LABELS = {
    "forward_selected": "DNN forward selection",
    "backward_selected": "Historical selection",
    "buy_hold": "Buy and hold (reset each fold)",
    "cash": "Cash (zero interest)",
}
COLORS = ["#BB86FC", "#03DAC6", "#E0E0E0", "#CF6679"]


def font_css() -> str:
    rules = []
    for name, file in (("Roboto", "sans.woff2"), ("Roboto Mono", "mono.woff2")):
        content = files("dnn_forward_lab").joinpath("assets", file).read_bytes()
        encoded = base64.b64encode(content).decode()
        rules.append(
            f"@font-face{{font-family:'{name}';src:url(data:font/woff2;base64,{encoded}) format('woff2');font-weight:400;font-display:swap;}}"
        )
    return "\n".join(rules)


def plot_performance(
    returns: pd.DataFrame, output: Path, initial_cash: float, compact: bool = False
) -> bytes:
    font = files("dnn_forward_lab").joinpath("assets", "Roboto.ttf")
    font_manager.fontManager.addfont(str(font))
    style = {
        "figure.facecolor": "#121212",
        "axes.facecolor": "#121212",
        "axes.edgecolor": "#555555",
        "axes.labelcolor": "#DEDEDE",
        "text.color": "#DEDEDE",
        "xtick.color": "#BDBDBD",
        "ytick.color": "#BDBDBD",
        "grid.color": "#383838",
        "font.family": "Roboto",
        "font.size": 11 if compact else 10,
        "svg.fonttype": "path",
        "svg.hashsalt": "dnn-forward-lab-v1",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(4, 6) if compact else (11, 7),
            sharex=True,
            layout="constrained",
            gridspec_kw={"height_ratios": [2, 1]},
        )
        for i, name in enumerate(returns):
            # Include initial capital so an immediate first-bar loss appears in drawdown.
            wealth = np.r_[1.0, np.cumprod(1 + returns[name].to_numpy())]
            x = np.arange(len(wealth))
            drawdown = (wealth / np.maximum.accumulate(wealth) - 1) * 100
            axes[0].plot(
                x,
                wealth * initial_cash,
                color=COLORS[i],
                label=LABELS[name],
                linewidth=1.8,
                linestyle=["-", "--", "-.", ":"][i],
            )
            axes[1].plot(
                x, drawdown, color=COLORS[i], linewidth=1.4, linestyle=["-", "--", "-.", ":"][i]
            )
        axes[0].set_ylabel("Equity (input currency)")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_xlabel(
            "Test bar (0 = initial capital)"
            if compact
            else "Out-of-sample trading bar (0 = initial capital)"
        )
        if compact:
            axes[0].legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.02))
        else:
            axes[0].legend(frameon=False, fontsize=9, loc="best")
        for ax in axes:
            ax.grid(axis="y", alpha=0.7)
            ax.spines[["top", "right"]].set_visible(False)
        if not compact:
            fig.suptitle(
                f"Observed evaluation: {returns.index[0].date()} to {returns.index[-1].date()}",
                x=0.5,
            )
        stem = "performance-mobile" if compact else "performance"
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", metadata={"Date": None})
        if not compact:
            fig.savefig(output / f"{stem}.png", dpi=160, metadata={"Software": "DNN Forward Lab"})
        plt.close(fig)
        content = buffer.getvalue()
        (output / f"{stem}.svg").write_bytes(content)
        return content


def render_report(summary: dict, returns: pd.DataFrame, output: Path) -> Path:
    m, a, folds = summary["manifest"], summary["aggregate"], summary["folds"]
    svg = plot_performance(returns, output, m["config"]["initial_cash"])
    img = "data:image/svg+xml;base64," + base64.b64encode(svg).decode()
    mobile_svg = plot_performance(returns, output, m["config"]["initial_cash"], compact=True)
    mobile_img = "data:image/svg+xml;base64," + base64.b64encode(mobile_svg).decode()
    forward, historical = (
        a["forward_selected"]["total_return"],
        a["backward_selected"]["total_return"],
    )
    repaired_count = sum(fold["repaired_forecast_bars"] for fold in folds)
    delta = (forward - historical) * 100
    comparison = "exceeded" if delta > 1e-8 else "trailed" if delta < -1e-8 else "matched"
    headline = f"DNN selection {comparison} historical selection in this sample."
    escape = html.escape

    def number(value, percent=False):
        return (
            "Undefined"
            if value is None
            else f"{value * (100 if percent else 1):,.2f}{'%' if percent else ''}"
        )

    rows = "".join(
        f"<tr><th scope='row'>{LABELS[name]}</th><td>{number(metrics['total_return'], True)}</td>"
        f"<td>{number(metrics['max_drawdown'], True)}</td><td>{number(metrics['sharpe'])}</td>"
        f"<td>{number(metrics['sortino'])}</td><td>{metrics['trades']}</td></tr>"
        for name, metrics in a.items()
    )
    fold_rows = "".join(
        f"<tr><th scope='row'>{fold['fold']}</th><td>{fold['cutoff']}</td>"
        f"<td>{escape(fold['selected']['forward_selected'])}</td>"
        f"<td>{escape(fold['selected']['backward_selected'])}</td>"
        f"<td>{number(fold['performance']['forward_selected']['total_return'], True)}</td>"
        f"<td>{number(fold['performance']['backward_selected']['total_return'], True)}</td>"
        f"<td>{fold['repaired_forecast_bars']}/{m['config']['horizon']}</td></tr>"
        for fold in folds
    )
    error_rows = "".join(
        f"<tr><th scope='row'>{fold['fold']}</th><td>{escape(method)}</td>"
        f"<td>{number(scores['close']['mae'])}</td><td>{number(scores['close']['rmse'])}</td>"
        f"<td>{number(scores['close']['mape_pct'])}%</td></tr>"
        for fold in folds
        for method, scores in fold["forecast_errors"].items()
    )
    config_rows = "".join(
        f"<tr><th scope='row'>{escape(key)}</th><td>{escape(str(value))}</td></tr>"
        for key, value in m["config"].items()
    )
    synthetic_note = (
        "Synthetic prices demonstrate software behavior only. These results provide no evidence of market profitability."
        if m["data_status"].startswith("synthetic")
        else "Input prices and adjustment conventions are user-supplied and not independently verified."
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DNN Forward Lab | {escape(m["asset"])} research report</title>
<style>{font_css()}
:root{{color-scheme:dark;--bg:#121212;--surface:#1E1E1E;--text:#DEDEDE;--muted:#BDBDBD;--primary:#BB86FC;--line:#454545}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 Roboto,Arial,sans-serif}}
main{{max-width:1160px;margin:auto;padding:48px 32px 64px}}header{{padding:24px 0 40px;border-bottom:1px solid var(--line)}}
.eyebrow{{font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:var(--primary)}}
h1{{font-size:clamp(32px,4vw,48px);line-height:1.16;font-weight:400;max-width:860px;margin:24px 0}}
h2{{font-size:26px;font-weight:400;margin:0 0 20px}}h3{{font-size:20px;font-weight:400}}p{{max-width:850px}}
.muted,figcaption{{color:var(--muted)}}.status{{display:inline-block;border:1px solid var(--line);padding:4px 12px;margin:12px 0}}
.answer{{display:grid;grid-template-columns:1fr 2fr;gap:32px;padding:32px 0}}
.metric{{font:400 clamp(40px,6vw,64px)/1.15 'Roboto Mono',monospace;margin:8px 0 12px}}
.context{{padding:24px;background:var(--surface);border-radius:8px}}section{{padding:32px 0;border-bottom:1px solid var(--line)}}
figure{{margin:24px 0}}img{{width:100%;height:auto}}figcaption{{font-size:14px;margin:12px 0}}
.scroll{{overflow:auto;max-width:100%}}table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{text-align:right;padding:14px 12px;border-bottom:1px solid var(--line);white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
th{{font-weight:400}}thead{{color:var(--muted)}}caption{{text-align:left;color:var(--muted);padding:0 0 12px}}
a{{color:var(--primary)}}a:focus-visible,summary:focus-visible,.scroll:focus-visible{{outline:2px solid var(--primary);outline-offset:4px}}
details{{margin-top:24px}}summary{{cursor:pointer;padding:12px 0;min-height:48px}}code{{font-family:'Roboto Mono',monospace;font-size:12px;overflow-wrap:anywhere}}
footer{{margin-top:40px;font-size:14px;color:var(--muted)}}
@media(max-width:640px){{main{{padding:24px 16px}}.answer{{grid-template-columns:1fr;gap:16px}}header{{padding-top:0}}th,td{{padding:12px 8px}}}}
@media print{{:root{{color-scheme:light;--bg:white;--surface:#F5F5F5;--text:#222;--muted:#555;--line:#AAA;--primary:#6200EE}}main{{padding:0}}section{{break-inside:avoid}}.scroll{{overflow:visible}}}}
</style></head><body><main>
<header><div class="eyebrow">DNN Forward Lab / Research report / v{escape(m["package_version"])}</div>
<span class="status">{escape(m["data_status"].upper())}</span>
<h1>{headline}</h1><p class="muted">{escape(m["asset"])} · {folds[0]["test_start"]} to {folds[-1]["test_end"]} · {len(returns)} evaluated daily bars · {len(folds)} expanding folds</p>
<div class="answer"><div><div>Return difference</div><div class="metric">{delta:+.2f}<small> pp</small></div><div class="muted">DNN minus historical selection, net of costs</div></div>
<div class="context"><p>{synthetic_note}</p><p>Strategy choices were frozen before each observed test window. {m["config"]["horizon"]}-bar recursive forecasts use no future prices.</p>
<p class="muted">This sample does not establish statistical significance, an investable edge, or reproduction of the paper's reported results. {repaired_count}/{len(returns)} forecast candles required geometry or positivity repairs; this is a model reliability diagnostic.</p></div></div></header>
<section><h2>01 / Observed results</h2><div class="scroll" tabindex="0" role="region" aria-label="Performance comparison table"><table>
<caption>Same observed windows, costs and capital. Ratios assume zero risk-free return.</caption>
<thead><tr><th>Method</th><th>Net return</th><th>Max drawdown</th><th>Sharpe*</th><th>Sortino*</th><th>Trades</th></tr></thead><tbody>{rows}</tbody></table></div>
<figure><picture><source media="(max-width:640px)" srcset="{mobile_img}"><img src="{img}" alt="Equity and drawdown across the observed evaluation bars; exact values are available in daily_returns.csv."></picture>
<figcaption>Equity compounds across folds. Each fold liquidates at its last close, including buy and hold. Max drawdown includes starting capital and is not annualized. *Sharpe and Sortino use {m["config"]["periods_per_year"]} periods/year; short-sample annualization is unstable.</figcaption></figure></section>
<section><h2>02 / Decisions at each cutoff</h2><div class="scroll" tabindex="0" role="region" aria-label="Fold decisions table"><table>
<caption>Repairs count projected forecast candles with invalid prices or OHLC geometry.</caption>
<thead><tr><th>Fold</th><th>History through</th><th>DNN choice</th><th>Historical choice</th><th>DNN return</th><th>Historical return</th><th>Repairs</th></tr></thead><tbody>{fold_rows}</tbody></table></div>
<p class="muted">Candidates, ranking scores, training curves, model arrays, forecasts and order ledgers are saved per fold. Selection uses net return with an alphabetical tie break.</p></section>
<section><h2>03 / Forecast diagnostics</h2><div class="scroll" tabindex="0" role="region" aria-label="Forecast errors table"><table>
<caption>Close-price errors over each full recursive forecast horizon; MAE/RMSE in input price units. All OHLC scores are in summary.json.</caption>
<thead><tr><th>Fold</th><th>Forecast model</th><th>MAE</th><th>RMSE</th><th>MAPE</th></tr></thead><tbody>{error_rows}</tbody></table></div></section>
<section><h2>04 / Method, provenance and limits</h2>
<p>Four independent feed-forward networks use {m["config"]["lookback"]} lagged prices, two ReLU hidden layers, Adam and L1 loss. Scaling uses only fitting data. The trailing {m["config"]["validation_size"]} historical bars select epochs by recursive validation MAE; those bars do not update weights.</p>
<p>Long/cash, fractional shares, no leverage. A signal at close executes at the next open. Commission: {m["config"]["commission_bps"]} bps per side. Slippage: {m["config"]["slippage_bps"]} bps per side. Terminal liquidation is scheduled at each fold's last close. Taxes, financing, volume limits and cash interest are excluded.</p>
<p>Source: {escape(m["source"])}<br>Adjustment: {escape(m["adjustment"])}<br>Unused trailing input rows: {m["unused_tail_rows"]}</p>
<p>Input fingerprint (canonical CSV SHA-256):<br><code>{m["data_sha256"]}</code></p>
<p>Inspired by <a href="https://arxiv.org/abs/2210.11532">Letteri et al. (2022)</a>. Exact reproduction is unavailable: original code was inaccessible during development and the paper leaves execution and tuning details unresolved. The strategy universe is an explicit reconstruction. Forecast projection and walk-forward evaluation are implementation extensions.</p>
<details><summary>Experiment configuration</summary><div class="scroll"><table><caption>Exact settings for this run</caption><tbody>{config_rows}</tbody></table></div></details></section>
<footer>Standalone report. No external scripts, fonts, trackers or network requests. Original charts are generated from the exported experiment data.</footer>
</main></body></html>"""
    path = output / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
