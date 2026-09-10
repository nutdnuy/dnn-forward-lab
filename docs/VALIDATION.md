# Validation evidence

Validation date: 2026-09-10. These checks verify software behavior, not an investment edge.

## Local checks

| Check | Result |
|---|---|
| Unit/integration suite, including optional ARIMA | 67 passed; no skipped tests |
| Combined line/branch coverage | 91.76% (85% gate) |
| Ruff lint and formatting | Passed |
| Wheel and source distribution build | Passed |
| Clean wheel installation outside source checkout | Complete synthetic experiment and packaged-font report passed |
| Browser: 1440 × 1100 and 390 × 844 | No page overflow or page errors; charts loaded |
| Report network requests | Zero external requests |
| Keyboard configuration disclosure | Passed at both viewport sizes |
| Independent output audit | Input/source hashes, return compounding, 30-bar forecasts and all order timestamps reconciled |

Browser checks used local headless Chromium. Tables intentionally scroll within their own regions on narrow screens. Mobile uses a separate chart layout generated from the same returns. Semantic headings, table captions, text disclosures, line styles and keyboard focus were inspected; this is not a claim of a complete accessibility certification.

## Environment

Python 3.12.12, macOS ARM64, CPU inference. Locked runtime dependencies for this sample:

- numpy: 2.4.6
- pandas: 2.3.3
- torch: 2.14.0
- scikit-learn: 1.9.0
- matplotlib: 3.11.1

Linux GitHub Actions is configured for Python 3.11, 3.12 and 3.13 with CPU-only PyTorch. Check the repository's Actions tab for the latest remote status.

## Sample provenance

- Dataset: seeded synthetic OHLC generator, seed 42; 450 weekday rows.
- Test: 90 bars over three non-overlapping 30-bar folds.
- Input SHA-256: `7ab3e3bd6f3e58ada4669345447426b1b2c9fdc085eaeaa8434679e8c27e9fab`.
- Source SHA-256 (sorted package Python filenames and bytes): `0fa556c7742ed2c64cfa247f8c83d25af4928c29388c22058e0385b2a0823b56`.
- Configuration: `configs/demo.toml`; full resolved settings in the sample manifest.
- Projection diagnostic: 60 of 90 forecast candles needed positivity/geometry repair. This substantial repair count and weak forecast errors in some folds limit model credibility, regardless of the synthetic return comparison.

No original ANF/EOG snapshot was available, no paper result was claimed as reproduced, and no broker orders were placed. Performance uncertainty, survivorship bias, corporate-action accuracy and actual market execution remain outside these software checks.
