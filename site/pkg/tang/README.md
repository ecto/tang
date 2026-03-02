# tang-site-wasm

WASM bindings for the [tang.dev](https://phyz.dev) interactive demo. Exposes dual-number autodiff to JavaScript for live visualization of functions and their tangent lines.

## What it does

Evaluates f(x) = x sin(x) using `Dual<f64>` and returns both the value and exact derivative — then generates SVG paths for plotting.

## Exports

| Function | Input | Output |
|----------|-------|--------|
| `dual_eval(x)` | f64 | `[f(x), f'(x)]` |
| `curve_svg_path(...)` | coordinate bounds | SVG path string |
| `tangent_svg(x, ...)` | x + coordinate bounds | tangent line endpoints |
| `to_svg(...)` | math coords | SVG pixel coords |

## Build

```bash
wasm-pack build crates/tang-site-wasm --target web
```

Optimized for code size (`opt-level = "s"`, LTO enabled).

## License

[MIT](../../LICENSE)
