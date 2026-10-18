/* tslint:disable */
/* eslint-disable */

/**
 * One damped-least-squares IK step toward (tx, ty), with the Jacobian from
 * two dual passes through `arm_tip`. Returns [q1, q2, error, elbow x, elbow y, tip x, tip y].
 */
export function arm_ik_step(q1: number, q2: number, tx: number, ty: number): Float64Array;

/**
 * Generate SVG path `d` attribute for f(x) = x * sin(x).
 * Maps math coordinates to SVG pixel coordinates.
 */
export function curve_svg_path(x_min: number, x_max: number, y_min: number, y_max: number, w: number, h: number, steps: number): string;

/**
 * Evaluate f(x) = x * sin(x) using Dual<f64>.
 * Returns [value, derivative].
 */
export function dual_eval(x: number): Float64Array;

/**
 * Parse `src` in x, y, z, optionally append its gradient, and emit a kernel.
 * Returns the source, or a line starting with `error:`.
 */
export function expr_kernel(src: string, dialect: string, grad: boolean): string;

/**
 * The live part of a demo's graph, as JSON:
 * `{"nodes":[{"op":"×","args":[0,1],"lit":false}],"outputs":[5,9]}` with
 * indices into `nodes`, which are in topological order.
 */
export function graph_layout(kind: string): string;

/**
 * Every live node's value for these inputs, in `graph_layout` order.
 */
export function graph_values(kind: string, inputs: Float64Array): Float64Array;

/**
 * Score a launch and its gradient, one dual pass per parameter.
 * Returns [miss², ∂/∂speed, ∂/∂heading, path xy...].
 */
export function orbit_eval(speed: number, heading: number, tx: number, ty: number): Float64Array;

/**
 * Generate text from a seed string with temperature sampling.
 */
export function poet_generate(seed: string, temperature: number): string;

/**
 * Initialize the Quantum Poet model and dataset.
 */
export function poet_init(): void;

/**
 * Run one training epoch. Returns average loss for the epoch.
 */
export function poet_train_epoch(): number;

/**
 * Compute tangent line endpoints in SVG coordinates.
 * Returns [x1, y1, x2, y2].
 */
export function tangent_svg(x: number, x_min: number, x_max: number, y_min: number, y_max: number, w: number, h: number, half_len: number): Float64Array;

/**
 * Convert math coordinates to SVG pixel coordinates.
 * Returns [sx, sy].
 */
export function to_svg(x: number, y: number, x_min: number, x_max: number, y_min: number, y_max: number, w: number, h: number): Float64Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly arm_ik_step: (a: number, b: number, c: number, d: number) => [number, number];
    readonly curve_svg_path: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly dual_eval: (a: number) => [number, number];
    readonly orbit_eval: (a: number, b: number, c: number, d: number) => [number, number];
    readonly poet_generate: (a: number, b: number, c: number) => [number, number];
    readonly poet_init: () => void;
    readonly poet_train_epoch: () => number;
    readonly tangent_svg: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly to_svg: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly graph_layout: (a: number, b: number) => [number, number];
    readonly graph_values: (a: number, b: number, c: number, d: number) => [number, number];
    readonly expr_kernel: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
