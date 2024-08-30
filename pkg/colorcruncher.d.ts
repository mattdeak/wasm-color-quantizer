/* tslint:disable */
/* eslint-disable */
/**
*/
export function start(): void;

export type Algorithm = "lloyd" | "hamerly" | "lloyd-gpu"
export type Initializer = "kmeans++" | "random";


/**
*/
export class ColorCruncher {
  free(): void;
/**
* @param {Uint8Array} data
* @returns {Promise<Uint8Array>}
*/
  quantizeImage(data: Uint8Array): Promise<Uint8Array>;
}
/**
*/
export class ColorCruncherBuilder {
  free(): void;
/**
*/
  constructor();
/**
* @param {number} max_colors
* @returns {ColorCruncherBuilder}
*/
  withMaxColors(max_colors: number): ColorCruncherBuilder;
/**
* @param {number} sample_rate
* @returns {ColorCruncherBuilder}
*/
  withSampleRate(sample_rate: number): ColorCruncherBuilder;
/**
* @param {number} tolerance
* @returns {ColorCruncherBuilder}
*/
  withTolerance(tolerance: number): ColorCruncherBuilder;
/**
* @param {number} max_iterations
* @returns {ColorCruncherBuilder}
*/
  withMaxIterations(max_iterations: number): ColorCruncherBuilder;
/**
* @param {string} initializer
* @returns {ColorCruncherBuilder}
*/
  withInitializer(initializer: string): ColorCruncherBuilder;
/**
* @param {string} algorithm
* @returns {ColorCruncherBuilder}
*/
  withAlgorithm(algorithm: string): ColorCruncherBuilder;
/**
* @param {bigint} seed
* @returns {ColorCruncherBuilder}
*/
  withSeed(seed: bigint): ColorCruncherBuilder;
/**
* @returns {Promise<ColorCruncher>}
*/
  build(): Promise<ColorCruncher>;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly start: () => void;
  readonly __wbg_colorcruncher_free: (a: number) => void;
  readonly __wbg_colorcruncherbuilder_free: (a: number) => void;
  readonly colorcruncherbuilder_new: () => number;
  readonly colorcruncherbuilder_withMaxColors: (a: number, b: number) => number;
  readonly colorcruncherbuilder_withSampleRate: (a: number, b: number) => number;
  readonly colorcruncherbuilder_withTolerance: (a: number, b: number) => number;
  readonly colorcruncherbuilder_withMaxIterations: (a: number, b: number) => number;
  readonly colorcruncherbuilder_withInitializer: (a: number, b: number, c: number) => number;
  readonly colorcruncherbuilder_withAlgorithm: (a: number, b: number, c: number) => number;
  readonly colorcruncherbuilder_withSeed: (a: number, b: number) => number;
  readonly colorcruncherbuilder_build: (a: number) => number;
  readonly colorcruncher_quantizeImage: (a: number, b: number, c: number) => number;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly _dyn_core__ops__function__FnMut__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__h6e435f3bc6e3d654: (a: number, b: number, c: number) => void;
  readonly _dyn_core__ops__function__FnMut__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__h870dc3eb3e49f78a: (a: number, b: number, c: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly wasm_bindgen__convert__closures__invoke2_mut__h849a5a7e5a0f14b2: (a: number, b: number, c: number, d: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
