/* tslint:disable */
/* eslint-disable */
/**
*/
export function start(): void;

export type Algorithm = "lloyd" | "hamerly" | "lloyd-assignment-cube" | "lloyd-assignment-gpu";
export type Initializer = "kmeans++" | "random";


/**
*/
export class ColorCruncher {
  free(): void;
/**
* @param {number} max_colors
* @param {number} sample_rate
*/
  constructor(max_colors: number, sample_rate: number);
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
* @returns {ColorCruncherBuilder}
*/
  static _new(): ColorCruncherBuilder;
/**
* @param {number} max_colors
* @returns {ColorCruncherBuilder}
*/
  withMaxColors(max_colors: number): ColorCruncherBuilder;
/**
* @param {number} max_colors
*/
  setMaxColors(max_colors: number): void;
/**
* @param {number} sample_rate
* @returns {ColorCruncherBuilder}
*/
  withSampleRate(sample_rate: number): ColorCruncherBuilder;
/**
* @param {number} sample_rate
*/
  setSampleRate(sample_rate: number): void;
/**
* @param {number} tolerance
* @returns {ColorCruncherBuilder}
*/
  withTolerance(tolerance: number): ColorCruncherBuilder;
/**
* @param {number} tolerance
*/
  setTolerance(tolerance: number): void;
/**
* @param {number} max_iterations
* @returns {ColorCruncherBuilder}
*/
  withMaxIterations(max_iterations: number): ColorCruncherBuilder;
/**
* @param {number} max_iterations
*/
  setMaxIterations(max_iterations: number): void;
/**
* @param {string} initializer
* @returns {ColorCruncherBuilder}
*/
  withInitializer(initializer: string): ColorCruncherBuilder;
/**
* @param {string} initializer
*/
  setInitializer(initializer: string): void;
/**
* @param {string} algorithm
* @returns {ColorCruncherBuilder}
*/
  withAlgorithm(algorithm: string): ColorCruncherBuilder;
/**
* @param {string} algorithm
*/
  setAlgorithm(algorithm: string): void;
/**
* @param {bigint} seed
* @returns {ColorCruncherBuilder}
*/
  withSeed(seed: bigint): ColorCruncherBuilder;
/**
* @param {bigint} seed
*/
  setSeed(seed: bigint): void;
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
  readonly colorcruncherbuilder__new: () => number;
  readonly colorcruncherbuilder_withMaxColors: (a: number, b: number) => number;
  readonly colorcruncherbuilder_setMaxColors: (a: number, b: number) => void;
  readonly colorcruncherbuilder_withSampleRate: (a: number, b: number) => number;
  readonly colorcruncherbuilder_setSampleRate: (a: number, b: number) => void;
  readonly colorcruncherbuilder_withTolerance: (a: number, b: number) => number;
  readonly colorcruncherbuilder_setTolerance: (a: number, b: number) => void;
  readonly colorcruncherbuilder_withMaxIterations: (a: number, b: number) => number;
  readonly colorcruncherbuilder_setMaxIterations: (a: number, b: number) => void;
  readonly colorcruncherbuilder_withInitializer: (a: number, b: number, c: number) => number;
  readonly colorcruncherbuilder_setInitializer: (a: number, b: number, c: number) => void;
  readonly colorcruncherbuilder_withAlgorithm: (a: number, b: number, c: number) => number;
  readonly colorcruncherbuilder_setAlgorithm: (a: number, b: number, c: number) => void;
  readonly colorcruncherbuilder_withSeed: (a: number, b: number) => number;
  readonly colorcruncherbuilder_setSeed: (a: number, b: number) => void;
  readonly colorcruncherbuilder_build: (a: number) => number;
  readonly colorcruncher_builder: (a: number, b: number) => number;
  readonly colorcruncher_quantizeImage: (a: number, b: number, c: number) => number;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly _dyn_core__ops__function__FnMut__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__h09cf73a106909c04: (a: number, b: number, c: number) => void;
  readonly _dyn_core__ops__function__FnMut__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__h70355f2cee0091f7: (a: number, b: number, c: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly wasm_bindgen__convert__closures__invoke2_mut__h54e00a299b7bd240: (a: number, b: number, c: number, d: number) => void;
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
