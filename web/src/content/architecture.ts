/**
 * The two shipped model configs, resolved through `llmfs.config.load_config`.
 *
 * Through the loader, so `_base_` inheritance and every default have already
 * been applied. A page that re-read the YAML would be reimplementing the loader;
 * a page that typed these in would be guessing at what the loader does.
 *
 * Do not hand-edit — regenerate with `llmfs-export-web`. `tests/test_web_export.py`
 * asserts this file is still what the generator emits, so a stale copy fails CI
 * rather than shipping.
 */
export const ARCHITECTURES = {
  "gpt2": {
    "config": {
      "vocabSize": 50304,
      "nLayer": 12,
      "nHead": 12,
      "nKvHead": 12,
      "nEmbd": 768,
      "blockSize": 1024,
      "norm": "layernorm",
      "posEmb": "learned",
      "mlp": "gelu",
      "tieEmbeddings": true,
      "bias": true,
      "mlpRatio": 4.0,
      "mlpHiddenMultipleOf": 256,
      "normEps": 1e-05,
      "ropeTheta": 10000.0
    },
    "headDim": 64,
    "mlpHidden": 3072,
    "source": "configs/gpt2-124m.yaml"
  },
  "llama": {
    "config": {
      "vocabSize": 50304,
      "nLayer": 12,
      "nHead": 12,
      "nKvHead": 4,
      "nEmbd": 768,
      "blockSize": 1024,
      "norm": "rmsnorm",
      "posEmb": "rope",
      "mlp": "swiglu",
      "tieEmbeddings": true,
      "bias": false,
      "mlpRatio": 4.0,
      "mlpHiddenMultipleOf": 256,
      "normEps": 1e-05,
      "ropeTheta": 10000.0
    },
    "headDim": 64,
    "mlpHidden": 2048,
    "source": "configs/llama-124m.yaml"
  }
} as const;
