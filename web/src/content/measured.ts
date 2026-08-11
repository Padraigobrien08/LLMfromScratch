/**
 * Measured results, generated from `results/*.json`. Do not hand-edit.
 *
 *     llmfs-export-web
 *
 * Every figure here was produced by a run whose artifact is committed in `results/`,
 * and `tests/test_web_export.py` asserts this file is still what the generator emits.
 * A page that imports from here cannot quote a number the repository does not hold;
 * a page that retypes one can, which is why nothing on the site should retype one.
 */
export const MEASURED = {
  "tests": {
    "python": 334,
    "browser": 84
  },
  "reproduction": {
    "split": "val",
    "step": 19000,
    "loss": 3.050325984352098,
    "targetLoss": 3.29,
    "perplexity": 21.1222288164529,
    "tokensEvaluated": 99999744,
    "hellaswag": {
      "accNorm": 0.3043218482374029,
      "acc": 0.28908583947420835,
      "chance": 0.25,
      "reference": 0.2955,
      "nEvaluated": 10042
    },
    "gpu": "NVIDIA H100 80GB HBM3"
  },
  "ablations": {
    "arms": 12,
    "seeds": 3,
    "runs": 39,
    "gpuHours": 7.555965511642603,
    "baselineLoss": 3.911562124888102,
    "noiseFloor": 0.004298686981201172,
    "baselineSeeds": 3
  },
  "scaling": {
    "label": "5090x8",
    "config": "gpt2-124m",
    "steps": 50,
    "warmup": 15,
    "hasNvlink": false,
    "interconnect": "PCIe",
    "points": [
      {
        "worldSize": 1,
        "gradAccum": 32,
        "tokensPerSec": 185927.67103368638,
        "tokensPerSecPerGpu": 185927.67103368638,
        "stepTimeMs": 2819.849229999818,
        "efficiency": 1.0,
        "lossFirst": 10.951740264892578,
        "maxLossDeltaVs1Gpu": null
      },
      {
        "worldSize": 2,
        "gradAccum": 16,
        "tokensPerSec": 366468.9325698284,
        "tokensPerSecPerGpu": 183234.4662849142,
        "stepTimeMs": 1430.6478759972379,
        "efficiency": 0.9855147717722758,
        "lossFirst": 10.951740264892578,
        "maxLossDeltaVs1Gpu": 1.621246337890625e-05
      },
      {
        "worldSize": 4,
        "gradAccum": 8,
        "tokensPerSec": 721904.0356655323,
        "tokensPerSecPerGpu": 180476.00891638309,
        "stepTimeMs": 726.2571949977428,
        "efficiency": 0.970678586533171,
        "lossFirst": 10.951740264892578,
        "maxLossDeltaVs1Gpu": 1.430511474609375e-05
      },
      {
        "worldSize": 8,
        "gradAccum": 4,
        "tokensPerSec": 1414340.4433039958,
        "tokensPerSecPerGpu": 176792.55541299947,
        "stepTimeMs": 370.6943420038442,
        "efficiency": 0.9508673691769537,
        "lossFirst": 10.951739311218262,
        "maxLossDeltaVs1Gpu": 4.38690185546875e-05
      }
    ]
  },
  "accumulation": {
    "fit": {
      "a": 1.9752077117223488,
      "b": 11.134219740847406,
      "fittedFrom": [
        8,
        4
      ]
    },
    "points": [
      {
        "accum": 8,
        "tokensPerStep": 1048576,
        "tokensPerSec": 1440266.7111526092,
        "efficiency": 0.9663301482067173,
        "predicted": false,
        "predictedEfficiency": 0.9663301482067171
      },
      {
        "accum": 4,
        "tokensPerStep": 524288,
        "tokensPerSec": 1410959.562322512,
        "efficiency": 0.952412373530658,
        "predicted": false,
        "predictedEfficiency": 0.952412373530658
      },
      {
        "accum": 2,
        "tokensPerStep": 262144,
        "tokensPerSec": 1363110.9580351762,
        "efficiency": 0.9205856802799002,
        "predicted": true,
        "predictedEfficiency": 0.9245768241785395
      },
      {
        "accum": 1,
        "tokensPerStep": 131072,
        "tokensPerSec": 1270772.1431248619,
        "efficiency": 0.8604086654437935,
        "predicted": true,
        "predictedEfficiency": 0.8689057254743026
      }
    ]
  },
  "quantization": {
    "device": "cuda",
    "schemes": [
      {
        "name": "fp32 baseline",
        "bits": null,
        "groupSize": null,
        "memoryMib": 474.837890625,
        "compression": 1.0,
        "perplexity": 19.090686066879346,
        "deltaPerplexity": 0.0,
        "decodeTokS": 191.7555400059163
      },
      {
        "name": "int8 per-tensor",
        "bits": 8,
        "groupSize": -1,
        "memoryMib": 232.470703125,
        "compression": 2.042570888468809,
        "perplexity": 19.10554525985343,
        "deltaPerplexity": 0.014859192974082447,
        "decodeTokS": 142.47704301415823
      },
      {
        "name": "int8 g128",
        "bits": 8,
        "groupSize": 128,
        "memoryMib": 236.900390625,
        "compression": 2.004377828893671,
        "perplexity": 19.10318688530482,
        "deltaPerplexity": 0.01250081842547246,
        "decodeTokS": 139.77704520153742
      },
      {
        "name": "int4 per-tensor",
        "bits": 4,
        "groupSize": -1,
        "memoryMib": 191.970703125,
        "compression": 2.4734914385129567,
        "perplexity": 22.664289419087474,
        "deltaPerplexity": 3.573603352208128,
        "decodeTokS": 94.39607296818704
      },
      {
        "name": "int4 g128",
        "bits": 4,
        "groupSize": 128,
        "memoryMib": 196.400390625,
        "compression": 2.41770339210597,
        "perplexity": 20.442163359084685,
        "deltaPerplexity": 1.3514772922053382,
        "decodeTokS": 94.084451007825
      },
      {
        "name": "int4 g32",
        "bits": 4,
        "groupSize": 32,
        "memoryMib": 211.587890625,
        "compression": 2.2441638281963945,
        "perplexity": 20.29736203603998,
        "deltaPerplexity": 1.206675969160635,
        "decodeTokS": 91.04388889105338
      }
    ]
  },
  "speculative": {
    "device": "cuda",
    "losslessRuns": 18,
    "divergedRuns": 0,
    "best": {
      "prompt": "code-ish",
      "drafter": "prompt-lookup",
      "k": 8,
      "speedup": 5.345114694714658,
      "acceptanceRate": 0.9739130434782609,
      "tokensPerTargetForward": 7.529411764705882
    },
    "rows": [
      {
        "prompt": "prose",
        "drafter": "prompt-lookup",
        "k": 2,
        "tokensPerSec": 353.64064744131935,
        "speedup": 1.612437039451491,
        "acceptanceRate": 0.8625,
        "tokensPerTargetForward": 2.1333333333333333
      },
      {
        "prompt": "prose",
        "drafter": "prompt-lookup",
        "k": 4,
        "tokensPerSec": 507.7859010739356,
        "speedup": 2.3152677751465935,
        "acceptanceRate": 0.7592592592592593,
        "tokensPerTargetForward": 2.723404255319149
      },
      {
        "prompt": "prose",
        "drafter": "prompt-lookup",
        "k": 8,
        "tokensPerSec": 599.7550144094596,
        "speedup": 2.7346042001324062,
        "acceptanceRate": 0.6666666666666666,
        "tokensPerTargetForward": 3.282051282051282
      },
      {
        "prompt": "prose",
        "drafter": "model-draft",
        "k": 2,
        "tokensPerSec": 166.9612809079342,
        "speedup": 0.7612658653298324,
        "acceptanceRate": 0.6944444444444444,
        "tokensPerTargetForward": 2.3703703703703702
      },
      {
        "prompt": "prose",
        "drafter": "model-draft",
        "k": 4,
        "tokensPerSec": 134.97881607238313,
        "speedup": 0.6154406857671395,
        "acceptanceRate": 0.5088757396449705,
        "tokensPerTargetForward": 2.9767441860465116
      },
      {
        "prompt": "prose",
        "drafter": "model-draft",
        "k": 8,
        "tokensPerSec": 101.08199765449257,
        "speedup": 0.4608869433395597,
        "acceptanceRate": 0.3612167300380228,
        "tokensPerTargetForward": 3.764705882352941
      },
      {
        "prompt": "repetitive",
        "drafter": "prompt-lookup",
        "k": 2,
        "tokensPerSec": 347.76473104625086,
        "speedup": 1.5380670692315304,
        "acceptanceRate": 0.7558139534883721,
        "tokensPerTargetForward": 2.0
      },
      {
        "prompt": "repetitive",
        "drafter": "prompt-lookup",
        "k": 4,
        "tokensPerSec": 540.180822513119,
        "speedup": 2.389070139568966,
        "acceptanceRate": 1.0,
        "tokensPerTargetForward": 2.9767441860465116
      },
      {
        "prompt": "repetitive",
        "drafter": "prompt-lookup",
        "k": 8,
        "tokensPerSec": 537.3494988677811,
        "speedup": 2.3765479794058884,
        "acceptanceRate": 0.6615384615384615,
        "tokensPerTargetForward": 2.9767441860465116
      },
      {
        "prompt": "repetitive",
        "drafter": "model-draft",
        "k": 2,
        "tokensPerSec": 212.62685403468427,
        "speedup": 0.9403896744824115,
        "acceptanceRate": 0.9883720930232558,
        "tokensPerTargetForward": 2.9767441860465116
      },
      {
        "prompt": "repetitive",
        "drafter": "model-draft",
        "k": 4,
        "tokensPerSec": 150.31602711268988,
        "speedup": 0.6648061480650674,
        "acceptanceRate": 0.5806451612903226,
        "tokensPerTargetForward": 3.282051282051282
      },
      {
        "prompt": "repetitive",
        "drafter": "model-draft",
        "k": 8,
        "tokensPerSec": 158.57852784208032,
        "speedup": 0.7013488999512177,
        "acceptanceRate": 0.6257309941520468,
        "tokensPerTargetForward": 5.818181818181818
      },
      {
        "prompt": "code-ish",
        "drafter": "prompt-lookup",
        "k": 2,
        "tokensPerSec": 395.3625358105224,
        "speedup": 1.7698473847639584,
        "acceptanceRate": 0.8928571428571429,
        "tokensPerTargetForward": 2.3703703703703702
      },
      {
        "prompt": "code-ish",
        "drafter": "prompt-lookup",
        "k": 4,
        "tokensPerSec": 385.89869597052734,
        "speedup": 1.7274823383229687,
        "acceptanceRate": 0.528169014084507,
        "tokensPerTargetForward": 2.3703703703703702
      },
      {
        "prompt": "code-ish",
        "drafter": "prompt-lookup",
        "k": 8,
        "tokensPerSec": 1194.034083442915,
        "speedup": 5.345114694714658,
        "acceptanceRate": 0.9739130434782609,
        "tokensPerTargetForward": 7.529411764705882
      },
      {
        "prompt": "code-ish",
        "drafter": "model-draft",
        "k": 2,
        "tokensPerSec": 207.9224640940272,
        "speedup": 0.9307685882681911,
        "acceptanceRate": 0.9438202247191011,
        "tokensPerTargetForward": 2.8444444444444446
      },
      {
        "prompt": "code-ish",
        "drafter": "model-draft",
        "k": 4,
        "tokensPerSec": 195.15957060476597,
        "speedup": 0.8736352698123033,
        "acceptanceRate": 0.8389830508474576,
        "tokensPerTargetForward": 4.266666666666667
      },
      {
        "prompt": "code-ish",
        "drafter": "model-draft",
        "k": 8,
        "tokensPerSec": 242.06806805295705,
        "speedup": 1.0836219883608598,
        "acceptanceRate": 0.957983193277311,
        "tokensPerTargetForward": 8.533333333333333
      }
    ]
  }
} as const;
