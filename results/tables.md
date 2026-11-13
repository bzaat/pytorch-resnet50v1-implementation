## Per-seed results

| Experiment | Seed | Val best epoch / last | Val acc at best (%) | Test acc at best-val (%) | Test loss at best-val | Test acc at last epoch (%) | Test loss at last | Train acc at last (augmented, %) | Train time (min) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A | 42 | 41 / 50 | 38.72 | 30.74 | 3.1512 | 31.40 | 3.5155 | 72.25 | 28.67 |
| A | 43 | 48 / 50 | 35.33 | 29.38 | 3.7715 | 29.87 | 3.6746 | 76.70 | 30.50 |
| A | 44 | 48 / 50 | 38.72 | 32.27 | 3.5487 | 32.57 | 3.7862 | 72.08 | 31.02 |
| B | 42 | 10 / 10 | 95.52 | 92.70 | 0.2906 | 92.70 | 0.2906 | 99.35 | 3.09 |
| B | 43 | 9 / 10 | 94.70 | 92.23 | 0.3342 | 92.26 | 0.3305 | 99.46 | 3.09 |
| B | 44 | 9 / 10 | 94.57 | 92.23 | 0.3002 | 92.40 | 0.3012 | 99.59 | 3.19 |
| C | 42 | 4 / 10 | 100.00 | 99.62 | 0.0116 | 99.56 | 0.0152 | 99.93 | 2.91 |
| C | 43 | 7 / 10 | 99.86 | 99.73 | 0.0116 | 99.62 | 0.0132 | 99.93 | 2.88 |
| C | 44 | 3 / 10 | 99.86 | 99.59 | 0.0208 | 99.65 | 0.0127 | 100.00 | 2.89 |

## Aggregate (mean ± sample std, n = number of seeds)

| Experiment | n | Test acc, best-val checkpoint (%) | Test acc, last epoch (%) | Val acc at best (%) | Train acc at last (augmented, %) | Train time (min) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| A. ResNet-50 from scratch, 37 breeds, 50 epochs | 3 | 30.80 ± 1.45 | 31.28 ± 1.35 | 37.59 ± 1.96 | 73.68 ± 2.62 | 30.06 ± 1.23 |
| B. Pretrained ResNet-50 (IMAGENET1K_V2), 37 breeds, 10 epochs | 3 | 92.39 ± 0.27 | 92.45 ± 0.22 | 94.93 ± 0.51 | 99.47 ± 0.12 | 3.12 ± 0.06 |
| C. Pretrained ResNet-50 (IMAGENET1K_V2), cat vs dog, 10 epochs | 3 | 99.65 ± 0.07 | 99.61 ± 0.04 | 99.91 ± 0.08 | 99.95 ± 0.04 | 2.89 ± 0.01 |
