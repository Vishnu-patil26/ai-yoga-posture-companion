# Angle deviation of wrong poses

Source: 'Yoga for all' (Zenodo 7818789, CC BY 4.0). For every labelled wrong
folder, each joint angle is compared with the matching *right* step.
`deviation` = median(wrong) - median(right), in degrees (stance in torso
lengths). `z` = deviation in robust standard deviations of the right step.
`AUC` = chance that a wrong image has the larger value (0.5 = no signal).
Only the three strongest features per fault are shown.

**Limits.** Each pose in this dataset has one or two volunteers, so these are
within-person deviations; the tables show *which* angles betray *which*
fault, not population-wide thresholds (see docs/COLLECTION_PROTOCOL.md).

## bhujangasana

| Wrong step | Fault | n | Feature | Right | Wrong | Deviation | z | AUC |
|---|---|---|---|---|---|---|---|---|
| 1 | Head | 12 | stance width | 0.152 | 0.908 | +0.8 | +16.5 | 1.00 |
|  | |  | neck alignment | 14.044 | 2.733 | -11.3 | -0.7 | 0.14 |
|  | |  | straighter elbow | 174.2 | 165.726 | -8.5 | -1.3 | 0.26 |
| 1 | Legs | 11 | stance width | 0.152 | 0.847 | +0.7 | +15.2 | 0.96 |
|  | |  | straighter knee | 171.287 | 168.207 | -3.1 | -0.4 | 0.38 |
|  | |  | lower arm | 119.246 | 112.212 | -7.0 | -0.2 | 0.40 |
| 2 | Hand and Legs | 12 | stance width | 0.207 | 0.565 | +0.4 | +6.5 | 0.95 |
|  | |  | higher arm | 149.735 | 162.482 | +12.7 | +1.0 | 0.72 |
|  | |  | straighter knee | 174.596 | 170.215 | -4.4 | -1.1 | 0.29 |
| 2 | Legs | 12 | neck alignment | 7.197 | 32.134 | +24.9 | +4.0 | 1.00 |
|  | |  | straighter elbow | 78.82 | 147.828 | +69.0 | +1.5 | 0.90 |
|  | |  | bent knee | 164.046 | 160.172 | -3.9 | -0.5 | 0.26 |
| 3 | Head | 12 | stance width | 0.152 | 0.918 | +0.8 | +16.7 | 1.00 |
|  | |  | lower arm | 119.246 | 97.828 | -21.4 | -0.7 | 0.24 |
|  | |  | higher arm | 96.542 | 79.785 | -16.8 | -0.8 | 0.31 |
| 3 | Legs | 11 | stance width | 0.152 | 0.854 | +0.7 | +15.3 | 1.00 |
|  | |  | lower arm | 119.246 | 157.356 | +38.1 | +1.3 | 0.70 |
|  | |  | straighter elbow | 174.2 | 171.029 | -3.2 | -0.5 | 0.31 |

## marjaryasana

| Wrong step | Fault | n | Feature | Right | Wrong | Deviation | z | AUC |
|---|---|---|---|---|---|---|---|---|
| 1 | Hand and Neck | 12 | higher arm | 170.109 | 145.464 | -24.6 | -4.2 | 0.13 |
|  | |  | lower arm | 175.135 | 160.276 | -14.9 | -3.0 | 0.19 |
|  | |  | straighter elbow | 167.448 | 132.817 | -34.6 | -3.2 | 0.21 |
| 2 | Legs | 12 | higher arm | 169.261 | 161.183 | -8.1 | -1.8 | 0.30 |
|  | |  | lower arm | 172.945 | 168.947 | -4.0 | -0.9 | 0.33 |
|  | |  | bent elbow | 153.295 | 137.148 | -16.1 | -0.8 | 0.35 |
| 3 | Legs | 10 | lower arm | 177.823 | 171.172 | -6.7 | -2.5 | 0.18 |
|  | |  | higher arm | 172.267 | 162.664 | -9.6 | -2.0 | 0.19 |
|  | |  | stance width | 0.568 | 0.268 | -0.3 | -1.1 | 0.21 |
| 4 | Hand and Legs | 11 | lower arm | 175.38 | 168.659 | -6.7 | -1.5 | 0.20 |
|  | |  | bent elbow | 152.777 | 131.472 | -21.3 | -1.2 | 0.24 |
|  | |  | higher arm | 171.524 | 164.983 | -6.5 | -1.3 | 0.25 |

## tadasana

| Wrong step | Fault | n | Feature | Right | Wrong | Deviation | z | AUC |
|---|---|---|---|---|---|---|---|---|
| 1 | Legs | 12 | stance width | 0.139 | 0.503 | +0.4 | +37.8 | 1.00 |
|  | |  | bent knee | 170.509 | 172.785 | +2.3 | +0.6 | 0.75 |
|  | |  | lower arm | 176.591 | 172.762 | -3.8 | -1.6 | 0.31 |
| 2 | Legs and Hand | 12 | stance width | 0.144 | 0.385 | +0.2 | +11.7 | 0.97 |
|  | |  | straighter elbow | 164.483 | 137.028 | -27.5 | -1.4 | 0.31 |
|  | |  | bent elbow | 143.947 | 85.688 | -58.3 | -2.5 | 0.35 |
| 3 | Legs and Hand | 12 | straighter elbow | 175.933 | 152.252 | -23.7 | -6.5 | 0.08 |
|  | |  | bent elbow | 169.542 | 143.533 | -26.0 | -3.6 | 0.12 |
|  | |  | stance width | 0.152 | 0.258 | +0.1 | +2.1 | 0.88 |
| 4 | Hands and Legs | 12 | stance width | 0.144 | 0.385 | +0.2 | +11.7 | 0.97 |
|  | |  | straighter elbow | 164.483 | 137.028 | -27.5 | -1.4 | 0.31 |
|  | |  | bent elbow | 143.947 | 85.688 | -58.3 | -2.5 | 0.35 |
| 5 | Standing | 12 | stance width | 0.139 | 0.503 | +0.4 | +37.8 | 1.00 |
|  | |  | bent knee | 170.509 | 172.785 | +2.3 | +0.6 | 0.75 |
|  | |  | lower arm | 176.591 | 172.762 | -3.8 | -1.6 | 0.31 |
