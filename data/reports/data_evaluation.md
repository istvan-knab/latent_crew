# Data evaluation

Created 2026-10-08T16:36:19, git commit `31fe608`, Python 3.14.8, numpy 2.5.3, Pillow 12.3.0.  
Manifest `manifests/datasets.json` SHA-256 `145493b4387a7f806139c500e88d62bbecff33ce05e39eaeb47d418f76c39a01`.  
Samples checked: all. Runtime 643.1 s. Full details: `data_evaluation.json`; flagged samples: `problematic_samples.jsonl`.

## Summary

| dataset | samples | train | test | flagged samples | errors | warnings |
|---|---|---|---|---|---|---|
| sunrgbd | 10335 | 5285 | 5050 | 652 | 0 | 652 |
| nyu_depth_v2 | 1449 | 795 | 654 | 58 | 0 | 58 |

## SUN RGB-D (sunrgbd)

Version V1. Downloads 7.0 GiB, processed 3.1 GiB in 51677 files. Train ratio 0.511.

### Files

| field | format | dtype | resolutions | missing | unreadable |
|---|---|---|---|---|---|
| rgb | JPEG | uint8 | 730x530 (3784), 561x427 (2003), 591x441 (3389), 681x531 (1159) | 0 | 0 |
| depth | PNG | uint16 | 730x530 (3784), 561x427 (2003), 591x441 (3389), 681x531 (1159) | 0 | 0 |
| depth_raw | PNG | uint16 | 730x530 (3784), 561x427 (2003), 591x441 (3389), 681x531 (1159) | 0 | 0 |
| semantic | PNG | uint8 | 730x530 (3784), 561x427 (2003), 591x441 (3389), 681x531 (1159) | 0 | 0 |
| instance | PNG | uint16 | 730x530 (3784), 561x427 (2003), 591x441 (3389), 681x531 (1159) | 0 | 0 |

Index: 0 duplicate ids, 0 duplicate paths, 0 unknown splits; 0 incomplete samples.

### RGB (0-1 scale)

| channel | mean | std | min | p1 | median | p99 | max |
|---|---|---|---|---|---|---|---|
| R | 0.4963 | 0.2778 | 0.0 | 0.0 | 0.4941 | 1.0 | 1.0 |
| G | 0.4667 | 0.2862 | 0.0 | 0.0 | 0.4471 | 1.0 | 1.0 |
| B | 0.4408 | 0.2911 | 0.0 | 0.0 | 0.4078 | 1.0 | 1.0 |

### Depth (mm, valid pixels)

| field | min | p1 | median | mean | std | p99 | max | invalid pixels | valid per sample (mean) |
|---|---|---|---|---|---|---|---|---|---|
| depth | 351.0 | 684.0 | 2085.0 | 2398.3 | 1248.9 | 6993.0 | 10000.0 | 0.0 % | 100.0 % |
| depth_raw | 351.0 | 734.0 | 2038.0 | 2306.3 | 1122.5 | 6375.0 | 10000.0 | 17.4 % | 84.5 % |

### Labels

37 classes (37 seen), unlabelled pixels 25.0 %, 5.8 classes and 12.6 instances per image (mean). 12 classes cover 90 % of labelled pixels; 3 rare classes.

| class | share of labelled pixels |
|---|---|
| wall | 26.6 % |
| floor | 23.7 % |
| chair | 11.4 % |
| table | 8.1 % |
| bed | 3.6 % |
| sofa | 3.1 % |
| desk | 3.0 % |
| window | 3.0 % |
| cabinet | 2.8 % |
| door | 2.6 % |

| boxes | images | boxes | per image (mean) | invalid | distinct names |
|---|---|---|---|---|---|
| 2D | 10335 | 80220 | 7.762 | 0 | 1068 |
| 3D | 10335 | 64657 | 6.2561 | 8 | 1143 |

### Consistency

| sensor | train | test |
|---|---|---|
| kinect_v2 | 1924 | 1860 |
| xtion | 1701 | 1688 |
| kinect_v1 | 1073 | 930 |
| realsense | 587 | 572 |

| scene | train | test |
|---|---|---|
| bedroom | 558 | 526 |
| office | 547 | 499 |
| classroom | 511 | 512 |
| furniture_store | 585 | 380 |
| rest_space | 492 | 432 |
| bathroom | 331 | 293 |
| living_room | 274 | 250 |
| kitchen | 232 | 266 |
| dining_area | 195 | 202 |
| library | 162 | 219 |
| corridor | 222 | 151 |
| conference_room | 99 | 191 |
| idk | 152 | 124 |
| lab | 40 | 218 |
| discussion_area | 97 | 104 |
| dining_room | 108 | 92 |
| study_space | 72 | 120 |
| computer_room | 114 | 65 |
| lecture_theatre | 133 | 43 |
| home_office | 73 | 96 |
| office_kitchen | 52 | 30 |
| printer_room | 43 | 34 |
| recreation_room | 24 | 36 |
| bookstore | 25 | 11 |
| playroom | 17 | 14 |
| office_dining | 3 | 26 |
| reception_room | 15 | 12 |
| study | 14 | 12 |
| basement | 8 | 18 |
| mail_room | 5 | 17 |
| gym | 6 | 15 |
| laundromat | 6 | 12 |
| reception | 12 | 5 |
| cafeteria | 9 | 4 |
| storage_room | 6 | 5 |
| dinette | 8 | 2 |
| lobby | 6 | 3 |
| stairs | 4 | 5 |
| home | 5 | 1 |
| hotel_room | 6 | 0 |
| exhibition | 6 | 0 |
| coffee_room | 4 | 1 |
| music_room | 0 | 3 |
| dancing_room | 2 | 1 |
| indoor_balcony | 2 | 0 |

Only in train: hotel_room, exhibition, indoor_balcony. Only in test: music_room.

Best RGB-depth shift (px), samples per sensor:

| sensor | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| kinect_v1 | 225 | 317 | 314 | 351 | 274 | 180 | 328 |
| kinect_v2 | 2697 | 536 | 308 | 149 | 39 | 20 | 26 |
| realsense | 241 | 254 | 156 | 138 | 120 | 94 | 153 |
| xtion | 298 | 578 | 864 | 712 | 425 | 181 | 257 |

Duplicates: 0 exact RGB copies; near duplicates 0 within train, 0 within test, 0 across train/test.

Overlap with NYU Depth V2: 1449 samples, split differs for 0.

### Flagged samples

| check | issue | severity | samples | examples |
|---|---|---|---|---|
| rgb | dark | warning | 34 | sun_001921, sun_002010, sun_002012 |
| depth | low_valid | warning | 372 | sun_000013, sun_000134, sun_000135 |
| labels | unlabeled | warning | 3 | sun_005407, sun_008010, sun_008104 |
| labels | bad_box_3d | warning | 8 | sun_002094, sun_007334, sun_007446 |
| alignment | misaligned | warning | 238 | sun_001055, sun_001155, sun_001363 |

## NYU Depth V2 (nyu_depth_v2)

Version labeled. Downloads 2.8 GiB, processed 618.6 MiB in 7247 files. Train ratio 0.549.

### Files

| field | format | dtype | resolutions | missing | unreadable |
|---|---|---|---|---|---|
| rgb | JPEG | uint8 | 640x480 (1449) | 0 | 0 |
| depth | PNG | uint16 | 640x480 (1449) | 0 | 0 |
| depth_raw | PNG | uint16 | 640x480 (1449) | 0 | 0 |
| semantic | PNG | uint16 | 640x480 (1449) | 0 | 0 |
| instance | PNG | uint8 | 640x480 (1449) | 0 | 0 |

Index: 0 duplicate ids, 0 duplicate paths, 0 unknown splits; 0 incomplete samples.

### RGB (0-1 scale)

| channel | mean | std | min | p1 | median | p99 | max |
|---|---|---|---|---|---|---|---|
| R | 0.4803 | 0.289 | 0.0 | 0.0039 | 0.4588 | 1.0 | 1.0 |
| G | 0.4109 | 0.2956 | 0.0 | 0.0039 | 0.3686 | 1.0 | 1.0 |
| B | 0.392 | 0.3085 | 0.0 | 0.0 | 0.3373 | 1.0 | 1.0 |

### Depth (mm, valid pixels)

| field | min | p1 | median | mean | std | p99 | max | invalid pixels | valid per sample (mean) |
|---|---|---|---|---|---|---|---|---|---|
| depth | 713.0 | 826.0 | 2506.0 | 2796.3 | 1386.1 | 7555.0 | 9995.0 | 0.0 % | 100.0 % |
| depth_raw | 713.0 | 863.0 | 2456.0 | 2727.0 | 1318.3 | 7243.0 | 10000.0 | 31.4 % | 68.6 % |

### Labels

894 classes (894 seen), unlabelled pixels 17.4 %, 14.1 classes and 24.2 instances per image (mean). 39 classes cover 90 % of labelled pixels; 834 rare classes.

| class | share of labelled pixels |
|---|---|
| wall | 25.9 % |
| floor | 11.0 % |
| cabinet | 7.5 % |
| bed | 4.6 % |
| chair | 4.0 % |
| sofa | 3.2 % |
| door | 2.6 % |
| table | 2.6 % |
| window | 2.6 % |
| picture | 2.5 % |

### Consistency

| sensor | train | test |
|---|---|---|
| kinect_v1 | 795 | 654 |

| scene | train | test |
|---|---|---|
| bedroom | 192 | 191 |
| kitchen | 119 | 106 |
| living_room | 114 | 107 |
| bathroom | 63 | 58 |
| dining_room | 62 | 55 |
| office | 40 | 38 |
| home_office | 26 | 24 |
| classroom | 26 | 23 |
| bookstore | 25 | 11 |
| playroom | 17 | 14 |
| furniture_store | 27 | 0 |
| study | 14 | 11 |
| reception_room | 12 | 5 |
| office_kitchen | 6 | 4 |
| study_room | 5 | 2 |
| basement | 7 | 0 |
| computer_lab | 3 | 3 |
| cafe | 5 | 0 |
| conference_room | 5 | 0 |
| home_storage | 5 | 0 |
| student_lounge | 5 | 0 |
| dinette | 4 | 0 |
| foyer | 2 | 2 |
| excercise_room | 3 | 0 |
| laundry_room | 3 | 0 |
| printer_room | 3 | 0 |
| indoor_balcony | 2 | 0 |

Only in train: furniture_store, basement, cafe, conference_room, home_storage, student_lounge, dinette, excercise_room, laundry_room, printer_room, indoor_balcony. Only in test: -.

Best RGB-depth shift (px), samples per sensor:

| sensor | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| kinect_v1 | 1412 | 6 | 8 | 4 | 5 | 4 | 9 |

Duplicates: 0 exact RGB copies; near duplicates 0 within train, 0 within test, 0 across train/test.

### Flagged samples

| check | issue | severity | samples | examples |
|---|---|---|---|---|
| rgb | dark | warning | 2 | nyu_000386, nyu_000644 |
| depth | low_valid | warning | 56 | nyu_000014, nyu_000117, nyu_000121 |
