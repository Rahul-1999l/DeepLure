# Suspicious Design Groups Audit Report

This file documents inferred design groups that display weak SIFT keypoint connectivity or potential false-positive matching caused by repeated generic border textures or background features.

| Design ID | Image Count | Min Pairwise SIFT Matches | Suspicion Rationale | Recommended Action |
| :---: | :---: | :---: | :--- | :--- |
| **Design_001** | 8 | 17 | Contains weak pairwise SIFT match of 17 (close to threshold 35). Affected: `img_284975.jpg`, `img_37489.jpg`, `img_386972.jpg`, `img_519993.jpg`, `img_524137.jpg`, `img_602896.jpg`, `img_987571.jpg`, `img_987632.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_004** | 5 | 0 | Contains weak pairwise SIFT match of 0 (close to threshold 35). Affected: `h_img_35259.jpg`, `img_489839.jpg`, `img_694883.jpg`, `img_789127.jpg`, `img_90832.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_006** | 5 | 3 | Contains weak pairwise SIFT match of 3 (close to threshold 35). Affected: `h_img_76108.jpg`, `img_270633.jpg`, `img_667809.jpg`, `img_881252.jpg`, `img_915919.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_007** | 5 | 0 | Contains weak pairwise SIFT match of 0 (close to threshold 35). Affected: `h_img_93057.jpg`, `img_277936.jpg`, `img_289910.jpg`, `img_475101.jpg`, `img_578936.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_008** | 5 | 6 | Contains weak pairwise SIFT match of 6 (close to threshold 35). Affected: `img_10959.jpg`, `img_275183.jpg`, `img_40145.jpg`, `img_556574.jpg`, `img_7150.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_009** | 5 | 9 | Contains weak pairwise SIFT match of 9 (close to threshold 35). Affected: `img_110820.jpg`, `img_385960.jpg`, `img_718326.jpg`, `img_874118.jpg`, `img_917853.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_010** | 5 | 31 | Contains weak pairwise SIFT match of 31 (close to threshold 35). Affected: `img_115196.jpg`, `img_153688.jpg`, `img_246391.jpg`, `img_442389.jpg`, `img_741740.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_011** | 5 | 18 | Contains weak pairwise SIFT match of 18 (close to threshold 35). Affected: `img_175884.jpg`, `img_339968.jpg`, `img_764354.jpg`, `img_961476.jpg`, `img_978752.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_012** | 5 | 10 | Contains weak pairwise SIFT match of 10 (close to threshold 35). Affected: `img_221213.jpg`, `img_239226.jpg`, `img_364174.jpg`, `img_67123.jpg`, `img_927947.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_013** | 5 | 3 | Contains weak pairwise SIFT match of 3 (close to threshold 35). Affected: `img_260919.jpg`, `img_429922.jpg`, `img_565213.jpg`, `img_927463.jpg`, `img_955789.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_016** | 5 | 23 | Contains weak pairwise SIFT match of 23 (close to threshold 35). Affected: `img_364907.jpg`, `img_638094.jpg`, `img_828414.jpg`, `img_85121.jpg`, `img_892336.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_018** | 4 | 2 | Contains weak pairwise SIFT match of 2 (close to threshold 35). Affected: `h_img_31665.jpg`, `img_120749.jpg`, `img_48825.jpg`, `img_678178.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_019** | 4 | 4 | Contains weak pairwise SIFT match of 4 (close to threshold 35). Affected: `img_130255.jpg`, `img_307484.jpg`, `img_325344.jpg`, `img_478598.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_020** | 4 | 17 | Contains weak pairwise SIFT match of 17 (close to threshold 35). Affected: `img_48234.jpg`, `img_526657.jpg`, `img_5447.jpg`, `img_897203.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_021** | 3 | 0 | Contains weak pairwise SIFT match of 0 (close to threshold 35). Affected: `h_img_149526.jpg`, `img_411083.jpg`, `img_738703.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_022** | 3 | 0 | Contains weak pairwise SIFT match of 0 (close to threshold 35). Affected: `h_img_593526.jpg`, `img_257108.jpg`, `img_845893.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_023** | 3 | 17 | Contains weak pairwise SIFT match of 17 (close to threshold 35). Affected: `img_246854.jpg`, `img_2469.jpg`, `img_804542.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_024** | 3 | 5 | Contains weak pairwise SIFT match of 5 (close to threshold 35). Affected: `img_314642.jpg`, `img_55880.jpg`, `img_649782.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_025** | 3 | 1 | Contains weak pairwise SIFT match of 1 (close to threshold 35). Affected: `img_614731.jpg`, `img_635738.jpg`, `img_68235.jpg` | Manual visual verification or split if colorway motifs differ. |
| **Design_026** | 2 | 39 | Contains weak pairwise SIFT match of 39 (close to threshold 35). Affected: `img_180214.jpg`, `img_822729.jpg` | Manual visual verification or split if colorway motifs differ. |

## Singleton Near-Match Audit

The following 44 singleton images were checked against all 165 dataset images to verify whether any structural partners exist below the initial threshold (35 SIFT matches):

| Singleton Image | Highest Match Partner | SIFT Keypoint Matches | Audit Conclusion |
| :--- | :--- | :---: | :--- |
| `img_142355.jpg` | `h_img_31665.jpg` | 27 | Near match below threshold; true singleton or weak variant. |
| `img_259893.jpg` | `img_770334.jpg` | 27 | Near match below threshold; true singleton or weak variant. |
| `img_316470.jpg` | `img_366604.jpg` | 30 | Near match below threshold; true singleton or weak variant. |
| `img_32852.jpg` | `img_246854.jpg` | 28 | Near match below threshold; true singleton or weak variant. |
| `img_366604.jpg` | `img_316470.jpg` | 30 | Near match below threshold; true singleton or weak variant. |
| `img_583723.jpg` | `img_649782.jpg` | 26 | Near match below threshold; true singleton or weak variant. |
| `img_586052.jpg` | `img_386972.jpg` | 30 | Near match below threshold; true singleton or weak variant. |
| `img_586537.jpg` | `img_649782.jpg` | 28 | Near match below threshold; true singleton or weak variant. |
| `img_770334.jpg` | `img_259893.jpg` | 27 | Near match below threshold; true singleton or weak variant. |
| `img_776505.jpg` | `img_822729.jpg` | 28 | Near match below threshold; true singleton or weak variant. |
