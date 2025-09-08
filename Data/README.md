

## Data

All related resources are available at the following link:

- **Link:** https://pan.baidu.com/s/1y1R6UFfzVm1-2l4e0ya1_w
- **Extraction code:** `py95`
   *(Shared via Baidu Netdisk)*

If you know a better free way to share these resources, please feel free to send me an email.

### Contents of the link

- Two real-world subsets
- One synthetic dataset
- Pretrained checkpoints
- Other related resources

### Notes

- The environments used to train TPENet SWIN-B and TPENet SWIN-S were lost.
- The TPENet SWIN-L model was retrained on a rebuilt environment (see `requirements.txt` from the MS<sup>3</sup>PE repository).
- The datasets split files are also included in this repository.
- Several real images in the two real-world subsets have been updated compared to those used in the paper:
  - Some inappropriate boundaries were corrected.
  - Both the old and new versions are stored in the shared link.

### Updated Evaluation

We evaluated the updated real datasets using our models. The results for updated **OB-DIODE** and updated **OB-EntitySeg** are shown below:

#### OB-DIODE

| Method | ODS  | OIS  | AP   |
| ------ | ---- | ---- | ---- |
| SWIN-L |  86.4    |  87.8    |  91.4    |
| SWIN-B |  85.4    |  87.8    |  89.9    |
| SWIN-S |  84.1    |  86.9    |  89.4    |

#### OB-EntitySeg

| Method | ODS  | OIS  | AP   |
| ------ | ---- | ---- | ---- |
| SWIN-L |  79.4    |  81.1    |  84.9    |
| SWIN-B |  80.6    |  81.7    |  85.3    |
| SWIN-S |  81.2    |  82.3    |  86.4    | 


