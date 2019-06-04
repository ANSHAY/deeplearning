# The code for Fender apron defect vs healthy classification using tensorflow can be found here.
#### The dataset is divided into 2 parts - training, validation.  The stats for this dataset are -  
### Without data augmentation
> 'Nrows':150, 'Ncols':150, 'BATCH_SIZE':10, 'NUM_EPOCHS':100, 'FILTER_SIZE':(3,3)  

> Model: (c16, mp2, c32, mp2, c64, mp2, f, d256, d1)

| Data | Accuracy |
| --- | --- |
| Training | 55.72 |
| Validation | 55.10 |

---
> 'Nrows':300, 'Ncols':300, 'BATCH_SIZE':10, 'NUM_EPOCHS':100, 'FILTER_SIZE':(3,3)  

> Model: (c16, mp2, c32, mp2, c64, mp2, c64, mp2, f, d128, d1)

| Data | Accuracy |
| --- | ---|
| Training | 99.00 |
| Validation | 67.35 |
> Model saturated in 20 epochs. Started overfitting
---
---
### With Data augmentation
>rotation_range=40,width_shift_range=40,height_shift_range=40, zoom_range=1, horizontal_flip=True, vertical_flip=True

> 'Nrows':300, 'Ncols':300, 'BATCH_SIZE':10, 'NUM_EPOCHS':50, 'FILTER_SIZE':(3,3)  

> Model: (c16, mp2, c32, mp2, c64, mp2, c64, mp2, f, d128, d1)

| Data | Accuracy |
| --- | --- |
| Training | 55.72 |
| Validation | 55.1 |
> Model saturated in 10 epochs. Couldn't learn much
---

>rotation_range=40,width_shift_range=40,height_shift_range=40, zoom_range=1, horizontal_flip=True, vertical_flip=True

> 'Nrows':300, 'Ncols':300, 'BATCH_SIZE':10, 'NUM_EPOCHS':70, 'FILTER_SIZE':(5,5)  

> Model: (c16, mp2, c32, mp2, c64, mp2, c64, mp2, f, d256, d1)

| Data | Accuracy |
| --- | --- |
| Training | 55.72 |
| Validation | 55.1 |
> Model saturated in 10 epochs. Couldn't learn much
---

>rotation_range=40,width_shift_range=40,height_shift_range=40, zoom_range=1, horizontal_flip=True, vertical_flip=True

> 'Nrows':300, 'Ncols':300, 'BATCH_SIZE':20, 'NUM_EPOCHS':70, 'FILTER_SIZE':(5,5)  

> Model: (c16, mp2, c32, mp2, c64, mp2, f, d512, d1)

| Data | Accuracy |
| --- | --- |
| Training | 55.72 |
| Validation | 55.1 |
> Model saturated in 10 epochs. Couldn't learn much
