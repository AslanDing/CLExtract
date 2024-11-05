# This code is for the work of CLExtract 

<p align="center">
<img src="./pictures/framework0.PNG" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall framework of our work.
</p>

## Requirements
numpy, scikit-learn, torch, tqdm, matplotlib

## Data
Because of the sensibility of the satellite communication data, the data is not concluded in this repository. If you want to get more information, please contact the corresponding author. Thanks for understanding.


## pipeline of this work
- get the data(clear headers)
- preprocessing the data(preprocessing.py)
- CLExtract pretrain(./src/CLExtract_pretrain.py)
- CLExtract finetune(./src/CLExtract_finetune.py)
