# SJTU ECE6501G Principles of Medical Imaging

## Topic 5 Glioma Classification

### Baseline Binary Classification

We use the self-attention UNet. To train the model, please use  
`python train.py --config randn.yml`  
for a quick check.

To do binary classification, please use  
`python predict.py`  
to get masks of images.
