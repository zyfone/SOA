# Sampling-Free Universal Domain Adaptive Object Detection with Synergistic Optimization Alignment

## Requirements

* Python >= 3.8
* PyTorch >= 2.1.0+cu118
* torchvision >= 0.16.0+cu118
* CUDA 11.8
* GCC >= 4.9 (Linux) or Visual Studio 2019+ (Windows)

Install dependencies:

```bash
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

pip install ninja yacs cython matplotlib tqdm opencv-python pycocotools
```

Check CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Build

```bash
python setup.py build develop
```

## Dataset

Dataset paths are configured in:

```text
maskrcnn_benchmark/config/paths_catalog.py
```

Modify the entries in `DatasetCatalog.DATASETS`, for example:

```python
"openset_voc07_0.5": {
    "data_dir": "D:/datasets/VOCdevkit/VOC2007/",
    "split": "trainval"
},
```

## Training

```bash
sh train_script/train_voc05.sh > voc05.out 
sh train_script/train_voc2water.sh > voc2water.out 
sh train_script/train_water2voc.sh > water2voc.out 
sh train_script/train_voc075.sh > voc075.out 
sh train_script/train_voc025.sh > voc025.out
sh train_script/train_city2foggy.sh > foggy.out
```

## Testing

```bash
python tools/test_net.py \
    --config-file configs/UniDAOD/da_rcnn_fpn_voc05.yaml \
    MODEL.WEIGHT output_dic/baseline/voc05/model_final.pkl
```
