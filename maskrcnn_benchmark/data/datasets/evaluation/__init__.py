from maskrcnn_benchmark.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation

from .voc_openset import voc_evaluation_openset

def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )

    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    # elif isinstance(dataset, datasets.WaterColorDataset_test):
    #     return voc_evaluation(**args)
    # elif isinstance(dataset, datasets.Openset_Clipart_test):
    #     return voc_evaluation(**args)
    elif isinstance(dataset, datasets.WaterColorDataset_test):
        return voc_evaluation_openset(**args)
    elif isinstance(dataset, datasets.Openset_Clipart_test):
        return voc_evaluation_openset(**args)
    elif isinstance(dataset, datasets.Water2voc_test):
        return voc_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
