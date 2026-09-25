import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList


class Openset_Clipart_test(torch.utils.data.Dataset):

    CLASSES_075 = ('__background__',  # always index 0, 
                    #source and target domains common classes
                    "boat","bottle","bus","car","cat","chair","cow","diningtable",
                    "dog","horse","motorbike","person","pottedplant","sheep","sofa",
                    # uda source domain private
                    "train", "tvmonitor") 
    
    CLASSES_075_common = ('__background__',  # always index 0, 
                    #source and target domains common classes
                    "boat","bottle","bus","car","cat","chair","cow","diningtable",
                    "dog","horse","motorbike","person","pottedplant","sheep","sofa") 
    #source domain private class: ‘train’and‘tvmonitor’

    CLASSES_05 = ('__background__',  # always index 0
                # source and target domains common classes
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                # uda source domain private
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                )

    CLASSES_05_common = ('__background__',  # always index 0
                # source and target domains common classes
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person')
    
    #source domain private class:  ‘aeroplane’,‘bicycle’,‘bird’,‘boat’, and ‘bottle’.

    CLASSES_025 =('__background__',  # always index 0  
                    #source and target domains common classes 
                    "aeroplane","bicycle","bird","boat","bottle",
                    # uda source domain private
                    "bus","car","cat","chair","cow","diningtable","dog") 
    
    CLASSES_025_common =('__background__',  # always index 0  
                    #source and target domains common classes 
                    "aeroplane","bicycle","bird","boat","bottle") 
    

    def __init__(self, data_dir, split, use_difficult=False, transforms=None,openset=None,is_source=True):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = True #use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        if openset=="ALL":
            cls = Openset_Clipart_test.CLASSES
        elif openset=="0.75":
            cls = Openset_Clipart_test.CLASSES_075
            print("common class: {}".format(Openset_Clipart_test.CLASSES_075))
            self.common_class=Openset_Clipart_test.CLASSES_075_common
        elif openset=="0.5":
            cls = Openset_Clipart_test.CLASSES_05
            print("common class: {}".format(Openset_Clipart_test.CLASSES_05))
            self.common_class=Openset_Clipart_test.CLASSES_05_common
        elif openset=="0.25":
            cls = Openset_Clipart_test.CLASSES_025
            print("common class: {}".format(Openset_Clipart_test.CLASSES_025))
            self.common_class=Openset_Clipart_test.CLASSES_025_common
        
        # cls=self.common_class
        self.cls=cls
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        

        print("\n")
        print("##################################################")
        print("Dataset Openset_Clipart_test_"+openset,":",len(self.ids))
        #add new code to remove empty bboxe s
        self.ids=self.remove_empty_bbox()
        # get_bbox_len after
        print("Filtration Openset_Clipart_test_"+openset,":",len(self.ids))
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        print(data_dir,split,openset)
        print("class",cls)
        print("\n")


    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])

        target.add_field("difficult", anno["difficult"])
        return target
    
    def get_groundtruth_openset(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        
        index_ignore=[index for index,i in enumerate(anno["labels"]) if self.cls[i] in self.common_class]
        # print(index_ignore)
        # print(anno["labels"])
        # print("common: {}".format(self.common_class))
        target = BoxList(anno["boxes"][index_ignore], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"][index_ignore])
        target.add_field("difficult", anno["difficult"][index_ignore])
    
        return target
    
    def remove_empty_bbox(self):
        #add new code to remove empty bboxe 
        print(" remove empty bboxe ......")
        new_ids_list=[]
        for i,id_new in enumerate(self.ids):
            img_id = self.ids[i]
            anno = ET.parse(self._annopath % img_id).getroot()
            anno = self._preprocess_annotation(anno)
            if len(anno["boxes"])>0:
                new_ids_list.append(id_new)
        return new_ids_list

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()

            if not name in self.class_to_ind.keys():
                continue

            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text, 
                bb.find("ymin").text, 
                bb.find("xmax").text, 
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    # def map_class_id_to_class_name(self, class_id):
    #     return Openset_Clipart_test.CLASSES[class_id]
    def map_class_id_to_class_name(self, class_id):
        return self.cls[class_id]
        # try:
        #     return self.cls[class_id]
        # except:
        #     print("############################")
        #     print("ERROR  class")
        #     print(class_id)
        #     print(self.cls)
        #     print("\n")
        #     return "None"