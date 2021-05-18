import mmcv
import numpy as np
import json

from .builder import DATASETS
from .base_dataset import BaseDataset

@DATASETS.register_module()
class MyDataset(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = json.load(f)
            all_imgs = samples["images"]
            for img in all_imgs:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': img["file_name"]}
                info['gt_label'] = np.array(img["category_id"], dtype=np.int64)
                data_infos.append(info)
            return data_infos