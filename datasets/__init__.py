from typing import Dict, Type
from datasets.base_dataset import BaseDataset
from datasets.mean import MeanDataset
from datasets.regression import RegressionDataset


dataset_name_to_DatasetClass: Dict[str, Type[BaseDataset]] = {
    "Mean Dataset": MeanDataset,
    "Regression Dataset": RegressionDataset,
}
