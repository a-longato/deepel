import torch
from torch.utils.data import Dataset as TorchDataset
import random
from problog.logic import Term, Constant
from deepproblog.dataset import Dataset
from deepproblog.query import Query

def loading_abstraction_extended(
    data_path,
    label_path,
    binary_target_path,
    instance_length=2048,
    dtype=torch.float32,
    target_labels_to_extract=None,
    attribute_indices_to_extract=None,
):
    with open(data_path, 'r') as f:
        raw_data = f.read().replace('\n', ' ')
    numbers = [float(n) for n in raw_data.strip().split() if n]
    data_tensor = torch.tensor(numbers, dtype=dtype)
    num_total_instances = len(data_tensor) // instance_length
    data_tensor = data_tensor.view(num_total_instances, instance_length)

    with open(label_path, 'r') as f:
        original_label_list = [int(line.strip()) for line in f if line.strip()]

    with open(binary_target_path, 'r') as f:
        all_binary_vectors_full = [
            [int(bit) for bit in line.split()]
            for line in f if line.strip()
        ]

    if target_labels_to_extract is not None:
        target_label_to_new_idx = {label_val: i for i, label_val in enumerate(target_labels_to_extract)}
        
        instance_indices_to_keep = []
        final_label_indices_for_instances = []
        for i, original_label_val in enumerate(original_label_list):
            if original_label_val in target_label_to_new_idx:
                instance_indices_to_keep.append(i)
                final_label_indices_for_instances.append(target_label_to_new_idx[original_label_val])
        
        final_data_instances = data_tensor[instance_indices_to_keep]
        selected_class_binary_vectors = [all_binary_vectors_full[label_val - 1] for label_val in target_labels_to_extract]
    else:
        final_data_instances = data_tensor
        final_label_indices_for_instances = [x - 1 for x in original_label_list]
        selected_class_binary_vectors = all_binary_vectors_full

    if attribute_indices_to_extract is not None:
        final_filtered_binary_vectors = [
            [bv[i] for i in attribute_indices_to_extract] for bv in selected_class_binary_vectors
        ]
    else:
        final_filtered_binary_vectors = selected_class_binary_vectors
            
    return final_data_instances, final_label_indices_for_instances, final_filtered_binary_vectors


class BinaryTargetDataset(Dataset):
    def __init__(self, data, labels, binary_attributes):
        """
        Args:
            data (torch.Tensor): Tensor of shape (num_samples, instance_feature_length)
            labels (list): List of 0-based indices, where labels[i] is an index into binary_attributes
                           for the i-th sample in `data`.
            binary_attributes (list of lists): Ground truth binary vectors for each CLASS.
                                               Shape: (num_classes, num_attributes_per_class).
                                               It's assumed these are the *active* attributes.
        """
        self.data = data
        self.label_indices = labels
        self.binary_vectors = binary_attributes

        self.num_samples = data.shape[0]
        self.instance_feature_length = data.shape[1]
        self.num_attributes = len(binary_attributes[0])
        self.total_attribute_instances = self.num_samples * self.num_attributes

    def __len__(self):
        return self.total_attribute_instances

    def __getitem__(self, idx):
        sample_idx = idx // self.num_attributes
        attr_idx = idx % self.num_attributes

        x = self.data[sample_idx]
        class_idx_for_sample = self.label_indices[sample_idx]
        binary_value = self.binary_vectors[class_idx_for_sample][attr_idx]
        
        return x, class_idx_for_sample, sample_idx, attr_idx, binary_value

    def to_query(self, idx):
        sample_idx = idx // self.num_attributes
        attr_idx = idx % self.num_attributes

        class_idx_for_sample = self.label_indices[sample_idx]
        binary_value = self.binary_vectors[class_idx_for_sample][attr_idx]

        term_name = f"t{attr_idx}"
        return Query(
            Term(term_name, Term("tensor", Term("dataset", Constant(sample_idx))), Constant(binary_value))
        )


class BinaryTargetInterface:
    def __init__(self, dataset: BinaryTargetDataset):
        self.dataset = dataset

    def _extract_sample_idx(self, key_object):
        """
        Helper to extract sample_idx from various Pyswip/Problog term structures
        that DeepProbLog might use as keys for a tensor source.
        """
        if isinstance(key_object, tuple):
            return self._extract_sample_idx(key_object[0])
        elif isinstance(key_object, Term) and key_object.functor == 'tensor':
            return int(key_object.args[0].args[0].value)
        elif isinstance(key_object, Term) and key_object.functor == 'dataset':
            return int(key_object.args[0].value)
        elif isinstance(key_object, Constant):
            return int(key_object.value)
        elif isinstance(key_object, int):
            return key_object

    def __getitem__(self, item_from_dpl):
        key_to_parse = item_from_dpl[0] if isinstance(item_from_dpl, list) else item_from_dpl
        sample_idx = self._extract_sample_idx(key_to_parse)
        return self.dataset.data[sample_idx]
    

class AnimalCategorizer(Dataset, TorchDataset):
    def __init__(self, dataset_name, function_name: str, seed=None, sample_indices=None):
        self.dataset = dataset_name
        self.function_name = function_name
        self.seed = seed
        self.num_attributes = self.dataset.num_attributes
        self.labels = self.dataset.label_indices
        
        if sample_indices is not None:
            self.sample_indices = sample_indices
        else:
            self.sample_indices = list(range(self.dataset.num_samples))
        
        if seed is not None:
            random.Random(seed).shuffle(self.sample_indices)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index: int):
        sample_idx = self.sample_indices[index]
        input_tensor, label, *_ = self.dataset[sample_idx * self.num_attributes]
        return input_tensor, label
    
    def to_query(self, index: int) -> Query:
        sample_idx = self.sample_indices[index]
        label = self.labels[sample_idx]
        subs = {Term("a"): Term("tensor", Term("dataset", Constant(sample_idx)))}
        return Query(Term(self.function_name, Term("a"), Constant(label)), subs)