import streamlit as st
import torch
import tempfile
import os
import io
import json
import contextlib
from collections import Counter
from sklearn.model_selection import train_test_split
from functools import partial
from unittest.mock import patch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

try:
    from data_manager import loading_abstraction_extended, BinaryTargetDataset, BinaryTargetInterface, AnimalCategorizer
    from neural import OptimizedMLP
    from conf_matrix_prob import get_confusion_matrix_and_write_files
    from confidence_interval import confidence_interval
except ImportError as e:
    st.error(f"Missing local module dependency: {e}. Ensure the folder structure is correct.")

TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42

DEFAULT_DATA_PATH = 'Animals_with_Attributes2/ResNet101/AwA2-features.txt'
DEFAULT_LABEL_PATH = 'Animals_with_Attributes2/ResNet101/AwA2-labels.txt'
DEFAULT_BINARY_TARGET_PATH = 'Animals_with_Attributes2/predicate-matrix-binary.txt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="DeepEL UI", layout="wide")

STANDARD_EXPERIMENTS = {
    'is_terrestrial': {
        "query": "is_terrestrial",
        "mlps": "32, 42",
        "mapping": {
            "default": 1,
            "mappings": {"0": [2, 8, 16, 17, 23, 29, 35, 46, 49]}
        }
    },
    'is_aquatic': {
        "query": "is_aquatic",
        "mlps": "10, 11, 27, 43",
        "mapping": {
            "default": 0,
            "mappings": {
                "1": [2, 8, 13, 17, 23, 46, 49],
                "2": [3, 35, 44]
            }
        }
    },
    'is_ungulate': {
        "query": "is_ungulate",
        "mlps": "18, 22, 32, 35, 42",
        "mapping": {
            "default": 0,
            "mappings": {
                "1": [6, 22, 37, 41],
                "2": [0, 15, 20, 27, 30, 36, 39, 48]
            }
        }
    },
    'is_carnivore': {
        "query": "is_carnivore",
        "mlps": "33, 34, 37",
        "mapping": {
            "default": 0,
            "mappings": {"1": [1, 2, 7, 9, 12, 14, 21, 23, 29, 31, 33, 34, 35, 40, 42, 44, 45]}
        }
    },
    'predict (Big Experiment)': {
        "query": "predict",
        "mlps": "17, 18, 27, 32, 35, 42, 43",
        "mapping": {
            "default": 3,
            "mappings": {
                "0": [16, 19, 24],
                "1": [2, 3, 8, 13, 17, 23, 35, 44, 46, 49],
                "2": [0, 6, 15, 20, 22, 27, 30, 36, 37, 39, 41, 48]
            }
        }
    }
}


@st.cache_data
def generate_labels_from_config(label_indices, config):
    """Generates task labels dynamically based on the provided JSON mapping configuration."""
    mappings = config.get("mappings", {})
    default_val = config.get("default", 0)

    lookup = {}
    for target_class_str, orig_labels in mappings.items():
        target_class = int(target_class_str)
        for orig in orig_labels:
            lookup[orig] = target_class
            
    return [lookup.get(label, default_val) for label in label_indices]

@st.cache_resource(show_spinner="Loading AwA2 Dataset (This may take a minute)...")
def load_cached_data(data_path, label_path, binary_target_path, indices):
    data, label_indices, binary_vectors = loading_abstraction_extended(
        data_path, label_path, binary_target_path,
        attribute_indices_to_extract=indices
    )
    return data, label_indices, binary_vectors

st.title("Dashboard for DeepEL Neuro-Symbolic Experiments (AwA2 Dataset)")
st.markdown("Recreate existing DeepEL experiments or define entirely new queries, MLPs, and labeling logic.")

with st.sidebar:
    st.header("1. Setup Mode")
    mode = st.radio("Experiment Mode", ["Recreate Standard", "Custom / New"])
    
    if mode == "Recreate Standard":
        selected_standard = st.selectbox("Select Standard Experiment", list(STANDARD_EXPERIMENTS.keys()))
        config_data = STANDARD_EXPERIMENTS[selected_standard]
        
        query_name = st.text_input("Query Name", config_data['query'], disabled=True)
        mlp_ids_str = st.text_input("MLP IDs (comma-separated)", config_data['mlps'], disabled=True)
        mapping_json_str = st.text_area(
            "Label Mapping Rules (JSON)", 
            json.dumps(config_data['mapping'], indent=2), 
            height=200, disabled=True)
    else:
        query_name = st.text_input("Query Name (e.g. is_custom)", "is_custom")
        mlp_ids_str = st.text_input("MLP IDs (comma-separated)", "10, 15")
        
        default_custom_mapping = {
            "default": 0,
            "mappings": {
                "1": [1, 5, 10]}}
        mapping_json_str = st.text_area(
            "Target Label Mapping Rules (JSON)", 
            json.dumps(default_custom_mapping, indent=2), 
            height=200,
            help="Define how original AwA2 class IDs map to your target query labels. 'default' catches unmapped IDs.")

    st.header("2. Execution Settings")
    inference_mode = st.radio("Workflow", ["Pretrained Inference", "Train from Scratch"])
    
    train_epochs = 10
    if inference_mode == "Train from Scratch":
        train_epochs = st.slider("Training Epochs", min_value=1, max_value=50, value=10)
    
    st.header("3. Upload Knowledge Base")
    uploaded_kb = st.file_uploader("Upload .pl or .txt file", type=['pl', 'txt'])

    with st.expander("Override Data Paths"):
        DATA_PATH = st.text_input("Features Path", DEFAULT_DATA_PATH)
        LABEL_PATH = st.text_input("Labels Path", DEFAULT_LABEL_PATH)
        BIN_PATH = st.text_input("Binary Matrix Path", DEFAULT_BINARY_TARGET_PATH)

run_button = st.button("Run Experiment", type="primary")


if run_button:
    if not uploaded_kb:
        st.error("Please upload a Knowledge Base file first.")
        st.stop()
        
    try:
        mlp_indices = sorted([int(x.strip()) for x in mlp_ids_str.split(",") if x.strip().isdigit()])
        if not mlp_indices:
            st.error("Invalid MLP IDs provided.")
            st.stop()
            
        try:
            mapping_config = json.loads(mapping_json_str)
        except json.JSONDecodeError:
            st.error("Invalid JSON provided in Label Mapping Rules.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pl") as tmp_kb:
            tmp_kb.write(uploaded_kb.getvalue())
            kb_filepath = tmp_kb.name

        os.makedirs('experiments_log_demo', exist_ok=True)
        os.makedirs('snapshot_demo', exist_ok=True)
        actual_path = f'experiments_log_demo/{query_name}_actual.txt'
        predicted_path = f'experiments_log_demo/{query_name}_predicted.txt'

        with st.spinner("Executing Pipeline... Check the app logs if training takes long."):
            data, label_indices, binary_vectors = load_cached_data(DATA_PATH, LABEL_PATH, BIN_PATH, mlp_indices)

            labels = generate_labels_from_config(label_indices, mapping_config)

            original_sample_indices = list(range(len(label_indices)))
            train_indices, test_indices = train_test_split(
                original_sample_indices, test_size=TEST_SPLIT_RATIO,
                random_state=RANDOM_SEED, stratify=labels)

            dataset = BinaryTargetDataset(data, labels, binary_vectors)
            interface = BinaryTargetInterface(dataset)
            
            train_set = AnimalCategorizer(dataset, query_name, RANDOM_SEED, train_indices)
            test_set = AnimalCategorizer(dataset, query_name, RANDOM_SEED, test_indices)

            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                
                if inference_mode == "Pretrained Inference":
                    networks = {}
                    for i in mlp_indices:
                        network = OptimizedMLP()
                        network.load_state_dict(torch.load(f'pretrained_mlps/net{i}.pth', map_location=device))
                        net = Network(network, f"net{i}", batching=True)
                        net.optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
                        networks[f"net{i}"] = net
                    model = Model(kb_filepath, list(networks.values()))
                    model.set_engine(ExactEngine(model), cache=True)
                    model.add_tensor_source("dataset", interface)
                
                elif inference_mode == "Train from Scratch":
                    networks = {}
                    for i in mlp_indices:
                        network = OptimizedMLP()
                        net = Network(network, f"net{i}", batching=True)
                        net.optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
                        if torch.cuda.is_available():
                            net.cuda()
                        networks[f"net{i}"] = net
                    model = Model(kb_filepath, list(networks.values()))
                    model.set_engine(ExactEngine(model), cache=True)
                    model.add_tensor_source("dataset", interface)
                    print(f"Starting Training Loop for {train_epochs} Epochs")
                    loader = DataLoader(train_set, 24, shuffle=True)
                    with patch('signal.signal', return_value=None):
                        train_model(model, loader, train_epochs, log_iter=50, profile=0)
                    model.save_state(f"snapshot_demo/scratch_model_{query_name}.pth")

                    if not torch.cuda.is_available():
                        torch.load = partial(torch.load, map_location="cpu")
                        model.load_state(f"snapshot_demo/scratch_model_{query_name}.pth")

                print("\nRunning Evaluation")
                _, errors_indices = get_confusion_matrix_and_write_files(
                    model, test_set, actual_path, predicted_path, verbose=1)
                
                miss_labels = [label_indices[x] for x in errors_indices]
                print("\nError Label Distribution (Original AwA2 Classes)")
                print(Counter(miss_labels))
                
                print("\nConfidence Intervals")
                confidence_interval(actual_path, predicted_path)

            console_output = output_buffer.getvalue()

        st.success(f"Experiment for `{query_name}` completed successfully!")
        st.subheader("Console Output & Results")
        st.code(console_output, language='text')

        if os.path.exists(kb_filepath):
            os.remove(kb_filepath)

    except Exception as e:
        st.error(f"An error occurred during execution: {str(e)}")