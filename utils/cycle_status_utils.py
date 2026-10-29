import yaml
import os
import sys

try:
    import torch_xla
    _USE_XLA = True
    from google.cloud import storage
    # bucket = storage.Client().bucket('generative_active_learning')
    GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "generative_active_learning")
    bucket = storage.Client().bucket(GCS_BUCKET_NAME)

except ImportError:
    _USE_XLA = False

class CycleStatusUtils:
    @staticmethod
    def create_cycle_status(cycle_dir):
        file_path = f"{cycle_dir}/cycle_status.yaml"
        """Create a default cycle status YAML file."""
        default_status = {
            'training_phase_done': False,
            'inference_phase_done': False,
            'finalize_phase_done': False
        }
        if not _USE_XLA:
            with open(file_path, 'w') as file:
                yaml.safe_dump(default_status, file)
        else:
            blob = bucket.blob(file_path)
            blob.upload_from_string(yaml.safe_dump(default_status))
            

    @staticmethod
    def load_cycle_status(cycle_dir):
        file_path = f"{cycle_dir}/cycle_status.yaml"
        """Load cycle status from a YAML file."""
        if not _USE_XLA:
            if not os.path.exists(file_path):
                CycleStatusUtils.create_cycle_status(cycle_dir)
            assert os.path.exists(file_path), f"Cycle status file not found at {file_path}"

            with open(file_path, 'r') as file:
                status_data = yaml.safe_load(file)
            return status_data
        else:
            blob = bucket.blob(file_path)
            if not blob.exists():
                CycleStatusUtils.create_cycle_status(cycle_dir)
                print(f"Created default cycle status file at {file_path} in GCS bucket.", file=sys.stderr)
            
            blob = bucket.blob(file_path)
            content = blob.download_as_string()
            status_data = yaml.safe_load(content)
            print("status_data:", status_data, file=sys.stderr)
            return status_data

    @staticmethod
    def save_cycle_status(cycle_dir, training_phase_done=None, inference_phase_done=None, finalize_phase_done=None):
        """Save cycle status to a YAML file."""
        status_data = CycleStatusUtils.load_cycle_status(cycle_dir)
        if training_phase_done is not None:
            status_data['training_phase_done'] = training_phase_done
        if inference_phase_done is not None:
            status_data['inference_phase_done'] = inference_phase_done
        if finalize_phase_done is not None:
            status_data['finalize_phase_done'] = finalize_phase_done

        if not _USE_XLA:
            file_path = f"{cycle_dir}/cycle_status.yaml"
            with open(file_path, 'w') as file:
                yaml.safe_dump(status_data, file)
        else:
            file_path = f"{cycle_dir}/cycle_status.yaml"
            blob = bucket.blob(file_path)
            blob.upload_from_string(yaml.safe_dump(status_data))

    @staticmethod
    def is_training_phase_done(cycle_dir):
        status_data = CycleStatusUtils.load_cycle_status(cycle_dir)
        return status_data.get('training_phase_done', False)
    
    @staticmethod
    def is_inference_phase_done(cycle_dir):
        status_data = CycleStatusUtils.load_cycle_status(cycle_dir)
        return status_data.get('inference_phase_done', False)
    
    @staticmethod
    def is_finalize_phase_done(cycle_dir):
        status_data = CycleStatusUtils.load_cycle_status(cycle_dir)
        return status_data.get('finalize_phase_done', False)
