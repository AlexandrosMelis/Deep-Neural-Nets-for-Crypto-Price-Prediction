from os.path import join
from enum import Enum
from pyexpat import model


class PathConfig(Enum):
    io_path = "io"
    data_processor_path = "data_processor"
    trainer_path = "pytorch_trainer"

    input_path = join(io_path, "input")
    output_path = join(io_path, "output")
    exports_path = join(output_path, "exports")
    raw_data_path = join(input_path, "raw_data")
    data_4h_interval_path = join(raw_data_path, "4h_interval")
    base_data_path = join(input_path, "base_data")
    metric_plots_path = join(exports_path, "metrics_plots")
    experiment_results_path = join(exports_path, "experiment_results")
    model_path = join(exports_path, "models")