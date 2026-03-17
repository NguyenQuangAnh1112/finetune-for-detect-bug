from transformers import TrainerCallback


class MLflowProgressCallback(TrainerCallback):
    def __init__(self):
        import mlflow

        self.mlflow = mlflow
        self.current_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.mlflow.log_metric(key, value, step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        self.mlflow.log_metric(
            "checkpoint_step", state.global_step, step=state.global_step
        )
