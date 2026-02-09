from typing import Any, Dict

import pandas as pd


class LossLogger:
    def __init__(self, log_file: str, save_interval: int = 10):
        self.log_file = log_file
        self.save_interval = save_interval
        self.logs = pd.DataFrame(columns=["step", "loss"])

    def __call__(self, log: Dict[str, Any]):
        # Append log to DataFrame
        if len(self.logs) == 0:
            self.logs = pd.DataFrame([log])
        else:
            self.logs = pd.concat([self.logs, pd.DataFrame([log])], ignore_index=True)

        # Save to CSV periodically
        if log["step"] % self.save_interval == 0:
            self.logs.to_csv(self.log_file, index=False)
