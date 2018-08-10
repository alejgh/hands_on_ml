import os
import tensorflow as tf

from datetime import datetime


class TFLogger():
    def __init__(self, cls_name):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.log_dir = os.path.join("tf_logs", cls_name, "run-{}".format(now))
        self.summaries = dict()
        self.summaries_str = []

    def write_summaries(self, step):
        for summary_str in self.summaries_str:
            self.file_writer.add_summary(summary_str, step)

    def add_summary(self, name, tensor, tensor_constructor):
        self.summaries[name] = tensor_constructor(name, tensor)

    def eval(self, summary_name, **kwargs):
        self.summaries_str.append(self.summaries[summary_name].eval(**kwargs))

    def __enter__(self):
        self.file_writer = tf.summary.FileWriter(self.log_dir,
                                                 tf.get_default_graph())
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.file_writer.close()
