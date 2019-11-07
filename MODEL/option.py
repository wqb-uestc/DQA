import json
import random
import os


class Config(object):

    def __init__(self):
        self.path = {
            'IVIPC': os.path.join('dataset', 'IVIPC_DQA'), # YOUR PATH
        }

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


