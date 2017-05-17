""" Functions and classes for parsing config files and command line arguments.
"""


def parse_cmd_args():
    """Return parsed command line arguments."""
    import argparse

    p = argparse.ArgumentParser(description='')
    p.add_argument('-cf', '--config_file', type=str, default="dev",
                   metavar='config_file_name::str', help='Config file name.')
    args = p.parse_args()
    return args


class Messenger(object):
    """ Turns a dictionary into a nested object.

        Instead of `cmdl["agent"]["name"]` we can do now `cmdl.agent.name`
    """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

        for key, value in kwargs.copy().items():
            if isinstance(value, dict):
                self.__dict__[key] = Messenger(**value)


def parse_config_file(args):
    import yaml

    config_path = "./configs/%s.yaml" % args.config_file
    f = open(config_path)
    config_data = yaml.load(f, Loader=yaml.SafeLoader)
    f.close()

    # import pprint
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(config_data)
    return Messenger(**config_data)
