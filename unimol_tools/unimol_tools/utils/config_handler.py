# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import yaml
import os
from addict import Dict

from .base_logger import logger


class YamlHandler:
    '''A clss to read and write the yaml file'''
    def __init__(self, file_path):
        """
        A custom logger class that provides logging functionality to console and file.

        :param file_path: (str) The yaml file path of the config.
        """
        if not os.path.exists(file_path):
            raise FileExistsError(OSError)
        self.file_path = file_path
    def read_yaml(self, encoding='utf-8'):
        """ read yaml file and convert to easydict

        :param encoding: (str) encoding method uses utf-8 by default
        :return: Dict (addict), the usage of Dict is the same as dict
        """
        with open(self.file_path, encoding=encoding) as f:
            return Dict(yaml.load(f.read(), Loader=yaml.FullLoader))
    def write_yaml(self, data, out_file_path, encoding='utf-8'):
        """ write dict or easydict to yaml file(auto write to self.file_path)

        :param data: (dict or Dict(addict)) dict containing the contents of the yaml file
        """
        with open(out_file_path, encoding=encoding, mode='w') as f:
            return yaml.dump(addict2dict(data) if isinstance(data, Dict) else data,
                stream=f,
                allow_unicode=True)


def addict2dict(addict_obj):
    '''convert addict to dict

    :param addict_obj: (Dict(addict)) the addict obj that you want to convert to dict

    :return: (Dict) converted result
    '''
    dict_obj = {}
    for key, vals in addict_obj.items():
        dict_obj[key] = addict2dict(vals) if isinstance(vals, Dict) else vals
    return dict_obj


if __name__ == '__main__':
    yaml_handler = YamlHandler('../config/default.yaml')
    config = yaml_handler.read_yaml()
    print(config.Modelhub)
