# coding: utf-8
# created by deng on 7/24/2018

import json

def load(json_url):
    """ load python object form json file

    Args:
        json_url: url to json file

    Returns:
        python object

    """
    with open(json_url, "r", encoding="utf-8") as json_file:
        obj = json.load(json_file)
    return obj


def dump(obj, json_url):
    """ dump python object into json file

    Args:
        obj: python object, more information here
             https://docs.python.org/2/library/json.html#encoders-and-decoders
        json_url: url to save json file

    """
    with open(json_url, "w", encoding="utf-8", newline='\n') as json_file:
        json.dump(obj, json_file, separators=[',\n', ': '])


def sort_dict_by_value(dic, reverse=False):
    """ sort a dict by value

    Args:
        dic: the dict to be sorted
        reverse: reverse order or not

    Returns:
        sorted dict

    """
    return dict(sorted(dic.items(), key=lambda x: x[1], reverse=reverse))


def main():
    pass


if __name__ == '__main__':
    main()
