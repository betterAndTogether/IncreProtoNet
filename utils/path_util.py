# coding: utf-8
# created by deng on 7/23/2018

from os.path import dirname, join, normpath, exists, basename
import time
import os

# 得到项目根目录
project_root_url = normpath(join(dirname(__file__), '..'))

def from_project_root(rel_path):
    """ 根据相对项目根目录的路径返回绝对路径

    Args:
        rel_path: 相对路径

    Returns:
        str: 绝对路径

    """
    return normpath(join(project_root_url, rel_path))


def date_suffix(file_type=""):
    """ 返回当前的日期后缀，如'180723.csv'

    Args:
        file_type: 文件类型, 如'.csv', 为""则只返回日期

    Returns:
        str: 日期后缀

    """
    suffix = time.strftime("%y%m%d", time.localtime())
    suffix += file_type
    return suffix

# 获取文件夹中所有的文件名
def fectch_filenames(dir_path):
    """
    :param dir_path: 文件夹路径
    :return:
    """
    filenames = []
    sub_dirs = [] # 子文件夹名
    for roots,dirs,files in os.walk(dir_path):
        filenames.extend(files)
        sub_dirs.extend(dirs)

    return filenames,sub_dirs


def main():
    print(project_root_url)
    print(from_project_root('.gitignore'))
    print(from_project_root('data/test.py'))
    print(date_suffix('.csv'))
    print(date_suffix(""))
    pass


if __name__ == '__main__':
    main()
