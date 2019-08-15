# coding=utf-8
from shutil import copyfile

import random


def shuffle(file_names):
    """
    将文章处理成r2l
    :param files:
    :return:
    """
    for file in file_names:
        new_file_name = file + '_temp'
        copyfile(file, new_file_name)
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context.endswith('\n'):
                        context = context[:-1]
                    if context.__contains__('\n'):
                        print(context)
                    if context:
                        context_list = context.split(' ')
                        random.shuffle(context_list)
                        context = ' '.join(context_list)
                    f_target.write(context)
                    f_target.write('\n')


def r2l(file_names):
    """
    将文章处理成r2l
    :param files:
    :return:
    """
    for file in file_names:
        new_file_name = file + '_temp'
        copyfile(file, new_file_name)
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context.endswith('\n'):
                        context = context[:-1]
                    if context.__contains__('\n'):
                        print(context)
                    if context:
                        context_list = context.split(' ')
                        context_list.reverse()
                        context = ' '.join(context_list)
                    f_target.write(context)
                    f_target.write('\n')


def first_2(file_names):
    """
    将文章处理成r2l
    :param files:
    :return:
    """
    for file in file_names:
        new_file_name = file + '_temp'
        copyfile(file, new_file_name)
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context:
                        if context.endswith('\n'):
                            context = context[:-1]
                        if context.__contains__('\n'):
                            print(context)
                        context_list = context.split(' ')
                        if len(context_list) > 1:
                            context_list.append(context_list[0])
                            context_list.pop(0)
                            context = ' '.join(context_list)
                    f_target.write(context)
                    f_target.write('\n')


def first_2_r2l(file_names):
    """
    将文章处理成r2l
    :param files:
    :return:
    """
    for file in file_names:
        new_file_name = file + '_temp'
        copyfile(file, new_file_name)
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context:
                        if context.endswith('\n'):
                            context = context[:-1]
                        if context.__contains__('\n'):
                            print(context)
                        context_list = context.split(' ')
                        if len(context_list) > 1:
                            # 将第一个单词拼接在最后 并使用r2l
                            context_list.append(context_list[0])
                            context_list.pop(0)
                            context_list.reverse()
                            context = ' '.join(context_list)
                    f_target.write(context)
                    f_target.write('\n')


def remove_1(file_names):
    """
    删除第一个词语
    :param files:
    :return:
    """
    for file in file_names:
        new_file_name = file + '_temp'
        copyfile(file, new_file_name)
        with open(file, 'w') as f_target:
            with open(new_file_name, 'r') as f_src:
                for context in f_src.readlines():
                    if context:
                        if context.endswith('\n'):
                            context = context[:-1]
                        if context.__contains__('\n'):
                            print(context)
                        context_list = context.split(' ')
                        if len(context_list) > 1:
                            context_list.pop(0)
                            context = ' '.join(context_list)
                    f_target.write(context)
                    f_target.write('\n')


if __name__ == '__main__':
    # file_names = [
    #     "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/r2l/mt03.ref0",
    #     "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/r2l/mt03.ref1",
    #     "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/r2l/mt03.ref2",
    #     "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/r2l/mt03.ref3",
    #     "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/train/r2l/train.en"
    # ]
    # r2l(file_names)
    file_names = ["/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2/mt03.ref0",
                  "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2/mt03.ref1",
                  "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2/mt03.ref2",
                  "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2/mt03.ref3",
                  "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/train/first_2/train.en"]
    first_2(file_names)
    # file_names = ["/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2_r2l/mt03.ref0",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2_r2l/mt03.ref1",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2_r2l/mt03.ref2",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/first_2_r2l/mt03.ref3",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/train/first_2_r2l/train.en"]
    # first_2_r2l(file_names)

    # file_names = ["/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/remove_1/mt03.ref0",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/remove_1/mt03.ref1",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/remove_1/mt03.ref2",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/test/remove_1/mt03.ref3",
    #               "/home/wangdq/njunlp_code/data/nist_zh-en_1.34m/train/remove_1/train.en"]
    # remove_1(file_names)
