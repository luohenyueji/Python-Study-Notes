# -*- coding: utf-8 -*-
"""
猫眼top100电影排行榜抓取http://maoyan.com/board/4
Created on Thu Sep 17 19:40:58 2020

@author: luohenyueji
"""

import requests
import re
import json
import time
import csv


# ----- 抓取首页


def get_one_page(url):
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0"}
    # 获得对象
    # 需要伪装成浏览器，不然会解析失败
    response = requests.get(url, headers=headers)
    # 状态
    if response.status_code == 200:
        # 设置编码
        response.encoding = "utf-8"
        return response.text
    return None


# ----- 解析网页
def parse_one_page(html):
    # 将一页的10个电影信息都提取出来
    # re.s的作用是使.匹配包括换行符在内的所有字符
    pattern = re.compile(
        '<dd>.*?board-index.*?>(.*?)</i>'
        + '.*?data-src="(.*?)".*?name.*?a.*?>(.*?)</a>'
        + '.*?star.*?>(.*?)</p>.*?releasetime.*?>(.*?)</p>'
        + '.*?integer.*?>(.*?)</i>.*?fraction.*?>(.*?)</i>.*?</dd>',
        re.S)
    items = re.findall(pattern, html)
    for item in items:
        # yield关键字使用见https://blog.csdn.net/mieleizhi0522/article/details/82142856
        yield {
            'index': item[0],
            'image': item[1],
            # 去除.strip空格
            'title': item[2].strip(),
            # 先去除空格和主演：然后判断是否为空
            'actor': item[3].strip()[3:] if len(item[3]) > 3 else '',
            'time': item[4].strip()[5:] if len(item[4]) > 5 else '',
            # item[5]整数，item[6]小数
            'score': item[5].strip() + item[6].strip()
        }


# ----- 保存数据为json文件
def write_to_json(content):
    with open('result.txt', 'a', encoding='utf-8') as f:
        # ensure_ascii不将中文格式化
        f.write(json.dumps(content, ensure_ascii=False) + '\n')


# ----- 保存数据为csv文件
def write_to_csvFile(content):
    with open("result.csv", 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=content.keys())
        # 如果是第一行，写入表头
        if int(content['index']) == 1:
            writer.writeheader()
        writer.writerow(content)


def main(offset):
    # 地址
    url = 'http://maoyan.com/board/4?offset=' + str(offset)
    # 获得网页源代码
    html = get_one_page(url)
    # 解析数据
    for item in parse_one_page(html):
        print(item)
        write_to_json(item)
        write_to_csvFile(item)


if __name__ == '__main__':
    for i in range(10):
        # 分页爬取
        main(offset=i * 10)
        # 延时爬取
        time.sleep(1)
