# -*- coding: utf-8 -*-
"""

Created on Wed Sep 23 23:53:41 2020
豆瓣音乐排行榜爬取
@author: luohenyueji
"""

from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent
import csv
import json


# ----- 解析数据
def parseHtml(url):
    ua = UserAgent()
    # 随机产生UserAgent，定义请求头，防止反爬
    # 详情见https://blog.csdn.net/u013421629/article/details/78168430
    headers = {'User-Agent': ua.random}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    # 使用css选择器获取class="col5"节点下面的所有li节点
    for index, li in enumerate(soup.select(".col5 li")):
        if index < 10:
            yield {
                # 序号
                '歌曲排名': li.span.text,
                # 歌曲名，使用方法选择器
                "歌曲名": li.find(class_="icon-play").a.text,
                # 演唱者
                '演唱者': li.find(class_="intro").p.text.strip().split()[0],
                # 播放次数
                '播放次数': li.find(class_="intro").p.text.strip().split()[-1],
                # 上榜时间
                '上榜时间': li.find(class_="days").text.strip()
            }
        else:
            yield {
                # 序号
                '歌曲排名': li.span.text,
                # 歌曲名，使用方法选择器
                "歌曲名": li.find(class_="icon-play").a.text,
                # 演唱者
                '演唱者': li.find(class_="intro").p.contents[-1].strip().split()[0],
                # 播放次数
                '播放次数': li.find(class_="intro").p.contents[-1].strip().split()[-1],
                # 上榜时间
                '上榜时间': li.find(class_="days").text.strip()
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
        if int(content['歌曲排名']) == 1:
            writer.writeheader()
        writer.writerow(content)


def main():
    # 豆瓣音乐排行榜
    url = "https://music.douban.com/chart"
    for item in parseHtml(url):
        print(item)
        write_to_json(item)
        write_to_csvFile(item)


if __name__ == '__main__':
    main()
