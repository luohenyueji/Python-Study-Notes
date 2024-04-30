# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:13:44 2020
全景网分类图片爬取
@author: luohenyueji
"""

import os
import queue
import threading
import time

import requests
from bs4 import BeautifulSoup

# 链接名，根据category种类号110003设定，如果要查看其他的种类见 https://www.quanjing.com/category
string = 'https://www.quanjing.com/category/110003/'
# 队列
url_queue = queue.Queue()


# ----- 获取下载网页链接
def get_url(page):
    for i in range(1, page + 1):
        url = string + '{}.html'.format(i)
        url_queue.put(url)
    print(url_queue.queue)


# ----- 下载网页
def spider(url_queue):
    # 队列为空则返回
    if url_queue.empty():
        return
    # 获得网页链接
    url = url_queue.get()
    # 获得网页名
    floder_name = os.path.split(url)[1].split('.')[0]
    # 创建文件夹
    os.makedirs('第{0}页'.format(floder_name), exist_ok=True)
    html = requests.get(url=url, verify=False).text
    soup = BeautifulSoup(html, 'lxml')
    # 解析
    ul = soup.find_all('a', attrs={"class": "item lazy"})
    for i, child in enumerate(ul):
        # 获得当前标签下的子标签的图片链接
        downurl = child.img['src']
        # 下载
        result = requests.get(url=downurl, verify=False).content
        with open('第{0}页\{1}.jpg'.format(floder_name, i), 'ab') as f:
            f.write(result)
        print('第{0}页第{1}张存储完成'.format(floder_name, i))

    if not url_queue.empty():
        spider(url_queue)


def main(queue_count=3):
    queue_list = []
    # 线程数
    queue_count = queue_count
    for i in range(queue_count):
        t = threading.Thread(target=spider, args=(url_queue,))
        queue_list.append(t)
    for t in queue_list:
        t.start()
    for t in queue_list:
        t.join()


if __name__ == '__main__':
    # 需要爬取的页数
    page = 3
    get_url(page)
    start_time = time.time()
    main()
    print("用时：%f s" % (time.time() - start_time))
