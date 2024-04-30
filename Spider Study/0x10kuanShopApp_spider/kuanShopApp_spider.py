# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 04:26:09 2020
酷安应用商店应用数据爬取
@author: luohenyueji
"""

import requests
import queue
import threading
import re
from lxml import etree
import csv
from copy import deepcopy


class KuAn(object):

    def __init__(self, category, page):
        if category not in ['apk', 'game']:
            raise ValueError('category参数不在范围内')
        # 类别
        self.category = category
        self.page = page
        self.header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36'}
        # 写入csv头
        self.csv_header = ['应用名称', '下载链接']
        with open('{}.csv'.format(self.category), 'w', newline='', encoding='utf-8-sig') as f:
            csv_file = csv.writer(f)
            csv_file.writerow(self.csv_header)

        # url
        self.url = 'https://www.coolapk.com'
        self.base_url = 'https://www.coolapk.com/{}'.format(category)

        # 队列
        # 要爬取的网页
        self.page_url_queue = queue.Queue()
        # 单个应用网页地址
        self.detail_url_queue = queue.Queue()
        self.save_queue = queue.Queue()

    # ----- 获得单个应用的页面地址
    def get_detail_url_fun(self):
        while True:
            # 取出页面
            page_url = self.page_url_queue.get()
            req = requests.get(url=page_url, headers=self.header)
            if req.status_code == 200:
                req.encoding = req.apparent_encoding
                html = etree.HTML(req.text)
                # 获得链接
                if self.category == 'apk':
                    path = html.xpath('//*[@class="app_left_list"]/a/@href')
                elif self.category == 'game':
                    path = html.xpath('//*[@class="game_left_three"]/a/@href')
                for _ in path:
                    # 单个应用网页地址
                    detail_url = self.url + _
                    print('正在获取详情链接:', detail_url)
                    # 保存数据
                    self.detail_url_queue.put(deepcopy(detail_url))
            # 告诉 page_url_queue.join()任务完成
            # 参考 https://blog.csdn.net/qq_43577241/article/details/104442854
            self.page_url_queue.task_done()

            if self.page_url_queue.empty():
                break

    # ----- 获得单个应用的下载地址
    def get_download_url_fun(self):
        while True:
            detail_url = self.detail_url_queue.get()
            req = requests.get(url=detail_url, headers=self.header)
            if req.status_code == 200:
                req.encoding = 'utf-8'
                # 下载链接获取需要仔细寻找，可能无法直接下载
                url_reg = "'(.*?)&from=from-web"
                name_reg = '<p class="detail_app_title">(.*?)<'
                # 获取下载链接
                download_url = re.findall(url_reg, req.text)[0] + '&from=from-web'
                # 获取应用名字
                name = re.findall(name_reg, req.text)[0]

                data = {'name': name, 'url': download_url}
                print('获取到数据:', data)
                self.save_queue.put(data)
            self.detail_url_queue.task_done()

    # ----- 保存数据
    def save_data_fun(self):
        while True:
            data = self.save_queue.get()
            name = data.get('name')
            url = data.get('url')
            with open('{}.csv'.format(self.category), 'a+', newline='', encoding='utf-8-sig') as f:
                csv_file = csv.writer(f)
                csv_file.writerow([name, url])
            self.save_queue.task_done()

    def run(self):
        for _ in range(1, self.page + 1):
            # 设定网页
            page_url = self.base_url + '?p={}'.format(_)
            print('下发页面url', page_url)
            # 要爬取的网页
            self.page_url_queue.put(page_url)

        self.get_detail_url_fun()
        thread_list = []
        # 两个线程获得单个应用的页面地址
        for _ in range(2):
            get_detail_url = threading.Thread(target=self.get_detail_url_fun)
            thread_list.append(get_detail_url)

        # 五个线程获得单个应用的下载地址
        for _ in range(5):
            get_download_url = threading.Thread(target=self.get_download_url_fun)
            thread_list.append(get_download_url)

        # 两个线程保存单个应用的下载地址
        for _ in range(2):
            save_data = threading.Thread(target=self.save_data_fun)
            thread_list.append(save_data)

        for t in thread_list:
            # 设置为守护进程 主进程中的代码执行完毕之后，子进程自动结束
            t.setDaemon(True)
            t.start()
        for q in [self.page_url_queue, self.detail_url_queue, self.save_queue]:
            # 直到 queue中的数据均被删除或者处理
            # 参考 https://blog.csdn.net/dashoumeixi/article/details/80946509
            q.join()

        print('爬取完成，结束')


if __name__ == '__main__':
    KuAn(category='game', page=2).run()
