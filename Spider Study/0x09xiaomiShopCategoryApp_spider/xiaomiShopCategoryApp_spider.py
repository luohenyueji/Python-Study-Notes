# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 01:03:10 2020
小米应用商店分类应用数据爬取
@author: luohenyueji
"""

import requests
import csv
import queue
from fake_useragent import UserAgent


class XiaoMiShop:

    def __init__(self, category=15, max_page=50):
        """
        :param category: category下载类别
        :param max_page: max_page最大下载页数
        """
        # 网址解析说明见 https://blog.csdn.net/weixin_42521211/article/details/106965550
        # 网址解析地址
        self.base_url = 'http://app.mi.com/categotyAllListApi'
        # 下载地址
        self.base_download = 'http://app.mi.com/download/'

        # csv要保存的头部信息
        self.csv_header = ['ID', '应用名称', '应用子类', '下载链接']
        self.max_page = max_page
        # 下载类别
        self.category = category
        self.queue = queue.Queue()

    # ----- 清洗数据
    def clean_data(self, data):
        for i in data:
            app = {}
            app['ID'] = i.get('appId')
            app['应用名称'] = i.get('displayName')
            app['应用子类'] = i.get('level1CategoryName')
            app['下载链接'] = self.base_download + str(i.get('appId'))
            self.queue.put(app)

    # ----- 获得网页信息
    def request(self, page):
        # pageSize表示每页数量，page表示当前页数
        param = {
            'page': page,
            'categoryId': int(self.category),
            'pageSize': 30
        }
        # 随机产生UserAgent，定义请求头，防止反爬
        headers = {'User-Agent': UserAgent().random}
        req = requests.get(url=self.base_url, params=param, headers=headers)
        req.encoding = req.apparent_encoding
        return req

    # ----- 失败重新爬取
    def spider_by_page(self, page, retry=3):
        # retry 重试次数
        if retry > 0:
            print('重试第{}页'.format(page))
            req = self.request(page=page)
            try:
                data = req.json()['data']
                if data:
                    self.clean_data(data)
                    print('第{}页重试成功'.format(page))
            except:
                self.spider_by_page(page=page, retry=retry - 1)

    # ----- 爬取数据
    def spider(self):
        # 失败页面
        fail_page = []
        for _ in range(self.max_page):
            print('正在爬取第{}页'.format(_))
            # 获得网页信息
            req = self.request(_)
            try:
                data = req.json()['data']
            except:
                data = []
                fail_page.append(_)
            if data:
                # 清洗数据
                self.clean_data(data)
            else:
                continue
        # 失败重新爬取
        if fail_page:
            print('出错的页数：', fail_page)
            for _ in fail_page:
                self.spider_by_page(page=_)
        else:
            print('没有出错')

    def run(self):
        self.spider()
        data_list = []
        # 逐条信息提取
        while not self.queue.empty():
            data_list.append(self.queue.get())
        # 保存数据
        with open('{}.csv'.format(self.category), 'w', newline='', encoding='utf-8-sig') as f:
            f_csv = csv.DictWriter(f, self.csv_header)
            f_csv.writeheader()
            f_csv.writerows(data_list)
        print('文件保存成功,打开{}.csv查看'.format(self.category))


if __name__ == '__main__':
    XiaoMiShop(category=15, max_page=2).run()
