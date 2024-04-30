# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 05:45:46 2020
诗词名句网数据爬取，无数据保存
@author: luohenyueji
"""

import queue
import re
import threading

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


# ----- 浏览器头
def get_header():
    return {
        'User-Agent': UserAgent().random,
        'Connection': 'close'
    }


# ----- 多线程爬取，由于可能导致数据爬取不全，数据诗词总数约为20w+数据
class Shici(object):

    def __init__(self, thread=5):
        # 诗人网页
        self.poet_queue = queue.Queue()
        # 线程数
        self.thread = thread
        self.base_url = 'http://www.shicimingju.com'

    # ----- 查询每个诗人的网页
    def get_poet_url(self):
        # 查询每个诗人的网页
        # 具体作者查询https://www.shicimingju.com/category/all
        for i in range(4, 5):
            url = 'http://www.shicimingju.com/chaxun/zuozhe/{}.html'.format(i)

            self.poet_queue.put(url)

    # ----- 爬取信息
    def Spider(self):
        # 当诗人列表不为空
        while not self.poet_queue.empty():
            # 获得诗人诗词链接
            url = self.poet_queue.get()
            req = requests.get(url, headers=get_header())
            if req.status_code == 200:

                req.encoding = 'utf-8'
                soup = BeautifulSoup(req.text, 'lxml')
                # 作者名
                name = soup.h4.text
                # 作者朝代
                dynasty = soup.select(".aside_left .aside_val")[0].text.strip()
                if len(dynasty) == 0:
                    dynasty = '未知'
                # 生平介绍
                introduction = soup.find(attrs={"class": "des"}).text.strip()
                # 诗词数量
                poem_num = soup.select(".aside_right .aside_val")[0].text.strip()[:-1]
                # 当前作者每首诗的网址
                poet_url_list = []
                # 20表示每页诗词数量
                for i in range(1, int(int(poem_num) / 20) + 2):
                    # 诗人id
                    poet_id = re.sub("\D", "", url)
                    # 每页诗词网页
                    poet_page_url = 'http://www.shicimingju.com/chaxun/zuozhe/{}_{}.html'.format(poet_id, i)
                    req1 = requests.get(url=poet_page_url, headers=get_header())
                    if req1.status_code == 200:
                        req1.encoding = 'utf-8'
                        list_html = BeautifulSoup(req1.text, 'lxml')
                        # 诗词具体链接
                        poet_url_list += list_html.find_all('h3')
                # 获得作者每部诗词网页的链接
                poet_url_list = map(lambda x: self.base_url + x.a['href'].strip(), poet_url_list)
                for url in poet_url_list:
                    print(url)
                    # 获得具体诗词页的内容
                    req2 = requests.get(url, headers=get_header())
                    if req2.status_code == 200:
                        req2.encoding = 'utf-8'
                        poet_html = BeautifulSoup(req2.text, 'lxml')
                        # 诗词标题
                        title = poet_html.h1.text
                        # 内容
                        content = poet_html.find(class_='item_content')
                        # 解析
                        analysis = poet_html.find(class_='shangxi_content')
                        if not content:
                            content = ""
                        else:
                            content = content.text.strip()
                        if not analysis:
                            analysis = ''
                        else:
                            analysis = analysis.text.strip()

    def run(self):
        self.get_poet_url()
        thread_list = []

        # 爬取文章
        for i in range(self.thread):
            t = threading.Thread(target=self.Spider)
            thread_list.append(t)
        for t in thread_list:
            t.setDaemon(True)
            t.start()
        for t in thread_list:
            t.join()
        # 检查是否还有未爬取的链接
        self.Spider()


if __name__ == '__main__':
    Shici().run()
