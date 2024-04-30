# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 19:57:30 2020
豆瓣选电影页面信息爬取
@author: luohenyueji
"""

import json
import time
import requests
from fake_useragent import UserAgent
from requests.exceptions import RequestException
import csv


# ----- 获得网页
def get_one_page(page_start):
    # 定义请求url
    # url获得方法见
    # https://www.cnblogs.com/dcpeng/p/13589433.html
    url = "https://movie.douban.com/j/search_subjects"

    ua = UserAgent()
    # 随机产生UserAgent，定义请求头，防止反爬
    # 详情见https://blog.csdn.net/u013421629/article/details/78168430
    headers = {'User-Agent': ua.random}
    params = {
        # 类型
        "type": "movie",
        # 标签
        "tag": "热门",
        # 排序方式
        "sort": "recommend",
        # 每页显示数量
        "page_limit": "20",
        # 开始页
        "page_start": page_start
    }
    try:
        response = requests.get(
            url=url,
            headers=headers,
            params=params,
            verify=False
        )
        if response.status_code == 200:
            # 方式一:直接转换json方法
            results = response.json()
            # # 方式二: 手动转换
            # # 获取字节串
            # content = response.content
            # # 转换成字符串
            # string = content.decode('utf-8')
            # # 把字符串转成python数据类型
            # results = json.loads(string)
            return results
        return None
    except RequestException:
        return None


# ----- 解析数据    
def parse_one_page(results, page_start):
    for i, item in enumerate(results["subjects"]):
        yield {
            # 序号
            'index': i + 1 + page_start,
            # 名字
            "title": item["title"],
            # 豆瓣评分
            'rate': item["rate"],
            # 豆瓣链接
            'url': item["url"]
        }


# ----- 保存为json数据
def write_to_json(content):
    with open('result.txt', 'a', encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False) + '\n')


# ----- 保存数据为csv文件
def write_to_csvFile(content):
    with open("result.csv", 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=content.keys())
        # 如果是第一行，写入表头
        if int(content['index']) == 1:
            writer.writeheader()
        writer.writerow(content)


def main(page_start):
    results = get_one_page(page_start)
    for item in parse_one_page(results, page_start):
        print(item)
        write_to_json(item)
        write_to_csvFile(item)


if __name__ == '__main__':
    # 循环构建请求参数并且发送请求
    for i in range(0, 100, 20):
        main(i)
        time.sleep(1)
