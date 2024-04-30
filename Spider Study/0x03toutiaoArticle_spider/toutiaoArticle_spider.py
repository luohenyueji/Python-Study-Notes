# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:20:40 2020
今日头条单篇文章爬取
@author: luohenyueji
"""

import requests
from hashlib import md5
import os
import re

# 需要cookie
headers = {
    'cookie': 'csrftoken=1490f6d92e97ce79f9e52dc4f3222608; ttcid=22698125819f4938826fc916af6b7e7355; SLARDAR_WEB_ID=f754d5f8-83ce-4577-8f77-a232e1708142; tt_webid=6856774172980561421; WEATHER_CITY=%E5%8C%97%E4%BA%AC; tt_webid=6856774172980561421; __tasessionId=v8lvfouyt1596594815875; s_v_web_id=kdgrbwsk_9Ussu5RZ_AUmC_4DO5_8s8w_21Pv7qDIVeeE; tt_scid=AyiNMhl4GyKjhxNFpcm5AWgbRD7dsl-Zu4nBHWPBkHFf6lAynwUzX3zbMRIWr.De95f9',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/84.0.4147.105 Safari/537.36 '
}


# ----- 获取网页
def get_page(offset):
    # timestamp和_signature分割可以不传
    params = {
        'aid': 24,
        'app_name': 'web_search',
        # 控制翻页的参数
        'offset': offset,
        'format': 'json',
        # 搜索图片的关键词
        'keyword': '手机',
        'autoload': 'true',
        'count': 20,
        'en_qc': 1,
        'cur_tab': 1,
        'from': 'search_tab',
        'pd': 'synthesis',
    }
    url = 'http://www.toutiao.com/api/search/content/'
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json()
    except requests.ConnectionError as e:
        print(e)
        return None


# ----- 获得图像
def get_images(json):
    if json.get('data'):
        for item in json.get('data'):
            title = item.get('title')
            images = item.get('image_list')
            # 如果为空
            if images is None or title is None:
                continue
            for image in images:
                # 获得原图链接
                origin_image = re.sub("list.*?pgc-image", "large/pgc-image",
                                      image.get('url'))
                yield {
                    'image': origin_image,
                    'title': title
                }

            # ----- 保存图片


# 修正文件名
def correct_title(title):
    # 避免文件名含有不合法字符
    # 文件名最好不要含有.，否则有的系统会报错
    error_set = ['/', '\\', ':', '*', '?', '"', '|', '<', '>', '.']
    for c in title:
        if c in error_set:
            title = title.replace(c, '')
    return title


def save_image(item):
    dir_name = "手机/" + correct_title(item.get('title'))
    # 创建文件夹
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    try:
        response = requests.get(item.get('image'))
        if response.status_code == 200:
            # 获得图片
            file_path = '{0}/{1}.{2}'.format(dir_name,
                                             md5(response.content).hexdigest(), 'jpg')
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                print('Already Downloaded', file_path)
    except requests.ConnectionError:
        print('Failed to Save Image')


def main(offset):
    json = get_page(offset)
    for item in get_images(json):
        print(item)
        save_image(item)


if __name__ == '__main__':
    # 控制翻页
    for i in range(0, 6):
        print("第" + str(i + 1) + "页开始下载！！！")
        offset = i * 20
        main(offset)
