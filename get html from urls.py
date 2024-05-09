import os
import requests
from requests.exceptions import Timeout

# 添加自定义头部
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Referer': 'https://ntp.msn.cn/'
}

def fetch_url_content(url):
    try:
        response = requests.get(url, headers=headers, timeout=15)  # 设置超时时间为15秒
        if response.status_code == 200:
            return response.text, True
        else:
            return f"Failed to fetch URL: {url}. Status code: {response.status_code}", False
    except Timeout:
        return f"Failed to fetch URL: {url}. Timeout occurred.", False
    except requests.RequestException as e:
        return f"Failed to fetch URL: {url}. Error: {e}", False

def main(start, end):
    success_count = 0
    fail_count = 0

    if not os.path.exists('output'):
        os.makedirs('output')

    with open('urls.txt', 'r', encoding='utf-8') as urls_file:
        urls = urls_file.readlines()
        for i, url in enumerate(urls, start=1):
            if start <= i <= end:
                url = url.strip()
                content, success = fetch_url_content(url)
                if success:
                    success_count += 1
                    with open(os.path.join('output', f'html_{i}.txt'), 'w', encoding='utf-8') as txtfile:
                        txtfile.write(content)
                    print(f"Successfully fetched URL {i}: {url}  {success_count}/{fail_count}")
                else:
                    fail_count += 1
                    with open(os.path.join('output', f'html_{i}.txt'), 'w', encoding='utf-8') as txtfile:
                        txtfile.write('')
                    print(f"Failed to fetch URL {i}: {url} (Created empty HTML file)  {success_count}/{fail_count}")

if __name__ == "__main__":
    start = int(input("Enter the start line number: "))
    end = int(input("Enter the end line number: "))
    main(start, end)
