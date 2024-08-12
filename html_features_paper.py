import torch
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from collections import Counter
import re

def extract_html_feature_1(html_content, base_url):
    """
    提取HTML内容中的内部和外部链接特征，并返回一个PyTorch一维张量。

    参数:
    - html_content: str, HTML页面的原始内容。
    - base_url: str, 页面的基础URL，用于判断链接是内部还是外部。

    返回:
    - torch.Tensor, 包含内部链接和外部链接数量的一维张量。
    """
    # 解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取所有<a>标签的href属性
    links = soup.find_all('a', href=True)

    # 解析基础URL
    base_url_parts = urlparse(base_url)
    base_domain = base_url_parts.netloc

    internal_links_count = 0
    external_links_count = 0

    # 遍历所有链接，判断是内部链接还是外部链接
    for link in links:
        link_url = link['href']
        # 忽略页面内的锚点链接和javascript伪协议链接
        if link_url.startswith('#') or link_url.startswith('javascript:'):
            continue
        # 解析链接的URL
        link_parts = urlparse(link_url)
        # 检查链接是否是绝对URL，如果不是，则认为是相对路径，属于内部链接
        if not link_parts.netloc:
            internal_links_count += 1
        elif link_parts.netloc == base_domain:
            internal_links_count += 1
        else:
            external_links_count += 1

    # 将特征转换为PyTorch一维张量
    features = torch.tensor([internal_links_count, external_links_count], dtype=torch.float32)

    return features

def extract_html_feature_2(html_content):
    """
    提取HTML内容中的空链接数量特征，并返回一个PyTorch一维张量。

    参数:
    - html_content: str, HTML页面的原始内容。

    返回:
    - torch.Tensor, 包含空链接数量的一维张量。
    """
    # 解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取所有<a>标签的href属性
    links = soup.find_all('a', href=True)

    # 计算空链接的数量
    empty_links_count = sum(1 for link in links if not link['href'] or link['href'] == '#')

    # 将特征转换为PyTorch一维张量
    features = torch.tensor([empty_links_count], dtype=torch.float32)

    return features

def extract_html_feature_3(html_content):
    """
    检测HTML内容中是否存在登录表单，并返回一个PyTorch一维张量。

    参数:
    - html_content: str, HTML页面的原始内容。

    返回:
    - torch.Tensor, 包含是否存在登录表单的布尔值特征的一维张量。
    """
    # 解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 定义用于匹配登录表单的关键词
    login_keywords = ['password', 'pass', 'login', 'signin']

    # 标识是否存在登录表单
    has_login_form = False

    # 检查所有<form>标签
    forms = soup.find_all('form')
    for form in forms:
        # 检查<form>标签内的所有<input>子标签
        inputs = form.find_all('input')
        for input_tag in inputs:
            # 检查<input>标签的type属性和name/值属性是否包含登录关键词
            if input_tag.get('type') == 'password' or any(keyword in (input_tag.get('name') or input_tag.get('value') or '').lower() for keyword in login_keywords):
                has_login_form = True
                break

    # 如果存在登录表单，将特征设置为1，否则为0
    login_form_feature = 1 if has_login_form else 0

    # 将特征转换为PyTorch一维张量
    features = torch.tensor([login_form_feature], dtype=torch.float32)

    return features

def extract_html_feature_4(html_content):
    """
    提取HTML内容中特定标签内容的长度特征，并返回一个PyTorch一维张量。

    参数:
    - html_content: str, HTML页面的原始内容。

    返回:
    - torch.Tensor, 包含特定标签内容长度和整个HTML代码长度的一维张量。
    """
    # 解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 定义需要计算长度的标签
    tags = ['style', 'script', 'link', '!--', 'form']

    # 计算每个标签内容的长度
    tag_lengths = []
    for tag in tags:
        if tag == '!--':  # 注释标签需要特殊处理
            comments = soup.find_all(string=lambda text: isinstance(text, str) and ('<!--' in text or '-->' in text))
            tag_length = sum(len(comment) for comment in comments)
        else:
            tag_elements = soup.find_all(tag)
            tag_length = sum(len(str(element)) for element in tag_elements)
        tag_lengths.append(tag_length)

    # 计算整个HTML代码的长度
    html_length = len(html_content)

    # 将所有特征转换为PyTorch一维张量
    features = torch.tensor(tag_lengths + [html_length], dtype=torch.float32)

    return features

def extract_html_feature_5(html_content):
    """
    检测HTML内容中是否包含弹出警告窗口的代码，并返回一个PyTorch一维张量。

    参数:
    - html_content: str, HTML页面的原始内容。

    返回:
    - torch.Tensor, 包含是否存在警告窗口的布尔值特征的一维张量。
    """
    # 使用正则表达式搜索'alert('，它会匹配任何非空的alert函数调用
    has_alert = re.search(r'alert\(.*?\)', html_content)

    # 如果找到匹配项，has_alert将不是None，表示存在警告窗口
    alert_window_feature = 1 if has_alert else 0

    # 将特征转换为PyTorch一维张量
    features = torch.tensor([alert_window_feature], dtype=torch.float32)

    return features

def extract_html_feature_6(html_content):
    """
    检测HTML内容中是否存在重定向字符串，并返回一个PyTorch一维张量。

    参数:
    - html_content: str, HTML页面的原始内容。

    返回:
    - torch.Tensor, 包含是否存在重定向字符串的布尔值特征的一维张量。
    """
    # 定义用于检测重定向的关键词
    redirect_keyword = "redirect"

    # 检查HTML内容中是否包含redirect关键词
    has_redirect = redirect_keyword in html_content.lower()

    # 如果存在重定向字符串，将特征设置为1，否则为0
    redirection_feature = 1 if has_redirect else 0

    # 将特征转换为PyTorch一维张量
    features = torch.tensor([redirection_feature], dtype=torch.float32)

    return features

def extract_html_feature_7(html_content):
    """
    提取HTML内容中隐藏或限制信息的特征，并返回一个PyTorch一维张量。

    参数:
    - html_content: str, HTML页面的原始内容。

    返回:
    - torch.Tensor, 包含特殊代码数量的一维张量。
    """
    # 解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 检查特定标签和属性
    hidden_div_count = len(soup.find_all('div', style=lambda value: value and ('hidden' in value or 'none' in value)))
    disabled_button_count = len(soup.find_all('button', attrs={'disabled': 'disabled'}))
    hidden_input_count = len(soup.find_all('input', type='hidden'))
    disabled_input_count = len(soup.find_all('input', attrs={'disabled': 'disabled'}))
    prefilled_input_count = len(soup.find_all('input', value=lambda value: value and 'hello' in value))

    # 将特征转换为PyTorch一维张量
    features = torch.tensor([
        hidden_div_count,
        disabled_button_count,
        hidden_input_count,
        disabled_input_count,
        prefilled_input_count
    ], dtype=torch.float32)

    return features


def extract_html_feature_8(html_content, url):
    """
    检测网页标题中的品牌名称是否与URL中的品牌名称一致，并返回一个PyTorch一维张量。

    参数:
    - html_content: str, HTML页面的原始内容。
    - url: str, 网页的URL。

    返回:
    - torch.Tensor, 包含品牌名称一致性的布尔值特征的一维张量。
    """
    # 解析HTML内容，提取<title>标签的文本
    soup = BeautifulSoup(html_content, 'html.parser')
    title_text = soup.title.string if soup.title else ""

    # 解析URL，获取域名
    parsed_url = urlparse(url)
    domain = parsed_url.netloc if parsed_url.netloc else parsed_url.path

    # 提取URL品牌名称
    url_brand = domain.split('.')[-2] if domain.count('.') > 1 else domain

    # 检查标题文本中是否包含URL品牌名称
    has_consistent_brand = url_brand.lower() in title_text.lower()

    # 如果标题中包含URL品牌名称，将特征设置为1，否则为0
    brand_consistency_feature = 1 if has_consistent_brand else 0

    # 将特征转换为PyTorch一维张量
    features = torch.tensor([brand_consistency_feature], dtype=torch.float32)

    return features

def extract_html_feature_9(html_content, url):
    """
    检测最频繁链接品牌名称是否与URL品牌名称一致，并返回相关特征的PyTorch一维张量。

    参数:
    - html_content: str, HTML页面的原始内容。
    - url: str, 网页的URL。

    返回:
    - torch.Tensor, 包含最频繁链接品牌名称出现次数和一致性特征的一维张量。
    """
    # 解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取所有<a>标签的href属性
    links = soup.find_all('a', href=True)
    link_domains = [urlparse(link['href']).netloc for link in links if urlparse(link['href']).netloc]

    # 计算每个品牌名称出现的次数
    brand_counter = Counter(link_domains)
    most_common_brand, most_common_count = brand_counter.most_common(1)[0] if brand_counter else (None, 0)

    # 解析URL，获取URL品牌名称
    parsed_url = urlparse(url)
    url_brand = parsed_url.netloc.split('.')[-2] if '.' in parsed_url.netloc else parsed_url.netloc

    # 检查最频繁的链接品牌名称是否与URL品牌名称一致
    is_consistent = (most_common_brand == url_brand)

    # 将特征转换为PyTorch一维张量
    features = torch.tensor([most_common_count, int(is_consistent)], dtype=torch.float32)

    return features

def extract_html_feature_10(html_content, base_url):
    """
    提取HTML页面中使用到的内部和外部资源的数量，并返回一个PyTorch一维张量。

    参数:
    - html_content: str, HTML页面的原始内容。
    - base_url: str, 页面的基础URL，用于判断资源链接是内部还是外部。

    返回:
    - torch.Tensor, 包含内部和外部资源数量的一维张量。
    """
    # 解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 解析基础URL
    base_url_parts = urlparse(base_url)
    base_domain = base_url_parts.netloc

    # 定义用于检测资源的标签和属性
    resource_tags = ['link', 'img', 'script', 'noscript']
    internal_resources_count = {tag: 0 for tag in resource_tags}
    external_resources_count = {tag: 0 for tag in resource_tags}

    # 遍历所有资源标签，统计内部和外部资源
    for tag in resource_tags:
        if tag == 'script':
            resources = soup.find_all(tag, src=True)
        elif tag in ['link', 'img', 'noscript']:
            resources = soup.find_all(tag, href=True)
        else:
            resources = soup.find_all(tag)  # 如果不需要特定的属性
        for resource in resources:
            resource_url = resource.get('href') or resource.get('src')
            if resource_url:
                # 将相对URL转换为绝对URL
                full_resource_url = urljoin(base_url, resource_url)
                # 解析资源URL
                resource_url_parts = urlparse(full_resource_url)
                # 判断资源是否为外部资源
                if resource_url_parts.netloc == base_domain:
                    internal_resources_count[tag] += 1
                else:
                    external_resources_count[tag] += 1

    # 将统计结果转换为列表
    internal_resource_features = [internal_resources_count[tag] for tag in resource_tags]
    external_resource_features = [external_resources_count[tag] for tag in resource_tags]

    # 将特征转换为PyTorch一维张量
    features = torch.tensor(internal_resource_features + external_resource_features, dtype=torch.float32)

    return features

def extract_html_feature_11(html_content, url):
    """
    提取URL品牌名称在HTML代码中的出现次数。

    参数:
    - html_content: str, HTML页面的原始内容。
    - url: str, 网页的URL。

    返回:
    - torch.Tensor, 包含URL品牌名称出现次数的一维张量。
    """
    # 解析URL以提取品牌名称
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    # 提取域名作为品牌名称，忽略顶级域名
    url_brand = domain.split('.')[-2] if domain.count('.') > 1 else domain

    # 使用正则表达式来计算品牌名称在HTML内容中的出现次数
    # 这里我们假设品牌名称由字母和数字组成，并可能包含连字符（-）
    # 根据实际情况，正则表达式可能需要调整
    pattern = re.compile(re.escape(url_brand), re.IGNORECASE)
    matches = pattern.findall(html_content)

    # 计算出现次数并转换为PyTorch张量
    brand_name_count = len(matches)
    features = torch.tensor([brand_name_count], dtype=torch.float32)

    return features

def combine_features(html_content, url):
    """
    将所有特征提取函数的结果融合成一个一维张量。

    参数:
    - html_content: str, HTML页面的原始内容。
    - url: str, 网页的URL。

    返回:
    - torch.Tensor, 包含所有特征的一维张量。
    """
    # 提取第1个特征
    feature_1 = extract_html_feature_1(html_content, url)

    # 提取第2个特征
    feature_2 = extract_html_feature_2(html_content)

    feature_3 = extract_html_feature_3(html_content)

    feature_4 = extract_html_feature_4(html_content)

    feature_5 = extract_html_feature_5(html_content)

    feature_6 = extract_html_feature_6(html_content)

    feature_7 = extract_html_feature_7(html_content)

    feature_8 = extract_html_feature_8(html_content, url)

    feature_9 = extract_html_feature_9(html_content, url)

    feature_10 = extract_html_feature_10(html_content, url)

    feature_11 = extract_html_feature_11(html_content, url)
    # 将所有特征张量合并为一个一维张量
    # 假设每个特征提取函数返回的是一个一维张量
    combined_features = torch.cat([feature_1, feature_2, feature_3, feature_4,
                                   feature_5, feature_6, feature_7,
                                   feature_8, feature_9, feature_10,
                                   feature_11], dim=0)

    return combined_features
