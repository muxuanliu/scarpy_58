import scrapy
from data_58.items import Data58Item

class EmploySpider(scrapy.Spider):
    name = 'employ'
    allowed_domains = ['cc.58.com','m.58.com']
    start_urls = ['https://cc.58.com/pugongjg/']

    base_url = 'https://cc.58.com/'
    page_list = ["sijiwl/","nxiaoshou/","ncanyin/","mendianlsh/","anbaoxf/","yunyingkf/","renshixzhcw/","fuwuy/","shengchanzhz/","nchuanmei","yiliaojk/"]

    # 一级爬虫位置
    # name = //div[@class="job_name clearfix"]//a//span[@class="name"]
    # 二级地址
    # // div[ @class ="job_name clearfix"] // a[@ target="_blank"] / @ href
    # 二级爬虫位置
    # // div[ @class ="des"] / text()
    # 重定向后--------------------------------
    # 一级爬虫位置
    # // div[ @class ="list_wrap"] / a // div[@ class ="info-title"]
    # 二级爬虫地址
    # // div[ @class ="list_wrap"] / a / @ href
    # 二级爬虫位置
    # // div[ @class ="job_ms_content_inner"]
    def parse(self, response):
        for path in self.page_list:
            url = self.base_url +path
            # 发起请求，要求使用parse_level_one处理url
            yield scrapy.Request(url,callback=self.parse_level_one,meta={'label':path})
    def parse_level_one(self,response):
        # # 提取一级爬虫位置的文本
        # print("判断url是否正确：",response.url)
        # print("判断状态码是否正确：",response.status)
        # print("判断响应头是否正确：",response.headers)
        job_names = response.xpath('// div[ @class ="list_wrap"] / a // div[@ class ="info-title"]/text()').extract()
        job_urls = response.xpath('// div[ @class ="list_wrap"] / a / @ href').getall()
        for job_name,job_url in zip(job_names,job_urls):
            abs_job_url = response.urljoin(job_url)
            label = response.meta.get('label')
            yield scrapy.Request(url = abs_job_url,callback=self.parse_level_two,meta={'label':label,'job_name':job_name})

    def parse_level_two(self, response):

        try:
            # 尝试提取数据
            label = response.meta.get('label')
            job_name = response.meta.get('job_name')
            description = response.xpath('//div[@class="job_ms_content_inner"]/text()').get()  # 确保这里是 get() 而不是 getall()
            # 检查是否成功提取了数据
            if not description:
                self.logger.error(f"No description found for job_name: {job_name}")
                return

            # 创建 Item 实例
            employ = Data58Item(label=label, job_name=job_name, description=description)
            yield employ

        except Exception as e:
            # 记录异常信息
            self.logger.error(f"Error processing page {response.url}: {e}")


