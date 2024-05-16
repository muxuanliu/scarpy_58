# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class Data58Pipeline:
    def open_spider(self,spider):
        self.fp = open('nlp_58.json','w',encoding = 'utf-8')
    def process_item(self, item, spider):
        self.fp.write((str(item)))
        return item
    def close_spider(self,spider):
        self.fp.close()

from scrapy.utils.project import get_project_settings
import pymysql

import pymysql
from scrapy.exceptions import DropItem

class MysqlPipeline():
    def open_spider(self, spider):
        self.db = pymysql.connect(
            host='localhost',
            user='root',
            password='passwd',
            db='nlp_58',
            charset='utf8',
        )
        self.cursor = self.db.cursor()

    def close_spider(self, spider):
        self.db.close()

    def process_item(self, item, spider):
        if item['label'] and item['job_name'] and item['description']:
            data = {
                'label': item['label'],
                'job_name': item['job_name'],
                'description': item['description']
            }
            sql = "INSERT INTO employ_info (label, job_name, description) VALUES (%s, %s, %s)"
            self.cursor.execute(sql, tuple(data.values()))
            self.db.commit()
        else:
            return DropItem('Missing Data58Item fields')
        return item