# coding=utf-8


import uuid

from cassandra.cluster import Cluster  # 引入Cluster模块


class MnistDeepImage(object):

    def __init__(self):
        # self.cluster = Cluster(['some-cassandra']) # 连接容器
        # self.cassandra_addr = '192.168.6.103'
        self.cassandra_addr = 'some-cassandra'
        print(self.create_keyspace_and_table())

        self.cluster = Cluster([self.cassandra_addr])

        self.session = self.cluster.connect('deepnn')  # 指定连接keyspace，相当于sql中use dbname

    def create_keyspace_and_table(self):
        try:
            cluster = Cluster([self.cassandra_addr])
            session = cluster.connect()
            session.execute("CREATE KEYSPACE deepnn WITH replication = {'class':'SimpleStrategy', 'replication_factor' : 3};")
            session.execute('use deepnn;')
            # 创建table
            session.execute('create table deepnn.images(image_id varchar primary key,req_time varchar,mnist_result int,file_name varchar);')

            self.session = session
            return True
        except Exception as err:
            print(err)
            return False

    def find_all(self):
        rows = self.session.execute('select * from deepnn.images')  # 查询所有列
        return rows

    def insert_data(self, req_time, file_name, mnist_result):
        # ['image_id', "file_name", 2, 'req_time']
        res = self.session.execute(
            """
            INSERT INTO images (image_id, file_name, mnist_result, req_time)
            VALUES (%s, %s, %s, %s)
            """, (str(uuid.uuid1()), file_name, mnist_result, req_time)

        )
        print(res)

if __name__ == '__main__':
    handle = MnistDeepImage()
    handle.create_keyspace_and_table()
