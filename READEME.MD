### 部署

1、安装docker，docker-compose

2、传代码到/home目录

3、启动 cassandra 

​      docker-compose up -d cassandra

4、启动数据库过后一段时间启动web

​	docker-compose up -d web

### 使用

#### 1、通过接口传文件进行测试

```
[root@localhost peng]# curl -XPOST -F "file=@/home/peng/files/mnist_digits_images/9/5.bmp" http://0.0.0.0:5000/mnist
result: 9
```

#### 2、进入cassandra容器中查看结果

```
* docker exec -it some-cassandra bash (进入容器)
** cqlsh（使用sql shell）
*** use deepnn; （使用keyspce deepnn）
*** select * from images; （从images库查看全部数据）
```



#### 视频录制步骤

```
1、启动docker
systemctl start docker
2、启动数据库进入查询已有记录

3、启动web

4、使用shell传图片

4、再次查询记录

```



