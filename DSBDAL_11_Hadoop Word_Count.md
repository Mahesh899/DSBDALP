# Hadoop Word Count Program:

## Open Command Prompt as Administrator

```
cd\
```

```
cd hadoop
```

```
hdfs namenode -format
```

```
start-dfs.cmd
```

```
jps
```

Output after this command:

9200 DataNode

10396 Jps

10636 NameNode

```
start-yarn
```

Output after this command:

starting yarn daemons

```
jps
```

Output after this command:
11024 NodeManager

6432 ResourceManager

9200 DataNode

11176 Jps

10636 NameNode

```
hadoop fs -mkdir /input1
```

Create a text file with some random text in file with name data.txt:

data.txt

```
google
facebook
instagram
youtube
google
hi
hello
bye
facebook
youtube
google
hello
instagram
bye
hi
thanks
welcome
thanks
google
```

After saving the above file as data.txt in c drive perform below command

```
hadoop fs -put c:\data.txt /input1
```

```
hadoop fs -ls /input1
```

```
hadoop dfs -cat /input1/data.txt
```

```
hadoop jar C:\hadoop\share\hadoop\mapreduce\hadoop-mapreduce-examples-3.2.4.jar wordcount /input1 /out
```

```
hadoop fs -cat /out/*
```
