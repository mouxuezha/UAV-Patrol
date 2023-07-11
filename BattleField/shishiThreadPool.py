# from concurrent.futures import ThreadPoolExecutor
# import threading
# import time
# # 定义一个准备作为线程任务的函数
# def action(max):
#     my_sum = 0
#     for i in range(max):
#         print(threading.current_thread().name + '  ' + str(i))
#         my_sum += i
#     return my_sum
# # 创建一个包含2条线程的线程池
# pool = ThreadPoolExecutor(max_workers=2)
# # 向线程池提交一个task, 50会作为action()函数的参数
# future1 = pool.submit(action, 50)
# # 向线程池再提交一个task, 100会作为action()函数的参数
# future2 = pool.submit(action, 100)
# # 判断future1代表的任务是否结束
# print(future1.done())
# time.sleep(3)
# # 判断future2代表的任务是否结束
# print(future2.done())
# # 查看future1代表的任务返回的结果
# print(future1.result())
# # 查看future2代表的任务返回的结果
# print(future2.result())
# # 关闭线程池
# pool.shutdown()

WEIZHI =r'E:/EnglishMulu/UAV-Patrol' 
import sys 
# sys.path.append(WEIZHI+r'/lib_cpp')
sys.path.append(r'E:/EnglishMulu/shishi_py/x64/Debug')
import numpy as np
import shishi_py
print(dir(shishi_py))

node_x = np.array([0.0,0.0])
node_y = np.array([0.0,1.0])
UAV_weizhi=np.array([1.0,1.0])

jieguo = shishi_py.jvli(node_x,node_y,UAV_weizhi)
print(jieguo)
print(type(jieguo))
print(jieguo.shape)

jieguo = shishi_py.jvli2(node_x,node_y,UAV_weizhi,10.0)
print(jieguo)
print(type(jieguo))
print(jieguo.shape)
# import example1
# dir(example1)