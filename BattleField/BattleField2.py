# this is to define battlefield 
from concurrent.futures import ThreadPoolExecutor
import threading
import time 

import numpy as np
import copy
from queue import Queue
WEIZHI =r'E:/EnglishMulu/UAV-Patrol' 
import sys 
sys.path.append(WEIZHI+r'/Support')
from huatu_support import huatu_support
from zuobiaoxi import zuobiaoxi
from zuobiaoxi import zuobiao

sys.path.append(WEIZHI+r'/UAV')
from UAV import UAV

# # sys.path.append(WEIZHI+r'/lib_cpp')
# sys.path.append(r'E:/EnglishMulu/shishi_py/x64/Debug')

# import shishi_py

class BattleField(object):
    def __init__(self,L_x=1000,L_y=1000,dL = 10 ,**kargs):
        print('Establishing battle field control, stand by')
        self.L_x = L_x 
        self.L_y = L_y 
        self.dL = dL 
        self.nodes=self.generate_nodes2(self.L_x,self.L_y,self.dL)
        self.nodes_observed =[]
        self.UAV_feiji = []  
        self.patrol_area = 0
    
    def generate_nodes2(self,L_x,L_y,dL,**kargs):
        L_x_iteration = 0 
        L_y_iteration = 0 
        nodes = [] 
        while(L_y_iteration<=L_y):
            L_y_iteration = L_y_iteration +dL
            L_x_iteration = 0 
            nodes_line = [] 
            while(L_x_iteration<=L_x):
                L_x_iteration = L_x_iteration + dL
                one_node_zuobiao = zuobiao(zhi=np.array([L_x_iteration,L_y_iteration]),zuobiaoxi=UAV.abs_coordinate)
                one_node = BattleField_node(zuobiao=one_node_zuobiao)
                nodes_line.append(one_node) 
            nodes.append(nodes_line)       
        # while(L_x_iteration<=L_x):
        #     L_x_iteration = L_x_iteration +dL
        #     L_y_iteration = 0 
        #     while(L_y_iteration<=L_y):
        #         L_y_iteration = L_y_iteration + dL
        #         one_node_zuobiao = zuobiao(zhi=np.array([L_x_iteration,L_y_iteration]),zuobiaoxi=UAV.abs_coordinate)
        #         one_node = BattleField_node(zuobiao=one_node_zuobiao)
        #         nodes.append(one_node)
        return nodes 
    
    def UAV_online(self,**kargs):
        if 'UAV_feiji' in kargs:
            self.UAV_feiji = kargs['UAV_feiji']
        else:
            self.UAV_feiji = UAV(location=np.array([500,100]), sudu_max=114514, omega_max = 1,r=1)
        print('BattleField: UAV online.')

    def visual_BattleField(self,**kargs):
        if 'huatu' in kargs:
            huatu = kargs['huatu']
        else:
            huatu = huatu_support()
            huatu.ax = huatu.plot_coordinate(huatu.ax,x_min=0,x_max=self.L_x,y_min=0,y_max=self.L_y)

        # for node in self.nodes:
        #     huatu = node.visual_node(huatu) # too slow. G 
        x_observed = []
        y_observed = [] 
        x_unobserved = []
        y_unobserved = []
        x_inside = [] 
        y_inside = [] 
        for node_line in self.nodes:
            for node in node_line:
                # node.running(self.UAV_feiji)
                if node.flag_observed==True:
                    x_observed.append(node.zuobiao.zhi[0][0])
                    y_observed.append(node.zuobiao.zhi[0][1])
                else:
                    x_unobserved.append(node.zuobiao.zhi[0][0])
                    y_unobserved.append(node.zuobiao.zhi[0][1])
                if node.flag_inside== True:
                    x_inside.append(node.zuobiao.zhi[0][0])
                    y_inside.append(node.zuobiao.zhi[0][1])

        huatu.ax = huatu.plot_point(huatu.ax,[x_observed,y_observed],yanse=huatu.get_color(index = 0),marker='o',markersize=0.5)
        huatu.ax = huatu.plot_point(huatu.ax,[x_unobserved,y_unobserved],yanse=huatu.get_color(index = 1),marker='o',markersize=0.5)   
        huatu.ax = huatu.plot_point(huatu.ax,[x_inside,y_inside],yanse=huatu.get_color(index = 2),marker='o',markersize=0.5)  

        # then UAV 
        huatu = self.UAV_feiji.visual_UAV(huatu = huatu)

        return huatu

    def running(self,d_omega=1,d_a=1,first_step = False,**kargs):

        if first_step:
            pass # dont fly the UAV in first step.
        else:
            self.UAV_feiji.time_goes_by(d_omega=d_omega,d_a=d_a)
            BattleField_node.running_clock(self.UAV_feiji.dt)
        
        x_index_low = self.__get_index_low(0)
        x_index_high = self.__get_index_high(0)
        y_index_low = self.__get_index_low(1)
        y_index_high = self.__get_index_high(1) 
        for x_index in range(x_index_low,x_index_high):
            for y_index in range(y_index_low,y_index_high):
                node = self.__get_node_single(x_index,y_index)
                node.running(self.UAV_feiji)
                if node.flag_observed: # add observed in the list (queue in fact)
                    self.nodes_observed.append(node)
        
        
        panju = False
        while panju: # remove out dated nodes.
            if len(self.nodes_observed)==0:
                # which means the UAV have fly away from the area, G
                break           
            self.nodes_observed[0].running_simple() 
            if not(self.nodes_observed[0].flag_observed):
                del self.nodes_observed[0]
            else:
                panju = False

    def __get_node_single(self,x_index,y_index):
        return self.nodes[y_index][x_index]

    def __get_index_low(self,ij):
        x_index_low = round((self.UAV_feiji.zuobiao.zhi[0,ij] - self.UAV_feiji.r_detect) /self.dL) -1 
        if x_index_low<0:
            x_index_low = 0    
        return x_index_low
    
    def __get_index_high(self,ij):
        x_index_high = round((self.UAV_feiji.zuobiao.zhi[0,ij] + self.UAV_feiji.r_detect) /self.dL) + 1
        if x_index_high>=len(self.nodes):
            x_index_high=len(self.nodes)-1
        return x_index_high
    # def generate_patrol_area1(self,**kargs):
    #     # total random strategy 
    #     # abolished, it might become something round.
    def generate_patrol_area(self,S_target=0,L_fanwei_min = 300.0,L_fanwei_max = 700.0,**kargs):
        # add locations randomly, untill the area meet the requirement

        # not randomly
        
        self.patrol_area = np.zeros((0,2)) 
        # first, get three points, sort, and check the area 
        node_xy = self.generate_patrol_first3nodes(L_fanwei_min = L_fanwei_min,L_fanwei_max = L_fanwei_max)
        S_all = self.get_area_from_nodes(node_xy) 
        # then add points 
        while S_all < S_target:
            extend_xy = node_xy
            extend_xy = np.append(extend_xy,extend_xy[0].reshape(1,2),axis=0)
            # get a point, 
            node_xy_add = np.random.uniform(L_fanwei_min,L_fanwei_max,(1,2))
            
            # decide its weizhi: totally jiaodu, it seem GGed. 
            # cos_jiaodu = np.array([]).reshape((0,1))
            # for i in range(len(node_xy)):
            #     cos_jiaodu_i = self.get_cos_nodes(extend_xy[i,:],extend_xy[i+1,:],node_xy_add)
            #     cos_jiaodu = np.append(cos_jiaodu,cos_jiaodu_i)
            # index_max = np.argmax(cos_jiaodu)

            # decide its weizhi: distance and jiaodu.
            jvli = np.array([]).reshape((0,1))
            for i in range(len(node_xy)):
                jvli_i = self.get_jvli_nodes(extend_xy[i,:],node_xy_add)
                jvli = np.append(jvli,jvli_i)
            index_min = np.argmin(jvli)
            # then jiaodu.
            cos1 = self.get_cos_nodes(extend_xy[index_min-1,:],extend_xy[index_min,:],node_xy_add)
            cos2 = self.get_cos_nodes(extend_xy[index_min,:],extend_xy[index_min+1,:],node_xy_add)
            if min(cos1,cos2)>-0.5:
                if (cos1>cos2):
                    # then, insert the point between index-1 and index
                    index = index_min-1
                else:
                    index = index_min
                self.patrol_area = node_xy
                # huatu = self.visual_patrol_area()
                # huatu.ax = huatu.plot_point(huatu.ax,node_xy_add[0],markersize=3)
                # huatu.show_figure()
                good_flag = self.check_edges(extend_xy,node_xy_add,index = index)
                # huatu = self.visual_patrol_area()
                # huatu.ax = huatu.plot_point(huatu.ax,node_xy_add[0],markersize=3)
                # huatu.show_figure()   
                if good_flag:
                    pass
                else:
                    continue
            else:
                continue

            # decide inside or outside 
            # index = index_min # index_max
            vector1 = extend_xy[index,:] - node_xy_add 
            vector2 = node_xy_add - extend_xy[index+1,:]
            chaji = np.cross(vector1,vector2)
            if chaji>=0:
                # good, this point is out of the existing shape
                node_xy = np.append(np.append(node_xy[0:index+1,:],node_xy_add,axis=0),node_xy[index+1:,:],axis=0)
                S_all = self.get_area_from_nodes(node_xy)
                
                # self.patrol_area = node_xy
                # huatu = self.visual_patrol_area()
                # huatu.show_figure()
                pass 
            else:
                # bad, this point is in the existing shape. there must be another one and nothing would happen here.
                pass

        self.patrol_area = node_xy

        return node_xy

    def generate_patrol_area2(self,S_target=0,L_fanwei_min = 300.0,L_fanwei_max = 700.0,**kargs):
        # add according to the points.
        self.patrol_area = np.zeros((0,2)) 
        # first, get three points, sort, and check the area 
        node_xy = self.generate_patrol_first3nodes(L_fanwei_min = L_fanwei_min,L_fanwei_max = L_fanwei_max)
        S_all = self.get_area_from_nodes(node_xy) 
        # then add points 
        while S_all < S_target:
            extend_xy = node_xy
            extend_xy = np.append(extend_xy,extend_xy[0].reshape(1,2),axis=0)
            # randome edge:
            index = np.random.randint(low=0,high=len(node_xy))
            # random bili:
            bili = np.random.uniform(0.0,1.0)
            # random changdu:
            changdu = np.random.uniform(0.1,0.5)*(L_fanwei_max+L_fanwei_min)/2
            # go.
            vector_bian = copy.deepcopy(extend_xy[index+1,:] - extend_xy[index,:])
            vector_bian = vector_bian.reshape(1,2)
            point_bian =  extend_xy[index,:] + vector_bian * bili 

            vector_chuizi = copy.deepcopy(np.fliplr(vector_bian))
            vector_chuizi[0,0] = vector_chuizi[0,0] * (-1) 
            vector_chuizi = vector_chuizi/np.linalg.norm(vector_chuizi)

            if np.cross(vector_bian,vector_chuizi)<0:
                # good direction 
                pass 
            else:
                # bad direction
                vector_chuizi = vector_chuizi *(-1)
            
            # then get the adding point
            node_xy_add = point_bian + changdu * vector_chuizi
            node_xy = np.append(np.append(node_xy[0:index+1,:],node_xy_add,axis=0),node_xy[index+1:,:],axis=0)
            S_all = self.get_area_from_nodes(node_xy) 
                       

            pass 
        node_xy = self.check_shape(node_xy)
        node_xy = self.scale_shape(node_xy,S_target)
        self.patrol_area = node_xy

        # renew the flag of nodes.
        for node_line in self.nodes:
            for node_i in node_line:
                node_i = self.get_neiwai_nodes(node_i,node_xy)
        # self.patrol_area = node_xy
        # huatu = self.visual_patrol_area()
        # huatu.show_figure()
        # flag_good = self.check_shape2(node_xy)
        if flag_good:
            return node_xy
        else:
            return [] 
        # return node_xy

    def generate_patrol_area2_gai(self,S_target=0,L_fanwei_min = 300.0,L_fanwei_max = 700.0,**kargs):
        node_xy = [] 
        while(len(node_xy)==0):
            node_xy = self.generate_patrol_area2(S_target=S_target,L_fanwei_min=L_fanwei_min,L_fanwei_max=L_fanwei_max,**kargs)
        self.patrol_area = node_xy
        huatu = self.visual_patrol_area()
        huatu.show_figure()
        flag_good = self.check_shape2(node_xy)            
        return node_xy

    def generate_patrol_first3nodes(self, L_fanwei_min = 300.0,L_fanwei_max = 700.0):
        node_xy = np.random.uniform(L_fanwei_min,L_fanwei_max,(3,2))
        e_x = np.array([1.0,0.0]) 
        cos_1 = self.get_cos_vector(node_xy[0,:]-node_xy[1,:],e_x)
        cos_2 = self.get_cos_vector(node_xy[0,:]-node_xy[2,:],e_x)
        if (cos_1>cos_2)and(node_xy[0,1]-node_xy[1,1]>=0):
            # good order. 
            pass
        elif (cos_1<cos_2)and(node_xy[0,1]-node_xy[2,1]<=0):
            # also good order
            pass
        elif (node_xy[0,1]-node_xy[1,1]<=0) and (node_xy[0,1]-node_xy[2,1]>=0):
            # also good order 
            pass
        else:
            # bad order, change
            temp = copy.deepcopy(node_xy[1,:])
            node_xy[1,:] = copy.deepcopy(node_xy[2,:])
            node_xy[2,:] = temp
        return node_xy

    def get_area_from_nodes(self,node_xy,**kargs):
        # https://zhuanlan.zhihu.com/p/110025234
        extend_xy = node_xy
        extend_xy = np.append(extend_xy,node_xy[0].reshape(1,2),axis=0)
        geshu = len(node_xy)
        he = 0 
        for i in range(geshu):
            he =he + extend_xy[i,0]*extend_xy[i+1,1] - extend_xy[i,1]*extend_xy[i+1,0]

        S_all = 1/2.0 * np.abs(he) 
        return S_all

    def get_cos_nodes(self,node_i,node_ijia1,node_xy_add,**kargs):
        vector1 = node_xy_add - node_i 
        vector2 = node_ijia1 - node_xy_add
        cos_jiaodu = self.get_cos_vector(vector1,vector2)
        return cos_jiaodu
    def get_cos_vector(self,vector1,vector2):
        cos_jiaodu = np.dot(vector1,vector2.T)/np.linalg.norm(vector1)/np.linalg.norm(vector2)
        return cos_jiaodu        
    def get_jvli_nodes(self,node_i,node_xy_add,**kargs):
        vector = node_i - node_xy_add
        jvli = np.linalg.norm(vector)
        return jvli

    def check_edges(self,extend_xy,node_xy_add,**kargs):
        geshu = len(extend_xy)-1
        index = kargs['index']
        good_flag = True  
        # for i in range(geshu):
        #     for j in range(geshu):
        #         if (j!=i) and (j!=i+1):
        #             notgood_flag_i = self.check_edge_sinlge( extend_xy[i,:], extend_xy[i+1,0],node_xy_add,extend_xy[j,:])
        #             good_flag = good_flag and not(notgood_flag_i) 
        for i in range(geshu):
            if (i!=index) and (i!=index+1):
                notgood_flag_i = self.check_edge_sinlge( extend_xy[i,:], extend_xy[i+1,:],node_xy_add,extend_xy[index,:])
                notgood_flag_ijia1 = self.check_edge_sinlge( extend_xy[i,:], extend_xy[i+1,:],node_xy_add,extend_xy[index+1,:])
                good_flag = good_flag and not(notgood_flag_i) and not(notgood_flag_ijia1)
        return good_flag

    def check_edge_sinlge(self,node_i,node_ijia1,node_xy_add,node_xy_add_i,**kargs):
        # cross each other.
        vector1 = node_ijia1 - node_i
        vector2 = node_xy_add - node_xy_add_i

        # one side: 
        vector_panju11 = node_xy_add - node_i
        vector_panju12 = node_xy_add_i - node_i
        zuobian1 = np.cross(vector1,vector_panju11)
        youbian1 = np.cross(vector1,vector_panju12)

        # the other side:
        vector_panju21 = node_ijia1 - node_xy_add_i
        vector_panju22 = node_i - node_xy_add_i 
        zuobian2 = np.cross(vector2,vector_panju21)
        youbian2 = np.cross(vector2,vector_panju22)

        if (zuobian1*youbian1<-0.001) and (zuobian2*youbian2<-0.001):
            # which means cross. 
            return True
        else:
            return False

    def check_shape(self,node_xy):
        # this is to avoid concave

        n_removed = 114514
        while(n_removed>0):
            n_removed = 0 
            extend_xy = node_xy
            extend_xy = np.append(extend_xy,extend_xy[0].reshape(1,2),axis=0)  
            geshu = len(node_xy)  
            node_xy2 = np.array([]).reshape(0,2)
            vector1 = extend_xy[0,:] - extend_xy[-1,:]
            for i in range(geshu):
                vector2 = copy.deepcopy(vector1)
                vector1 = extend_xy[i+1,:] - extend_xy[i,:]
                cross_zhi = np.cross(vector2,vector1)
                if cross_zhi>=0:
                    # then it is zhengchang, nothing would happen.
                    node_xy2 = np.append(node_xy2,extend_xy[i,:].reshape(1,2),axis=0)
                else:
                    # then remove the point.
                    n_removed = n_removed+1
                    pass
            node_xy = copy.deepcopy(node_xy2) 
            
        return node_xy2

    def check_shape2(self,node_xy):
        # this is to avoid concave, again. 
        extend_xy = node_xy
        extend_xy = np.append(extend_xy,extend_xy[0].reshape(1,2),axis=0)  
        geshu = len(node_xy)  
        vector1 = extend_xy[0,:] - extend_xy[-1,:]
        jiaodu_accumulate = 0 
        for i in range(geshu):
            vector2 = copy.deepcopy(vector1)
            vector1 = extend_xy[i+1,:] - extend_xy[i,:]          
            chengji1 = vector1.dot(vector2) 
            yuxian = chengji1/vector2.abs()/vector1.abs()/1.0
            cross_zhi = np.cross(vector2,vector1)
            if cross_zhi>=0:
                jiaodu = np.arccos(yuxian)
            else:
                jiaodu = np.arccos(yuxian) * (-1.0)
            jiaodu_accumulate = jiaodu_accumulate + jiaodu
        
        if jiaodu_accumulate>np.pi*2 + 0.1:
            # which means there are more than one circle
            flag_good = False
        else: 
            flag_good = True
        return flag_good

    def scale_shape(self,node_xy,S_target):
        S_all = self.get_area_from_nodes(node_xy)
        bili = (S_target / S_all )**0.5

        zhongxin = np.mean(node_xy,axis=0)
        node_xy_new = copy.deepcopy(node_xy) 
        geshu = len(node_xy)
        for i in range(geshu):
            vector = node_xy[i,:] - zhongxin
            vector_scaled = vector * bili 
            node_xy_new[i,:] = zhongxin + vector_scaled
        return node_xy_new

    def visual_patrol_area(self,**kargs):
        if 'huatu' in kargs:
            huatu = kargs['huatu']
        else:
            huatu = huatu_support()
            huatu.ax = huatu.plot_coordinate(huatu.ax,x_min=0,x_max=self.L_x,y_min=0,y_max=self.L_y)

        huatu.ax = huatu.plot_Polygon(huatu.ax,self.patrol_area,yanse=huatu.get_color(index = 3))
        # huatu.ax = huatu.plot_line_2D(huatu.ax,self.patrol_area[:,0],self.patrol_area[:,1])
        
        return huatu

    def get_neiwai_nodes(self,node_i,node_xy) :
        extend_xy = node_xy
        extend_xy = np.append(extend_xy,extend_xy[0].reshape(1,2),axis=0)  
        geshu = len(node_xy)
        flag_inside = True
        for i in range(geshu):
            vector1 = extend_xy[i+1,:] - extend_xy[i,:]
            vector2 = node_i.zuobiao.zhi -  extend_xy[i,:]
            cross1 =  np.cross(vector1,vector2)
            if cross1>0:
                flag_inside = flag_inside and True
                node_i.flag_inside = flag_inside
            else:
                node_i.flag_inside = False
                break 
        return node_i

    def get_nodes_array(self):
        L_x = self.L_x
        L_y = self.L_y
        dL = self.dL
        hangshu =  int(L_x/dL) +1
        flag_observed_array = np.zeros((hangshu,hangshu),dtype=int)
        flag_inside_array = np.zeros((hangshu,hangshu),dtype=int)

        for index_x in range(hangshu):
            for index_y in range(hangshu):
                node = self.__get_node_single(index_x,index_y)
                flag_observed_array[index_x,index_y] = node.flag_observed
                flag_inside_array[index_x,index_y] =node.flag_inside

        return flag_observed_array, flag_inside_array

    def get_UAV_location(self):
        location = self.UAV_feiji.zuobiao.zhi
        return location
    
    def get_UAV_direction(self):
        sudu = self.UAV_feiji.sudu.zhi
        sudu_norm = sudu / np.linalg.norm(sudu)
        return sudu_norm

    def get_UAV_omega(self):
        omega = self.UAV_feiji.omega
        return omega        

    def check_UAV(self):
        # check the location of UAV and avoid it go outside the BattleField 
        flag = True
        flag = flag and (self.UAV_feiji.zuobiao.zhi > (0-self.dL))
        flag = flag and (self.UAV_feiji.zuobiao.zhi < (self.L_x+self.dL))
        flag = flag and (self.UAV_feiji.zuobiao.zhi > (0 - self.dL))


class BattleField_node(object):
    node_number = 0
    time_now = 0

    @staticmethod
    def running_clock(dt):
        BattleField_node.time_now = BattleField_node.time_now + dt 

    def __init__(self,zuobiao,**kargs):
        BattleField_node.node_number =  BattleField_node.node_number + 1
        self.index = BattleField_node.node_number
        self.zuobiao = zuobiao
        self.time_interval = 1 
        self.time_detected_list = [0.0] 
        self.flag_observed = False
        self.flag_inside = False 

    def running(self,UAV_feiji,**kargs):
        jvli_vector = self.zuobiao.zhi - UAV_feiji.zuobiao.zhi
        jvli_abs = np.linalg.norm(jvli_vector) 

        if jvli_abs<UAV_feiji.r_detect:
            # which means this node is under observation
            self.time_detected_list.append(BattleField_node.time_now)
            self.flag_observed = True
        else:
            # which means this node is not under observation now
            time_after_observed = BattleField_node.time_now - self.time_detected_list[-1]
            if time_after_observed > self.time_interval:
                self.flag_observed = False

    def running_simple(self):
        time_after_observed = BattleField_node.time_now - self.time_detected_list[-1]
        if time_after_observed > self.time_interval:
            self.flag_observed = False 

    def visual_node(self,huatu):
        if self.flag_observed==True:
            yanse = huatu.get_color(index = 0)
        else:
            yanse = huatu.get_color(index = 1)
        huatu.ax = huatu.plot_point(huatu.ax,self.zuobiao.zhi[0],yanse=yanse,marker='*',markersize=1)
        return huatu
    
if __name__ == '__main__':
    location = r"E:\EnglishMulu\UAV-Patrol\BattleField" 
    shishi1 = UAV(location=np.array([100,100]), sudu_max=114514, omega_max = 1,r=100)
    shishi1.set_chuzhi(sudu_0=np.array([500,0]),a_0=0,omega_0=0,dt = 0.1)
    shishi_BattleField = BattleField(L_x=1000,L_y=1000)
    shishi_BattleField.UAV_online(UAV_feiji = shishi1)
    shishi_BattleField.generate_patrol_area2_gai(S_target=160000,L_fanwei_min = 100.0,L_fanwei_max = 900.0)

    shishi_BattleField.running(d_omega=1,d_a=0,first_step=True)
    shishi_BattleField.running(d_omega=1,d_a=0)
    # shishi_BattleField.running_mul(d_omega=1,d_a=0)

    flag_observed_array, flag_inside_array = shishi_BattleField.get_nodes_array() 
    
    huatu = shishi_BattleField.visual_BattleField()
    huatu = shishi_BattleField.visual_patrol_area(huatu=huatu)
    
    huatu.save_all(location=r"E:\EnglishMulu\UAV-Patrol\BattleField",name='shishi')

    # for i in range(30):
    #     huatu = shishi_BattleField.visual_BattleField()
    #     name = 'shishi'+str(i)
    #     huatu.save_all(location=location,name=name) 
    #     shishi_BattleField.running(d_omega=1,d_a=0)

    # name_list = [] 
    # for i in range(30):
    #     name = r'\shishi'+str(i)+'.png' 
    #     name_list.append(r'E:\EnglishMulu\UAV-Patrol\BattleField\void'+name)
    # huatu.generate_gif(location=location+r'\void',image_list=name_list)   

