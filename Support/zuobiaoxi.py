# this is a zuobiaoxi for UAV-Patrol
WEIZHI =r'E:/EnglishMulu/UAV-Patrol'
import numpy as np
import sys 
sys.path.append(WEIZHI+r'/Support')
from huatu_support import huatu_support

class zuobiaoxi(object):
    zuobiaoxi_index = 0-1-1
    def __init__(self,dim=2,Po = np.array([0,0]),name = 'undefined',**kargs):
        zuobiaoxi.zuobiaoxi_index=zuobiaoxi.zuobiaoxi_index+1
        self.dim = dim 
        self.e_vector = []
        self.e_array = np.zeros((dim,dim))  
        self.Po = Po
        self.name = name
        self.index = zuobiaoxi.zuobiaoxi_index
        if dim == 2:
            self.initiate_2D(**kargs)
        elif dim == 3:
            self.initiate_3D(**kargs)
        else:
            raise Exception("zuobiaoxi: invalid dimension number")
        
        flag = np.dot(self.e_vector[0],self.e_vector[1])
        if abs(flag)>0.00000001:
            raise Exception("zuobiaoxi: invalid base vector, G!")    
    def initiate_2D(self,**kargs):
        if 'e_vector' in kargs:
            self.e_vector = kargs['e_vector']
            for i in range(self.dim):
                self.e_vector[i] = self.e_vector[i] / np.linalg.norm(self.e_vector[i])
            self.e_array[:,0] = self.e_vector[0]
            self.e_array[:,1] = self.e_vector[1]
        else:
            e1 = np.array([1,0])
            e2 = np.array([0,1])
            self.e_vector.append(e1)
            self.e_vector.append(e2)
            self.e_array[:,0] = e1 
            self.e_array[:,1] = e2  
    
    def initiate_3D(self,**kargs):
        if 'e_vector' in kargs:
            self.e_vector = kargs['e_vector']
            for i in range(self.dim):
                self.e_vector[i] = self.e_vector[i] / np.linalg.norm(self.e_vector[i])
            self.e_array[:,0] = self.e_vector[0] 
            self.e_array[:,1] = self.e_vector[1] 
            self.e_array[:,2] = self.e_vector[2] 
        else:
            e1 = np.array([1,0,0])
            e2 = np.array([0,1,0])
            e3 = np.array([0,0,1])
            self.e_vector.append(e1)
            self.e_vector.append(e2)
            self.e_vector.append(e3)
            self.e_array[:,0] = e1 
            self.e_array[:,1] = e2  
            self.e_array[:,2] = e3  
        
    def visual_coordinate(self,**kargs):
        if 'huatu' in kargs:
            huatu = kargs['huatu']
        else:
            huatu = huatu_support()
            huatu.ax = huatu.plot_coordinate(huatu.ax)
        yanse = huatu.get_color(index=self.index)
        if self.dim==2:
            huatu.ax = huatu.plot_arrow(huatu.ax,self.Po,self.e_vector[0],edgecolor=yanse)
            huatu.ax = huatu.plot_arrow(huatu.ax,self.Po,self.e_vector[1],edgecolor=yanse)
            # huatu.show_figure() 
            # huatu.save_all(location=r"E:\EnglishMulu\UAV-Patrol\Support")
        return huatu             

    def transfer(self,zuobiao_this,new_zuobiaoxi):
        # function zhi = transfer(obj,vector,new)
        # zhi = vector *obj.e_vector * new.e_vector' ; 
        zhi_this = zuobiao_this.zhi
        zhi_absolute = np.matmul(zhi_this,self.e_array)
        zhi_new = np.matmul(zhi_absolute,new_zuobiaoxi.e_array.T)
        
        zuobiao_new =zuobiao(zhi=zhi_new,zuobiaoxi=new_zuobiaoxi)
        return zuobiao_new

class zuobiao(object):
    # this is a yangjian zuobiao
    def __init__(self,zhi=np.array([0,0]),zuobiaoxi = zuobiaoxi(dim=2), **kargs):
        # declaration 'zuobiaoxi = zuobiaoxi(dim=2)' would make the zuobiaoxi_index +1
        self.zhi = zhi 
        self.zuobiaoxi=zuobiaoxi
        if len(zhi) == 1:
            pass
        elif len(zhi) == zuobiaoxi.dim:
            self.zhi=self.zhi.reshape(1,zuobiaoxi.dim)
        else:
            raise Exception('zuobiao: invalid dimension')
    
    def sum(self,zuobiao2):
        # vector add in the same coordinate.
        # zuobiao.sum(zuobiao2): zuobiao = zuobiao + zuobiao2
        zuobiao2_this = zuobiao2.zuobiaoxi.transfer(zuobiao2,self.zuobiaoxi)
        self.zhi = self.zhi + zuobiao2_this.zhi
        return self

    def dot(self,zuobiao2):
        # vector add in the same coordinate.
        # zuobiao.dot(zuobiao2): jieguo = zuobiao * zuobiao2
        if type(zuobiao2)==zuobiao:
            zuobiao2_this = zuobiao2.zuobiaoxi.transfer(zuobiao2,self.zuobiaoxi)
            zuobiao2_this_zhi = zuobiao2_this.zhi
        else:
            zuobiao2_this_zhi = zuobiao2
        jieguo = np.dot(self.zhi,zuobiao2_this_zhi)      
        return jieguo 

    def abs(self):
        jieguo = np.linalg.norm(self.zhi)
        return jieguo

    def visual_zuobiao_vector(self,**kargs):

        zhi_absolute = np.matmul(self.zhi,self.zuobiaoxi.e_array)
        if 'huatu' in kargs:
            huatu = kargs['huatu']
        else:
            huatu = huatu_support()
            huatu.ax = huatu.plot_coordinate(huatu.ax) 
        yanse = huatu.get_color(index=self.zuobiaoxi.index)       
        huatu = self.zuobiaoxi.visual_coordinate(huatu=huatu)
        huatu.ax = huatu.plot_arrow(huatu.ax,self.zuobiaoxi.Po,zhi_absolute,edgecolor=yanse)
        return huatu       
    
    def get_zhi(self,zhi_new,**kargs):
        self.zhi = zhi_new

if __name__ == '__main__':
    # this is to test zuobiaoxi
    e1 = np.array([1,1])
    e2 = np.array([1,-1])
    Po = np.array([5,5])
    e_list = [] 
    e_list.append(e1)
    e_list.append(e2)

    shishi1 = zuobiaoxi(dim=2,e_vector=e_list,Po=Po,name='test')
    # huatu = shishi1.visual_coordinate()
    
    shishi2 = zuobiaoxi(dim=2) 
    # huatu = shishi2.visual_coordinate(huatu=huatu)
    
    shishi1_vector_zhi=np.array([1,1])
    shishi1_vector = zuobiao(zhi=shishi1_vector_zhi,zuobiaoxi=shishi1)
    huatu = shishi1_vector.visual_zuobiao_vector()

    shishi2_vector = zuobiao(zhi=shishi1_vector_zhi,zuobiaoxi=shishi2)
    huatu = shishi2_vector.visual_zuobiao_vector(huatu=huatu)

    shishi2_vector.sum(shishi1_vector)
    huatu = shishi2_vector.visual_zuobiao_vector(huatu=huatu)

    huatu.save_all(location=r"E:\EnglishMulu\UAV-Patrol\Support")
    print("zuobiaoxi: finish shishi")
