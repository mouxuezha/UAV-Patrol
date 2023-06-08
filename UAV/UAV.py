# this is UAV itself.

WEIZHI =r'E:/EnglishMulu/UAV-Patrol'
import numpy as np
import sys 
sys.path.append(WEIZHI+r'/support')
from huatu_support import huatu_support
from zuobiaoxi import zuobiaoxi
from zuobiaoxi import zuobiao

class UAV(object):
    abs_coordinate = zuobiaoxi(dim=2,name='abs_coordinate')
    location_list = np.array([]).reshape((0,2)) 
    def __init__(self,location=np.array([5,1]),sudu_max=114514,omega_max = 1,r=1,a_max=1 , **kargs):
        self.coordinate = zuobiaoxi(dim=2,Po=location,name='UAV_coordinate') 
        self.zuobiao = zuobiao(zhi=location,zuobiaoxi=self.coordinate)
        self.sudu = zuobiao(zhi=np.array([1,1]),zuobiaoxi=self.coordinate) # m/s
        self.omega_v_max = omega_max # rad/s for sudu, nishizheng is positive.
        self.a_max = a_max # m/(s^2) for sudu
        self.sudu_max = sudu_max 
        self.omega = 0 
        self.a = 0 
        self.time = 0 
        self.dt = 0.1 
        self.theta = self.get_theta_from_sudu(self.sudu) # rad

        self.r_detect = r # m detected distance.
        self.picture = r'E:\EnglishMulu\UAV-Patrol\UAV\UAV-sample.png'

    def set_chuzhi(self,sudu_0=np.array([1,1]),a_0=0,omega_0=0,dt = 0.1):
        self.sudu = zuobiao(zhi=sudu_0,zuobiaoxi=self.coordinate) # m/s
        self.omega = omega_0 
        self.a = a_0
        self.dt = dt
        self.theta = self.get_theta_from_sudu(self.sudu) # rad

    def visual_UAV(self,**kargs):
        if 'huatu' in kargs:
            huatu = kargs['huatu']
        else:
            huatu = huatu_support(title='UAVtest')
            huatu.ax = huatu.plot_coordinate(huatu.ax)
        yanse = huatu.get_color(index=self.coordinate.index) 
        # zuobiao_abs = self.coordinate.transfer(self.zuobiao,UAV.abs_coordinate)
        # x = zuobiao_abs.zhi[0][0] - 0.3
        # y = zuobiao_abs.zhi[0][1] - 0.1
        x = self.coordinate.Po[0] - 0.3
        y = self.coordinate.Po[1] - 0.1
        # rotate = -np.pi/4
        # rotate = self.get_theta_from_sudu(self.sudu)
        rotate = self.theta

        huatu.ax = huatu.plot_image(huatu.ax,im_location=self.picture,xy=(x,y),zoom=0.02,rotate=rotate)
        huatu.ax = huatu.plot_circle(huatu.ax,self.zuobiao.zhi[0],self.r_detect) 
        if len(UAV.location_list)>0:
            huatu.ax = huatu.plot_line_2D(huatu.ax,UAV.location_list[:,0],UAV.location_list[:,1])
        return huatu

    def get_theta_from_sudu(self,sudu,**kargs):
        # this is to get theta from zuobiao:sudu.
        chengji1 = sudu.dot(UAV.abs_coordinate.e_vector[0]) 
        chengji2 = sudu.dot(UAV.abs_coordinate.e_vector[1]) 
        yuxian = chengji1/sudu.abs()/1.0
        theta = np.arccos(yuxian)

        if chengji2<0:
            theta = theta*(-1) 

        return theta # rad

    def time_goes_by(self,d_omega=0,d_a=0,**kargs):

        self.zuobiao.zhi = self.zuobiao.zhi + self.dt*self.sudu.zhi
        self.coordinate.Po = self.zuobiao.zhi[0] 
        UAV.location_list = np.append(UAV.location_list,self.zuobiao.zhi,axis=0)

        abs_sudu = self.xianzhi(self.sudu.abs() + self.dt * self.a,self.sudu_max)
        self.a =  self.xianzhi(self.a  + self.dt * d_a,self.a_max)

        self.theta = self.theta+self.omega*self.dt
        if self.theta[0]>np.pi:
            self.theta = self.theta - 2*np.pi
        elif self.theta[0]<(-np.pi):
            self.theta = self.theta + 2*np.pi  

        self.omega = self.xianzhi(self.omega + self.dt*d_omega, self.omega_v_max)

        sudu_zhi_new = np.array([abs_sudu * np.cos(self.theta) * 1.0,
                                 abs_sudu * np.sin(self.theta) * 1.0],dtype=float).reshape((1,2))
        # self.sudu.zhi[0][0] = abs_sudu * np.cos(self.theta) * 1.0
        # self.sudu.zhi[0][1] = abs_sudu * np.sin(self.theta) * 1.0
        self.sudu.get_zhi(sudu_zhi_new)

    def xianzhi(self,liang,liang_max):
        if liang>liang_max:
            liang = liang_max
        elif liang<(liang_max*-1):
            liang = liang_max*-1
        return liang
    
if __name__ == '__main__':
    location = r"E:\EnglishMulu\UAV-Patrol\UAV" 
    shishi1 = UAV(location=np.array([5,1]), sudu_max=114514, omega_max = 1,r=1)
    shishi1.set_chuzhi(sudu_0=np.array([5,0]),a_0=0,omega_0=5,dt = 0.1)
    huatu = shishi1.visual_UAV()
    # huatu = shishi1.sudu.visual_zuobiao_vector(huatu=huatu)
    # print(shishi1.get_theta_from_sudu(shishi1.sudu))
    # huatu.save_all(location=r"E:\EnglishMulu\UAV-Patrol\UAV")
    # shishi1.time_goes_by(d_omega=1,d_a=1)
    # huatu = shishi1.visual_UAV()
    # huatu.save_all(location=r"E:\EnglishMulu\UAV-Patrol\UAV",name='void2')
    # for i in range(30):
    #     huatu = shishi1.visual_UAV()
    #     name = 'shishi'+str(i)
    #     huatu.save_all(location=location,name=name) 
    #     shishi1.time_goes_by(d_omega=1,d_a=0)    
    name_list = [] 
    for i in range(30):
        name = r'\shishi'+str(i)+'.png' 
        name_list.append(location+r'\UAVtest'+name)
    huatu.generate_gif(location=location+r'\UAVtest',image_list=name_list)   