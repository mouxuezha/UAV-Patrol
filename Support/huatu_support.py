# this is a new huatu for UAV-Patrol
import matplotlib.pyplot as plt
import os 
import numpy as np 
from matplotlib.font_manager import FontProperties
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import (OffsetImage,AnnotationBbox)

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon

class huatu_support():
    def __init__(self,title='void',**kargs):
        fig, ax = plt.subplots()
        self.fig = fig 
        self.ax = ax 
        self.set_chicun(self.fig)
        self.title = title
        self.zihao = 12

        self.flag_zhongwen = False
        self.zhongwen_font = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc',size=self.zihao)
        self.legend_list = [] 
        
    def set_chicun(self,fig,**kargs):
        bili = 0.397 # transfer from cm into inches.

        if 'width' in kargs:
            fig.set_figwidth(kargs['width']*bili)
        else:
            
            fig.set_figwidth(7.4*bili)

        if 'height' in kargs :
            fig.set_figheight(kargs['height']*bili)
        else :
            fig.set_figheight(7.4*bili)
        
        if 'adjust' in kargs:
            shuzi = kargs['adjust']
            plt.subplots_adjust(bottom=shuzi[0], right=shuzi[1], left = shuzi[2],top=shuzi[3])
        else:
            plt.subplots_adjust(bottom=0.15, right=0.95, left = 0.15,top=0.95)

    def save_all(self,location=None,model='png',name='void'):
        #make a folder for saveing the data and figure
        wenjianjia = location + '/' + self.title
        try:
            os.mkdir(wenjianjia)
        except:
            print('huatu_UAV: wenjianjia already there.')
        if model == 'png':
            wenjianming_tu = wenjianjia + '/' + name + '.png'
        elif model == 'gif':
            pass 
        self.fig.savefig(wenjianming_tu,dpi=1200)
    
    def plot_arrow(self,ax,Po,vector,facecolor='k', edgecolor='k',**kargs):
        # this is to draw an arrow
        if len(vector)==1:
            vector = vector.reshape(len(Po),)
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.arrow.html?highlight=axes%20arrow#matplotlib.axes.Axes.arrow

        # dx = vector[0]-Po[0]
        # dy = vector[1]-Po[1]
        dx = vector[0]
        dy = vector[1]
        ax.arrow(Po[0],Po[1],dx,dy,head_width=0.05, head_length=0.1, fc=facecolor, ec=edgecolor)
        self.ax = ax 
        return ax

    def plot_coordinate(self,ax,x_min=0,x_max=10,y_min=0,y_max=10,zihao=12,x_name="x",y_name="y"):
        # this is for biankuang

        x_changdu = x_max - x_min
        y_changdu = y_max - y_min

        x_one_grid = self.get_one_grid(0.2*x_changdu)
        if abs(x_one_grid)>0.51:
            x_one_grid = round(x_one_grid)
        y_one_grid = self.get_one_grid(0.2*y_changdu)
        if abs(y_one_grid)>0.51:
            y_one_grid = round(y_one_grid)

        x_offset = x_min % x_one_grid
        y_offset = y_min% y_one_grid

        ax.set_xlim(x_min-x_offset-0.05*x_changdu, x_max+0.05*x_changdu)
        # ax.set_xlim(x_min-x_offset-0.5*x_one_grid, x_max+0.05*x_changdu)
        ax.set_ylim(y_min-y_offset-0.05*y_changdu, y_max+0.05*y_changdu)
        # ax.set_ylim(y_min-y_offset, y_max+0.05*y_changdu)
        x_label = np.arange(x_min-x_offset, x_max,x_one_grid)
        y_label = np.arange(y_min-y_offset,y_max,y_one_grid)

        ax.set_xticks(x_label)
        ax.set_yticks(y_label)
        
        if self.flag_zhongwen:
            prop = self.zhongwen_font
            xylabel_prop = self.zhongwen_font
        else: 
            prop = {"family" : "Times New Roman" ,'size' : str(zihao-2)}
            xylabel_prop = "Times New Roman"
        
        ax.set_xlabel(x_name,fontsize=zihao,fontproperties=xylabel_prop)
        ax.set_ylabel(y_name,fontsize=zihao,fontproperties=xylabel_prop)

        plt.yticks(fontproperties = 'Times New Roman', size = zihao-2)
        plt.xticks(fontproperties = 'Times New Roman', size = zihao-2) 
 
        ax.tick_params(axis='y', labelcolor='k',labelsize=zihao-2)
        ax.tick_params(axis='x', labelcolor='k',labelsize=zihao-2)

        self.ax = ax 
        return ax      

    def get_one_grid(self,buchang):
        # make the ireegular buchang to be regular, for example 114.514 to 100;
        # baoli chabiao
        for i in range(-3,5,1):
            if (buchang>(10**i)) & (buchang<(10**i*1.5)):
                buchang = 10**i 
                break 
            elif (buchang>(10**i*1.5)) & (buchang<10**i*3.5):
                buchang = 10**i*2
                break
            elif (buchang>(10**i*3.5)) & (buchang<10**i*7.5):
                buchang = 10**i*5
                break
            elif (buchang>(10**i*7.5)) & (buchang<10**i*10):
                buchang = 10**i*10
                break
        
        if abs(buchang)<0.000001:
            buchang = 0.5 
        
        return buchang
 
    def show_figure(self):
        plt.show()

    def get_color(self,index=0,**kargs):
        self.yanse = ['C0','C1','C2','C3','C4','C5','C6']
        self.xianxing = ['solid','dotted','dashed','dashdot']
        self.biaozhi = ['o','s','D','^','v','.','*']    

        return self.yanse[index%len(self.yanse)]    

    def plot_image(self,ax,im_location,xy=(0.5,0.5),zoom=0.1,rotate = np.pi/4,**kargs):
        
        # https://matplotlib.org/stable/gallery/text_labels_and_annotations/demo_annotation_box.html
        # good good look, good good learn
        with get_sample_data(im_location) as file:
            arr_img = plt.imread(file)
        
        img = self.xuanzhuan_image(im_location,rotate=rotate)

        # ax.imshow(arr_img, transform=tr) 
        # plt.show()
        imagebox = OffsetImage(img, zoom=zoom)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, xy,
                            xybox= xy,
                            xycoords='data',
                            # boxcoords="offset points",
                            pad=0.0,
                            frameon=False,
                            arrowprops=dict(
                                arrowstyle="-",
                                connectionstyle="arc3",ec='w',fc='w',)
                            )

        ax.add_artist(ab)
        # plt.show()
        self.ax = ax 
        return ax
    
    def xuanzhuan_image(self,im_location,rotate = np.pi/4):
        degree = rotate/np.pi*180 
        im = Image.open(im_location)
        im = im.rotate(degree) # Rotates counter clock-wise.
        # implot = plt.imshow(im)
        return im 
        # plt.show()
    
    def plot_circle(self,ax,Po,r):
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Circle.html#matplotlib.patches.Circle

        circle = Circle((Po[0], Po[1]), r,facecolor='None',edgecolor='k',linestyle=':',linewidth=0.5)
        ax.add_artist(circle)
        # plt.show()
        self.ax = ax 
        return ax
    def plot_line_2D(self,ax,x,y,yanse='k',linewidth=0.5,label=None,linestyle='--',marker=None,markersize=2,**kagrs):
        ax.plot(x, y,color=yanse,linewidth=linewidth,label=label,linestyle=linestyle,marker = marker,markersize = markersize)
        return ax 
    def plot_point(self,ax,zuobiao_zhi,yanse='k',marker='*',markersize=1,**kargs):
        ax.plot(zuobiao_zhi[0],zuobiao_zhi[1],color=yanse,marker = marker,markersize = markersize,markeredgewidth=0.1,linestyle='None')
        return ax

    def generate_gif(self,location,**kargs):
        # transfer pictures in one folder to one gif and save.
        import imageio 
        
        if 'image_list' in kargs:
            image_list = kargs['image_list']
        else:
            image_list = [location +'/'+ img for img in os.listdir(location)]

        shishi = [] 
        if 'endswith' in kargs:
            enddwith = kargs['endswith']
        else:
            enddwith = '.png'
        for image_name in image_list:
            if image_name.endswith(enddwith):
                print('mxairfoil: read picture, '+image_name)
                shishi.append(imageio.imread(image_name))
        
        if 'name' in kargs:
            name = location + '/'+kargs['name']+'.gif'
        else:
            name = location + '/dynamic_tu.gif'

        if 'duration' in kargs:
            duration = kargs['duration']
        else:
            duration = 0.1

        imageio.mimsave(name,shishi,'GIF',duration=duration)

        return
    
    def plot_Polygon(self,ax,xy,yanse='k',**kargs):
        duobianxing = Polygon(xy, closed=True,alpha=0.2,color=yanse,linestyle='None')
        ax.add_artist(duobianxing)
        # plt.show()
        self.ax = ax 
        return ax        
