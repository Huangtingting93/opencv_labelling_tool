import matplotlib.pyplot as plt
import numpy as np
import cv2

def show(img,pts_list=[],msize=10,color=None, show=True,show_label=False):
    imgcp = img.copy()
    RED = (255,0,0)
    GREEN = (0,255,0)
    BLUE = (0,0,255)
    BLACK = (0,0,0)
    YELLOW = (0,255,255)
    colors_3channel = [RED,BLUE,GREEN,BLACK,YELLOW]
    colors_1channel = [5,10,15,20,25]
    colors  = ['r.','g.','c.','b.']
    if pts_list is None:
        return img
    if isinstance(pts_list,list):
        pts_num = len(pts_list)
    elif isinstance(pts_list,np.ndarray):
        pts_num = 1
        pts_list = [pts_list]
    
    if len(img.shape) == 1 or (len(img.shape)>1 and img.shape[1] == 2):
        msize = msize//2
        img = img.reshape(-1,2)
        pts_list.append(img)
        for i,pts in enumerate(pts_list):
            msize = max(msize-1,1)
            pts = pts.reshape(-1,2)
            color = colors[i%len(colors)]
            plt.plot(pts[:,0],pts[:,1],color,markersize=msize)
            if show_label:
                chunk_i = max(1,int(pts.shape[0]/100)) #max 100 loop
                for k,pt in enumerate(pts):
                    if k%chunk_i == 0:
                        plt.text(pt[0],pt[1],str(k))
        plt.show()
    elif img.shape[1] > 2:
        for i,pts in enumerate(pts_list):    
            if len(img.shape) > 2: # 3channel
                if color is None:
                    color =  colors_3channel[i%len(colors_3channel)]
            else: # mask 
                color = colors_1channel[i%len(colors_1channel)]
            if pts is not None:
                pts = pts.astype(int)
                pts = pts.reshape(-1,2)
                for pt in pts:
                    cv2.circle(imgcp, (pt[0], pt[1]), 0, color, msize)
                color = None
                if show_label:
                    chunk_i = max(1,int(pts.shape[0]/100)) #max 100 loop
                    for k,pt in enumerate(pts):
                        if k%chunk_i == 0:
                            plt.text(pt[0],pt[1],str(k))
        if show:
            plt.imshow(imgcp),plt.show()
        return imgcp
     