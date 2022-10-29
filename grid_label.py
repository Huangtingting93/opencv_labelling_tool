from builtins import breakpoint
from tkinter import N
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
import string    
from util import show
from pprint import pprint

X_INIT_NUM = 5
Y_INIT_NUM = 5
OPT_EXTEND = 'extend grid'
OPT_MOVE = 'move grid'
OPT_X_NUM = 'grid x num'
OPT_Y_NUM = 'grid y num'
OPT_CIRCLE_SIZE= 'circle size'
OPT_DOT_DETECT_THRESHOD = "dot threshold"
OPT_CORNER = "adjust corner"

def transfrom(points, M):
    points_one = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed = (M @ points_one.T).T
    transformed = transformed / transformed[:, 2, np.newaxis]
    transformed = transformed[:, :2]
    return transformed


def detect_corners(img,t=0.02):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, t*dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(
        centroids), (5, 5), (-1, -1), criteria)
    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)

    dots = res[:, [0, 1]]
    # dots = np.vstack((res[:, [0, 1]], res[:, [2, 3]]))
    return dots

class Grid:
    def __init__(self) -> None:
        self.x_num = X_INIT_NUM
        self.y_num = Y_INIT_NUM
        self.x_gap = 1
        self.y_gap = 1
        self.x_max = self.x_num * self.x_gap
        self.y_max = self.y_num * self.y_gap
        self.x_offset = 0
        self.y_offset = 0
        self.lock_t = False # lock gap and offset 
        self.coords_x_offset = 0
        self.coords_y_offset = 0
        self.generate_grid()
    
    def __str__(self):
        return f'grid {self.x_num} x {self.y_num} ,size: {self.x_max} x {self.y_max} ,offset: {self.x_offset} ,{self.y_offset}'

    def lock(self):
        self.lock_t = True

    def unlock(self):
        self.lock_t = False

    def update_x_y_max(self):
        self.x_num = max(2,self.x_num)
        self.y_num = max(2,self.y_num)

        self.x_gap = max(1,self.x_gap)
        self.y_gap = max(1,self.y_gap)

        if not self.lock_t: # lock x_max and y_max
            self.x_max = self.x_num * self.x_gap
            self.y_max = self.y_num * self.y_gap


    def generate_grid(self):
        self.update_x_y_max()
        # x = np.linspace(0, self.x_max, self.x_num)
        # y = np.linspace(0, self.y_max, self.y_num)

        x = np.arange(0,self.x_max,self.x_gap).astype('float')
        y = np.arange(0,self.y_max,self.y_gap).astype('float')

  
        x_arr = np.repeat(x, self.y_num)
        y_arr = np.tile(y, self.x_num)
        
        # create plane points on xoy plane
        grid = np.vstack((x_arr, y_arr)).T
        grid[:,0] += self.x_offset
        grid[:,1] += self.y_offset
 
        self.grid_corner_ids = [0,self.y_num-1,self.x_num*self.y_num-1,(self.x_num-1)*self.y_num]
        grid_corners = grid[self.grid_corner_ids]
        if not self.lock_t: self.grid_corners = grid_corners
       
        faces = []
        idx = self.x_num * self.y_num
        idx = np.arange(idx).reshape(self.x_num, self.y_num)
        self.coords = [(i,j) for i in range(self.x_num) for j in range(self.y_num)]
        self.coords = np.asarray(self.coords).reshape(-1,2)
        self.coords[:,0] += self.coords_x_offset
        self.coords[:,1] += self.coords_y_offset 

        id_list = idx[:-1, :-1]
        for i in id_list.flatten():
            faces.append([i, i+self.y_num])
            faces.append([i, i+1])
            faces.append([i+1, i+1+self.y_num])
            faces.append([i+self.y_num, i+1+self.y_num])
        faces = np.asarray(np.unique(faces,axis=1))
        self.grid = grid
        
        self.grid_faces = faces
        return grid, self.grid_corners, faces,self.coords 

   
    def set_x_num(self,x):
        self.x_num = x
        self.update_x_y_max()
    
    def set_y_num(self,y):
        self.y_num = y
        self.update_x_y_max()

    def increase_x_num(self,n=1):
        self.x_num += n
        self.update_x_y_max()

    def decrease_x_num(self,n=1):
        self.x_num -= n
        self.update_x_y_max()

    def increase_y_num(self,n=1):
        self.y_num += n
        self.update_x_y_max()

    def decrease_y_num(self,n=1):
        self.y_num -= n
        self.update_x_y_max()

    def increase_x_gap(self,gap=1):
        if not self.lock_t:self.x_gap += gap
        self.update_x_y_max()

    def decrease_x_gap(self,gap=1):
        if not self.lock_t:self.x_gap -= gap
        self.update_x_y_max()

    def increase_y_gap(self,gap=1):
        if not self.lock_t:self.y_gap += gap
        self.update_x_y_max()

    def decrease_t_gap(self,gap=1):
        if not self.lock_t: self.t_gap -= gap
        self.update_x_y_max()

    def increase_x_offset(self,offset=0.1):
        if not self.lock_t: self.x_offset += offset
        self.update_x_y_max()

    def decrease_x_offset(self,offset=0.1):
        if not self.lock_t: self.x_offset -= offset
        self.update_x_y_max()

    def increase_y_offset(self,offset=0.1):
        if not self.lock_t: self.y_offset += offset
        self.update_x_y_max()
    
    def decrease_y_offset(self,offset=0.1):
        if not self.lock_t: self.y_offset -= offset
        self.update_x_y_max()

class Labeler:
    def __init__(self,orig_image) -> None:
        self.circle_r = 2
        self.prefix = ''
        self.dot_collector  = []
        self._corner_points = None
        self.stop_collect = 0
        self._source_dots = []
        self.grid_inside = Grid()
        self.orig_image = orig_image.copy() # layer 0 original layer
        self._image_vis_dots = orig_image.copy() # layer 1 with dots 
        self._image_vis_corner = orig_image.copy() # layer 2 with 4 corner points 
        self._image_vis_collect = orig_image.copy() # layer 3 with 4 mesh and collected dots

    @property
    def image_vis_dots(self):
        return self._image_vis_dots
    @image_vis_dots.setter    # when update layer 1 update layer 2
    def image_vis_dots(self,image_vis_dots):
        self._image_vis_dots = image_vis_dots
        self.draw_corners()

    @property
    def image_vis_corner(self):
        return self._image_vis_corner
    @image_vis_corner.setter    # when update layer 2 update layer 3
    def image_vis_corner(self,image_vis_corner):
        self._image_vis_corner = image_vis_corner
        self.collect_dots()

    @property
    def source_dots(self):
        return self._source_dots
    @source_dots.setter
    def source_dots(self,source_dots):
        print('drawing dots')
        self._source_dots = source_dots
        image_tmp = self.orig_image.copy()
        self.image_vis_dots  = show(image_tmp,self._source_dots,msize=8,show=False,color=(255,255,0)) # layer 0 with dots

    @property
    def corner_points(self):
        return self._corner_points

    @corner_points.setter
    def corner_points(self,corner_points):
        print('set corner')
        self._corner_points = np.asarray(corner_points).reshape(-1,2)
        self._corner_points = self._corner_points.astype('float32')  # user defined
       
        if self._corner_points.shape[0] == 4:
            min_idx = np.argsort(np.sum(self._corner_points,axis=1))[0]
            self._corner_points = np.vstack((self._corner_points[min_idx:],self._corner_points[:min_idx]))
            
            print('valid corners')
            self.find_M() # init M
        self.draw_corners()

    def draw_corners(self):
        print('drawing corners')
        # show target corners
        image_tmp = self._image_vis_dots.copy()
        if self._corner_points is not None:
            for k,pt in enumerate(self._corner_points):
                x,y = int(pt[0]),int(pt[1])
                cv2.circle(image_tmp, (x, y), 10, (0,0,255), -1)
                cv2.circle(image_tmp, (x, y), 5, (0,255,0), -1)
                plt.text(x,y,str(k)) #TODO, plt coordinates
        self.image_vis_corner = image_tmp
  

    def set_prefix(self,s):
        self.prefix = s
        self.update_dot_prefix()

    def increase_corner_x(self,i):
        self.corner_points[i][0] += 1

    def decrease_corner_x(self,i):
        self.corner_points[i][0] -= 1

    def increase_corner_y(self,i):    
        self.corner_points[i][1] += 1
        
    def decrease_corner_y(self,i):    
        self.corner_points[i][1] -= 1


    def find_M(self):
        if self.__dict__.get('grid_extend',None) is None:
            grid, grid_corners,grid_faces,coords = self.grid_inside.generate_grid()
            grid_corners = grid_corners.astype('float32')
           
            # self.M = M.copy()
        else:
            grid, grid_corners,grid_faces,coords = self.grid_extend.generate_grid()
            # M = self.M.copy() 

        M = cv2.getPerspectiveTransform(self.grid_inside.grid_corners.astype('float32'),self.corner_points)

        grid_t = transfrom(grid, M)
        self.grid_t = grid_t
        return grid_t, grid_corners,grid_faces,coords
    
    def extend_grid(self):
        if self.__dict__.get('grid_extend',None) is None:
            # self.grid_inside.lock()
            self.grid_extend = copy.deepcopy(self.grid_inside)
            # self.grid_extend.unlock()

    def extend_x_num(self,n=1,i=1):
        # i ==-1 : left, i==1: right
        self.grid_extend.increase_x_num(n)
        if i == -1:
            self.grid_extend.coords_x_offset += i*n
            # offset = self.grid_inside.x_max/(self.grid_inside.x_num-1) * i
            offset = self.grid_inside.x_gap * i
            self.grid_extend.increase_x_offset(offset)

    def extend_y_num(self,n=1,i=1):
        # i ==-1 : left, i==1: right
        self.grid_extend.increase_y_num(n)
     
        if i == -1:
            self.grid_extend.coords_y_offset += i*n
            # offset = self.grid_inside.y_max/(self.grid_inside.y_num-1) * i
            offset = self.grid_inside.y_gap * i
            self.grid_extend.increase_y_offset(offset)

    def set_circle_r(self,r):
        self.circle_r = r

    def collect_dots(self,show_flag=False):
        # print('drawing mesh and collected dots')
        image_tmp = self.image_vis_corner.copy() # layer 2 with grids 
        
        if self._corner_points is None or self._corner_points.shape[0]!=4 or self.stop_collect:
            return self.image_vis_corner
        
        self.dot_collector = []
        # find transformed grids
        grid_t, _,grid_faces,coords = self.find_M()

        # show mesh faces
        for face in grid_faces:
            points = grid_t[face].astype('int')
            cv2.line(image_tmp, tuple(points[0]), tuple(points[1]), (0,0,255), 3)

        for k,pt in enumerate(grid_t):
            x,y = int(pt[0]),int(pt[1])
            cv2.circle(image_tmp, (x, y), self.circle_r, (0,0,255), 2) # draw collect circle
            # collect dot via circle
            collect_mask = np.zeros((image_tmp.shape[0], image_tmp.shape[1]))
            cv2.circle(collect_mask,(x, y), self.circle_r, 1, -1)
            dot_ids = np.where(collect_mask[self._source_dots[:,1],self._source_dots[:,0]]!=0)
            if len(dot_ids[0])>0:
                picked_dot = self._source_dots[dot_ids].reshape(-1,2)
                if picked_dot.shape[0]>1:
                    picked_dot = picked_dot[np.argmin(np.linalg.norm(picked_dot - np.array([x,y]),axis=1))]
                picked_dot = picked_dot.reshape(2,)
                cc = coords[k]
                dot_label = self.prefix + '_'+ str(cc[0]) + ',' + str(cc[1])
                self.dot_collector.append([dot_label,picked_dot])
                cv2.circle(image_tmp, (int(picked_dot[0]),int(picked_dot[1])), 5, (0,255,255), -1)   
        
        for k, (label,dot) in enumerate(self.dot_collector):
            x,y = int(dot[0]),int(dot[1])
            cv2.circle(image_tmp, (x,y), 5, (0,255,255), -1)   
            cv2.putText(image_tmp, label, (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 0, 0), 1, cv2.LINE_AA)   

        if show_flag:
            show(image_tmp)
        self.image_vis_collect = image_tmp
        return self.image_vis_collect
    
    def update_dot_prefix(self):
        dot_collector = []
        for (label,dot) in self.dot_collector:
            c = label.split('_')[-1]
            label_new = self.prefix + '_'+ c
            dot_collector.append([label_new,dot])
        self.dot_collector = dot_collector

class Label_Window():
    def __init__(self,img_path):
        self.img_path = img_path
        self.image_raw = cv2.imread(img_path)
        # self.image_dots = self.image_raw.copy()
        self.t = 0.02
        self.detect_dots()
        self.init_labeler()
       
        self.image_vis = self.labeler.image_vis_dots.copy()
        self.window_name = 'bg grid labelling'
        self.corners_id = 0
        self.image_corners = []
        self.text_file = ('.').join(img_path.split('.')[:-1]) + '.txt'
        
    def detect_dots(self):
        print('detecting dots')
        self.detected_dots = detect_corners(self.image_raw,self.t)

    def set_detect_dot_t(self,t):
        self.t = max(0.0001,t)
        self.t = min(0.5,self.t)

    def init_labeler(self):
        self.labeler = Labeler(self.image_raw)
        self.labeler.source_dots = self.detected_dots.copy()

    def print_text(self):
        text = [
        "************************************************************",
        'processing: '+ img_path,
        "************************************************************",
       
        "enter: finish one operater",

        "r : clean image, start click corners; left click 4 corners points start with the origin and use anti-clockwise sequence",
        "n : enter origin name e.g. A0",
        "0/1/2/3 + w/s/a/d: adjust corners points",
        "x/y + w/s/a/d: adjust grids x/y number to map grid with the grid in image",
        "e + w/s/a/d: extend grids",
        "c + w/s increase/decrease circle size or dot detection threshold to collect dots",
        "o + w/s increase/decrease corner detection threshold",
        "i : save dots",
        "p : print grid size and corners detection threhold",
        "Press q: exist"
        "************************************************************"]
        for t in text:
            print(t)

   
    def reset_image_vis(self):
        self.image_vis = self.labeler.image_vis_dots.copy()
        

    def main(self):
        self.print_text()
        self.features_colloction = {}

        def click_corners(event, x, y, flags, self):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.image_corners) < 4:
                    self.image_corners.append([x,y]) 
                self.labeler.corner_points = self.image_corners
                   
        def keyboard_input():
            text = ""
            letters = string.ascii_lowercase + string.digits
            while True:
                key = cv2.waitKey(1)
                for letter in letters:
                    if key == ord(letter):
                        text = text + letter
                if key == ord("\n") or key == ord("\r"): # Enter Key
                    print('Take in finish',text)
                    break
                    
            return text

        self.func_mapping = click_corners  
        ids = [ord('0'),ord('1'),ord('2'),ord('3')]
        operator = OPT_X_NUM
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL) 
        while True:    
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyWindow(self.window_name)
                return self.features_colloction
            if key == ord("x"):
                operator = OPT_X_NUM
            elif key == ord("y"):
                operator = OPT_Y_NUM
            elif key == ord("e"):
                operator = OPT_EXTEND
            elif key == ord("c"):
                operator = OPT_CIRCLE_SIZE
            elif key == ord("o"):
                operator = OPT_DOT_DETECT_THRESHOD
            elif key == ord("m"):
                operator = OPT_MOVE
            elif key in ids:
                corner_i = ids.index(key)
                operator = OPT_CORNER

            if key == ord('p'):
                print(self.labeler.grid_inside)
                print(self.t)
           
            if key == ord('n'):
                print('plz input prefix')
                s = keyboard_input()
                self.labeler.set_prefix(s)
                self.labeler.stop_collect = 0
            
            if key == ord('r'):
                self.image_corners = []
                self.func_mapping = click_corners  
                self.init_labeler()
                self.reset_image_vis()

            if key == ord('i'):
                key = cv2.waitKey(1)
                while self.labeler.prefix in self.features_colloction or self.labeler.prefix == '':
                    self.labeler.stop_collect = 1
                    print(self.features_colloction.keys())
                    print('plz key in new name')
                    s = keyboard_input()
                    if s == 'q':
                        break
                    self.labeler.set_prefix(s)
      
                print('insert:',self.labeler.prefix)
                if self.labeler.prefix != '':
                    self.features_colloction.update({self.labeler.prefix:self.labeler.dot_collector})
                    self.labeler.stop_collect = 0
                    print(self.features_colloction.keys())
                    with open(self.text_file,'a') as f:
                        # for dots_set in self.features_colloction.values():
                        dots_set = self.labeler.dot_collector
                        f.write('*'*10+'\n')
                        for label, coord in dots_set:
                            f.write(label+'\t')
                            f.write(str(coord[0])+'\t'+str(coord[1]))
                            f.write('\n')


            if key == ord('w'):
                if operator == OPT_DOT_DETECT_THRESHOD:
                    # self.func_mapping = detect_dot_again
                    self.set_detect_dot_t(self.t+0.001)
                    self.detect_dots()
                          
                elif operator == OPT_X_NUM:
                    self.labeler.grid_inside.increase_x_num()   
                    
                elif operator == OPT_Y_NUM:
                    self.labeler.grid_inside.increase_y_num()
                    
                elif operator == OPT_CIRCLE_SIZE:
                    self.labeler.set_circle_r(self.labeler.circle_r+1)

                elif operator == OPT_EXTEND:
                    self.labeler.extend_grid()
                    self.labeler.extend_x_num(i=-1)       
              
                    
                elif operator == OPT_CORNER:
                    self.labeler.decrease_corner_y(corner_i)
                    
                    x,y = int(self.labeler.corner_points[corner_i][0]),int(self.labeler.corner_points[corner_i][1])
                    cv2.circle(self.image_vis, (x, y), 5,(255,0,255), 3)
                
                elif operator == OPT_MOVE:
                    self.labeler.extend_grid()
                    self.labeler.grid_extend.increase_y_offset()

                    

            elif key == ord('s'):
                if operator == OPT_DOT_DETECT_THRESHOD:
                    self.set_detect_dot_t(self.t-0.001)
                    self.detect_dots()                  
                    
                elif operator == OPT_X_NUM:    
                    self.labeler.grid_inside.decrease_x_num()
                   
                elif operator == OPT_Y_NUM:
                    self.labeler.grid_inside.decrease_y_num()
                                    
                elif operator == OPT_CIRCLE_SIZE:
                    self.labeler.set_circle_r(self.labeler.circle_r-1)              
                    
                elif operator == OPT_EXTEND: 
                    self.labeler.extend_grid()
                    self.labeler.extend_x_num(i=1)
                                   
                elif operator == OPT_CORNER:
                    self.labeler.increase_corner_y(corner_i)
                    
                    x,y = int(self.labeler.corner_points[corner_i][0]),int(self.labeler.corner_points[corner_i][1])
                    cv2.circle(self.image_vis, (x, y), 5, (255,0,255), 3)

                elif operator == OPT_MOVE:
                    self.labeler.extend_grid()
                    self.labeler.grid_extend.decrease_y_offset()
                

            elif key == ord('a'):
                if operator == OPT_EXTEND: 
                    self.labeler.extend_grid()
                    self.labeler.extend_y_num(i=-1)

                elif operator == OPT_CORNER:
                    self.labeler.decrease_corner_x(corner_i)
                    
                    x,y = int(self.labeler.corner_points[corner_i][0]),int(self.labeler.corner_points[corner_i][1])
                    cv2.circle(self.image_vis, (x, y), 5, (255,0,255), 3)

                elif operator == OPT_MOVE:
                    self.labeler.extend_grid()
                    self.labeler.grid_extend.increase_x_offset()
                    
            elif key == ord('d'):
                if operator == OPT_EXTEND: 
                    self.labeler.extend_grid()
                    self.labeler.extend_y_num(i=1)

                elif operator == OPT_CORNER:
                    self.labeler.increase_corner_x(corner_i)
                       
                    x,y = int(self.labeler.corner_points[corner_i][0]),int(self.labeler.corner_points[corner_i][1])
                    cv2.circle(self.image_vis, (x, y), 5, (255,0,255), 3)

                elif operator == OPT_MOVE:
                    self.labeler.extend_grid()
                    self.labeler.grid_extend.decrease_x_offset()    
            
            self.image_vis = self.labeler.collect_dots()            
            cv2.imshow(self.window_name, self.image_vis)
            cv2.resizeWindow(self.window_name, 900, 900) 
            cv2.setMouseCallback(self.window_name, self.func_mapping, self)

if __name__ == "__main__":
    img_path = 'chess_board.jpg'
    test = Label_Window(img_path)
    test.main()
    print(test.features_colloction.keys())
    

