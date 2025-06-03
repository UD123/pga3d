
'''

PGA3D Display Manager

==================

Shows different geometric objects in 3D.


Usage / Install : 
    See README.md


'''

import numpy as np
import unittest
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('dark_background')
# disable vefore compile
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

import logging 
logger         = logging.getLogger("pga3d")

#%% Help functions
def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)
    return M_inv

def euler_angles_to_rotation_matrix(theta, isdegree=True) :
    "transfers xyz euler representation to 3D rot matrix"
    if isdegree:
        theta = np.deg2rad(theta)
    if len(theta) != 3:
        raise ValueError("Input must be a 3-element vector representing Euler angles in radians.")


    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])

    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])

    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def compute_transform_matrix(extrinsics, patternCentric):
    "transformation to 4D matrix"            
    R           = euler_angles_to_rotation_matrix(extrinsics[3:6]) 
    H           = np.eye(4,4)
    H[0:3,0:3]  = R
    H[0:3,3]    = extrinsics[0:3]
    H          = inverse_homogeneoux_matrix(H) if patternCentric else H
    return H  

def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    M[1,1] = 0
    M[1,2] = 1
    M[2,1] = -1
    M[2,2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))
    


#%% Objects
# 3d Axis frame
class Frame:
    def __init__(self, fname = 'frame'):
        self.name               = fname  # H,W,Depth
        self.id                 = 0      # several objects of the same type  
        self.extrisincs         = np.zeros((0,6))
        self.params_model       = {'width':10, 'height':10,'depth':10}
        self.x_frame            = self.create_frame_model(20)
        self.h_frame            = [None]*len(self.x_frame) # assocoation to graphics   
        self.h_text             = None
        self.h_model            = [None]*len(self.x_frame) # assocoation to graphics
        self.show_axis          = True
        self.show_text          = True
        self.color              = 'C0' # default color
        self.min_values         = np.tile(np.inf,3)
        self.max_values         = np.tile(-np.inf,3)

    def create_frame_model(self, height = 10):
        # create frame axis
        X_frame1        = np.ones((4,2))
        X_frame1[0:3,0] = [0, 0, 0]
        X_frame1[0:3,1] = [height, 0, 0]
    
        X_frame2        = np.ones((4,2))
        X_frame2[0:3,0] = [0, 0, 0]
        X_frame2[0:3,1] = [0, height, 0]
    
        X_frame3        = np.ones((4,2))
        X_frame3[0:3,0] = [0, 0, 0]
        X_frame3[0:3,1] = [0, 0, height]
        
        return [X_frame1, X_frame2, X_frame3]  
    
    def draw_text(self, ax, cMo):
        "draws text in the given axis"
        if self.show_text is False:
            return
        
        clr     = self.color 
        # transform to the first coordinate
        X       = cMo.dot(self.x_frame[0])

        if self.h_text is None:
            self.h_text = ax.text(X[0,0], X[1,0]-10, X[2,0]+20, str(self.id), color=clr)
        else:
            self.h_text.set_x(X[0,0])
            self.h_text.set_y( X[1,0]-10)
            self.h_text.set_3d_properties(X[2,0]+20)   

    def update_min_max(self, X): 
        "updates min and max values"
        if X is None:
            return
        
        self.min_values = np.minimum(self.min_values, X.min(1))
        self.max_values = np.maximum(self.max_values, X.max(1))   

    def draw_axis(self, ax, cMo):
        "attach axis to the frame"
        if self.show_axis is False:
            return
                    
        part_num     = len(self.x_frame)
        clr          = 'rgb'
        for i in range(part_num):
            # transform
            X       = cMo.dot(self.x_frame[i])
            
            if self.h_frame[i] is None:
                self.h_frame[i],  = ax.plot3D(X[0,:], X[1,:], X[2,:], color=clr[i])
            else:
                self.h_frame[i].set_data(X[0,:], X[1,:])  
                self.h_frame[i].set_3d_properties(X[2,:])
            
            self.update_min_max(X[0:3,:])

        return 

    def draw_model(self, ax, cMo):
        "draws model when defined"
        if self.x_model[0] is None:
            return
        
        x_moving    = self.x_model
        clr         = self.color
        part_num     = len(x_moving)
        for i in range(part_num):
            # transform
            X       = cMo.dot(x_moving[i])
            
            # draw
            if self.h_model[i] is None:
                self.h_model[i],  = ax.plot3D(X[0,:], X[1,:], X[2,:], color=clr)
            else:
                self.h_model[i].set_data(X[0,:], X[1,:])
                self.h_model[i].set_3d_properties(X[2,:])

            self.update_min_max(X[0:3,:])

        return    

    def draw_all(self, ax, patternCentric = False):
        "draws the frame in the given axis"
        extrinsics          = self.extrinsics
        
        # compute transform
        cMo                 = compute_transform_matrix(extrinsics, patternCentric)

        # show model
        self.draw_model(ax, cMo)      
                
        # show text
        self.draw_text(ax, cMo)
                
        # show axis
        self.draw_axis(ax, cMo)

        return 
    
    def set_extrinsics(self, extrinsics):
        "set extrinsics for the frame"
        if extrinsics is None:
            self.extrinsics = np.zeros((0,6))
        else:
            self.extrinsics = extrinsics


class Box3D(Frame):
    def __init__(self, size = (200,500,300)):
        super().__init__()
        self.name               = 'board'  # H,W,Depth
        self.params_model       = {'width':size[0], 'height':size[1],'depth':size[2]}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):   
        # show board

        #f_scale = params_model['depth']
        dx        = self.params_model['width']
        dy        = self.params_model['height']
        dz        = self.params_model['depth']
    
        # draw calibration board
        X_board         = np.ones((4,14))
        #X_board_cam = np.ones((extrinsics.shape[0],4,5))
        X_board[0:3,0] = [0,0,0]
        X_board[0:3,1] = [dx,0,0]
        X_board[0:3,2] = [dx,dy,0]
        X_board[0:3,3] = [0,dy,0]
        X_board[0:3,4] = [0,0,0]
        X_board[0:3,5] = [0,0,dz]
        X_board[0:3,6] = [dx,0,dz]
        X_board[0:3,7] = [dx,dy,dz]
        X_board[0:3,8] = [0,dy,dz]
        X_board[0:3,9] = [0,0,dz]
        X_board[0:3,10] = [dx,0,dz]
        X_board[0:3,11] = [dx,0,0]
        X_board[0:3,12] = [0,0,0]
        X_board[0:3,13] = [0,0,dz]
        # output should be a list
        outv          = [X_board]
         # draw board with axis
        return outv    
    
class Line(Frame):
    def __init__(self):
        super().__init__()
        self.name               = 'line'  # H,W,Depth
        self.params_model       = {'width':0, 'height':800,'depth':5}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):  
        # show line

        height          = self.params_model['height']
        X_frame1        = np.ones((4,2))
        X_frame1[0:3,0] = [0, 0, 0]
        X_frame1[0:3,1] = [0, 0, height]

        # output should be a list
        outv            = [X_frame1]
        return outv  
    
class Ray(Frame):
    def __init__(self):
        super().__init__()
        self.name               = 'ray'  # H,W,Depth
        self.params_model       = {'width':10, 'height':10,'depth':10}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):   

        # show multiple rays
        # width, height, depth must be arrays of the same size

        width          = self.params_model['width']
        height         = self.params_model['height']
        depth          = self.params_model['depth']
        
        outv           = []
        for k in range(len(width)):
            X_frame1        = np.ones((4,2))
            X_frame1[0:3,0] = [0, 0, 0]
            X_frame1[0:3,1] = [width[k], height[k], depth[k]]
    
            # output should be a list
            outv.append(X_frame1)
            
        return outv    
    
class Point(Frame):
    def __init__(self):
        super().__init__()
        self.name               = 'point'  # H,W,Depth
        self.params_model       = {'width':5, 'height':5,'depth':5}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):    

        # show multiple points
        # width, height, depth must be arrays of the same size

        width          = self.params_model['width']
        height         = self.params_model['height']
        depth          = self.params_model['depth']
        
        outv           = []
        # draw point as a square
        X_point         = np.ones((4,10))
        X_point[0:3,0] = [0,0,0]
        X_point[0:3,1] = [width,0,0]
        X_point[0:3,2] = [0,height,0]
        X_point[0:3,3] = [0,0,0]
        X_point[0:3,4] = [0,0,depth]
        X_point[0:3,5] = [0,height,0]
        X_point[0:3,6] = [0,0,0]
        X_point[0:3,7] = [0,0,depth]
        X_point[0:3,8] = [width,0,0]
        X_point[0:3,9] = [0,0,0]  
        
        # output should be a list
        outv.append(X_point)
            
        return outv


# --------------------------------
#%% Main
class PGA3D_DISPLAY:

    def __init__(self):

        # show
        self.fig        = None
        self.plt        = None
        self.ax         = None    

        self.h_data     = None        
        self.h_pose     = None        
        self.h_text     = None

        # graphics
        self.min_values         = np.array([0,0,0]) # min axis range
        self.max_values         = np.array([1,1,1]) # max axis range
        
        # object list to remmenber
        self.object_list        = []        

        logger.debug(f'Created')
      
    def init_scene(self): 
        # 3D scene

        fig_num                 = 1
        object_num              = 1
        self.min_values         = np.array([0,0,0]) # min axis range
        self.max_values         = np.array([1,1,1]) # max axis range
        
        self.object_list        = []        


        # init figure
        fig                     = plt.figure(fig_num)
        plt.clf() 
        plt.ion()    
        #fig.tight_layout()  
        #ax = fig.add_subplot(projection='3d')  

        #fig.canvas.set_window_title('3D Scene')
        try:
            ax = fig.gca(projection='3d')
        except:
            ax = fig.add_subplot(projection = '3d')
        fig.tight_layout()
        ax.set_proj_type('ortho')        

        # Plot data points for handler
        x,y,z       = np.array([0, 1]), np.array([0, 1]), np.array([0, 1])
        h_data      = ax.scatter(x,y,z, marker='.',label='Data Points')
        #ax          = plt.gca()

        # # plot tracker positions
        # h_pose      = []
        # for k in range(object_num):
        #     h,  = ax.plot([0], [0],[0],marker='x',color='C'+str(k))
        #     h_pose.append(h)

        # # plot tracker names
        # h_text      = []
        # for k in range(object_num):
        #     h  = ax.text(0 , 0, 0, str(k), fontsize=8)
        #     h_text.append(h)

        
        plt.title('3D Objects')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')

        self.scale_scene(ax)

        plt.draw()
        #plt.pause(0.1)  # Update the plot
        plt.show()
        
        self.h_data = h_data        
        # self.h_pose = h_pose        
        # self.h_text = h_text

        self.fig    = fig
        self.plt    = plt
        self.ax     = ax


        #print('Scene rendering is done')

        # for debug
        #logger.info('Press any button to continue...')
        #self.plt.waitforbuttonpress()        
        return ax

    def scale_scene(self, ax):
        # sets view range of the scene
        if ax is None:
            return
        
        X_min = self.min_values[0]
        X_max = self.max_values[0]
        Y_min = self.min_values[1]
        Y_max = self.max_values[1]
        Z_min = self.min_values[2]
        Z_max = self.max_values[2]

        range_x = np.array([X_max-X_min]).max() * 1.2 # to get some volume 2.0
        range_y = np.array([Y_max-Y_min]).max() * 1.2 
        range_z = np.array([Z_max-Z_min]).max() * 1.2
        max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 1.9 # to get some volume 2.0
        range_x = range_y = range_z = max_range
    
        mid_x = (X_max+X_min) * 0.5
        mid_y = (Y_max+Y_min) * 0.5
        mid_z = (Z_max+Z_min) * 0.5
        ax.set_xlim(mid_x - range_x, mid_x + range_x)
        ax.set_ylim(mid_y - range_y, mid_y + range_y)
        ax.set_zlim(mid_z - range_z, mid_z + range_z)  
        
        #ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
        #ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
        #set_axes_equal(ax) # IMPORTANT - this is also required
        #ax.axis('equal')   

    def draw_objects(self, ax, object_list, patternCentric = False):
        # check

        min_values  = self.min_values
        max_values  = self.max_values
        
        object_num  = len(object_list)
        if object_num < 1:
            return []
    
        #cm_subsection   = np.linspace(0.0, 1.0, object_num)
        #colors          = [ cm.rainbow(x) for x in cm_subsection ]
        #colors          = [ cm.Pastel1(x) for x in cm_subsection ]
        #rotSequence     = 'xyz' Paired

        for k in range(object_num):
            
            # get params
            object_list[k].draw_all(ax, patternCentric)
            min_values          = np.minimum(min_values, object_list[k].min_values)
            max_values          = np.maximum(max_values, object_list[k].max_values)

        self.min_values = min_values
        self.max_values = max_values
        self.scale_scene(ax)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  
        
        return object_list #min_values.ravel(), max_values.ravel()    
    
    
    def update_objects(self, object_list = None, obj_name = 'base', obj_num = 1, obj_pose = 6*[None] ):
        # update the existing object with new extrinsic data
        # checks
        if object_list is None:
            object_list = self.object_list_current
        
        
        objects_list_found = [x for x in object_list if x.name == obj_name]
        if len(objects_list_found) < obj_num:
            logger.info('No object %s with number %s is found' %(obj_name,obj_num))
            return
        
        # objects are always 1,2,3
        k = obj_num - 1 
        
        # update not none
        for i,v in enumerate(obj_pose):
            if v is not None : 
                objects_list_found[k].extrinsics[i] = v

        objects_list_found = self.draw_objects(self.ax, objects_list_found)
                
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()       
    
    def finish(self):
        # Close everything
        #plt.show()
        try:
            #cv.destroyWindow(self.estimator_name) 
            pass
        except:
            print('No window found')



# --------------------------------
#%% Tests
class test_pga3d_display(unittest.TestCase):

    def test_create(self):
        "check create and data show init"
        d               = PGA3D_DISPLAY()
        ax              = d.init_scene()
        d.finish()
        self.assertFalse(ax is None) 

    def test_show_board(self):
        cfg                 = None
        d                   = PGA3D_DISPLAY()
        ax                  = d.init_scene()

        # create board positions
        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        b2                  = np.array([0.0,   0.0,   10.0,  0.0,  0.0, -90.0 ]).reshape(1,6)    
        b3                  = np.array([0.0,   10.0,   0.0,  0.0,  45.0, 0.0 ]).reshape(1,6) 
        extrinsics_board    = np.vstack((b1,b2,b3))
      
        # import argparse
        board_list          = []
        for k in range(extrinsics_board.shape[0]):
            h_board         = Box3D()
            h_board.set_extrinsics(extrinsics_board[k,:])
            board_list.append(h_board)
        

        patternCentric      = False
        d.draw_objects(ax, board_list, patternCentric)

  

        d.finish()
        #self.assertEqual(isOk, True)
      

# --------------------------------
#%% Run Test
def RunTest():
    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(test_pga3d_display("test_create")) # ok
    suite.addTest(test_pga3d_display("test_show_board")) 

    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':


    RunTest()


