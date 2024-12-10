
'''

PGA3D Display Manager

==================

Shows different geometric objects in 3D.


Usage / Install : 
    See README.md


'''

import math as np
import unittest
import matplotlib.pyplot as plt

import logging 
logger         = logging.getLogger("pga3d")


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
       

        logger.debug(f'Created')
      
    def init_show(self): 
        # 3D scene

        fig_num     = 1
        object_num  = 1


        # init figure
        fig         = plt.figure(fig_num)
        plt.clf() 
        plt.ion()    
        #fig.tight_layout()  
        #ax = fig.add_subplot(projection='3d')  

        # Plot data points for handler
        h_data,     = plt.scatter([0, 1], [0, 1], [0, 1], 'b.')
        ax          = plt.gca()

        # plot tracker positions
        h_pose      = []
        for k in range(object_num):
            h,  = ax.plot([0], [0],marker='x',color='C'+str(k))
            h_pose.append(h)

        # plot tracker names
        h_text      = []
        for k in range(object_num):
            h  = ax.text(0 , 0, str(k), fontsize=8)
            h_text.append(h)

        
        plt.title('3D Objects')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.zlabel('X2')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.draw()
        #plt.pause(0.1)  # Update the plot
        plt.show()
        
        self.h_data = h_data        
        self.h_pose = h_pose        
        self.h_text = h_text

        self.fig    = fig
        self.plt    = plt
        self.ax     = ax
        logger.debug('Scene rendering is done')

        # for debug
        logger.info('Press any button to continue...')
        self.plt.waitforbuttonpress()        
        return ax 
    
    def draw_data(self, dataList):
        "assuming that everythong is initialized - show data"
        if dataList is None:
            return

        self.h_data.set_data(dataList[0, :], dataList[1, :])

    def draw_track(self, trackList):
        "assuming that everythong is initialized - show track info"
        if trackList is None:
            return

        object_num       = self.params["TrackNum"]  
        ct              = np.linspace(0, 2 * np.pi, 100)
        circle          = np.vstack((np.cos(ct), np.sin(ct)))
        small_shift     = 3e-2

        # plot tracker positions
        for k in range(object_num):

            # do not show init stages
            if trackList[k].state < TrackState.LAST_INIT:
               continue
            
            ypred, Spred, yhist     = trackList[k].get_show_info()
            
            u, s, v                 = np.linalg.svd(Spred)
            elipse                  = u @ np.diag(np.sqrt(s)) @ circle            
            
            # update drawing
            self.h_pose[k].set_data(ypred[0], ypred[1]) 
            self.h_text[k].set_x(ypred[0] + small_shift)
            self.h_text[k].set_y(ypred[1]) 
            #self.h_text[k].set_text('%d-%d' %(trackList[k].id,trackList[k].state)) 
            self.h_text[k].set_text('%d-%d' %(trackList[k].id,trackList[k].life_time)) 
            self.h_circle[k].set_data(elipse[0,:] + ypred[0], elipse[1,:] + ypred[1]) 
            self.h_history[k].set_data(yhist[0,:], yhist[1,:])  


    def show_info(self, trackList = None, dataList = None):
        "displays tracks and data"

        self.draw_data(dataList)
        self.draw_track(trackList)

        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events() 

        self.plt.draw()
        self.plt.pause(0.1)

        # for debug
        #logger.info('Press any button to continue...')
        #self.plt.waitforbuttonpress()


    def show_tracks_and_data(self, trackList, dataList, par = None):
        """
        Visualizes the data points and tracks.

        Args:
            trackList: List of track objects.
            dataList: 2D array containing measurement data (time x measurements).
            Par: Dictionary containing parameters.
            
        """
        par         = self.params if par is None else par

        ShowFigNum  = 1
        #AxisSc      = [par["Y1Bounds"][0], par["Y2Bounds"][0], par["Y1Bounds"][1], par["Y2Bounds"][1]]
        SmallShift  = 5e-3
        NumSigma    = np.sqrt(par["GateLevel"])

        # Plot data points
        plt.figure(ShowFigNum)
        plt.plot(dataList[0, :], dataList[1, :], 'b.')
        #plt.axis(AxisSc)
        plt.title('Data Points & Tracks')
        plt.xlabel('X1')
        plt.ylabel('X2')
        ax = plt.gca()
        ax.set_xlim([par["Y1Bounds"][0], par["Y1Bounds"][1]])
        ax.set_ylim([par["Y2Bounds"][0], par["Y2Bounds"][1]])
        
        # Plot tracks
        TrackNum            = len(trackList)

        for i in range(TrackNum):

            y, S    = trackList[i].predict()
            u, s, v = np.linalg.svd(S)
            elipse  = u @ np.diag(np.sqrt(s)) @ np.vstack((np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100))))

            # do not show certain states
            #if not any(trackList[i]["State"] == s for s in ValidStatesForShow):
            #    y = np.array([[np.nan], [np.nan]])

            plt.plot(elipse[0, :] + y[0], elipse[1, :] + y[1], 'r')
            plt.text(y[0] + SmallShift, y[1], str(i), fontsize=8)

        plt.draw()
        plt.pause(0.1)  # Update the plot
        #plt.clf()       
 
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
class TestPGA3D_DISPLAY(unittest.TestCase):

    def test_create(self):
        "check create and data show init"
        d       = PGA3D_DISPLAY()
        ax      = d.init_show()
        d.finish()
        self.assertFalse(ax is None) 

    def test_show_data(self):
        "check create and data show"
        d       = PGA3D_DISPLAY()
        ax      = d.init_show()
        trackP  = None
        for k in range(10):
            dataP   = np.random.rand(2,10)
            d.show_info(trackP, dataP)
        d.finish()
        self.assertFalse(ax is None) 

      

# --------------------------------
#%% Run Test
def RunTest():
    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestPGA3D_DISPLAY("test_create")) # ok
    suite.addTest(TestPGA3D_DISPLAY("test_show_data")) 

    
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    #print(__doc__)

    RunTest()
    #RunApp('iir',43).run()    

