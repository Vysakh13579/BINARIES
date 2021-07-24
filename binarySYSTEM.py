
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi,gravitational_constant


from IPython.display import clear_output




class star(object):
    def __init__(self, num_steps):
        self.x=np.zeros(num_steps, np.float64)
        self.y=np.zeros(num_steps, np.float64)
        self.vx=np.zeros(num_steps, np.float64)
        self.vy=np.zeros(num_steps, np.float64)
        self.t=np.zeros(num_steps, np.float64)
        
class BinarySystem(object):

    
    
    def __init__ (self, M1, M2, a, e, theta=0):
        """
        theta is an angle to rotate the semi-major axis wrt +x
        """
        self.G = 6.67428e-11        
        self.M_sun = 1.98892e30    
        self.AU = 1.49598e11       
        self.year = 3.1556926e7
        
        self.m1 = M1*self.M_sun
        self.m2 = M2*self.M_sun
        self.a = a*self.AU
        self.e = e
        self.theta = theta

        # determine the individual semi-major axes
        # a1 + a2 = a,  M1 a1 = M2 a2
        self.a1 = self.a/(1.0 + self.m1/self.m2)
        self.a2 = self.a - self.a1

        # we put the center of mass at the origin
        # we put star 1 on the -x axis and star 2 on the +x axis
        self.x1_init = -self.a1*(1.0 - self.e)*np.cos(self.theta)
        self.y1_init = -self.a1*(1.0 - self.e)*np.sin(self.theta)

        self.x2_init = self.a2*(1.0 - self.e)*np.cos(self.theta)
        self.y2_init = self.a2*(1.0 - self.e)*np.sin(self.theta)

        # Kepler's laws should tell us the orbital period
        # P^2 = 4 pi^2 (a_star1 + a_star2)^3 / (G (M_star1 + M_star2))
        self.P = np.sqrt(4*np.pi**2*(self.a1 + self.a2)**3/(self.G*(self.m1 + self.m2)))

        # compute the initial velocities velocities

        # first compute the velocity of the reduced mass at perihelion
        # (C&O Eq. 2.33)
        v_mu = np.sqrt( (self.G*(self.m1 + self.m2)/(self.a1 + self.a2)) *
                        (1.0 + self.e)/(1.0 - self.e) )

        # then v_star2 = (mu/m_star2)*v_mu
        self.vx2_init = -(self.m1/(self.m1 + self.m2))*v_mu*np.sin(self.theta)
        self.vy2_init =  (self.m1/(self.m1 + self.m2))*v_mu*np.cos(self.theta)

        # then v_star1 = (mu/m_star1)*v_mu
        self.vx1_init =  (self.m2/(self.m1 + self.m2))*v_mu*np.sin(self.theta)
        self.vy1_init = -(self.m2/(self.m1 + self.m2))*v_mu*np.cos(self.theta)


        

        self.orbit1 = None
        self.orbit2 = None


    def integrate(self, dt, tmax):
        

        # allocate storage for R-K intermediate results
        # y[0:3] will hold the star1 info, y[4:7] will hold the star2 info
        k1 = np.zeros(8, np.float64)
        k2 = np.zeros(8, np.float64)
        k3 = np.zeros(8, np.float64)
        k4 = np.zeros(8, np.float64)

        y = np.zeros(8, np.float64)

        t = 0.0

        # initial conditions

        # star 1
        y[0] = self.x1_init  # initial x position
        y[1] = self.y1_init  # initial y position

        y[2] = self.vx1_init # initial x-velocity
        y[3] = self.vy1_init # initial y-velocity

        # star 2
        y[4] = self.x2_init  # initial x position
        y[5] = self.y2_init  # initial y position

        y[6] = self.vx2_init # initial x-velocity
        y[7] = self.vy2_init # initial y-velocity


        # how many steps will we need?
        nsteps = int(tmax/dt)

        # solution storage
        s1 = star(nsteps+1)
        s2 = star(nsteps+1)

        s1.x[0] = self.x1_init
        s1.y[0] = self.y1_init
        s1.vx[0] = self.vx1_init
        s1.vy[0] = self.vy1_init

        s2.x[0] = self.x2_init
        s2.y[0] = self.y2_init
        s2.vx[0] = self.vx2_init
        s2.vy[0] = self.vy2_init

        s1.t[0] = s2.t[0] = t

        for n in range(1, nsteps+1):

            k1[:] = dt*self.deriv(t, y, self.m1, self.m2)
            k2[:] = dt*self.deriv(t+0.5*dt, y[:]+0.5*k1[:], self.m1, self.m2)
            k3[:] = dt*self.deriv(t+0.5*dt, y[:]+0.5*k2[:], self.m1, self.m2)
            k4[:] = dt*self.deriv(t+dt, y[:]+k3[:], self.m1, self.m2)

            y[:] += (1.0/6.0)*(k1[:] + 2.0*k2[:] + 2.0*k3[:] + k4[:])

            t = t + dt

            s1.x[n] = y[0]
            s1.y[n] = y[1]
            s1.vx[n] = y[2]
            s1.vy[n] = y[3]

            s2.x[n] = y[4]
            s2.y[n] = y[5]
            s2.vx[n] = y[6]
            s2.vy[n] = y[7]

            s1.t[n] = s2.t[n] = t

        self.m1orbit = s1
        self.m2orbit = s2

    
    def deriv(self,t, y, M_star1, M_star2):
        """ the RHS of our system """

        f = np.zeros(8, np.float64)

        # y[0] = x_star1, y[1] = y_star1, y[2] = vx_star1, y[3] = vy_star1
        # y[4] = x_star2, y[5] = y_star2, y[6] = vx_star2, y[7] = vy_star2

        # unpack
        x_star1 = y[0]
        y_star1 = y[1]

        vx_star1 = y[2]
        vy_star1 = y[3]

        x_star2 = y[4]
        y_star2 = y[5]

        vx_star2 = y[6]
        vy_star2 = y[7]


        # distance between stars
        r = np.sqrt((x_star2 - x_star1)**2 + (y_star2 - y_star1)**2)

        f[0] = vx_star1  # d(x_star1) / dt
        f[1] = vy_star1  # d(y_star1) / dt

        f[2] = -self.G*M_star2*(x_star1 - x_star2)/r**3  # d(vx_star1) / dt
        f[3] = -self.G*M_star2*(y_star1 - y_star2)/r**3  # d(vy_star1) / dt

        f[4] = vx_star2  # d(x_star2) / dt
        f[5] = vy_star2  # d(y_star2) / dt

        f[6] = -self.G*M_star1*(x_star2 - x_star1)/r**3  # d(vx_star2) / dt
        f[7] = -self.G*M_star1*(y_star2 - y_star1)/r**3  # d(vy_star2) / dt

        return f
    def BINARYplotter(self,number_of_orbits=2,frame='cm'):

        # set the timestep in terms of the orbital period
        dt = self.P/360.0
        tmax = number_of_orbits*self.P  # maximum integration time

        self.integrate(dt, tmax)
        s1=self.m1orbit
        s2=self.m2orbit
        for n in np.arange(0,len(s1.t),3):

            fig = plt.figure(1,figsize=[10,7])
            plt.style.use('seaborn')
            fig.clear()

            ax = fig.add_subplot(111)
            plt.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.975)
            ax.set_aspect("equal", "datalim")
            #ax.set_axis_off()
            if frame.upper()=='CM':
                ax.scatter([0], [0], s=150, marker="x", color="k",label='COM')
                # plot star 1's orbit and position
                symsize = 200
                ax.plot(s1.x, s1.y, color="blue")
                ax.scatter([s1.x[n]], [s1.y[n]], s=symsize, color="blue", zorder=100,label='Mass1')

                # plot star 2's orbit and position
                symsize = 200*(self.m2/self.m1)
                ax.plot(s2.x, s2.y, color="red")
                ax.scatter([s2.x[n]], [s2.y[n]], s=symsize, color="red", zorder=100,label='Mass2')
                ax.plot([s1.x[n],s2.x[n]],[s1.y[n],s2.y[n]],'k-')
                xmin = 1.25*min(s1.x.min(), s2.x.min())
                xmax = 1.25*max(s1.x.max(), s2.x.max())
                ymin = 1.25*min(s1.y.min(), s2.y.min())
                ymax = 1.25*max(s1.y.max(), s2.y.max())
                
            elif frame.upper()=='PRIMARY_STAR':
                ax.scatter(0-s1.x[n], 0-s1.y[n], s=150, marker="x", color="k",label='COM')
                # plot star 1's orbit and position
                symsize = 200
                ax.plot(s1.x-s1.x, s1.y-s1.y, color="blue")
                ax.scatter(s1.x[n]-s1.x[n], s1.y[n]-s1.y[n], s=symsize, color="blue", zorder=100,label='Mass1')

                # plot star 2's orbit and position
                symsize = 200*(self.m2/self.m1)
                ax.plot(s2.x-s1.x, s2.y-s1.y, color="red")
                ax.scatter(s2.x[n]-s1.x[n], s2.y[n]-s1.y[n], s=symsize, color="red", zorder=100,label='Mass2')
                ax.plot([s1.x[n]-s1.x[n],s2.x[n]-s1.x[n]],[s1.y[n]-s1.y[n],s2.y[n]-s1.y[n]],'k-')
               
                xmin = 1.25*min((s1.x-s1.x).min(), (s2.x-s1.x).min())
                xmax = 1.25*max((s1.x-s1.x).max(), (s2.x-s1.x).max())
                ymin = 1.25*min((s1.y-s1.y).min(), (s2.y-s1.y).min())
                ymax = 1.25*max((s1.y-s1.y).max(), (s2.y-s1.y).max())
                
            else:
                print("ENTER VALID REFERNCE OF FRAME")
                break
            # display time-s1.y[n]
            ax.text(0.05, 0.9,r"TIME: {:3.2f} years".format(s1.t[n]/(365*24*3600)), transform=ax.transAxes, color="k", fontsize="large")
        
            ax.legend()
            

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            plt.show()
            clear_output(wait=True)
            
            
    def calcTrueAnomaly(self,periastron,ascNode,number_of_orbits=2):
        dt = self.P/360.0
        tmax = number_of_orbits*self.P  # maximum integration time

        self.integrate(dt, tmax)
        s1=self.m1orbit
        s2=self.m2orbit
        self.m1_true_anomaly=np.zeros(int(tmax/dt)+1, np.float64)
        self.m2_true_anomaly=np.zeros(int(tmax/dt)+1, np.float64)
        
        for i in range(len(s1.x)):
            if (s1.x[i]>=0) and (s1.y[i]>=0):
                self.m1_true_anomaly[i]=2*np.pi-(periastron+ascNode)+np.arctan(s1.y[i]/s1.x[i])
            elif (s1.x[i]<=0) and (s1.y[i]>=0):
                self.m1_true_anomaly[i]=2*np.pi-(periastron+ascNode)+np.pi-np.arctan(-s1.y[i]/s1.x[i])
            elif (s1.x[i]<=0) and (s1.y[i]<=0):
                self.m1_true_anomaly[i]=2*np.pi-(periastron+ascNode)+np.pi+np.arctan(s1.y[i]/s1.x[i])
            elif (s1.x[i]>=0) and (s1.y[i]<=0):
                self.m1_true_anomaly[i]=2*np.pi-(periastron+ascNode)-np.arctan(-s1.y[i]/s1.x[i])
        
        for i in range(len(s2.x)):
            if (s2.x[i]>=0) and (s2.y[i]>=0):
                self.m2_true_anomaly[i]=2*np.pi-(periastron+ascNode)+np.arctan(s2.y[i]/s2.x[i])
            elif (s2.x[i]<=0) and (s2.y[i]>=0):
                self.m2_true_anomaly[i]=2*np.pi-(periastron+ascNode)+np.pi-np.arctan(-s2.y[i]/s2.x[i])
            elif (s2.x[i]<=0) and (s2.y[i]<=0):
                self.m2_true_anomaly[i]=2*np.pi-(periastron+ascNode)+np.pi+np.arctan(s2.y[i]/s2.x[i])
            elif (s2.x[i]>=0) and (s2.y[i]<=0):
                self.m2_true_anomaly[i]=2*np.pi-(periastron+ascNode)-np.arctan(-s2.y[i]/s2.x[i])        
        return self.m1_true_anomaly,self.m2_true_anomaly
    
    
    
    def radVELcurvePlot(self,inclination,argPeriastron,longAscNode,radialVEL=0,number_of_orbits=2):
        dt = self.P/360.0
        tmax = number_of_orbits*self.P  # maximum integration time

        self.integrate(dt, tmax)
        s1=self.m1orbit
        s2=self.m2orbit
        
        TAm1,TAm2=self.calcTrueAnomaly(argPeriastron,longAscNode)
        
        k1=2*np.pi*self.a1*np.sin(inclination)/(np.sqrt(1-self.e**2)*self.P)
        k2=2*np.pi*self.a2*np.sin(inclination)/(np.sqrt(1-self.e**2)*self.P)

        self.v1rf=radialVEL+(np.cos(argPeriastron+TAm1)+self.e*np.cos(argPeriastron))*k1
        self.v2rf=radialVEL+(np.cos(argPeriastron+TAm2)+self.e*np.cos(argPeriastron))*k2
        time=s1.t/(365*24*3600)

        for i in np.arange(0,len(s2.x),4):
            fig,ax=plt.subplots(1,2,figsize=[20,7])
            plt.style.use('seaborn')
            #plt.figure(1,figsize=[10,7])
            ax[1].plot(time[0:i],self.v1rf[0:i],'r-')
            ax[1].plot(time[0:i],self.v2rf[0:i],'b-')
            xmin = min(time)
            xmax = max(time)
            ymin = 1.25*min(self.v1rf.min(), self.v2rf.min())
            ymax = 1.25*max(self.v1rf.max(), self.v2rf.max())
            ax[1].set_xlim(xmin, xmax)
            ax[1].set_ylim(ymin, ymax)
            ax[1].axhline(y=0,xmax=max(time),c='black')
            ax[1].set_xlabel('Time(years)')
            ax[1].set_ylabel('Velocity(m/s)')
            
            symsize = 200
            ax[0].plot(s1.x, s1.y, color="blue")
            ax[0].scatter([s1.x[i]], [s1.y[i]], s=symsize, color="blue", zorder=100,label='Mass1')
            symsize = 200*(self.m2/self.m1)
            ax[0].plot(s2.x, s2.y, color="red")
            ax[0].scatter([s2.x[i]], [s2.y[i]], s=symsize, color="red", zorder=100,label='Mass2')
            
            
            #ax[0].text(0.05, 0.9,r"TIME: {:3.2f} years".format(s1.t[2*i]/(365*24*3600)), transform=ax[0].transAxes, color="k", fontsize="large")
            ax[0].set_aspect("equal", "datalim")
            ax[0].legend()
            xmin = 1.25*min(s1.x.min(), s2.x.min())
            xmax = 1.25*max(s1.x.max(), s2.x.max())
            ymin = 1.25*min(s1.y.min(), s2.y.min())
            ymax = 1.25*max(s1.y.max(), s2.y.max())

            ax[0].set_xlim(xmin, xmax)
            ax[0].set_ylim(ymin, ymax)
            plt.show()
            clear_output(wait=True)