import numpy as np

class flatPlate():
    """Define a flat plate geometry in a moving reference frame
    """
    def __init__(self, N, c):
        """Initialize with a number of plate elements

        Args:
            N (int): Number of plate elements
        """
        self.N = N
        self.c = c
        # Define the flat plate geometry in a moving frame
        # of reference x, z
        self.x = np.linspace(0,1, N+1)*self.c
        self.z = np.zeros((1,N+1))*self.c
        self.qc_x = np.arange(1/N*1/4,1+1/N*1/4, 1/N) *self.c # quarter chord
        self.cp_x = np.arange(1/N*3/4,1+1/N*3/4, 1/N) *self.c # collocation points
        self.qc_z = self.qc_x * 0
        self.cp_z = self.cp_x * 0

def VOR2D(GAMMA, X, Z, Xj, Zj):
    """Discrete ortex method (11.1.1 from Katz Plotkin)

    Args:
        GAMMA (float): Vortex Strength
        X (float): Velocity induced at this point location  
        Z (float): Velocity induced at this point location
        Xj (float): Vortex element located as this location
        Zj ([type]): Vortex element located at this location

    Returns:
        numpy array: Velocity induced at X, Z from vortex element Xj,Zj
    """
    rj2 = ((X-Xj)**2 + (Z-Zj)**2)
    output = GAMMA/(2*np.pi*rj2)*np.array([Z-Zj, Xj-X])
    return output

def transform_local_to_global(x,z,theta,X0,Z0):
    """ Transformation matrix from 13.107

    Args:
        x (float): Location in moving ref
        z (float): Location in moving ref
        theta (float): Rotation angle in radians
        X0 (float): Flight path
        Z0 (float): Flight path

    Returns:
        numpy array: Transformed array
    """
    rot = np.array([[np.cos(theta),np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])
    output = np.dot(rot,np.array([x,z])) + np.array([X0,Z0])
    return output

def transform_velocity_l_to_g(theta, xdot, zdot):
    """ Transformation matrix from 13.108

    Args:
        theta (float): Rotation angle in radians
        xdot (float): x velocity
        xdot (float): z velocity

    Returns:
        numpy array: Transformed array
    """
    rot = np.array([[np.cos(theta),np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])
    output = np.dot(rot,np.array([xdot,zdot]))
    return output

def inv_transform_velocity_l_to_g(theta, X0dot, Z0dot, thetadot = 0, x=0):
    """Transformation matrix from 13.116

    Args:
        theta (float): Rotation angle in radians
        X0dot (float): X0 velocity
        Z0dot (float): Z0 velocity
        thetadot (float): deriv of angle of attack
        x (numpy array): x

    Returns:
        numpy array: Transformed array
    """
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    output = np.dot(rot,np.array([-X0dot,-Z0dot])) + np.array([0,thetadot*x])
    return output