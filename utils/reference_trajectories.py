import numpy as np

class Figure8Reference:
    def __init__(self, A=0.5, w=2*np.pi/6, segment_type='full'):
        self.A = A
        self.w = w
        self.segment_type = segment_type  # 'full', 'straight', or 'curve'

    def generate_reference(self, t):
        """Generate reference state for time t"""
        t = float(t)
        smooth_start = np.minimum(t/1.0, 1.0)
        
        x_ref = np.zeros(12)
        
        if self.segment_type == 'straight':
            # Diagonal straight line motion (x and z changing)
            x_ref[0] = self.A * t * smooth_start                # x increases linearly
            x_ref[1] = 0.0                                      # y stays at 0
            x_ref[2] = self.A * t * smooth_start               # z increases linearly
            
            # Constant velocities for straight line
            x_ref[6] = self.A * smooth_start                   # constant x velocity
            x_ref[7] = 0.0                                     # no y velocity
            x_ref[8] = self.A * smooth_start                   # constant z velocity
            
        elif self.segment_type == 'curve':
            # Half circle motion in x-z plane
            theta = np.pi * np.minimum(t/3.0, 1.0)  # angle parameter (0 to π)
            x_ref[0] = self.A * np.cos(theta) * smooth_start   # x: cos for half circle
            x_ref[1] = 0.0                                     # y stays at 0
            x_ref[2] = self.A * np.sin(theta) * smooth_start   # z: sin for half circle
            
            # Velocities for curved motion
            x_ref[6] = -self.A * np.pi/3.0 * np.sin(theta) * smooth_start  # x_dot
            x_ref[7] = 0.0                                                  # y_dot
            x_ref[8] = self.A * np.pi/3.0 * np.cos(theta) * smooth_start   # z_dot
            
        else:  # 'full' - original figure-8
            x_ref[0] = self.A * np.sin(self.w*t) * smooth_start        # x
            x_ref[1] = 0.0                                             # y
            x_ref[2] = self.A * np.sin(2*self.w*t)/2 * smooth_start   # z
            
            # Velocities
            x_ref[6] = self.A * self.w * np.cos(self.w*t) * smooth_start
            x_ref[7] = 0.0
            x_ref[8] = self.A * self.w * np.cos(2*self.w*t) * smooth_start
        
        return x_ref

    def get_final_reference(self, t):
        """Get reference position at time t"""
        t = float(t)
        if self.segment_type == 'straight':
            return {'x': 0.0, 'z': self.A * np.sin(2*self.w*t)/2}
        elif self.segment_type == 'curve':
            return {'x': self.A * np.sin(self.w*t), 'z': 0.0}
        else:  # 'full'
            return {
                'x': self.A * np.sin(self.w*t),
                'z': self.A * np.sin(2*self.w*t)/2
            }

    def get_trajectory_points(self, t_array):
        """Get trajectory points for visualization (no smooth start)"""
        points = np.zeros((len(t_array), 12))
        for i, t in enumerate(t_array):
            t = float(t)
            if self.segment_type == 'straight':
                # Diagonal straight line
                points[i, 0] = self.A * t  # x increases linearly
                points[i, 2] = self.A * t  # z increases linearly
            elif self.segment_type == 'curve':
                # Half circle
                theta = np.pi * np.minimum(t/3.0, 1.0)
                points[i, 0] = self.A * np.cos(theta)  # x
                points[i, 2] = self.A * np.sin(theta)  # z
            else:  # 'full'
                points[i, 0] = self.A * np.sin(self.w*t)
                points[i, 2] = self.A * np.sin(2*self.w*t)/2
        return points

        return points

   