'''
Classes implementing various kinematic chains. This module is perhaps mis-located
as it does not have a direct BMI role but rather contains code which is useful in
supporting BMI control of kinematic chains.

This code depends on the 'robot' module (https://github.com/sgowda/robotics_toolbox)
'''
import numpy as np
try:
    import robot
except ImportError:
    import warnings
    warnings.warn("The 'robot' module cannot be found! See https://github.com/sgowda/robotics_toolbox")

import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import izip

import time

pi = np.pi

class KinematicChain(object):
    '''
    Arbitrary kinematic chain (i.e. spherical joint at the beginning of 
    each joint)
    '''
    def __init__(self, link_lengths=[10., 10.], name=''):
        self.n_links = len(link_lengths)
        self.link_lengths = link_lengths

        links = []
        for link_length in link_lengths:
            link1 = robot.Link(alpha=-pi/2)
            link2 = robot.Link(alpha=pi/2)
            link3 = robot.Link(d=-link_length)
            links += [link1, link2, link3]
        
        # By convention, we start the arm in the XY-plane
        links[1].offset = -pi/2 

        self.robot = robot.SerialLink(links)
        self.robot.name = name

    def calc_full_joint_angles(self, joint_angles):
        return joint_angles

    def full_angles_to_subset(self, joint_angles):
        return joint_angles

    def plot(self, joint_angles):
        joint_angles = self.calc_full_joint_angles(joint_angles)
        self.robot.plot(joint_angles)

    def forward_kinematics(self, joint_angles, **kwargs):
        '''
        Calculate forward kinematics using D-H parameter convention
        '''
        joint_angles = self.calc_full_joint_angles(joint_angles)
        t, allt = self.robot.fkine(joint_angles, **kwargs)
        self.joint_angles = joint_angles
        self.t = t
        self.allt = allt
        return t, allt

    def apply_joint_limits(self, joint_angles):
        return joint_angles

    def inverse_kinematics(self, target_pos, q_start=None, method='pso', **kwargs):
        if q_start == None:
            q_start = self.random_sample()
        return self.inverse_kinematics_pso(target_pos, q_start, **kwargs)
        # ik_method = getattr(self, 'inverse_kinematics_%s' % method)
        # return ik_method(q_start, target_pos)

    def inverse_kinematics_grad_descent(self, target_pos, starting_config, n_iter=1000, 
                           verbose=False, eps=0.01, return_path=False):
        '''
        Default inverse kinematics method is RRT since for redundant 
        kinematic chains, an infinite number of inverse kinematics solutions 
        exist
        '''

        q = starting_config
        start_time = time.time()
        endpoint_traj = np.zeros([n_iter, 3])

        joint_limited = np.zeros(len(q))

        for k in range(n_iter):
            # print k
            # calc endpoint position of the manipulator
            endpoint_traj[k] = self.endpoint_pos(q)

            current_cost = np.linalg.norm(endpoint_traj[k] - target_pos, 2)
            if current_cost < eps:
                print "Terminating early"
                break

            # calculate the jacobian
            J = self.jacobian(q)
            J_pos = J[0:3,:]

            # for joints that are at their limit, zero out the jacobian?
            # J_pos[:, np.nonzero(self.calc_full_joint_angles(joint_limited))] = 0

            # take a step from the current position toward the target pos using the inverse Jacobian
            J_inv = np.linalg.pinv(J_pos)
            # J_inv = J_pos.T

            xdot = (target_pos - endpoint_traj[k])#/np.linalg.norm(endpoint_traj[k] - target_pos) 

            # if current_cost < 3 or k > 10:
            #     stepsize = 0.001
            # else:
            #     stepsize = 0.01


            xdot = (target_pos - endpoint_traj[k])#/np.linalg.norm(endpoint_traj[k] - target_pos)
            # xdot = (endpoint_traj[k] - target_pos)/np.linalg.norm(endpoint_traj[k] - target_pos)
            qdot = 0.001*np.dot(J_inv, xdot)
            qdot = self.full_angles_to_subset(np.array(qdot).ravel())

            q += qdot

            # apply joint limits
            q, joint_limited = self.apply_joint_limits(q)

        end_time = time.time()
        runtime = end_time - start_time
        if verbose:
            print "Runtime: %g" % runtime
            print "# of iterations: %g" % k

        if return_path:
            return q, endpoint_traj[:k]
        else:
            return q

    def jacobian(self, joint_angles):
        '''
        Return the full jacobian 
        '''
        joint_angles = self.calc_full_joint_angles(joint_angles)
        J = self.robot.jacobn(joint_angles)
        return J

    def endpoint_pos(self, joint_angles):
        t, allt = self.forward_kinematics(joint_angles)
        return np.array(t[0:3,-1]).ravel()

    def random_sample(self):
        q_start = []
        for lim_min, lim_max in self.joint_limits:
            q_start.append(np.random.uniform(lim_min, lim_max))
        return np.array(q_start)

    def ik_cost(self, q, q_start, target_pos, weight=100):
        q_diff = q - q_start
        return np.linalg.norm(q_diff[0:2]) + weight*np.linalg.norm(self.endpoint_pos(q) - target_pos)

    def inverse_kinematics_pso(self, target_pos, q_start, time_limit=np.inf, verbose=False, eps=0.5, n_particles=10, n_iter=10):
        # Initialize the particles; 
        n_joints = self.n_joints

        particles_q = np.tile(q_start, [n_particles, 1])

        # if 0:
        #     # initialize the velocities to be biased around the direction the jacobian tells you is correct
        #     current_pos = self.endpoint_pos(q_start)
        #     int_displ = target_pos - current_pos
        #     print int_displ, target_pos
        #     J = self.jacobian(q_start)
        #     endpoint_vel = np.random.randn(n_particles, 3)# + int_displ
        #     particles_v = np.dot(J[0:3,1::3].T, endpoint_vel.T).T
        # else:
        #     # initialize particle velocities randomly

        
        particles_v = np.random.randn(n_particles, n_joints) #/ np.array([1., 1., 1, 1]) #np.array(self.link_lengths)

        cost_fn = lambda q: self.ik_cost(q, q_start, target_pos)

        gbest = particles_q.copy()
        gbestcost = np.array(map(cost_fn, gbest))
        pbest = gbest[np.argmin(gbestcost)]
        pbestcost = cost_fn(pbest)

        min_limits = np.array([x[0] for x in self.joint_limits])
        max_limits = np.array([x[1] for x in self.joint_limits])
        min_limits = np.tile(min_limits, [n_particles, 1])
        max_limits = np.tile(max_limits, [n_particles, 1])

        start_time = time.time()
        for k in range(n_iter):
            if time.time() - start_time > time_limit:
                break

            # update positions of particles
            particles_q += particles_v

            # apply joint limits
            min_viol = particles_q < min_limits
            max_viol = particles_q > max_limits
            particles_q[min_viol] = min_limits[min_viol]
            particles_q[max_viol] = max_limits[max_viol]

            # update the costs
            costs = np.array(map(cost_fn, particles_q))

            # update the 'bests'
            gbest[gbestcost > costs] = particles_q[gbestcost > costs]
            gbestcost[gbestcost > costs] = costs[gbestcost > costs]

            idx = np.argmin(gbestcost)
            pbest = gbest[idx]
            pbestcost = gbestcost[idx]

            # update the velocity
            phi1 = 1#np.random.rand()
            phi2 = 1#np.random.rand()
            w=0.25
            c1=0.5
            c2=0.25
            particles_v = w*particles_v + c1*phi1*(pbest - particles_q) + c2*phi2*(gbest - particles_q)

            error = np.linalg.norm(self.endpoint_pos(pbest) - target_pos)
            if error < eps:
                break
            
        end_time = time.time()
        if verbose: print "Runtime = %g, error = %g, n_iter=%d" % (end_time-start_time, error, k)

        return pbest

    def spatial_positions_of_joints(self, joint_angles):
        _, allt = self.forward_kinematics(joint_angles, return_allt=True)
        pos = allt[0:3, -1]
        pos = np.hstack([np.zeros([3,1]), pos])
        return pos

class PlanarXZKinematicChain(KinematicChain):
    '''
    Kinematic chain restricted to movement in the XZ-plane
    '''
    def calc_full_joint_angles(self, joint_angles):
        '''
        only some joints rotate in the planar kinematic chain

        '''
        if not len(joint_angles) == self.n_links:
            raise ValueError("Incorrect number of joint angles specified!")

        # There are really 3 angles per joint to allow 3D rotation at each joint
        joint_angles_full = np.zeros(self.n_links * 3)  
        joint_angles_full[1::3] = joint_angles
        return joint_angles_full 

    def full_angles_to_subset(self, joint_angles):
        return joint_angles[1::3]

    def apply_joint_limits(self, joint_angles):
        if not hasattr(self, 'joint_limits'):
            return joint_angles
        else:
            angles = []
            limit_hit = []
            for angle, (lim_min, lim_max) in izip(joint_angles, self.joint_limits):
                limit_hit.append(angle < lim_min or angle > lim_max)
                angle = max(lim_min, angle)
                angle = min(angle, lim_max)
                angles.append(angle)

            return np.array(angles), np.array(limit_hit)

    @property 
    def n_joints(self):
        return len(self.link_lengths)

    def spatial_positions_of_joints(self, *args, **kwargs):
        pos_all_joints = super(PlanarXZKinematicChain, self).spatial_positions_of_joints(*args, **kwargs)
        return pos_all_joints[:,::3]

    def create_ik_subchains(self):
        proximal_link_lengths = self.link_lengths[:2]
        distal_link_lengths = self.link_lengths[2:]
        self.proximal_chain = PlanarXZKinematicChain2Link(proximal_link_lengths)
        self.distal_chain = PlanarXZKinematicChain(distal_link_lengths)

    def inverse_kinematics(self, target_pos, **kwargs):
        if not hasattr(self, 'proximal_chain') or not hasattr(self, 'distal_chain'):
            self.create_ik_subchains()
        distal_angles = kwargs.pop('distal_angles', None)

        if distal_angles is None:
            # Sample randomly from the joint limits (-pi, pi) if not specified
            if not hasattr(self, 'joint_limits') or len(self.joint_limits) < len(self.link_lengths):
                joint_limits = [(-pi, pi)] * len(self.distal_chain.link_lengths)
            else:
                joint_limits = self.joint_limits[2:]
            distal_angles = np.array([np.random.uniform(*limits) for limits in joint_limits])

        distal_displ = self.distal_chain.endpoint_pos(distal_angles)
        proximal_endpoint_pos = target_pos - distal_displ
        proximal_angles = self.proximal_chain.inverse_kinematics(proximal_endpoint_pos).ravel()
        angles = distal_angles.copy()
        joint_angles = proximal_angles.tolist()
        angles[0] -= np.sum(proximal_angles)
        return np.hstack([proximal_angles, angles])

    def jacobian(self, theta):
        l = self.link_lengths
        N = len(theta)
        J = np.zeros([2, len(l)])
        for m in range(N):
            for i in range(m, N):
                J[0, m] += -l[i]*np.sin(sum(theta[:i+1]))
                J[1, m] +=  l[i]*np.cos(sum(theta[:i+1]))
        return J


class PlanarXZKinematicChain2Link(PlanarXZKinematicChain):
    def __init__(self, link_lengths, *args, **kwargs):
        if not len(link_lengths) == 2:
            raise ValueError("Can't instantiate a 2-link arm with > 2 links!")

        super(PlanarXZKinematicChain2Link, self).__init__(link_lengths, *args, **kwargs)

    def inverse_kinematics(self, pos, **kwargs):
        '''
        Inverse kinematics for a two-link kinematic chain. These equations can be solved
        deterministically. 
        '''
        l_upperarm, l_forearm = self.link_lengths 

        if np.ndim(pos) == 1:
            pos = pos.reshape(1,-1)

        # require the y-coordinate to be 0, i.e. flat on the screen
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        assert np.all(np.abs(np.array(y)) < 1e-10)

        L = np.sqrt(x**2 + z**2)
        cos_el_pflex = (L**2 - l_forearm**2 - l_upperarm**2) / (2*l_forearm*l_upperarm)

        cos_el_pflex[ (cos_el_pflex > 1) & (cos_el_pflex < 1 + 1e-9)] = 1
        el_pflex = np.arccos(cos_el_pflex)

        sh_pabd = np.arctan2(z, x) - np.arcsin(l_forearm * np.sin(np.pi - el_pflex) / L)
        return np.array([-sh_pabd, -el_pflex])