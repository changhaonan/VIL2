"""Control utilities for VIL2."""
from __future__ import annotations
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
from matplotlib import pyplot as plt


def gen_path_from_wpts(way_pts, tics: float, vlim=None, alim=None):
    """Generate smooth path from wpts.
    Note: this wpts is parameterized a traj starting from static to static.
    """
    if isinstance(way_pts, list):
        way_pts = np.array(way_pts)
    path = ta.SplineInterpolator(np.linspace(0, 1, way_pts.shape[0]), way_pts)

    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(
        alim, discretization_scheme=constraint.DiscretizationType.Interpolation
    )
    instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper="seidel")
    jnt_traj = instance.compute_trajectory(0, 0)

    # detailed waypoints
    duration = jnt_traj.duration
    num_steps = int(duration / tics)
    ts = np.linspace(0, duration, num_steps)
    qs = jnt_traj.eval(ts)
    return ts, qs


def generate_whole_path_from_wpts(way_pts: np.ndarray | list, stops: np.ndarray | list[float], tics: float, vlim=None, alim=None):
    """Whole path contains muliple segments, each segment is a traj starting from static to static.
    Stops implies how long to stay static at each waypoint. If it is zero, then the robot will not stop at that waypoint.
    """
    if isinstance(way_pts, list):
        way_pts = np.array(way_pts)
    if isinstance(stops, list):
        stops = np.array(stops)
    assert way_pts.shape[0] == stops.shape[0]
    assert np.all(stops >= 0)
    assert stops[0] == 0 and stops[-1] > 0, "First and last waypoint should be static."
    # devide the whole path into segments based on stops
    # each segment is a traj starting from static to static
    seg_list = [[]]
    for i in range(way_pts.shape[0]):
        if stops[i] == 0:
            # no stop, just append the waypoint
            seg_list[-1].append(way_pts[i])
        else:
            # stops at this waypoint
            seg_list[-1].append(way_pts[i])
            if i != (way_pts.shape[0] - 1):
                # not the last waypoint, create a new segment
                seg_list.append([way_pts[i]])
    # generate path for each segment
    ts_list = []
    qs_list = []
    t = 0.0
    for seg in seg_list:
        ts, qs = gen_path_from_wpts(seg, tics, vlim, alim)
        ts_list.append(ts + t)
        qs_list.append(qs)

        # update t
        t = ts[-1] + tics

        # add stops
        num_stop_steps = int(stops[i] / tics)
        stop_ts = np.linspace(t, t + stops[i], num_stop_steps)
        stop_qs = np.array([qs[-1]] * num_stop_steps)
        ts_list.append(stop_ts)
        qs_list.append(stop_qs)

        t = stop_ts[-1] + tics
    # merge
    ts = np.hstack(ts_list)
    qs = np.vstack(qs_list)
    return ts, qs


def show_trajs(ts, qs):
    colors = plt.cm.jet(np.linspace(0, 1, len(ts)))
    # draw path in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(qs[:, 0], qs[:, 1], qs[:, 2], c=colors)
    # ax.plot(path[:, 0], path[:, 1], path[:, 2])
    plt.show()

    # animate the movement
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(len(ts)):
        plt.cla()
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        ax.set_zlim(-1, 3)
        ax.scatter(qs[i, 0], qs[i, 1], qs[i, 2], c=colors[i])
        plt.pause(0.1)



if __name__ == "__main__":
    # test 1: generate path from waypoints
    wpts = [
        [0, 0, 0],
        [2, 1, 1],
        [1, 0, 1],
        [1, 2, 0],
    ]
    vlim = np.array([[-1, 1], [-1, 1], [-1, 1]])
    alim = np.array([[-1, 1], [-1, 1], [-1, 1]])
    tics = 0.1
    # ts, qs = gen_path_from_wpts(wpts, num_steps=100, vlim=vlim, alim=alim)

    # test 2: generate whole path from waypoints
    stops = [0, 10, 0, 10]  # whole traj is 100 steps, how many steps to stop at each waypoint
    ts, qs = generate_whole_path_from_wpts(wpts, stops, tics=0.1, vlim=vlim, alim=alim)
    show_trajs(ts, qs)