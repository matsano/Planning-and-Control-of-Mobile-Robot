"""
RRT_2D
@author: huiming zhou

Modified by David Filliat
"""

import os
import sys
import math
import numpy as np
import plotting, utils
import env

# parameters
showAnimation = True

# constants
NODE_LEFT = 0
NODE_RIGHT = 1
NODE_BELOW = 2
NODE_TOP = 3

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class Rrt:
    def __init__(self, environment, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]

        self.env = environment
        if showAnimation:
            self.plotting = plotting.Plotting(self.env, s_start, s_goal)
        self.utils = utils.Utils(self.env)

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def planning(self):
        iter_goal = None
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= self.step_len and iter_goal == None and not self.utils.is_collision(node_new, self.s_goal):
                    node_new = self.new_state(node_new, self.s_goal)
                    node_goal = node_new
                    iter_goal = i

        if iter_goal == None:
            return None, self.iter_max
        else:
            return self.extract_path(node_goal), iter_goal

    def generate_random_node(self, goal_sample_rate):
        if np.random.random() < goal_sample_rate:
            return self.s_goal
        delta = self.utils.delta
        
        # Percentage of points sampled using a simple variant of the OBRRT
        percentage_strategy = 0.3
        if np.random.random() <= percentage_strategy:
            new_node_generated = False
            while not new_node_generated:
                # Get random obstacle
                x_obs, y_obs, w_obs, h_obs = self.env.obs_rectangle[np.random.randint(len(self.env.obs_rectangle))]
                
                # Decide which node around the obstacle (within the obstacles edge) will be selected
                id_selected_node = np.random.randint(4)

                if id_selected_node == NODE_LEFT:
                    # The left edge of the obstacle
                    new_node = Node((np.random.uniform(x_obs - delta, x_obs),
                                     np.random.uniform(y_obs - delta, y_obs + h_obs + delta)))
                elif id_selected_node == NODE_RIGHT:
                    # The right edge of the obstacle
                    new_node = Node((np.random.uniform(x_obs + w_obs, x_obs + w_obs + delta),
                                     np.random.uniform(y_obs - delta, y_obs + h_obs + delta)))
                elif id_selected_node == NODE_BELOW:
                    # The edge below the obstacle
                    new_node = Node((np.random.uniform(x_obs - delta, x_obs + w_obs + delta),
                                     np.random.uniform(y_obs - delta, y_obs)))
                elif id_selected_node == NODE_TOP:
                    # The top edge of the obstacle
                    new_node = Node((np.random.uniform(x_obs - delta, x_obs + w_obs + delta),
                                     np.random.uniform(y_obs + h_obs, y_obs + h_obs + delta)))
                
                # Verify if the new node is not inside an obstacle
                if self.utils.is_inside_obs(new_node):
                    new_node_generated = True
        else:
            new_node = Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                             np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
            
        return new_node


    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def get_path_length(path):
    """
    Compute path length
    """
    length = 0
    for i,k in zip(path[0::], path[1::]):
        length += np.linalg.norm(np.array(i) - np.array(k)) # math.dist(i,k)
    return length


def main():
    x_start=(2, 2)  # Starting node
    x_goal=(49, 24)  # Goal node
    environment = env.Env2()

    rrt = Rrt(environment, x_start, x_goal, 2, 0.10, 3000)
    path, nb_iter = rrt.planning()

    if path:
        print('Found path in ' + str(nb_iter) + ' iterations, length : ' + str(get_path_length(path)))
        if showAnimation:
            rrt.plotting.animation(rrt.vertex, path, "RRT", True)
            plotting.plt.show()
    else:
        print("No Path Found in " + str(nb_iter) + " iterations!")
        if showAnimation:
            rrt.plotting.animation(rrt.vertex, [], "RRT", True)
            plotting.plt.show()

if __name__ == '__main__':
    main()
