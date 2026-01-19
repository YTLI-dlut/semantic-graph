from skimage import io
import matplotlib.pyplot as plt
import os
from skimage.measure import block_reduce

from sensor import *
from graph_generator import *
from node import *
import time

from parameter import *
if not train_mode:
    from test_parameter import *

np.set_printoptions(threshold=np.inf)

class Env():
    def __init__(self, map_index, num_robots, k_size=20, plot=False, test=False):
        # import environment ground truth from dungeon files
        self.test = test
        if self.test:
            self.map_dir = f'DungeonMaps/test'  # change to 'complex', 'medium', and 'easy'
        else:
            self.map_dir = f'DungeonMaps/train'
        self.map_list = os.listdir(self.map_dir)
        self.map_list.sort(reverse=True)
        self.map_index = map_index % np.size(self.map_list)
        self.ground_truth, self.start_position = self.import_ground_truth(
            self.map_dir + '/' + self.map_list[self.map_index])
        # self.start_position = np.array([190, 290])
        self.ground_truth_size = np.shape(self.ground_truth)  # (480, 640)
        self.num_robots = num_robots

        # initialize robot_belief
        self.robot_beliefs = []
        # self.robot_positions = [np.array([190, 290]), np.array([200, 290]), np.array([190, 300])]
        self.robot_positions = []
        self.new_robot_beliefs = []
        for k in range(num_robots):
            self.robot_beliefs.append(np.ones(self.ground_truth_size) * 127)
            self.robot_positions.append(self.start_position)
            self.new_robot_beliefs.append(np.ones(self.ground_truth_size) * 127)
        self.downsampled_belief = None
        self.old_robot_beliefs = []
        for k in range(self.num_robots):
            self.old_robot_beliefs.append(self.robot_beliefs[k].copy())
        self.combined_belief = np.ones(self.ground_truth_size) * 127

        # initialize parameters
        self.resolution = 4  # to downsample the map
        self.sensor_range = 70 # 80
        self.explored_rate = 0

        # initialize graph generator
        self.graph_generator = Graph_generator(map_size=self.ground_truth_size, sensor_range=self.sensor_range, k_size=k_size, plot=plot)
        self.graph_generator.route_node.append(self.start_position)
        self.node_coords, self.graph, self.node_utility, self.guidepost, self.node_k_flags = None, None, None, None, None
        self.frontiers = None
        self.k_frontiers = []

        self.stepi = 0
        self.final_iou = 0

        self.k_traj = [list() for i in range(self.num_robots)]

        self.begin()

        # plot related
        self.plot = plot
        self.frame_files = []
        self.iou_frame_files = []
        self.scatter_frame_files = []
        self.points = {}
        if self.plot:
            # initialize the route
            for i in range(self.num_robots):
                self.points['x'+str(i+1)] = [self.start_position[0]]
                self.points['y'+str(i+1)] = [self.start_position[1]]

    def find_index_from_coords(self, position):
        index = np.argmin(np.linalg.norm(self.node_coords - position, axis=1))
        return index
    
    def find_index_from_ground_truth_coords(self, position):
        index = np.argmin(np.linalg.norm(self.ground_truth_node_coords - position, axis=1))
        return index

    def begin(self):
        for k in range(self.num_robots):
            self.robot_beliefs[k] = self.update_robot_belief(self.robot_positions[k], self.sensor_range, self.robot_beliefs[k],
                                                     self.ground_truth)
        self.combined_belief = self.combine_belief(self.robot_beliefs)
                                                     
        # downsampled belief has lower resolution than robot belief
        self.downsampled_belief = block_reduce(self.combined_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)
        self.frontiers = self.find_frontier()
        self.old_combined_belief = copy.deepcopy(self.combined_belief)

        for i in range(self.num_robots):
            self.k_traj[i].append(self.start_position)

        self.node_coords, self.graph, self.node_utility, self.guidepost, self.node_k_flags = self.graph_generator.generate_graph(
            self.robot_positions, self.combined_belief, self.frontiers, self.k_traj)

        if USE_K_FLAGS:
            # update k_flags
            self.node_k_flags[self.find_index_from_coords(self.start_position)] = [1 for i in range(self.num_robots)]


    def step(self, robot_position, next_position, travel_dist, i_robot):
        # move the robot to the selected position and update its belief
        dist = np.linalg.norm(robot_position - next_position)
        travel_dist += dist

        robot_position = next_position
        self.graph_generator.route_node.append(robot_position)
        next_node_index = self.find_index_from_coords(robot_position)
        self.graph_generator.nodes_list[next_node_index].set_visited()
        i_robot_belief = self.update_robot_belief(robot_position, self.sensor_range, self.robot_beliefs[i_robot],
                                                    self.ground_truth)
        self.new_robot_beliefs[i_robot] = self.calculate_new_free_area(self.combined_belief, i_robot_belief)
        self.robot_beliefs[i_robot] = i_robot_belief

        self.combined_belief = self.combine_belief(self.robot_beliefs)
        self.downsampled_belief = block_reduce(self.combined_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)

        frontiers = self.find_frontier()
        self.explored_rate = self.evaluate_exploration_rate()
        self.final_iou, _ = self.belief_iou(self.robot_beliefs)
        # calculate the reward associated with the action
        self.robot_positions = robot_position
        reward = self.calculate_reward(dist, frontiers)

        if self.plot:
            self.points['x'+str(i_robot+1)].append(robot_position[0])
            self.points['y'+str(i_robot+1)].append(robot_position[1])

        # update the graph
        self.node_coords, self.graph, self.node_utility, self.guidepost, self.node_k_flags = self.graph_generator.update_graph(
            robot_position, self.combined_belief, self.old_combined_belief, frontiers, self.frontiers, self.new_robot_beliefs, self.k_traj)
        self.old_combined_belief = copy.deepcopy(self.combined_belief)

        self.frontiers = frontiers
        self.stepi += 1

        # check if done
        done = self.check_done()
        if done:
            reward += 30 # a finishing reward

        return reward, done, robot_position, travel_dist

    def import_ground_truth(self, map_index):
        # occupied 1, free 255, unexplored 127
        ground_truth = (io.imread(map_index, 1) * 255).astype(int)
        robot_location = np.nonzero(ground_truth == 208)
        robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])
        ground_truth = (ground_truth > 150)
        # save_ground_truth = ground_truth.copy()
        # io.imsave(map_index.split('/')[-1][:-4]+'_clean.png', save_ground_truth)
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_location

    def free_cells(self):
        index = np.where(self.ground_truth == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def update_robot_belief(self, robot_position, sensor_range, robot_belief, ground_truth):
        robot_beliefs = sensor_work(robot_position, sensor_range, robot_belief, ground_truth)
        return robot_beliefs

    def combine_belief(self, robot_beliefs):
        combined_belief = np.ones_like(robot_beliefs[0]) * 127
        for belief in robot_beliefs:
            combined_belief = np.where(belief == 127, combined_belief, belief)
        return combined_belief

    def check_done(self):
        done = False
        # if np.sum(self.node_utility) == 0:
        #     print('No more frontiers to explore')
        utility = np.copy(self.node_utility[:,0]) if USE_K_FLAGS else np.copy(self.node_utility)
        if self.test and np.sum(self.ground_truth == 255) - np.sum(self.combined_belief == 255) <= 250:
            done = True
        elif np.sum(utility) == 0:
            done = True
        return done

    def calculate_reward(self, dist, frontiers):
        reward = 0
        reward -= dist / 48
        # reward -= 1 # step penalty

        # check the num of observed frontiers
        frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
        pre_frontiers_to_check = self.frontiers[:, 0] + self.frontiers[:, 1] * 1j
        intersect = np.intersect1d(frontiers_to_check, pre_frontiers_to_check)
        frontiers_num = intersect.shape[0]
        pre_frontiers_num = pre_frontiers_to_check.shape[0]
        delta_num = pre_frontiers_num - frontiers_num

        reward += delta_num / 60
        
        if INTERSECTFRONTIER:
            cnt = 0
            for frontier in pre_frontiers_to_check:
                intersected = 0
                fpos = np.array([frontier.real, frontier.imag])
                for pos in self.robot_positions:
                    if np.linalg.norm(pos - fpos) < self.sensor_range and not self.graph_generator.check_collision(pos, fpos, self.ground_truth):
                        intersected += 1
                if intersected > 1:
                    cnt += 1
            reward -= cnt / 16


        # print('reward:', reward, 'dist:', dist, 'delta_num:', delta_num, 'frontiers_num:', frontiers_num, 'pre_frontiers_num:', pre_frontiers_num)
        # reward by IOU of each robot
        # if USE_NEW_REWARD:
        #     iou, _ = self.belief_iou(self.robot_beliefs)
        #     reward += (1-iou) * IOU_LAMBDA
        if USE_COLLISION:
            p = 0
            p_pos = []
            for k in range(self.num_robots):
                for pos in p_pos:
                    dis = np.linalg.norm(self.robot_positions[k] - pos)
                    collision = dis < 1
                    if collision:
                        p -= 1
                        print('Collision!')
                        break
                p_pos.append(self.robot_positions[k])
            reward += p / N_ROBOTS
        return reward
    
    def belief_iou(self, robot_beliefs):
        intersection = np.ones_like(self.ground_truth) * 255
        intersection = np.where(intersection.ravel(order='F') == 255)[0]
        union = np.zeros_like(self.ground_truth)
        union = np.where(union.ravel(order='F') == 255)[0]
        union_intersection = np.zeros_like(self.ground_truth)
        for k in range(self.num_robots):
            free_index = np.where(robot_beliefs[k].ravel(order='F') == 255)[0]
            for l in range(k+1, self.num_robots):
                intersection = np.intersect1d(np.where(robot_beliefs[l].ravel(order='F') == 255)[0], free_index)
                union_intersection = np.union1d(union_intersection, intersection)
            union = np.union1d(union, free_index)
        iou = union_intersection.shape[0] / union.shape[0] if union.shape[0] > 0 else 0
        union_intersection_2d = np.zeros_like(self.ground_truth)
        union_intersection_indexes = np.unravel_index(union_intersection, self.ground_truth.shape, order='F')
        union_intersection_2d[union_intersection_indexes] = 255
        return iou, union_intersection_2d

    def evaluate_exploration_rate(self):
        rate = np.sum(self.combined_belief == 255) / np.sum(self.ground_truth == 255)
        return rate

    def calculate_new_free_area(self, old_robot_belief, robot_belief):
        old_free_area = old_robot_belief == 255
        current_free_area = robot_belief == 255
        new_free_area = (current_free_area.astype(int) - old_free_area.astype(int)) * 255
        new_free_area = np.where(new_free_area == -255, 0, new_free_area)
        return new_free_area

    def calculate_path_length(self, path):
        dist = 0
        start = path[0]
        end = path[-1]
        for index in path:
            if index == end:
                break
            dist += np.linalg.norm(self.node_coords[start] - self.node_coords[index])
            start = index
        return dist

    def find_frontier(self):
        # find frontiers from downsampled_belief by checking nearby 8 cells for each cell
        y_len = self.downsampled_belief.shape[0]
        x_len = self.downsampled_belief.shape[1]
        mapping = self.downsampled_belief.copy()
        belief = self.downsampled_belief.copy()
        mapping = (mapping == 127) * 1
        mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                  mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
                                                                                                      2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(belief.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)

        map_x = x_len
        map_y = y_len
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T

        f = points[ind_to]
        f = f.astype(int)
        f = f * self.resolution

        return f

    def plot_env(self, n, path, step, travel_dist):
        plt.switch_backend('agg')
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
        # plt.ion()
        plt.cla()
        plt.suptitle('')
        plt.imshow(self.combined_belief, cmap='gray')
        plt.axis('off')
        # plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        # for i in range(len(self.graph_generator.x)):
        #    plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'orange', zorder=1)  # plot edges will take long time
        # for i in range(len(self.node_coords)):
        #     plt.text(self.node_coords[i][0], self.node_coords[i][1], str(int(i)), fontsize=8, color='r', zorder=6)
        # plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        for i in range(self.num_robots):
            plt.plot(self.points['x'+str(i+1)], self.points['y'+str(i+1)], colors[i], linewidth=2, zorder=9)
            plt.plot(self.points['x'+str(i+1)][-1], self.points['y'+str(i+1)][-1], 'mo', markersize=8, zorder=10)
            plt.plot(self.points['x'+str(i+1)][0], self.points['y'+str(i+1)][0], 'co', markersize=8)
        # plt.pause(0.1)
        # if USE_K_FLAGS:
        #     utility_norm = np.copy(self.node_utility[:,0])
        # else:
        #     utility_norm = np.copy(self.node_utility)
        # if (np.max(utility_norm) == np.min(utility_norm)) or (self.test and np.sum(self.ground_truth == 255) - np.sum(self.combined_belief == 255) <= 250):
        #     plt.savefig('{}/{}_{}_trajectory.png'.format(path, n, step, dpi=150))
        #     plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c='darkblue', zorder=5)
        # elif self.stepi == 128:
        #     plt.savefig('{}/{}_{}_trajectory.png'.format(path, n, step, dpi=150))
        #     plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=utility_norm, zorder=5, cmap='plasma')
        # else:
        #     plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=utility_norm, zorder=5, cmap='plasma')
        # plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g} Total step: {}'.format(self.explored_rate, travel_dist, self.stepi))
        plt.tight_layout()
        
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=300))
        # plt.show()
        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)
        plt.close()

    def plot_scatter_env(self, n, path, step, travel_dist):
        plt.switch_backend('agg')
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
        # plt.ion()
        plt.cla()
        plt.suptitle('')
        plt.imshow(self.combined_belief, cmap='gray')
        plt.axis('off')
        # plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        # for i in range(len(self.graph_generator.x)):
        #    plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'orange', zorder=1)  # plot edges will take long time
        for i in range(len(self.node_coords)):
            if USE_K_FLAGS:
                plt.text(self.node_coords[i][0], self.node_coords[i][1], str(int(self.node_utility[i][0])), fontsize=8, color='r', zorder=6)
            else:
                plt.text(self.node_coords[i][0], self.node_coords[i][1], str(int(self.node_utility[i])), fontsize=8, color='r', zorder=6)
        plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        for i in range(self.num_robots):
            plt.plot(self.points['x'+str(i+1)], self.points['y'+str(i+1)], colors[i], linewidth=2, zorder=9)
            plt.plot(self.points['x'+str(i+1)][-1], self.points['y'+str(i+1)][-1], 'mo', markersize=8, zorder=10)
            plt.plot(self.points['x'+str(i+1)][0], self.points['y'+str(i+1)][0], 'co', markersize=8)
        plt.pause(0.1)
        if USE_K_FLAGS:
            utility_norm = np.copy(self.node_utility[:,0])
        else:
            utility_norm = np.copy(self.node_utility)
        if (np.max(utility_norm) == np.min(utility_norm)) or (self.test and np.sum(self.ground_truth == 255) - np.sum(self.combined_belief == 255) <= 250):
            # plt.savefig('{}/{}_{}_trajectory.png'.format(path, n, step, dpi=150))
            plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c='darkblue', zorder=5)
        elif self.stepi == 128:
            # plt.savefig('{}/{}_{}_trajectory.png'.format(path, n, step, dpi=150))
            plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=utility_norm, zorder=5, cmap='plasma')
        else:
            plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=utility_norm, zorder=5, cmap='plasma')
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g} Total step: {}'.format(self.explored_rate, travel_dist, self.stepi))
        plt.tight_layout()
        
        plt.savefig('{}/{}_{}_scatters.png'.format(path, n, step, dpi=300))
        # plt.show()
        frame = '{}/{}_{}_scatters.png'.format(path, n, step)
        self.scatter_frame_files.append(frame)
        plt.close()

    def iou_frame(self):
        frame = np.copy(self.combined_belief)
        frame = np.where(frame == 127, 30, frame)
        for k in range(self.num_robots):
            frame = np.where(self.robot_beliefs[k] == 255, (k+1)*(255//N_ROBOTS+3), frame)
        iou, union_intersection_2d = self.belief_iou(self.robot_beliefs)
        frame = np.where(union_intersection_2d == 255, (k+2)*(255//N_ROBOTS+3), frame)
        return iou, frame

    def plot_iou(self,  n, path, step, travel_dist):
        plt.switch_backend('agg')
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
        # plt.ion()
        plt.cla()
        iou, frame = self.iou_frame()
        plt.imshow(frame, cmap='CMRmap')
        # plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        # for i in range(len(self.graph_generator.x)):
        #    plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'tan', zorder=1)  # plot edges will take long time
        
        # for i in range(len(self.node_coords)):
        #     plt.text(self.node_coords[i][0], self.node_coords[i][1], str(self.node_utility[i]), fontsize=8, color='r', zorder=6)
        plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        for i in range(self.num_robots):
            plt.plot(self.points['x'+str(i+1)], self.points['y'+str(i+1)], colors[i], linewidth=2, zorder=9)
            plt.plot(self.points['x'+str(i+1)][-1], self.points['y'+str(i+1)][-1], colors[i]+'o', markersize=8, zorder=10)
            plt.plot(self.points['x'+str(i+1)][0], self.points['y'+str(i+1)][0], 'co', markersize=8)
        # plt.pause(0.1)
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g} Total step: {} EIOU:{:.4g}'.format(self.explored_rate, travel_dist, self.stepi, iou))
        plt.tight_layout()
        
        plt.savefig('{}/{}_{}_ious.png'.format(path, n, step, dpi=150))
        # plt.show()
        frame = '{}/{}_{}_ious.png'.format(path, n, step)
        self.iou_frame_files.append(frame)

    def paper_plot_env(self, n, path, step, travel_dist):
        plt.switch_backend('agg')
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
        # plt.ion()
        plt.cla()
        plt.suptitle('')
        plt.imshow(self.combined_belief, cmap='gray')
        plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        for i in range(self.num_robots):
            plt.plot(self.points['x'+str(i+1)], self.points['y'+str(i+1)], colors[i], linewidth=2, zorder=9)
            plt.plot(self.points['x'+str(i+1)][-1], self.points['y'+str(i+1)][-1], colors[i]+'o', markersize=8, zorder=10)
            plt.plot(self.points['x'+str(i+1)][0], self.points['y'+str(i+1)][0], 'co', markersize=8)
        plt.axis('off')
        plt.savefig('{}/{}_{}_fig1.png'.format(path, n, step, dpi=300), transparent=True)
        plt.cla()

        if USE_K_FLAGS:
            utility_norm = np.copy(self.node_utility[:,0])
        else:
            utility_norm = np.copy(self.node_utility)
        plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=utility_norm, zorder=5, cmap='plasma')
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.savefig('{}/{}_{}_fig2.png'.format(path, n, step, dpi=150), transparent=True)
        
        for i in range(len(self.graph_generator.x)):
           plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'plum', zorder=1)  # plot edges will take long time
        plt.savefig('{}/{}_{}_fig3.png'.format(path, n, step, dpi=150), transparent=True)
        plt.cla()

        plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=utility_norm, zorder=5, cmap='plasma')
        # plt.axis('off')
        robotx = [self.points['x'+str(i+1)][-1] for i in range(self.num_robots)]
        roboty = [self.points['y'+str(i+1)][-1] for i in range(self.num_robots)]
        for i in range(self.num_robots):
            plt.plot(self.points['x'+str(i+1)][-1], self.points['y'+str(i+1)][-1], colors[i]+'o', markersize=8, zorder=10)
            plt.plot((robotx[i], robotx[i % self.num_robots]), (roboty[i], roboty[i % self.num_robots]), 'r', markersize=8, zorder=10)
            for j in self.graph_generator.nodes_list:
                if j.utility > 0:
                    plt.plot((robotx[i], j.coords[0]), (roboty[i], j.coords[1]), 'r', markersize=8, zorder=1)
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.savefig('{}/{}_{}_fig4.png'.format(path, n, step, dpi=150), transparent=True)
        # plt.pause(0.1)
        
        if (np.max(utility_norm) == np.min(utility_norm)) or (self.test and np.sum(self.ground_truth == 255) - np.sum(self.combined_belief == 255) <= 250):
            plt.savefig('{}/{}_{}_trajectory.png'.format(path, n, step, dpi=150))
            plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c='darkblue', zorder=5)
        elif self.stepi == 128:
            plt.savefig('{}/{}_{}_trajectory.png'.format(path, n, step, dpi=150))
            plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=utility_norm, zorder=5, cmap='plasma')
        else:
            plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=utility_norm, zorder=5, cmap='plasma')
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g} Total step: {}'.format(self.explored_rate, travel_dist, self.stepi))
        plt.tight_layout()
        
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=150))
        # plt.show()
        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)
        plt.close()

    def paper_plot_iou(self,  n, path, step, travel_dist):
        plt.switch_backend('agg')
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
        # plt.ion()
        plt.cla()
        iou, frame = self.iou_frame()
        plt.imshow(frame, cmap='CMRmap')
        plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        # for i in range(len(self.graph_generator.x)):
        #    plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'tan', zorder=1)  # plot edges will take long time
        
        # for i in range(len(self.node_coords)):
        #     plt.text(self.node_coords[i][0], self.node_coords[i][1], str(self.node_utility[i]), fontsize=8, color='r', zorder=6)
        plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        for i in range(self.num_robots):
            plt.plot(self.points['x'+str(i+1)], self.points['y'+str(i+1)], colors[i], linewidth=2, zorder=9)
            plt.plot(self.points['x'+str(i+1)][-1], self.points['y'+str(i+1)][-1], colors[i], markersize=8, zorder=10)
            plt.plot(self.points['x'+str(i+1)][0], self.points['y'+str(i+1)][0], 'co', markersize=8)
        # plt.pause(0.1)
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g} Total step: {} EIOU:{:.4g}'.format(self.explored_rate, travel_dist, self.stepi, iou))
        plt.tight_layout()
        
        plt.savefig('{}/{}_{}_ious.png'.format(path, n, step, dpi=150))
        # plt.show()
        frame = '{}/{}_{}_ious.png'.format(path, n, step)
        self.iou_frame_files.append(frame)