import ray
import numpy as np
import os
import torch

from model import PolicyNet
from test_worker import TestWorker
from test_parameter import *


def run_test():
    if not os.path.exists(trajectory_path):
        os.makedirs(trajectory_path)

    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(INPUT_DIM + int(USE_K_FLAGS)*N_ROBOTS, EMBEDDING_DIM).to(device)

    if device == 'cuda':
        checkpoint = torch.load(f'{model_path}/checkpoint.pth')
    else:
        checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location = torch.device('cpu'))

    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    # print(global_network.previous_downsample.state_dict())
    curr_test = 0

    dist_history = []
    step_history = []
    EIOU_history = []
    TIME_history = []
    l90_history = []
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(dist_history) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                dist_history.append(metrics['travel_dist'])
                step_history.append(metrics['steps'])
                EIOU_history.append(metrics['EIOU'])
                # l90_history.append(metrics['90'])
                # TIME_history.append(max(metrics['k_dist']))
            if curr_test < NUM_TEST:
                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                curr_test += 1

        print('|#Total test:', NUM_TEST)
        print('|#Average length:', np.array(dist_history).mean())
        # print(dist_history)
        print('|#Length std:', np.array(dist_history).std())
        print('|#Average steps:', np.array(step_history).mean())
        print('|#Steps std:', np.array(step_history).std())
        print('|#Average EIOU:', np.array(EIOU_history).mean())
        print('|#EIOU std:', np.array(EIOU_history).std())
        # print('|#Average TIME:', np.array(TIME_history).mean())
        # print('|#Average l90:', np.array(l90_history).mean())
        # print('|#l90 std:', np.array(l90_history).std())

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(INPUT_DIM + int(USE_K_FLAGS)*N_ROBOTS, EMBEDDING_DIM)
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        worker = TestWorker(self.meta_agent_id, self.local_network,episode_number, device=self.device, save_image=SAVE_GIFS, greedy=True)
        worker.work(episode_number)

        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_weights(weights)

        metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    for i in range(NUM_RUN):
        run_test()
