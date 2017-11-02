# Replication of the Atari experiments done in Mnih et al. _Human-level control through deep reinforcement learning_

from six.moves import xrange

import numpy as np
from torch import optim

from rl_baseline.registration import env_registry
from rl_baseline.envs.atari import AtariDqnEnvWrapper
from rl_baseline.methods.dqn import DqnDeepMindModel, DqnTrainer, ReplayBuffer
from rl_baseline.util import copy_params, create_tb_writer, Saver, logger, report_model_stats

class DqnAtariReplayBuffer(ReplayBuffer):
    '''A custom replay buffer that is more space-efficient than the vanilla replay buffer.'''
    def __init__(self, capacity, ob_space, ac_space):
        obs_shape = [capacity] + list(ob_space.shape)
        self.obs = np.zeros(obs_shape, dtype='uint8')
        self.acs = np.zeros(capacity, dtype='int')
        self.rs = np.zeros(capacity, dtype='float')
        # Only store the new frame and avoid storing the overlapping frames
        self.next_frames = np.zeros([capacity] + list(ob_space.shape[1:]), dtype='uint8')
        self.dones = np.zeros(capacity, dtype='bool')
        self._count = 0
        self.capacity = capacity
        # Should be less than 33GB instead of over 400GB with the vanilla replay buffer
        logger.debug('Replay buffer took %i bytes', self.obs.nbytes + self.next_frames.nbytes + self.acs.nbytes + self.rs.nbytes + self.dones.nbytes)

    def add(self, ob, ac, r, next_ob, done):
        i = self.next_index
        self.obs[i] = ob
        self.acs[i] = ac
        self.rs[i] = r
        # Save the last frame in `next_ob`
        self.next_frames[i] = next_ob[-1]
        self.dones[i] = done

        self._count += 1

    @staticmethod
    def assemble_next_obs(obs, next_frames):
        # Concatenate the overlapping frames and the new frame
        return np.concatenate([obs[:, 1:], np.expand_dims(next_frames, 1)], axis=1)

    def sample_sars(self, size):
        '''Sample with replacement (s, a, r, s') tuples of `size`-length arrays.'''
        idxes = [np.random.randint(self.occupancy) for _ in xrange(size)]
        return self.obs[idxes], self.acs[idxes], self.rs[idxes], self.assemble_next_obs(self.obs[idxes], self.next_frames[idxes]), self.dones[idxes]


if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', default='Pong')
    parser.add_argument('-l', '--log-dir', default='/tmp/dqn-atari/pong')
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=None, help='GPU id for GPU computation. Default to use CPU.')
    parser.add_argument('-r', '--restore', dest='restore_from', help='Path to a checkpoint to be restored.')

    args = parser.parse_args()

    # Hyperparameters used in Extended Data Table 1
    # The unused parameters are commented out
    minibatch_size = 32
    replay_memory_size = 10 ** 6
    agent_history_length = 4
    target_network_update_frequency = 10 ** 4
    discount_factor = 0.99
    # action_repeat = 4
    update_frequency = 4
    learning_rate = 0.00025
    gradient_momentum = 0.95
    squared_gradient_momentum = 0.95
    min_squared_gradient = 0.01
    initial_exploration = 1
    final_exploration = 0.1
    final_exploration_frame = 10 ** 6
    replay_start_size = 50000
    # no_op_max = 30

    # The delta clipping done at https://github.com/deepmind/dqn/blob/master/dqn/NeuralQLearner.lua#L224 is equivalent to using huber loss
    loss_type = 'huber'

    # Make a properly preprocessed environment
    env = env_registry['gym.%sDeterministic-v4' % args.env].make()
    env = AtariDqnEnvWrapper(
        env=env,
        n_frames=agent_history_length,
        frame_width=84,
        clip_reward=1,
    )
    logger.info(env)
    logger.info(env.observation_space)
    logger.info(env.action_space)
    # Model according to supplement and official published code:
    # https://github.com/deepmind/dqn/blob/master/dqn/convnet_atari3.lua
    model = DqnDeepMindModel(env.observation_space, env.action_space)
    report_model_stats(model)
    target_model = DqnDeepMindModel(env.observation_space, env.action_space)
    if args.gpu_id is not None:
        # Move models to GPU
        model = model.cuda(args.gpu_id)
        target_model = target_model.cuda(args.gpu_id)
    opt = optim.RMSprop(
        params=model.parameters(),
        lr=learning_rate,
        momentum=gradient_momentum,
        eps=min_squared_gradient,
        alpha=squared_gradient_momentum,
    )
    # TODO Restore from a checkpoint
    if args.restore_from is not None:
        logger.info('Restoring checkpoint from %s', args.restore_from)
        checkpoint = torch.load(args.restore_from)
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])

    # Logging utilities
    # train_summary_dir = os.path.join(args.log_dir, 'train')
    # train_writer = create_tb_writer(train_summary_dir)
    # eval_summary_dir = os.path.join(args.log_dir, 'eval')
    # eval_writer = create_tb_writer(eval_summary_dir)
    train_writer, eval_writer = None, None
    saver = Saver(args.log_dir, model, opt, None, None)
    eval_env = env_registry['gym.%sDeterministic-v4' % args.env].make()
    eval_env = AtariDqnEnvWrapper(
        env=eval_env,
        n_frames=agent_history_length,
        frame_width=84,
        clip_reward=1,
    )

    # Training
    trainer = DqnTrainer(
        env=env,
        model=model,
        target_model=target_model,
        optimizer=opt,
        capacity=0,
        criterion=loss_type,
        max_grad_norm=0,
        target_update_interval=target_network_update_frequency,
        exploration_type='epsilon',
        initial_exploration=initial_exploration,
        terminal_exploration=final_exploration,
        exploration_start=replay_start_size,
        exploration_length=final_exploration_frame,
        minimal_replay_buffer_occupancy=replay_start_size,
        double_dqn=False,
        update_interval=update_frequency,
        batch_size=minibatch_size,
        gamma=discount_factor,
        eval_env=eval_env,
        train_summary_writer=train_writer,
        eval_summary_writer=eval_writer,
        saver=saver,
        gpu_id=args.gpu_id,
    )
    # HACK use a more efficient custom replay buffer
    trainer.replay_buffer = DqnAtariReplayBuffer(replay_memory_size, env.observation_space, env.action_space)

    # Extended table 2, 50M frames
    long_training_frames = 50 * 10**6
    # Extended table 3, 10M frames
    short_training_frames = 10 * 10**6

    trainer.train_for(
        max_ticks=short_training_frames,
        episode_report_interval=1,
        step_report_interval=1,
        checkpoint_interval=10000,
        eval_interval=0,
        n_eval_episodes=2,
        render=False,
    )
