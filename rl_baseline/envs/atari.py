from gym import spaces
import numpy as np
from scipy.misc import imresize
from gym.envs.registration import registry as gym_env_registry

from rl_baseline.core import Wrapper, Spec
from rl_baseline.registry import env_registry

class AtariDqnEnvWrapper(Wrapper):
    '''Environment wrapper that produces observations and rewards in the specification of Mnih et al.'''
    def __init__(self, env, n_frames=4, frame_width=84, clip_reward=1):
        '''
        Args:
            clip_reward : float
                The maximum magnitude of the reward. If `clip_reward` is 0, disable reward clipping.
        '''
        super().__init__(env)
        self.frame_width = frame_width
        self.n_frames = n_frames
        self._reset_frame_buffer()
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.n_frames, self.frame_width, self.frame_width))
        self.clip_reward = clip_reward

    def _dqn_preprocess(self, ob):
        # Grayscale
        ob = ob.mean(-1)
        # Resize to 84 x 84
        ob = imresize(ob, (self.frame_width, self.frame_width))
        return ob

    def _reset_frame_buffer(self):
        self.frame_buffer = np.zeros((self.n_frames, self.frame_width, self.frame_width), dtype='uint8')

    def _add_frame_to_buffer(self, ob):
        # Shift the frames forward by one, dropping the first one (oldest)
        self.frame_buffer[:-1] = self.frame_buffer[1:]
        self.frame_buffer[-1] = self._dqn_preprocess(ob)

    def _reset(self):
        self._reset_frame_buffer()
        ob = super()._reset()
        self._add_frame_to_buffer(ob)
        return self.frame_buffer

    def _step(self, ac):
        ob, r, done, extra = super()._step(ac)
        self._add_frame_to_buffer(ob)
        if self.clip_reward != 0:
            r = np.clip(r, -self.clip_reward, self.clip_reward)
        return self.frame_buffer, r, done, extra
