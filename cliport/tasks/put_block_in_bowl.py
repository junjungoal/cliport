"""Put Blocks in Bowl Task."""

import re
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import random
import pybullet as p


class PutBlockInBowlUnseenColors(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the {pick} block in a {place} bowl"
        self.lang_re_template = r"put the (.*) in a (.*)"
        self.task_completed_desc = "done placing blocks in bowls."

    def reset(self, env):
        super().reset(env)
        # n_bowls = np.random.randint(1, 4)
        # n_blocks = np.random.randint(1, n_bowls + 1)
        n_bowls = 1
        n_blocks = 1

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, 2)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[1] + [1])
            bowl_poses.append(bowl_pose)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[0] + [1])
            blocks.append((block_id, (0, None)))

        # Goal: put each block in a different bowl.
        self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))),
                           bowl_poses, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(pick=selected_color_names[0],
                                                         place=selected_color_names[1]))

        self.geometric_predicates['On'] = {}
        self.geometric_predicates['On']['{} block'.format(selected_color_names[0])] = 'table'
        self.geometric_predicates['On']['{} bowl'.format(selected_color_names[1])] = 'table'
        self.present_objects.append('{} block'.format(selected_color_names[0]))
        self.present_objects.append('{} bowl'.format(selected_color_names[1]))
        self.present_objects.append('robot')
        self.present_objects.append('table')

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 1

        # Colors of distractor objects.
        distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        distractor_bowl_color_names = [c for c in utils.COLORS if c not in selected_color_names]
        distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        distractor_block_color_names = [c for c in utils.COLORS if c not in selected_color_names]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        while n_distractors < max_distractors:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_block_colors if is_block else distractor_bowl_colors
            color_names = distractor_block_color_names if is_block else distractor_bowl_color_names
            pose = self.get_random_pose(env, size)
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose)
            color = colors[n_distractors % len(colors)]
            color_name = color_names[n_distractors % len(colors)]

            obj_type = 'block' if is_block else 'bowl'
            self.present_objects.append('{} {}'.format(color_name, obj_type))
            self.geometric_predicates['On']['{} {}'.format(color_name, obj_type)] = 'table'
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1

    def update_predicates(self):
        state = self.primitive.state
        m = re.match(self.lang_re_template, self.lang_goals[0])
        if state == 'grasp':
            if m.group(1) in self.geometric_predicates['On'].keys():
                del self.geometric_predicates['On'][m.group(1)]
            self.geometric_predicates['Grasped'] = m.group(1)
        elif state == 'place':
            if 'Grasped' in self.geometric_predicates.keys():
                del self.geometric_predicates['Grasped']
            self.geometric_predicates['On'][m.group(1)] = m.group(2)

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutBlockInBowlSeenColors(PutBlockInBowlUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class PutBlockInBowlFull(PutBlockInBowlUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors
