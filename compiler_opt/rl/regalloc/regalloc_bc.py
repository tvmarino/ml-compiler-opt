import os
import shutil
import subprocess
import contextlib
import json

import gin
import tensorflow as tf
from absl import app, flags, logging
import numpy as np

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import log_reader
from compiler_opt.rl import env

from tf_agents.trajectories import time_step
from compiler_opt.rl.imitation_learning import generate_bc_trajectories_lib

from compiler_opt.rl.regalloc import config

ModuleExplorer = generate_bc_trajectories_lib.ModuleExplorer
ModuleWorkerResultProcessor = generate_bc_trajectories_lib.ModuleWorkerResultProcessor
ProfilingDictValueType = generate_bc_trajectories_lib.ProfilingDictValueType


class RegallocTask(env.MLGOTask):
  """Implementation of the inlining-for-size MLGOTask."""

  module_name: str | None = None
  target_module_root_path: str | None = None
  compile_only: bool = True
  num_perf_reps: int = 8

  def __init__(
      self,
      default_reward_key: str = 'default',
      base_target_path: str = '/mlgo_regalloc_clang/thinlto_build/clang_corpus_target/objs/',
      # base_target_path: str = '/tmp/clang_corpus/objs',
      baseline_path: str = '/mlgo_regalloc_clang/thinlto_build/clang_corpus/objs',
      # baseline_path: str = '/tmp/clang_corpus_bak/objs',
      link_command_path: str = '/mlgo_regalloc_clang/thinlto_build/clang_corpus/link.sh',
      # link_command_path: str = '/tmp/clang_corpus/link.sh',
      # perf5_command_path: str = '/mlgo_regalloc_clang/thinlto_build/perf5_miba.sh'
      perf5_command_path: str = '/mlgo_regalloc_clang/thinlto_build/perf5_command.sh'
  ):
    super().__init__()
    self._default_reward_key: str = default_reward_key
    self._base_target_path: str = base_target_path
    self._baseline_path: str = baseline_path
    # with open(link_command_path, encoding='utf-8') as link_command_file:
    #   link_command = link_command_file.read()
    # link_command = link_command.strip('\n')
    # link_command = link_command.strip('\t')
    # self._link_command: list[str] = link_command.split(' ')
    # run_sh_path = os.path.join(base_target_path, 'run_ar.sh')
    # self._llvm_ar_command: list[str] = ['bash'] + [run_sh_path]
    self._link_command = ['bash'] + [link_command_path]
    # with open(perf5_command_path, encoding='utf-8') as perf5_command_file:
    #   perf5_command = perf5_command_file.read()
    # perf5_command = perf5_command.strip('\n')
    # perf5_command = perf5_command.strip('\t')
    # self._perf5_command: list[str] = perf5_command.split(' ')
    self._perf5_command: list[str] = ['bash'] + [perf5_command_path]

  def get_cmdline(self, clang_path: str, base_args: list[str],
                  interactive_base_path: str | None,
                  working_dir: str) -> list[str]:
    if interactive_base_path:
      interactive_args = [
          '-mllvm',
          '-regalloc-enable-advisor=release',
          '-mllvm',
          f'-regalloc-evict-interactive-channel-base={interactive_base_path}',
          '-mllvm',
          '-regalloc-evict-interactive-include-default',
      ]
    else:
      interactive_args = []
    compiled_module_path = os.path.join(working_dir, self.module_name)
    return [clang_path
           ] + base_args + interactive_args + ['-o', compiled_module_path]

  def get_module_scores(self, working_dir: str) -> dict[str, float]:

    assert self.module_name is not None
    assert self.target_module_root_path is not None

    if self.compile_only:
      return {self._default_reward_key: -1., 'variance': 0.}

    baseline_path = os.path.join(self._baseline_path, 'blaze-out')
    # base_target_path = os.path.join(self._base_target_path, 'blaze-out')
    base_target_path = self._base_target_path

    shutil.copy(
        os.path.join(working_dir, self.module_name),
        self.target_module_root_path,
    )
    subprocess.run(
      'echo 1 | sudo tee /proc/sys/vm/drop_caches'.split(' '),
      capture_output=True,
      check=True,
    )
    # subprocess.run(
    #     self._llvm_ar_command,
    #     cwd=self._base_target_path,
    #     capture_output=True,
    #     check=True)
    subprocess.run(
        self._link_command,
        # cwd=self._base_target_path,
        cwd='/google/src/cloud/tvmarinov/regalloc_workflow/google3',
        capture_output=True,
        check=True)

    subprocess.run(['bash', '/mlgo_regalloc_clang/thinlto_build/miba_scp.sh'],
                 capture_output=True,
                 check=True)
    perf5_outs = []
    for _ in range(self.num_perf_reps):
      while True:
        try:
          perf5_completed = subprocess.run(
              self._perf5_command, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
          print(e)
          continue
        # TODO(tvmarinov): why is this stderr
        perf5_out = perf5_completed.stderr
        perf5_out = [
            i for i in str(perf5_out.decode('utf-8')).split('\n\n')[2].split(' ')
            if i != ''
        ]
        if '(' not in perf5_out[-1]:
          break
      cycles = float(perf5_out[0].replace(',', ''))
      logging.info('cycles: %f', cycles)
      # variance = float(perf5_out[-2].replace('%', ''))
      perf5_outs.append(cycles)
    buckets = np.array_split(perf5_outs, 4)
    bucket_means = [np.median(bucket) for bucket in buckets]
    cycles = np.median(perf5_outs)
    variance = np.std(bucket_means)

    subprocess.run(['rsync', '-av', baseline_path, base_target_path],
             capture_output=False,
             check=True)

    return {self._default_reward_key: cycles, 'variance': variance/cycles}


def default_policy(curr_obs_dict: time_step.TimeStep) -> np.ndarray:
  curr_obs = curr_obs_dict.observation
  action = np.array(curr_obs['default_decision'])
  return action


class ExploreOnHot:
  """Explore based on the value of features which reflect hotness."""

  def __init__(self, hot_functions_path):
    self._hot_functions: dict[str, float] = {}
    with open(hot_functions_path, encoding='utf-8') as hot_functions:
      hot_functions = hot_functions.read()
      lines = hot_functions.strip().split('\n')
      line_num = 0
      # Parse header
      for line in lines:
        line_num += 1
        if line.startswith(
            '# Overhead'):  # Corrected to match the actual header line
          # Found the header line for the table, stop parsing header
          break
      for line in lines[line_num + 2:]:
        try:
          overhead = float(line.split()[0][:-1])
        except (IndexError, ValueError):
          continue
        func_name = line.split()[-1]
        if func_name in self._hot_functions:
          self._hot_functions[func_name] += overhead
        else:
          self._hot_functions[func_name] = overhead

  def __call__(self, obs_dict_seq: list[env.TimeStep]):

    all_obs = []
    for obs_dict in obs_dict_seq:
      mult_fact = 0.
      if obs_dict.context in self._hot_functions:
        mult_fact = self._hot_functions[obs_dict.context]
      all_obs.append(mult_fact *
                     np.max(obs_dict.obs['hottest_bb_freq_by_max'].numpy()))
    all_obs_dict = {}
    index = 0
    for obs in all_obs:
      if obs not in all_obs_dict:
        all_obs_dict[obs] = [index]
      else:
        all_obs_dict[obs].append(index)
      index += 1
    key_order = np.sort(-1.0*np.array(list(all_obs_dict.keys())))
    exploration_steps = []
    for key in key_order:
      exploration_steps.extend(all_obs_dict[float(-key)])

    return exploration_steps


def main(_):
  clang_path = '/mlgo_regalloc_clang/thinlto_build/clang'
  explicit_temps_dir = '/mlgo_regalloc_clang/thinlto_build/modules/explicit_temps'
  base_target_path = '/mlgo_regalloc_clang/thinlto_build/clang_corpus_target/objs/'
  cps = corpus.Corpus(
      data_path='/mlgo_regalloc_clang/thinlto_build/clang_corpus',
      replace_flags={
          '-fprofile-instrument-use-path':
              '/mlgo_regalloc_clang/thinlto_build/clang_corpus/MergedCS.profdata'
      })
  module_worker_result_processor = ModuleWorkerResultProcessor(
      persistent_objects_path='/mlgo_regalloc_clang/thinlto_build/persistent_objs'
  )
  partition_list = [np.inf]
  # partition_list = [
  #   25133353369.6, 25159745126.4, 25186690662.4,
  #   25231208448.0, 25285903974.4, 25361701683.2,
  #   25374454988.8, 25403411906.56
  # ]
  succeeded = []

  corpus_elements = cps.module_specs
  work = [
      cps.load_module_spec(corpus_element) for corpus_element in corpus_elements
  ]

  compile_only = False
  if not compile_only:
    # policies = [default_policy]
    policies = []
    policy_paths = [
        '/mlgo_regalloc_clang/policies/saved_model1_wraped/',
        '/mlgo_regalloc_clang/policies/thinlto_es_expl_retired_0/',
    ]
    for policy_path in policy_paths:
      tf_policy = tf.saved_model.load(policy_path, tags=None, options=None)
      policies.append(
          generate_bc_trajectories_lib.policy_action_wrapper(tf_policy))
    # policies.append(default_policy)
    explore_policy_paths = [
      '/mlgo_regalloc_clang/policies/es_thinlto_weighted/',
      '/mlgo_regalloc_clang/policies/thinlto_es_expl_retired_0/',
    ]
    # explore_policies = [None]
    explore_policies = []
    for policy_path in explore_policy_paths:
      tf_policy = tf.saved_model.load(policy_path, tags=None, options=None)
      explore_policies.append(
          generate_bc_trajectories_lib.policy_distr_wrapper(tf_policy))
  else:
    policies = []
    # policies = [default_policy]
    policy_paths = ['/mlgo_regalloc_clang/policies/saved_model1_wraped/']
    # policy_paths = []
    for policy_path in policy_paths:
      tf_policy = tf.saved_model.load(policy_path, tags=None, options=None)
      policies.append(
          generate_bc_trajectories_lib.policy_action_wrapper(tf_policy))
    explore_policies = [None]

  output_path = '/mlgo_regalloc_clang/training_data/records'
  file_name = 'thinlto_es_expl_retired_01.tfrecord'
  profiling_file_path = '/mlgo_regalloc_clang/training_data/profiles/thinlto_es_expl_retired_01'
  total_profiles_max: list[ProfilingDictValueType | None] = []
  total_profiles_pol: list[ProfilingDictValueType | None] = []
  tf_rec_path = (
      os.path.join(output_path, file_name)
      if output_path else contextlib.nullcontext())
  tfrecord_context = (
      tf.io.TFRecordWriter(tf_rec_path)
      if output_path else contextlib.nullcontext())

  with tfrecord_context as tfrecord_writer:
    for loaded_module_spec in work:
      target_module_path = os.path.join(base_target_path,
                                        loaded_module_spec.name)
      target_module_root_path = target_module_path.split('/')
      target_module_root_path = '/'.join(target_module_root_path[:-1]) + '/'
      # target_module_bak_path = target_module_path + '.bak'
      # shutil.copy(target_module_path, target_module_bak_path)
      module_name = loaded_module_spec.name.split('/')[-1]

      time_step_spec, action_spec = config.get_regalloc_signature_spec()
      RegallocTask.module_name = module_name
      RegallocTask.target_module_root_path = target_module_root_path
      RegallocTask.compile_only = compile_only
      RegallocTask.num_perf_reps = 8

      explore_on_hot = ExploreOnHot(
          hot_functions_path='/mlgo_regalloc_clang/thinlto_build/perf_thinlto.txt'
      )
      exploration_worker = ModuleExplorer(
          loaded_module_spec=loaded_module_spec,
          clang_path=clang_path,
          mlgo_task_type=RegallocTask,
          exploration_frac=1.0,
          max_exploration_steps=10,
          explore_on_features=None,
          obs_action_specs=(time_step_spec, action_spec),
          explicit_temps_dir=explicit_temps_dir,
          reward_key='default',
          explore_on_features_full_seq={
              'hottest_bb_freq_by_max': explore_on_hot
          }
          # gin_config_str=gin.config_str()
      )

      exploration_results = []
      try:
        for policy, exploration_policy in zip(policies, explore_policies):
          exploration_res = exploration_worker.explore_function(
              policy, exploration_policy)
          exploration_results.append(exploration_res)
        succeeded.append(
            module_worker_result_processor.process_succeeded(
                succeeded=exploration_results,
                spec_name=loaded_module_spec.name,
                partitions=partition_list,
            ))
      except (ValueError, AssertionError, TypeError, FileNotFoundError):
        # os.replace(target_module_bak_path, target_module_path)
        continue

      while succeeded:
        records, profiles_max, profiles_pol = succeeded.pop()
        total_profiles_max.append(profiles_max)
        total_profiles_pol.append(profiles_pol)
      if tfrecord_writer:
        tfrecord_writer.write(records.SerializeToString())
      # TODO(@tvmarinov): Move the copying of the default module to the reward fn
      # os.replace(target_module_bak_path, target_module_path)

  if profiling_file_path:
    max_profiles_path = profiling_file_path + '_max.json'
    pol_profiles_path = profiling_file_path + '_pol.json'
    with open(max_profiles_path, 'w+', encoding='utf-8') as prof_writer_max:
      with open(pol_profiles_path, 'w+', encoding='utf-8') as prof_writer_pol:
        json.dump(total_profiles_max, prof_writer_max, indent=2)
        json.dump(total_profiles_pol, prof_writer_pol, indent=2)


if __name__ == '__main__':
  app.run(main)
