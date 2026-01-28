import os
import subprocess

from absl import app, flags, logging
import numpy as np

_BASELINE_PATH = flags.DEFINE_string(
  'baseline_path',
  '/mlgo_regalloc_clang/thinlto_build/clang_corpus/objs/blaze-out',
  # '/tmp/clang_corpus_bak/objs/blaze-out/',
  ('Path to baseline to copy over into target'
   'before overwriting with target modules.')
)

_BUILD_TARGET_PATH = flags.DEFINE_string(
    'build_target_path',
    '/mlgo_regalloc_clang/thinlto_build/clang_corpus_target/objs/',
    # '/tmp/clang_corpus_bak/objs/blaze-out/',
    'Path to target build where modules will be overwritten'
)

_TARGET_PATH = flags.DEFINE_string(
    'target_path', None,
    'Path to target modules to evaluate'
)

_NUM_REPETITIONS = flags.DEFINE_integer(
    'num_repetitions', 1,
    'Number of times to run the eval for'
)

_LINK_COMMAND_PATH = flags.DEFINE_string(
   'link_command_path',
  '/mlgo_regalloc_clang/thinlto_build/clang_corpus/link.sh',
   'Path to a text file containing the linking command.'
)

_PERF5_COMMAND_PATH = flags.DEFINE_string(
   'perf5_command_path',
   '/mlgo_regalloc_clang/thinlto_build/perf5_miba.sh',
  #  '/mlgo_regalloc_clang/thinlto_build/perf5_command.sh',
   'Path to a text file containing the perf5 command.'
)

_AR_COMMAND_SH = flags.DEFINE_string(
   'ar_command_sh', '/mlgo_regalloc_clang/llvm-project/build_final_target/run_ar.sh',
   ('Name of a bash script to run the ar command pre-linking.',
    'Script must be in the baseline_path folder')
)


def main(_):
  baseline_path: str = _BASELINE_PATH.value
  base_target_path: str = _BUILD_TARGET_PATH.value
  target_path: str = _TARGET_PATH.value
  num_perf_reps: int = _NUM_REPETITIONS.value
  link_command_path: str = _LINK_COMMAND_PATH.value
  perf5_command_path: str = _PERF5_COMMAND_PATH.value
  ar_command_sh_name: str = _AR_COMMAND_SH.value

  # with open(link_command_path, encoding='utf-8') as link_command_file:
  #   link_command = link_command_file.read()
  # link_command = link_command.strip('\n')
  # link_command = link_command.strip('\t')
  # link_command: list[str] = link_command.split(' ')

  link_command: list[str] = ['bash'] + [link_command_path]
#   llvm_ar_command: list[str] = ['bash'] + [ar_command_sh_name]
  # with open(perf5_command_path, encoding='utf-8') as perf5_command_file:
  #   perf5_command = perf5_command_file.read()
  # perf5_command = perf5_command.strip('\n')
  # perf5_command = perf5_command.strip('\t')
  # perf5_command: list[str] = perf5_command.split(' ')
  perf5_command: list[str] = ['bash'] + [perf5_command_path]
  rsync_command = (f'rsync -av {target_path}'
                   f' {base_target_path}')
  rsync_command: list[str] = rsync_command.split(' ')

  subprocess.run(['rsync', '-av', baseline_path, base_target_path],
                 capture_output=False,
                 check=True)

  subprocess.run(rsync_command, capture_output=False, check=True)

  # subprocess.run(
  #     llvm_ar_command, cwd=base_target_path, capture_output=True, check=True)

  # subprocess.run(
  #     link_command, cwd=base_target_path, capture_output=True, check=True)
  subprocess.run(
      link_command,
      cwd='/google/src/cloud/tvmarinov/regalloc_workflow/google3',
      capture_output=True,
      check=True)

  subprocess.run(['bash', '/mlgo_regalloc_clang/thinlto_build/miba_scp.sh'],
                 capture_output=True,
                 check=True)
  subprocess.run(
      'echo 1 | sudo tee /proc/sys/vm/drop_caches'.split(' '),
      capture_output=True,
      check=True)

  perf5_outs = []
  for _ in range(num_perf_reps):
    while True:
      try:
        perf5_completed = subprocess.run(
            perf5_command, capture_output=True, check=True)
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
  std = np.std(bucket_means)
  # std = np.std(perf5_outs)
  logging.info('Median cycles: %f, mean cycles: %f, std dev: %f', cycles,
               np.mean(perf5_outs), std / cycles)
#   return {self._default_reward_key: cycles, 'variance': variance/cycles}



if __name__ == '__main__':
  app.run(main)
