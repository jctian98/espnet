#!/usr/bin/python

import sys
import os
import subprocess
import concurrent.futures
import uuid
from pathlib import Path


cpu_per_gpu = 32
mem_per_gpu = 200
gpu_partition = "interactive_singlenode,backfill_singlenode,batch_singlenode,polar,polar3,polar4"
gpu_partition = "backfill_singlenode,batch_singlenode,polar,polar3,polar4"
gpu_partition_multinode = "polar,polar3,polar4"
cpu_partition = "cpu"
image = "/lustre/fsw/portfolios/adlr/users/sanggill/docker/unifugatto:250519.sqsh"
mounts = "/home/jinchuant,/lustre/fsw/portfolios/llmservice/users/jinchuant"
duration = 4


def parse_commands():
    args = sys.argv[1:]

    # if name specified
    if args[0] == "--name":
        name = args[1].replace("/", "_")
        args = args[2:]
    else:
        name = "exp"

    # If request GPU:
    if args[0] == "--gpu":
        ngpu = int(args[1])
        args = args[2:]
        partition = gpu_partition
        cpu = cpu_per_gpu * ngpu
        mem = mem_per_gpu * ngpu
    else:     
        # NOTE(Jinchuan): don't use cpu partition, use GPU anyway
        ngpu = 1
        partition = gpu_partition
        cpu = cpu_per_gpu
        mem = mem_per_gpu
    
    # thread is not needed
    if args[0] == "--num_threads":
        args = args[2:]
    
    # num of nodes
    if args[0] == "--num_nodes":
        num_nodes = int(args[1])
        args = args[2:]
        if num_nodes > 1:
            partition = gpu_partition_multinode
    else:
        num_nodes = 1
    
    # if array processing
    if args[0].startswith("JOB="):
        job_array = args[0].replace("JOB=", "")
        job_start, job_end = job_array.split(":")
        job_start, job_end = int(job_start), int(job_end) + 1
        args = args[1:]
    else:
        job_start, job_end = 1, 2

    # log_file must be the final argument before the real command.
    log_file = args[0]
    args = args[1:]
    print(f"Launch experiment with log file {log_file}")

    cmd = " ".join(args)

    all_submit_cmd = list()
    for idx in range(job_start, job_end):

        if ngpu == 1: # batch job
            this_cmd = cmd.replace("JOB", f"{str(idx)}")
        else: # training, the JOB should be kept
            this_cmd = cmd
        this_log_file = log_file.replace("JOB", str(idx))

        submit_cmd =  f"submit_job "
        submit_cmd += f"--email_mode fail "
        submit_cmd += f"-n {name}_{idx}_{uuid.uuid4()} "
        submit_cmd += f"--partition {partition} "
        submit_cmd += f"--nodes {num_nodes} "
        if ngpu == 8: # take the whole node: cpu/gpu/mem
            submit_cmd += "--exclusive "
        else:
            submit_cmd += f"--gpu {ngpu} "
            submit_cmd += f"--cpu {cpu} "
            submit_cmd += f"--mem {mem} "
        submit_cmd += f"--image {image} "
        submit_cmd += f"--mounts {mounts} "
        submit_cmd += f"--duration {duration} "
        submit_cmd += f"-c \"source path.sh; {this_cmd}\" " # important to call path.sh

        # Multi-GPU job should be the training job
        if ngpu > 1:
            # NOTE(Jinchuan): send the terminate signal just 1 hour after the job launch.
            # The job will only detect the termination signal when it finishes the 
            # scheduled steps
            submit_cmd += "--autoresume_method submit_job --pre_timeout_signal 235 "
        
        if "JOB" in log_file: # batch job for data processing, wait it to end
            submit_cmd += "-w "
            submit_cmd += f"> {this_log_file} 2>&1 "
        else:
            submit_cmd += f"--outfile {this_log_file} "
        
        print(submit_cmd)

        all_submit_cmd.append(submit_cmd)
    
    return (job_start, job_end), all_submit_cmd, log_file

def execute_command(cmd):
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return {
        'command': cmd,
        'success': process.returncode == 0,
        'return_code': process.returncode,
        'stdout': process.stdout,
        'stderr': process.stderr
    }

def main():
    (job_start, job_end), command_list, log_file = parse_commands()
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        # Submit all commands to be executed
        future_to_cmd = {executor.submit(execute_command, cmd): cmd for cmd in command_list}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_cmd):
            print('finish one job', future, flush=True)
            try:
                results.append(future.result())
            except Exception as exc:
                cmd = future_to_cmd[future]
                results.append({
                    'command': cmd,
                    'success': False,
                    'return_code': -1,
                    'stderr': f'Exception: {exc}'
                })
    
    # Sort results to match original command order
    results.sort(key=lambda x: command_list.index(x['command']))

    # Keep all results into logs after everything is done, which is not ideal
    succ_count = 0
    tot_count = len(results)
    fail_idx = list()
    for idx, result in zip(range(job_start, job_end), results):
        if result['success']:
            succ_count += 1
        else:
            fail_idx.append(idx)
        
    if succ_count == tot_count:
        print('all jobs finish successfully')
    else:
        print(
            f"{succ_count}/{tot_count} jobs are successful. Failure index: {fail_idx} "
            f"Log file: {log_file}"
        )

if __name__ == "__main__":
    main()