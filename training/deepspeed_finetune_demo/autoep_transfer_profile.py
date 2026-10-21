# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Move live AutoEP training weights to new layouts and measure every step.

The source is a real DeepSpeed engine after a few training steps: AutoEP experts with
expert-data-parallel (EDP) replicas or ZeRO-3 fragments, and dense parameters replicated
(ZeRO-0/1/2) or partitioned (ZeRO-3). Storage maps from the engine feed the affine transfer
planner and executor, and every target element is checked bit for bit against a reference
gathered with plain collectives, independent of the affine code.

Each run records the plan analysis, the executor's own counts, and measured time and memory.
Replica selectors here are examples used to compare plans; they are not DeepSpeed policy. The
output is raw input for a cost model: nothing here fits a model, estimates time, or ranks plans.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine_transfer import PlanBudget, plan_transfer
from deepspeed.checkpoint.affine_transfer_analysis import (LOCAL, PEER, ReplicaRecorder, analyze_transfers,
                                                           distribution, expected_stats, replica_report,
                                                           report_to_dict)
from deepspeed.checkpoint.affine_transfer_executor import (StagingBudget, TransferEndpoints, TransferRequest,
                                                           execute_transfers, plan_digest)
from deepspeed.checkpoint.autoep_affine import autoep_storage_affine_map, make_autoep_placement_descriptor
from deepspeed.checkpoint.zero_affine import zero_param_storage_affine_map, zero_storage_affine_map
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

EXPERT_PARAMS = ("w1", "w2", "w3")
DTYPE = torch.bfloat16
CONCURRENT = "concurrent"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", required=True, help="HF model id or path; only its config is used")
    parser.add_argument("--num_layers", type=int, default=2, help="truncate the model to this many layers")
    parser.add_argument("--autoep_size", type=int, default=4)
    parser.add_argument("--zero_stage", type=int, default=2, choices=(0, 1, 2, 3))
    parser.add_argument("--train_steps", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--scenarios", default="permute,ep8,redundant,dense_tp4")
    parser.add_argument("--policies", default="first,round_robin,local_first,balanced")
    parser.add_argument("--budget_mb", type=float, default=8.0, help="executor staging bytes per peer")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--calibrate", action="store_true", help="record raw link and local copy times by size")
    parser.add_argument("--output", default="autoep_transfer_profile.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Source engine
# ---------------------------------------------------------------------------


def build_engine(args):
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.num_hidden_layers = args.num_layers
    if hasattr(config, "mlp_only_layers"):
        config.mlp_only_layers = []
    torch.manual_seed(1234)
    model = AutoModelForCausalLM.from_config(config, dtype=DTYPE, trust_remote_code=True)
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": args.zero_stage
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "torch_adam": True
            }
        },
        "expert_parallel": {
            "enabled": True,
            "autoep_size": args.autoep_size
        },
        "zero_allow_untested_optimizer": True,
    }
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
    device = torch.device(get_accelerator().current_device_name())
    generator = torch.Generator().manual_seed(4321 + dist.get_rank())
    for _ in range(args.train_steps):
        tokens = torch.randint(0, config.vocab_size, (1, args.seq_len), generator=generator).to(device)
        loss = engine(input_ids=tokens, labels=tokens).loss
        engine.backward(loss)
        engine.step()
    return engine, config


def autoep_layers(engine):
    return [(name, module) for name, module in engine.module.named_modules() if isinstance(module, AutoEPMoELayer)]


def flat_storage(param, shard_numel):
    # ZeRO-3 keeps the rank's partition in ds_tensor with trailing padding; the map excludes it.
    if hasattr(param, "ds_id"):
        return param.ds_tensor.data.view(-1)[:shard_numel]
    return param.data.view(-1)


def full_parameter(param):
    """The whole local parameter, gathered with ZeRO's own collective when it is partitioned."""
    if hasattr(param, "ds_id"):
        with deepspeed.zero.GatheredParameters([param]):
            return param.data.detach().clone()
    return param.data.detach().clone()


# ---------------------------------------------------------------------------
# Transfer items: one per parameter, with source map, target map and a reference
# ---------------------------------------------------------------------------


class Item:

    def __init__(self, key, source_map, target_map, source_buffer, reference_for):
        self.key = key
        self.source_map = source_map
        self.target_map = target_map
        self.source_buffer = source_buffer
        # Called with a target location; returns the expected flat contents of that shard.
        self.reference_for = reference_for


def target_rank_of(ep_rank, target_ep_size, world):
    return ep_rank * (world // target_ep_size)


def expert_targets(scenario, num_experts, world):
    """Expert ids held by each target EP rank, and the global rank holding that EP rank."""
    if scenario == "permute":
        # Same EP size as a 4-way source, experts dealt round-robin: about a quarter stay put.
        size = 4
        experts = [[expert for expert in range(num_experts) if expert % size == rank] for rank in range(size)]
    elif scenario == "ep8":
        size = world
        per_rank = num_experts // size
        experts = [list(range(rank * per_rank, (rank + 1) * per_rank)) for rank in range(size)]
    elif scenario == "redundant":
        # EPLB-style: every rank also hosts the first few "hot" experts. AutoEP cannot run this
        # layout yet; the transfer still has to fill it, so it is checked on buffers only.
        size = world
        per_rank = num_experts // size
        hot = list(range(min(4, num_experts)))
        experts = []
        for rank in range(size):
            own = list(range(rank * per_rank, (rank + 1) * per_rank))
            experts.append(own + [expert for expert in hot if expert not in own])
    else:
        raise ValueError(scenario)
    holders = {rank: [target_rank_of(rank, size, world)] for rank in range(size)}
    return experts, holders


def expert_items(engine, scenario, world):
    items = []
    for layer_name, layer in autoep_layers(engine):
        storage_maps = layer.expert_storage_affine_maps()
        num_experts = layer.num_experts
        experts_by_target, holders = expert_targets(scenario, num_experts, world)
        descriptor = make_autoep_placement_descriptor(num_experts, experts_by_target)
        location_experts = {holders[rank][0]: experts for rank, experts in enumerate(experts_by_target)}
        for name in EXPERT_PARAMS:
            param = getattr(layer.experts, name)
            physical_shape = getattr(param, "ds_shape", param.shape)
            logical_shape = (num_experts, *physical_shape[1:])
            target_map = autoep_storage_affine_map(descriptor, logical_shape, holders)
            source_map = storage_maps[name]
            rank = dist.get_rank()
            source_buffer = None
            if rank in source_map.shard_shapes:
                source_buffer = flat_storage(param, math.prod(source_map.shard_shapes[rank]))
            reference = ExpertReference(layer, param, num_experts, location_experts)
            items.append(Item(f"{layer_name}.{name}", source_map, target_map, source_buffer, reference))
    return items


class ExpertReference:
    """Gather the whole ``[num_experts, ...]`` tensor with plain collectives and slice by expert id.

    AutoEP packs EP rank ``r``'s experts as the contiguous range ``r * L .. (r + 1) * L - 1``; this
    reference relies on that packing directly rather than on any affine map.
    """

    def __init__(self, layer, param, num_experts, location_experts):
        self.layer = layer
        self.param = param
        self.num_experts = num_experts
        self.location_experts = location_experts
        self.full = None

    def gather(self):
        packed = full_parameter(self.param).contiguous()
        parts = [torch.empty_like(packed) for _ in range(self.layer.ep_size)]
        dist.all_gather(parts, packed, group=self.layer.ep_group)
        self.full = torch.cat(parts, dim=0)
        assert self.full.shape[0] == self.num_experts

    def __call__(self, location):
        experts = self.location_experts[location]
        return torch.cat([self.full[expert].reshape(-1) for expert in experts])

    def release(self):
        self.full = None


class DenseReference:

    def __init__(self, param, target_map):
        self.param = param
        self.target_map = target_map
        self.full = None

    def gather(self):
        self.full = full_parameter(self.param).reshape(-1)

    def __call__(self, location):
        piece = self.target_map.pieces_by_rank[location][0]
        return self.full[piece.source_offset:piece.source_offset + math.prod(piece.shape)]

    def release(self):
        self.full = None


def dense_items(engine, scenario, world):
    if scenario != "dense_tp4":
        raise ValueError(scenario)
    expert_ids = set()
    for _, layer in autoep_layers(engine):
        expert_ids.update(id(param) for param in layer.experts.parameters())
    targets = [target_rank_of(rank, 4, world) for rank in range(4)]
    items = []
    rank = dist.get_rank()
    for name, param in engine.module.named_parameters():
        if id(param) in expert_ids:
            continue
        shape = getattr(param, "ds_shape", param.shape)
        rows = shape[0]
        row_numel = math.prod(shape[1:])
        if rows < len(targets):
            continue
        # A row split over four ranks, like a TP4 consumer; on the flattened tensor each row block
        # is one contiguous range, so a ZeRO-style contiguous map describes it exactly.
        per_rank = math.ceil(rows / len(targets)) * row_numel
        target_map = zero_storage_affine_map(math.prod(shape), targets, partition_numel=per_rank)
        source_map = zero_param_storage_affine_map(param)
        source_buffer = None
        if rank in source_map.shard_shapes:
            source_buffer = flat_storage(param, math.prod(source_map.shard_shapes[rank]))
        items.append(Item(name, source_map, target_map, source_buffer, DenseReference(param, target_map)))
    return items


# ---------------------------------------------------------------------------
# Example replica selectors (comparison only, not DeepSpeed policy)
# ---------------------------------------------------------------------------


def make_selector(policy, endpoints, element_size):
    if policy == "first":
        return None
    if policy == "round_robin":
        counter = [0]

        def round_robin(request):
            chosen = request.candidates[counter[0] % len(request.candidates)].source_rank
            counter[0] += 1
            return chosen

        return round_robin
    if policy == "local_first":

        def local_first(request):
            target_process = endpoints.target[request.target_rank]
            for candidate in request.candidates:
                if endpoints.source[candidate.source_rank] == target_process:
                    return candidate.source_rank
            return request.candidates[0].source_rank

        return local_first
    if policy == "balanced":
        load = {}

        def balanced(request):
            # Greedy: read from the holder that has sent the fewest bytes so far.
            chosen = min(request.candidates,
                         key=lambda candidate: (load.get(endpoints.source[candidate.source_rank], 0),
                                                candidate.source_rank)).source_rank
            process = endpoints.source[chosen]
            load[process] = load.get(process, 0) + math.prod(request.shape) * element_size
            return chosen

        return balanced
    raise ValueError(policy)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def synchronize():
    get_accelerator().synchronize()


def max_over_ranks(value):
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=get_accelerator().current_device_name())
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def sum_over_ranks(value):
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=get_accelerator().current_device_name())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def nan_targets(items, endpoints, device):
    rank = dist.get_rank()
    buffers = []
    for item in items:
        targets = {}
        for location, shape in item.target_map.shard_shapes.items():
            if endpoints.target[location] == rank:
                targets[location] = torch.full(shape, float("nan"), dtype=DTYPE, device=device)
        buffers.append(targets)
    return buffers


def requests_for(items, plans, target_buffers):
    rank = dist.get_rank()
    requests = []
    for item, plan, targets in zip(items, plans, target_buffers):
        sources = {rank: item.source_buffer} if item.source_buffer is not None else {}
        requests.append(TransferRequest(item.key, plan, sources, targets))
    return requests


def check_bitwise(items, target_buffers):
    """Count target elements that differ from the reference bit for bit, or were never written."""
    mismatched, unwritten, checked = 0, 0, 0
    for item, targets in zip(items, target_buffers):
        item.reference_for.gather()
        for location, buffer in targets.items():
            expected = item.reference_for(location).to(buffer.device)
            got = buffer.reshape(-1)
            assert got.numel() == expected.numel(), (item.key, location)
            unwritten += int(torch.isnan(got).sum().item()) - int(torch.isnan(expected).sum().item())
            bad = int((got.view(torch.int16) != expected.view(torch.int16)).sum().item())
            if bad and os.environ.get("XFER_DEBUG"):
                print(f"rank {dist.get_rank()} {item.key} loc {location}: {bad}/{got.numel()} bad, first at "
                      f"{int((got.view(torch.int16) != expected.view(torch.int16)).nonzero()[0])}", flush=True)
            mismatched += bad
            checked += got.numel()
        item.reference_for.release()
    return mismatched, unwritten, checked


def all_ranks(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def time_execute(requests, endpoints, budget):
    """Run one batch; return the executor's stats and every rank's elapsed seconds."""
    dist.barrier()
    synchronize()
    started = time.perf_counter()
    stats = execute_transfers(requests, endpoints, DTYPE, budget)
    synchronize()
    elapsed = time.perf_counter() - started
    return stats, all_ranks(elapsed)


# ---------------------------------------------------------------------------
# Calibration: raw per-path times by message size, for an external cost model
# ---------------------------------------------------------------------------


def exchange(send, recv, to_process, from_process):
    """Post one send and one receive in a single group call; either side may be None."""
    ops = []
    # Sends are posted before receives: oneCCL hangs on a group call that starts with a receive.
    if to_process is not None:
        ops.append(dist.P2POp(dist.isend, send, to_process))
    if from_process is not None:
        ops.append(dist.P2POp(dist.irecv, recv, from_process))
    if ops:
        for handle in dist.batch_isend_irecv(ops):
            handle.wait()


def timed_exchange(send, recv, to_process, from_process):
    dist.barrier()
    synchronize()
    started = time.perf_counter()
    exchange(send, recv, to_process, from_process)
    synchronize()
    return time.perf_counter() - started


def calibrate(device, repeats=10, warmup=2):
    """Record raw link times by message size, with no fit.

    - peer: every ordered rank pair alone, so that links through different switches and
      sockets are all sampled; the time of a sample is the slower of its two endpoints.
    - concurrent: every rank sends to rank + shift and receives from rank - shift at once,
      for every shift, so that links are sampled while they share switches and root ports.
    - local: a device-local copy on every rank.
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    sizes = [1 << shift for shift in range(12, 27, 2)]
    peer_times = {}
    concurrent_times = {}
    local_times = {}
    for size in sizes:
        numel = size // 2
        send = torch.ones(numel, dtype=DTYPE, device=device)
        recv = torch.empty(numel, dtype=DTYPE, device=device)
        for source in range(world):
            for target in range(world):
                if source == target:
                    continue
                times = []
                for repeat in range(warmup + repeats):
                    to_process = target if rank == source else None
                    from_process = source if rank == target else None
                    seconds = timed_exchange(send, recv, to_process, from_process)
                    if repeat >= warmup and rank in (source, target):
                        times.append(seconds)
                if times:
                    peer_times[(size, source, target)] = times
        for shift in range(1, world):
            times = []
            for repeat in range(warmup + repeats):
                seconds = timed_exchange(send, recv, (rank + shift) % world, (rank - shift) % world)
                if repeat >= warmup:
                    times.append(seconds)
            concurrent_times[(size, shift)] = times
        times = []
        for repeat in range(warmup + repeats):
            synchronize()
            started = time.perf_counter()
            recv.copy_(send)
            synchronize()
            if repeat >= warmup:
                times.append(time.perf_counter() - started)
        local_times[size] = times

    # Timings stay rank-local until here so that no collective runs inside a timed region.
    gathered = all_ranks((peer_times, concurrent_times, local_times))
    samples = {PEER: [], CONCURRENT: [], LOCAL: []}
    for size in sizes:
        for source in range(world):
            for target in range(world):
                if source == target:
                    continue
                source_times = gathered[source][0][(size, source, target)]
                target_times = gathered[target][0][(size, source, target)]
                samples[PEER].append({
                    "bytes": size,
                    "from_process": source,
                    "to_process": target,
                    "seconds": [max(pair) for pair in zip(source_times, target_times)],
                })
        for shift in range(1, world):
            seconds_by_process = [gathered[process][1][(size, shift)] for process in range(world)]
            samples[CONCURRENT].append({"bytes": size, "shift": shift, "seconds_by_process": seconds_by_process})
        for process in range(world):
            samples[LOCAL].append({"bytes": size, "process": process, "seconds": gathered[process][2][size]})
    return samples


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def manifest(args, world):
    accelerator = get_accelerator()
    return {
        "args": vars(args),
        "world_size": world,
        "device_name": accelerator.get_device_name(0) if hasattr(accelerator, "get_device_name") else None,
        "accelerator": accelerator.device_name(),
        "torch": torch.__version__,
        "deepspeed": deepspeed.__version__,
        "deepspeed_commit": git_commit(os.path.dirname(os.path.dirname(deepspeed.__file__))),
        "host": platform.node(),
    }


def run_scenario(engine, scenario, policies, args, device):
    world = dist.get_world_size()
    rank = dist.get_rank()
    items = dense_items(engine, scenario, world) if scenario.startswith("dense") else expert_items(
        engine, scenario, world)
    endpoints = TransferEndpoints(source={location: location
                                          for location in range(world)},
                                  target={location: location
                                          for location in range(world)})
    budget = StagingBudget(bytes_per_peer=int(args.budget_mb * (1 << 20)))
    element_size = torch.empty((), dtype=DTYPE).element_size()

    runs = []
    for policy in policies:
        recorder = ReplicaRecorder(make_selector(policy, endpoints, element_size))
        started = time.perf_counter()
        plans = [plan_transfer(item.target_map, item.source_map, PlanBudget(), replica_selector=recorder) for item in items]
        plan_seconds = max_over_ranks(time.perf_counter() - started)
        started = time.perf_counter()
        report = analyze_transfers(plans, endpoints, element_size, budget)
        replicas = replica_report(recorder, plans, endpoints, element_size)
        analyze_seconds = max_over_ranks(time.perf_counter() - started)

        targets = nan_targets(items, endpoints, device)
        get_accelerator().reset_peak_memory_stats()
        memory_before = get_accelerator().memory_allocated()
        stats, first_seconds = time_execute(requests_for(items, plans, targets), endpoints, budget)
        first_seconds = max(first_seconds)
        peak_extra = max_over_ranks(get_accelerator().max_memory_allocated() - memory_before)
        stats_match = asdict(stats) == expected_stats(report, rank)
        mismatched, unwritten, checked = check_bitwise(items, targets)
        runs.append({
            "plans": plans,
            "targets": targets,
            "record": {
                "scenario": scenario,
                "policy": policy,
                "plan_digests": [plan_digest(plan, endpoints, budget, element_size) for plan in plans],
                "plan_seconds": plan_seconds,
                "analyze_seconds": analyze_seconds,
                "first_execute_seconds": first_seconds,
                "peak_extra_allocated_bytes": peak_extra,
                "stats_match_all_ranks": sum_over_ranks(0 if stats_match else 1) == 0,
                "bitwise": {
                    "mismatched_elements": int(sum_over_ranks(mismatched)),
                    "unwritten_elements": int(sum_over_ranks(unwritten)),
                    "checked_elements": int(sum_over_ranks(checked)),
                },
                "analysis": report_to_dict(report, replicas, include_messages=True),
                "execute_seconds": [],
                "execute_seconds_by_rank": [],
            },
            "remote_bytes": report.remote_bytes,
        })

    # Interleave policies in each window so slow drift affects every policy alike.
    for repeat in range(args.warmup + args.repeats):
        for run in runs:
            _, seconds = time_execute(requests_for(items, run["plans"], run["targets"]), endpoints, budget)
            if repeat >= args.warmup:
                run["record"]["execute_seconds"].append(max(seconds))
                run["record"]["execute_seconds_by_rank"].append(seconds)

    records = []
    for run in runs:
        record = run["record"]
        timing = distribution(record["execute_seconds"])
        record["execute"] = timing
        median = timing.get("p50")
        if median:
            record["remote_bytes_per_second"] = run["remote_bytes"] / median
        records.append(record)
    return records


def main():
    args = parse_args()
    get_accelerator().set_device(int(os.environ.get("LOCAL_RANK", 0)))
    deepspeed.init_distributed()
    device = torch.device(get_accelerator().current_device_name())
    world = dist.get_world_size()
    engine, _ = build_engine(args)
    output = {"manifest": manifest(args, world)}

    if args.calibrate:
        output["calibration"] = calibrate(device)

    output["runs"] = []
    policies = args.policies.split(",")
    for scenario in args.scenarios.split(","):
        records = run_scenario(engine, scenario, policies, args, device)
        output["runs"].extend(records)
        if dist.get_rank() == 0:
            for record in records:
                bits = record["bitwise"]
                print(f"[{scenario:>10}/{record['policy']:<11}] stats_match={record['stats_match_all_ranks']} "
                      f"mismatched={bits['mismatched_elements']} unwritten={bits['unwritten_elements']} "
                      f"p50={record['execute'].get('p50', float('nan')) * 1e3:.2f} ms "
                      f"plan={record['plan_seconds'] * 1e3:.1f} ms",
                      flush=True)

    if dist.get_rank() == 0:
        with open(args.output, "w") as stream:
            json.dump(output, stream, indent=2, default=str)
        print(f"wrote {args.output}")
    failed = [
        record for record in output["runs"]
        if not record["stats_match_all_ranks"] or record["bitwise"]["mismatched_elements"]
        or record["bitwise"]["unwritten_elements"]
    ]
    engine.destroy()
    if failed:
        raise SystemExit(f"{len(failed)} runs failed correctness")


if __name__ == "__main__":
    main()
