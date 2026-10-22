# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Move live AutoEP training weights to new layouts and measure every step.

The source is a real DeepSpeed engine after a few training steps: AutoEP experts with
expert-data-parallel (EDP) replicas or ZeRO-3 fragments. Storage maps from the engine feed the
affine transfer planner and executor, and every target element is checked bit for bit against a reference
gathered with plain collectives, independent of the affine code.

Each run records the executor's own counts, and measured time and memory.
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
from deepspeed.checkpoint.affine_transfer_calibration import calibrate_links
from deepspeed.checkpoint.affine_transfer_executor import (StagingBudget, TransferEndpoints, TransferRequest,
                                                           execute_transfers, plan_digest)
from deepspeed.checkpoint.autoep_affine import autoep_storage_affine_map, make_autoep_placement_descriptor
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

EXPERT_PARAMS = ("w1", "w2", "w3")
DTYPE = torch.bfloat16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", required=True, help="HF model id or path; only its config is used")
    parser.add_argument("--num_layers", type=int, default=2, help="truncate the model to this many layers")
    parser.add_argument("--autoep_size", type=int, default=4)
    parser.add_argument("--zero_stage", type=int, default=2, choices=(0, 1, 2, 3))
    parser.add_argument("--train_steps", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--scenarios", default="permute,ep8,redundant")
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


def distribution(values):
    """Count, sum and order statistics, with nearest-rank percentiles."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "sum": 0}

    def percentile(fraction):
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "sum": sum(ordered),
        "min": ordered[0],
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


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
    items = expert_items(engine, scenario, world)
    endpoints = TransferEndpoints(source={location: location
                                          for location in range(world)},
                                  target={location: location
                                          for location in range(world)})
    budget = StagingBudget(bytes_per_peer=int(args.budget_mb * (1 << 20)))
    element_size = torch.empty((), dtype=DTYPE).element_size()

    runs = []
    for policy in policies:
        selector = make_selector(policy, endpoints, element_size)
        started = time.perf_counter()
        plans = [plan_transfer(item.target_map, item.source_map, PlanBudget(), replica_selector=selector) for item in items]
        plan_seconds = max_over_ranks(time.perf_counter() - started)

        targets = nan_targets(items, endpoints, device)
        get_accelerator().reset_peak_memory_stats()
        memory_before = get_accelerator().memory_allocated()
        stats, first_seconds = time_execute(requests_for(items, plans, targets), endpoints, budget)
        first_seconds = max(first_seconds)
        peak_extra = max_over_ranks(get_accelerator().max_memory_allocated() - memory_before)
        stats_by_rank = all_ranks(asdict(stats))
        mismatched, unwritten, checked = check_bitwise(items, targets)
        runs.append({
            "plans": plans,
            "targets": targets,
            "record": {
                "scenario": scenario,
                "policy": policy,
                "plan_digests": [plan_digest(plan, endpoints, budget, element_size) for plan in plans],
                "plan_seconds": plan_seconds,
                "first_execute_seconds": first_seconds,
                "peak_extra_allocated_bytes": peak_extra,
                "executor_stats_by_rank": stats_by_rank,
                "bitwise": {
                    "mismatched_elements": int(sum_over_ranks(mismatched)),
                    "unwritten_elements": int(sum_over_ranks(unwritten)),
                    "checked_elements": int(sum_over_ranks(checked)),
                },
                "execute_seconds": [],
                "execute_seconds_by_rank": [],
            },
            "remote_bytes": sum(rank_stats["sent_bytes"] for rank_stats in stats_by_rank),
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
        output["calibration"] = calibrate_links()

    output["runs"] = []
    policies = args.policies.split(",")
    for scenario in args.scenarios.split(","):
        records = run_scenario(engine, scenario, policies, args, device)
        output["runs"].extend(records)
        if dist.get_rank() == 0:
            for record in records:
                bits = record["bitwise"]
                stats = record["executor_stats_by_rank"]
                # The busiest sender and the longest round schedule, not total bytes, separate the selectors.
                max_sent = max(rank_stats["sent_bytes"] for rank_stats in stats)
                max_rounds = max(rank_stats["rounds"] for rank_stats in stats)
                print(f"[{scenario:>10}/{record['policy']:<11}] "
                      f"mismatched={bits['mismatched_elements']} unwritten={bits['unwritten_elements']} "
                      f"max_sent={max_sent / 2**20:.0f} MiB rounds={max_rounds} "
                      f"p50={record['execute'].get('p50', float('nan')) * 1e3:.2f} ms "
                      f"plan={record['plan_seconds'] * 1e3:.1f} ms",
                      flush=True)

    if dist.get_rank() == 0:
        with open(args.output, "w") as stream:
            json.dump(output, stream, indent=2, default=str)
        print(f"wrote {args.output}")
    failed = [
        record for record in output["runs"]
        if record["bitwise"]["mismatched_elements"] or record["bitwise"]["unwritten_elements"]
    ]
    engine.destroy()
    if failed:
        raise SystemExit(f"{len(failed)} runs failed correctness")


if __name__ == "__main__":
    main()
