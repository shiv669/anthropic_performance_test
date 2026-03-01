"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots, vliw=False):
        """
        - Pack ALU ops only when there are NO RAW/WAW hazards
        - Writes become visible only after cycle ends
        """

        instrs = []

        current_alu_bundle = []
        current_valu_bundle = []
        writes_in_cycle = set()

        def flush_alu():
            nonlocal current_alu_bundle, current_valu_bundle, writes_in_cycle
            if current_alu_bundle:
                instrs.append({"alu": current_alu_bundle})
            if current_valu_bundle:
                instrs.append({"valu": current_valu_bundle})
            current_alu_bundle = []
            current_valu_bundle = []
            writes_in_cycle = set()

        for engine, slot in slots:
            if engine == "alu":
                _, dest, src1, src2 = slot

                #WAW
                if dest in writes_in_cycle:
                    flush_alu()

                #RAW
                if src1 in writes_in_cycle or src2 in writes_in_cycle:
                    flush_alu()

                current_alu_bundle.append(slot)
                writes_in_cycle.add(dest)

            # Slot limit
                if len(current_alu_bundle) == SLOT_LIMITS["alu"]:
                    flush_alu()

            elif engine == "valu":
                _, dest, src1, src2 = slot

                #WAW
                if dest in writes_in_cycle:
                    flush_alu()

                #RAW
                if src1 in writes_in_cycle or src2 in writes_in_cycle:
                    flush_alu()

                current_valu_bundle.append(slot)
                writes_in_cycle.add(dest)

            # Slot limit
                if len(current_valu_bundle) == SLOT_LIMITS["valu"]:
                    flush_alu()

            elif engine == "load":
                #finish an alu bundle before load
                flush_alu()

                #emit load as its own cycle
                instrs.append({"load": [slot]})

                #reset dependency tracking
                writes_in_cycle = set()

                #record that this cycle writes to dest
                if slot[0] in ("load","const"):
                    dest = slot[1]
                    writes_in_cycle.add(dest)

                
                
            else:
            # Non-ALU ops are barriers
                flush_alu()
                instrs.append({engine: [slot]})
                writes_in_cycle = set()

        flush_alu()
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            # debug removed for performance

        return slots

    def build_kernel(
    self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero = self.scratch_const(0)
        one = self.scratch_const(1)
        two = self.scratch_const(2)

        self.add("flow", ("pause",))

        body = []

        # Vector scratch registers (VLEN-width for SIMD loads/stores)
        tmp_idx_v = self.alloc_scratch("tmp_idx_v", VLEN)
        tmp_val_v = self.alloc_scratch("tmp_val_v", VLEN)
        tmp_node_v = self.alloc_scratch("tmp_node_v", VLEN)
        tmp_addr_v = self.alloc_scratch("tmp_addr_v", VLEN)

        # Helper registers for per-lane compute
        tmp1_v = self.alloc_scratch("tmp1_v", VLEN)
        tmp_node_addr = self.alloc_scratch("tmp_node_addr")

        for r in range(rounds):
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)

                
                # Load VLEN indices at once
                body.append(("alu", ("+", tmp_addr_v, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("vload", tmp_idx_v, tmp_addr_v)))

                # Load VLEN values at once
                body.append(("alu", ("+", tmp_addr_v, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("vload", tmp_val_v, tmp_addr_v)))

                # Load all tree node values (scatter load per lane)
                for k in range(VLEN):
                    idx = i + k
                    if idx >= batch_size:
                        continue

                    body.append(("alu", ("+", tmp_node_addr, self.scratch["forest_values_p"], tmp_idx_v + k)))
                    body.append(("load", ("load", tmp_node_v + k, tmp_node_addr)))

                
                for k in range(VLEN):
                    idx = i + k
                    if idx >= batch_size:
                        continue

                    # XOR with node value
                    body.append(("alu", ("^", tmp_val_v + k, tmp_val_v + k, tmp_node_v + k)))
                    body.extend(self.build_hash(tmp_val_v + k, tmp1, tmp2, r, idx))

                    # Branch logic (modulo select)
                    body.append(("alu", ("%", tmp1, tmp_val_v + k, two)))
                    body.append(("alu", ("==", tmp1, tmp1, zero)))
                    body.append(("flow", ("select", tmp3, tmp1, one, two)))
                    body.append(("alu", ("*", tmp_idx_v + k, tmp_idx_v + k, two)))
                    body.append(("alu", ("+", tmp_idx_v + k, tmp_idx_v + k, tmp3)))

                    # Bounds check
                    body.append(("alu", ("<", tmp1, tmp_idx_v + k, self.scratch["n_nodes"])))
                    body.append(("flow", ("select", tmp_idx_v + k, tmp1, tmp_idx_v + k, zero)))

                
                # Store VLEN indices at once
                body.append(("alu", ("+", tmp_addr_v, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr_v, tmp_idx_v)))

                # Store VLEN values at once
                body.append(("alu", ("+", tmp_addr_v, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr_v, tmp_val_v)))

        self.instrs.extend(self.build(body))
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
