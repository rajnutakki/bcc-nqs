from typing import Sequence

import os
import sys
import time
import argparse

import numpy as np

import jax
import jax.numpy as jnp

import optax

import netket as nk
import netket.experimental as nkx
import nk_extensions as nke
from netket.utils.version_check import module_version

from nk_extensions.callbacks import SaveVariables

import vmc.utils.serialize as saver
from vmc.optimization.utils import process_print, to_sequence, add_module


if module_version("jax") >= (0, 5, 0):
    pass
else:
    pass


class BaseProtocol:
    callbacks = (nk.callbacks.InvalidLossStopping(),)
    solvers = {
        "cholesky": nk.optimizer.solver.cholesky,
        "pinv_smooth": nk.optimizer.solver.pinv_smooth,
        "LU": nk.optimizer.solver.LU,
    }

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--iters", type=int, action="append", help="Number of optimization steps"
        )
        parser.add_argument("--lr", type=float, action="append", help="Learning rate")
        parser.add_argument(
            "--diag_shift",
            type=float,
            action="append",
            help="Initial diagonal shift of schedule",
        )
        parser.add_argument(
            "--diag_shift_factor",
            type=float,
            action="append",
            help="The factor multiplied by diag_shift to give the final diag_shift",
        )
        parser.add_argument(
            "--save_every",
            type=int,
            help="Number of iterations between saving data",
        )
        parser.add_argument(
            "--save_base",
            type=str,
            default="",
            help="File to save optimization results to",
        )
        parser.add_argument(
            "--save_num", type=int, default=0, help="Number to append to save file name"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=-1,
            help="Random seed for initializing state and sampler",
        )
        parser.add_argument(
            "--time_it", type=int, default=0, help="Whether to time the optimization"
        )
        parser.add_argument(
            "--show_progress",
            type=int,
            default=0,
            help="Whether to show progress bar during optimization",
        )
        parser.add_argument(
            "--chunk_size",
            type=int,
            action="append",
            help="Number of samples to compute with at a time (reduces memory requirements), 0 = None",
        )
        parser.add_argument(
            "--lr_factor",
            type=float,
            action="append",
            help="Cosine decay scheduler factor for lr, min learning rate = lr_factor*lr",
        )
        parser.add_argument(
            "--post_iters",
            type=int,
            default=0,
            help="Number of iterations to perform post optimization to choose lowest energy state",
        )
        parser.add_argument(
            "--n_symmetry_stages",
            type=int,
            default=1,
            help="Which symmetry stage to ramp up to, 1 is without any symmetries, ",
        )
        parser.add_argument(
            "--solver",
            type=str,
            default="cholesky",
            help="Which type of solver to invert S or T matrix",
        )
        parser.add_argument(
            "--r", type=float, default=1e-14, help="rtol and rtol_smooth for pinv"
        )
        parser.add_argument(
            "--momentum", type=float, default=None, help="Momentum for SPRING"
        )
        parser.add_argument(
            "--norm_constraint_factor",
            type=float,
            default=None,
            help="Norm constraint factor for SPRING",
        )
        parser.add_argument(
            "--proj_reg",
            type=float,
            default=None,
            help="Projection regularization for SPRING",
        )
        parser.add_argument(
            "--load_base",
            type=str,
            default="",
            help="Folder to load vstate from, no load if empty",
        )

    @staticmethod
    def check_args(args):
        """
        Check and modify the args to have the desired form.
        Returns:
            passed: bool - whether passed the tests
            chunk_size: int or None - the processed chunk_size argument
        """
        if args["chunk_size"] == 0:
            args["chunk_size"] = None

        if isinstance(args["chunk_size"], Sequence) and not isinstance(
            args["chunk_size"], str
        ):
            args["chunk_size"] = [None if x == 0 else x for x in args["chunk_size"]]

        return True

    def __init__(
        self, system, network, args: dict, compile_step=False, log_mode="fail"
    ):
        # Check the arguments
        if not self.check_args(args):
            raise RuntimeError(
                "args failed check, verify save_every is a divisor of iters"
            )
        self.read_args(args)
        self.compile_step = compile_step

        if self.seed == -1:
            self.seed = int(np.random.randint(1, 100000, size=1)[0])
            print(
                f"No seed given, random seed generated = {self.seed}, of type {type(self.seed)}"
            )

        self.system = system
        self.network = network
        if self.norm_constraint_factor is not None:
            print("Using SGD with norm clipping optimizer")
            self.optimizer_t = lambda lr: nke.optimizer.sgd_norm_clipped(
                learning_rate=lr, norm_constraint=self.norm_constraint_factor
            )
        else:
            print("Using standard SGD optimizer")
            self.optimizer_t = nk.optimizer.Sgd
        i = 0
        while True:
            try:
                i += 1
                self.log_file = self.save_base + f"opt{i}"
                self.log = nk.logging.JsonLog(
                    self.log_file,
                    mode=log_mode,
                    write_every=self.save_every,
                    save_params=False,
                )
                break
            except ValueError:  # file exists
                pass

        (
            self.lr,
            self.iters,
            self.lr_factor,
            self.diag_shift,
            self.iters,
            self.diag_shift_factor,
            self.chunk_sizes,
        ) = [
            to_sequence(arg)
            for arg in (
                self.lr,
                self.iters,
                self.lr_factor,
                self.diag_shift,
                self.iters,
                self.diag_shift_factor,
                self.chunk_size,
            )
        ]

        print(f"Number of symmetry stages = {self.n_symmetry_stages}")
        assert self.n_symmetry_stages <= len(system.symmetrizing_functions)
        assert len(self.iters) >= self.n_symmetry_stages
        assert (
            len(self.iters)
            == len(self.lr)
            == len(self.diag_shift)
            == len(self.diag_shift_factor)
            == len(self.lr_factor)
            == len(self.chunk_sizes)
        )
        self.lr_schedulers = [
            optax.cosine_decay_schedule(
                init_value=self.lr[i],
                decay_steps=self.iters[i],
                alpha=self.lr_factor[i],
                exponent=1,
            )
            if self.iters[i] > 0
            else 0.1  # dummy if no iterations
            for i in range(self.n_symmetry_stages)
        ]
        self.diag_shift_schedulers = [
            optax.exponential_decay(
                init_value=self.diag_shift[i],
                transition_steps=self.iters[i],
                decay_rate=self.diag_shift_factor[i],
            )
            if self.iters[i] > 0
            else 0.1  # dummy if no iterations
            for i in range(self.n_symmetry_stages)
        ]
        self.SR_solver = eval("nk.optimizer.solver." + self.solver)
        if self.solver == "pinv_smooth":
            self.SR_solver = self.SR_solver(rtol=self.r, rtol_smooth=self.r)
        # symmetrized networks
        self.nets = [f(network.network) for f in system.symmetrizing_functions]

        print("Chunk sizes: ", self.chunk_sizes)

    def read_args(self, args: dict):
        """
        Set the variables self.{variable_name} = args["{variable_name}"]
        """
        for key, value in args.items():
            setattr(self, key, value)

    def run(self):
        """
        Run the protocol, first performing the optimization and then computing expectation values
        in post-optimization if specified
        """
        times, final_lr, final_diag_shift = self.optimize()
        saver.save(
            system=self.system,
            network=self.network,
            fname=self.save_base + "opt.json",
            save_type="opt",
            symmetry_stage=self.symmetry_stage,
        )
        self.save_vstate(folder="vstate")

        if self.post_iters > 0:
            min_index, min_energy = self.post_optimize(
                lr=final_lr, diag_shift=final_diag_shift
            )

            saver.save(
                system=self.system,
                network=self.network,
                fname=self.save_base + "post.json",
                save_type="post",
                min_index=min_index,
                min_energy=min_energy,
                symmetry_stage=self.symmetry_stage,
            )

        return times, self.vstate.n_parameters, self.log_file

    def save_sampler(self, folder: str = "vstate", sname: str = "sampler"):
        """
        Save the current sampler
        fname: name of the file to save the vstate to, saved to {self.save_base}{folder}/fname.nk
        """
        saver.save_sampler(
            sampler=self.sampler, save_base=self.save_base, folder=folder, sname=sname
        )

    def load_sampler(self, fname: str = "sampler.nk"):
        """
        Load the sampler from a file.
        fname: name of the file to load the vstate from
        """
        return saver.load_sampler(fname=fname)

    def save_vstate(
        self, folder: str = "vstate", vname: str = "vars", sname: str = "sampler"
    ):
        """
        Save the current vstate by saving its variables to {self.save_base}{folder}/fname.mpack
        and sampler to {self.save_base}{folder}/sname.nk

        """
        saver.save_vstate(
            vstate=self.vstate,
            sampler=self.sampler,
            save_base=self.save_base,
            folder=folder,
            vname=vname,
            sname=sname,
        )


class MCProtocol(BaseProtocol):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        """
        Add optimization/post-optimization arguments to the parser.
        These arguments are read into the class instance in __init__.
        """
        BaseProtocol.add_arguments(parser)
        parser.add_argument(
            "--samples_per_rank",
            type=int,
            help="Number of samples on each rank",
        )
        parser.add_argument(
            "--n_chains_per_rank",
            type=int,
            help="Number of MC chains per rank",
        )
        parser.add_argument(
            "--discard_fraction",
            type=float,
            help="Fraction of samples to discard per chain",
        )
        parser.add_argument(
            "--sweep_factor",
            type=int,
            default=1,
            help="Factor for number of MC steps per sweep",
        )
        parser.add_argument(
            "--load_stage",
            type=int,
            default=-1,
            help="Symmetry stage of vstate being loaded in",
        )
        parser.add_argument(
            "--load_stage_diff",
            type=int,
            default=1,
            help="Difference in symmetry stage of new vstate versus old",
        )
        parser.add_argument(
            "--reset_sampler",
            type=int,
            default=0,
            help="Whether to reset the sampler when loading in a vstate",
        )

    def __init__(self, system, network, args: dict, compile_step=True, log_mode="fail"):
        """
        Initialize all the objects for running the optimization protocol specified in args
        """
        super().__init__(system, network, args, compile_step, log_mode)
        # Dependent parameters
        self.n_samples = self.samples_per_rank * len(jax.devices())
        self.n_discard_per_chain = int(
            self.discard_fraction
            * self.samples_per_rank
            // (self.n_chains_per_rank * len(jax.devices()))
        )

        self.sampler_t = system.sampler_t

        self.sampler = self.sampler_t(
            hilbert=system.hilbert_space,
            n_chains_per_rank=self.n_chains_per_rank,
            sweep_size=system.graph.n_nodes * self.sweep_factor,
        )

        # Check maximum number of parameters vs no. of samples to decide whether to use SR or minSR
        params = self.nets[self.n_symmetry_stages - 1].init(
            jax.random.PRNGKey(5),
            system.hilbert_space.random_state(jax.random.PRNGKey(0), size=1),
        )
        max_nparams = nk.jax.tree_size(params)
        print(
            f"Maximum no. of parameters = {max_nparams}, total number of samples = {self.n_samples}"
        )
        use_minSR = max_nparams > self.n_samples
        print(f"Using minSR = {use_minSR}")
        self.hamiltonian = system.hamiltonian.to_jax_operator()
        self.driver_t = lambda opt, dshift, vstate, c_size: nkx.driver.VMC_SR(
            hamiltonian=self.hamiltonian,
            optimizer=opt,
            variational_state=vstate,
            diag_shift=dshift,
            linear_solver_fn=self.SR_solver,
            mode="complex",
            chunk_size_bwd=c_size,
            use_ntk=use_minSR,
            on_the_fly=False,
            momentum=self.momentum,
            proj_reg=self.proj_reg,
        )
        # capture to call self.driver_t(optimizer, diag_shift, variational_state)

    def optimize(self):
        process_print("Running optimization...")
        old_vars = None  # dummy
        self.symmetry_stage = 0
        times = []
        all_start_time = time.time()
        i = 0
        while self.symmetry_stage < self.n_symmetry_stages:
            self.symmetry_stage += 1
            if i == 0 and self.load_base != "":  # Load vstate from file
                self.vstate = self.load_vstate(
                    self.load_base,
                    load_stage=self.load_stage,
                    stage_diff=self.load_stage_diff,
                    new_sampler=self.reset_sampler,
                )
                self.symmetry_stage = (
                    self.load_stage + self.load_stage_diff
                )  # load in vstate whilst increasing symmetry stage
                # self.vstate.chunk_size = self.chunk_sizes[self.symmetry_stage - 1]
            else:
                self.vstate = nk.vqs.MCState(
                    self.sampler,
                    model=self.nets[self.symmetry_stage - 1],
                    n_samples_per_rank=self.samples_per_rank,
                    seed=self.seed,
                    sampler_seed=self.seed,
                    n_discard_per_chain=self.n_discard_per_chain,
                    chunk_size=self.chunk_sizes[self.symmetry_stage - 1],
                )
            self.chunk_size = self.chunk_sizes[self.symmetry_stage - 1]

            print(
                f"Symmetry stage {self.symmetry_stage}/{self.n_symmetry_stages} on process {jax.process_index()}:"
            )

            print("Performing initial thermalization of vstate...")
            self.vstate.sample(n_samples=self.n_samples * 10)
            print(f"<E> = {self.vstate.expect(self.hamiltonian)}")

            if i > 0:  # increasing symmetry stage
                updated_params = add_module(
                    old_params=old_vars["params"],
                    new_params=self.vstate.variables["params"],
                )
                old_vars["params"] = updated_params
                self.vstate.variables = old_vars
                assert old_vars == self.vstate.variables

            optimizer = self.optimizer_t(self.lr_schedulers[self.symmetry_stage - 1])
            driver = self.driver_t(
                optimizer,
                self.diag_shift_schedulers[self.symmetry_stage - 1],
                self.vstate,
                self.chunk_size,
            )

            if self.compile_step:
                process_print("Compiling...")
                start_time = time.time()
                driver.run(
                    n_iter=1,
                    out=self.log,
                    show_progress=self.show_progress,
                    timeit=self.time_it,
                    callback=self.callbacks,
                )
                end_time = time.time()
                process_print(f"Compilation time = {end_time - start_time:.0f}s")

            process_print("Running optimization...")
            start_time = time.time()
            all_start_time = time.time()

            callbacks = self.callbacks
            driver.run(
                n_iter=self.iters[self.symmetry_stage - 1],
                out=self.log,
                show_progress=self.show_progress,
                timeit=self.time_it,
                callback=callbacks,
            )

            self.save_vstate(folder=f"vstate{self.symmetry_stage}")

            if (
                driver.step_count < self.iters[self.symmetry_stage - 1]
            ):  # e.g if simulation stops due to invalidloss callback
                print(
                    f"Optimization not completed, only {driver.step_count}/{self.iters[self.symmetry_stage - 1]} iterations completed, exiting with code 5"
                )
                sys.exit(5)
            old_vars = self.vstate.variables
            end_time = time.time()
            times.append(end_time - start_time)
            process_print(f"Optimization time = {times[-1]:.0f}s")

            i += 1
        all_end_time = time.time()
        times.append(all_end_time - all_start_time)
        process_print("Finished optimization")
        final_lr = self.lr_schedulers[-1](driver.step_count)
        final_diag_shift = self.diag_shift_schedulers[-1](driver.step_count)

        return times, final_lr, final_diag_shift

    def post_optimization(
        self,
        post_iters: int,
        lr: float,
        diag_shift: float,
        momentum: float,
        save_base: str,
        chunk_size,
        optimizer_t,
        SR_solver,
        old_vstate,
        callbacks,
    ):
        """
        Run post_iters optimization steps with hyperparameters specified in arguments.
        The network variables at each iteration are saved and the iteration which had the minimum energy is determined, returning the index corresponding to this iteration
        """
        process_print(f"Performing {post_iters} additional optimization steps...")
        start = time.time()
        # Make directory for saving vstates
        post_save_base = save_base + "post/"
        os.makedirs(post_save_base, exist_ok=False)  # raises error if already exists
        callbacks = (
            SaveVariables(save_every=1, file_prefix=post_save_base),
        ) + callbacks

        # # New sampler for different number of chains
        # sampler = sampler_t(
        #     system.hilbert_space,
        #     n_chains_per_rank=n_chains_per_rank,
        #     sweep_size=system.graph.n_nodes,
        # )
        # # New vstate with new sampler
        # vstate = nk.vqs.MCState(
        #     sampler,
        #     model=old_vstate.model,
        #     n_samples=old_vstate.n_samples,
        #     seed=seed,
        #     n_discard_per_chain=old_vstate.n_samples
        #     // (
        #         n_chains_per_rank * jax.process_count()
        #     ),  # = n_samples_per_chain across all processes
        #     chunk_size=chunk_size,
        # )
        vstate = old_vstate
        vstate.variables = old_vstate.variables
        print("Sampling parameters:")
        print(f"n_chains = {vstate.sampler.n_chains}")
        print(f"n_samples = {vstate.n_samples}")
        print(f"n_discard_per_chain = {vstate.n_discard_per_chain}")
        optimizer = optimizer_t(lr)
        log = nk.logging.RuntimeLog()
        gs = nkx.driver.VMC_SR(
            self.hamiltonian,
            optimizer,
            linear_solver_fn=SR_solver,
            diag_shift=diag_shift,
            variational_state=vstate,
            chunk_size_bwd=chunk_size,
            momentum=momentum,
            use_ntk=True,
            on_the_fly=False,
        )
        gs.run(
            n_iter=post_iters,  # run optimization for post_iters steps
            out=log,
            callback=callbacks,
        )
        # Now find lowest energy state
        energies = np.real(log["Energy"]["Mean"])
        print(f"Final {post_iters} energies = {energies}")
        min_index = np.arange(len(energies))[energies == min(energies)][0]
        print(f"min_index={min_index}")
        end = time.time()
        print("Finished additional optimization steps")
        print(f"Post-processing time = {end - start:.0f}s")
        return int(min_index), min(energies)

    def post_optimize(self, lr: float, diag_shift: float):
        min_index, min_energy = self.post_optimization(
            post_iters=self.post_iters,
            lr=lr,
            diag_shift=diag_shift,
            momentum=self.momentum,
            save_base=self.save_base,
            chunk_size=self.chunk_size,
            optimizer_t=self.optimizer_t,
            SR_solver=self.SR_solver,
            old_vstate=self.vstate,
            callbacks=self.callbacks,
        )
        return min_index, min_energy

    def load_and_postoptimize(self):
        self.vstate = self.load_vstate(
            self.load_base, load_stage=self.load_stage, stage_diff=0
        )
        self.symmetry_stage = self.load_stage
        print(f"Loaded in vstate at symmetry stage {self.symmetry_stage}")
        self.chunk_size = self.chunk_sizes[self.symmetry_stage - 1]
        min_index, min_energy = self.post_optimize(
            lr=self.lr_schedulers[self.symmetry_stage - 1],
            diag_shift=self.diag_shift_schedulers[self.symmetry_stage - 1],
        )
        saver.save(
            system=self.system,
            network=self.network,
            fname=self.save_base + "post.json",
            save_type="post",
            min_index=min_index,
            min_energy=min_energy,
            symmetry_stage=self.symmetry_stage,
        )

    def test_load(self):
        self.vstate = self.load_vstate(
            self.load_base, load_stage=self.load_stage, increase_stage=True
        )
        print("Testing loaded vstate...")
        print("Computing log value ...")
        print(
            self.vstate.log_value(
                jnp.array(
                    self.system.hilbert_space.random_state(
                        jax.random.PRNGKey(0), size=5
                    )
                )
            )
        )
        print("Sampling...")
        samples = self.vstate.sample(chain_length=1)
        print(samples)
        optimizer = self.optimizer_t(self.lr_schedulers[1])
        driver = self.driver_t(
            optimizer,
            self.diag_shift_schedulers[1],
            self.vstate,
            self.chunk_size,
        )
        print("Performing one optimization step...")
        driver.run(n_iter=1)
        print("Done testing loaded vstate")

    def load_vstate(
        self,
        load_base: str,
        load_stage: int = -1,
        vname: str = "vars",
        sname: str = "sampler",
        stage_diff: int = 0,
        new_sampler=False,
    ):
        """
        Load a saved vstate, by loading its variables from {load_base}fname.mpack
        and sampler from {load_base}sname.nk.
        Resets the vstate to its zeroth symmetrization stage, whilst loading in with parameters
        formatted according to load_stage
        """
        self.chunk_size = self.chunk_sizes[load_stage - 1 + stage_diff]

        resume_vstate, self.sampler = saver.load_vstate(
            sampler=self.sampler,
            nets=self.nets,
            samples_per_rank=self.samples_per_rank,
            seed=self.seed,
            n_discard_per_chain=self.n_discard_per_chain,
            chunk_size=self.chunk_size,
            save_base=self.save_base,
            load_base=load_base,
            load_stage=load_stage,
            vname=vname,
            sname=sname,
            stage_diff=stage_diff,
            new_sampler=new_sampler,
        )
        return resume_vstate
