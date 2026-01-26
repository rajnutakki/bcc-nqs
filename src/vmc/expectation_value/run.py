import vmc.config.args as args
from vmc.expectation_value import expectation_value
import time

if __name__ == "__main__":
    parser = args.parser

    parser.add_argument(
        "--directory",
        type=str,
        help="Path to directory containing post.json and post/checkpoint (i.e output of optimization run)",
    )
    parser.add_argument(
        "--n_samples_per_chain",
        type=int,
    )
    parser.add_argument("--n_chains", type=int)
    parser.add_argument("--n_discard_per_chain", type=int)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--observables", type=str, action="append")
    parser.add_argument("--save_type", type=str, default="post")
    parser.add_argument(
        "--final_stage",
        type=int,
        default=0,
        help="What to increase the symmetry stage to to evaluate exp. values, default is 0 (behaves as None)",
    )
    parser.add_argument(
        "--symmetry_stage",
        type=int,
        default=0,
        help="What symmetry stage is being loaded in, default is 0 (behaves as None)",
    )
    parser.add_argument(
        "--save_file_name",
        type=str,
        default="expectation_values",
        help="Name of the file to save the expectation values to, saved as a json",
    )
    parser.add_argument(
        "--little_group_id",
        type=int,
        default=None,
        help="If provided, changes the little_group_id of the loaded system to this value",
    )
    parser.add_argument(
        "--spin_flip_symmetric",
        type=int,
        default=None,
        help="If provided, changes the spin_flip_symmetric of the loaded system to this value",
    )

    args = parser.parse_args()
    print("Arguments:", args)

    print("Computing expectation values...")
    start = time.time()
    results = expectation_value.compute(
        dirname=args.directory,
        n_samples_per_chain=args.n_samples_per_chain,
        n_chains=args.n_chains,
        n_discard_per_chain=args.n_discard_per_chain,
        chunk_size=args.chunk_size,
        observables=args.observables,
        save_type=args.save_type,
        final_stage=args.final_stage,
        symmetry_stage=args.symmetry_stage,
        save_file_name=args.save_file_name,
        little_group_id=args.little_group_id,
        spin_flip_symmetric=args.spin_flip_symmetric,
    )
    end = time.time()
    print("Finished computing expectation values")
    print(f"Time taken {end - start:.1f}s")
