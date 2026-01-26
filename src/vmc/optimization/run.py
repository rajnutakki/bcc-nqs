from vmc.optimization.protocols import MCProtocol
import vmc.config.args as args
from vmc.system import BCCHeisenberg
from nets.net import ViTNd

system_class = BCCHeisenberg
network_class = ViTNd
protocol_class = MCProtocol

if __name__ == "__main__":
    # For command line arguments
    parser = args.parser
    system_class.add_arguments(parser)
    network_class.add_arguments(parser)
    protocol_class.add_arguments(parser)
    # Process the command line parameters
    parsed_args = parser.parse_args()
    system_args = system_class.read_arguments(parsed_args)
    print(system_args)
    system = system_class(*system_args)
    network_args = network_class.read_arguments(parsed_args)
    network = network_class(**network_args, system=system)

    # Initialize the protocol
    protocol = protocol_class(
        system, network, vars(parsed_args), compile_step=False, log_mode="fail"
    )
    # Run it
    times, n_parameters, log_file = protocol.run()
