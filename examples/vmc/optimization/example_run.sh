#Examples of running the optimization protocol from the command line with a config file
export JAX_NUM_CPU_DEVICES=4
uv run ../../../src/vmc/optimization/run.py --config config.yaml
# uv run ../cleaner.py $PWD #clean up files 